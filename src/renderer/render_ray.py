import torch
import torch.nn.functional as F
import numpy as np
import time
import os
from tqdm import tqdm
import imageio

import raymarching

from src.utils.training_utils import batchify, batchify_func
from src.utils.loss_utils import to8b


def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d

def prepare_rays(args, H, W, K, pose, target, trainVGG, i, start, N_rand, target_mask = None, cam_info_others = None):

    rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)
    dw = None
    if args.mask_sample:
        if trainVGG:
            AssertionError("Not implemented yet")

        assert(target_mask is not None, "target_mask is None")

        nsampled_rays = 0
        mask_sample_ratio = 0.5
        coord_list = []
        nrays = N_rand

        coords_full = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)
        coords_full = torch.reshape(coords_full, [-1,2])
        target_mask = target_mask[..., 0]

        while nsampled_rays < nrays:
            n_mask = int((nrays - nsampled_rays) * mask_sample_ratio)
            n_bg = (nrays - nsampled_rays) - n_mask
            # sample rays on mask
            coord_mask =  torch.nonzero(target_mask == 1)
            import pdb
            pdb.set_trace()
            
            coord_mask = coord_mask[torch.randint(0, coord_mask.shape[0], size=(n_mask,), device=coords_full.device )]
            coord = coords_full[torch.randint(0, len(coords_full), size = (n_bg,), device = coords_full.device)].long()
            ''' debug
            # cv2.imwrite('debug/sample_mask_rays_full.png',(target_mask==1).cpu().numpy()*255)
            # cv2.imwrite('debug/target_mask.png',(target_mask).cpu().numpy()*255)
            # sample_points = torch.zeros_like(target_mask)
            # sample_points[coord_mask[:,0], coord_mask[:,1]] = 1
            # cv2.imwrite('debug/sample_mask_rays.png',(sample_points).cpu().numpy()*255)
            # sample_points = torch.zeros_like(target_mask)
            # sample_points[coord[:,0], coord[:,1]] = 1
            # cv2.imwrite('debug/sample_mask_rays_others.png',(sample_points).cpu().numpy()*255)
            '''


            coord = torch.cat([coord_mask, coord], dim=0)
            # coord = np.concatenate([coord_mask, coord], axis=0)


            coord_list.append(coord)
            nsampled_rays += len(coord)

        coord = torch.cat(coord_list, dim=0)

        select_coords = torch.reshape(coord, [-1, 2]).long()
    else:
        if trainVGG: # get a cropped img (dw,dw) to train vgg
            strides = args.vgg_strides + i%3 - 1
            
            # args.vgg_strides-1, args.vgg_strides, args.vgg_strides+1
            dw = int(max(20, min(40, N_rand ** 0.5 )))
            vgg_min_border = 10
            strides = min(strides, min(H-vgg_min_border,W-vgg_min_border)/dw)
            strides = int(strides)

            coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)
            if True:
                target_grey = torch.mean(torch.abs(target-args.white_bkgd), dim=-1, keepdim=True) # H,W,1
                img_wei = coords.to(torch.float32) * target_grey
                center_coord = torch.sum(img_wei, dim=(0,1)) / torch.sum(target_grey)
                center_coord = center_coord.cpu().numpy()
                # add random jitter
                random_R = dw*strides / 2.0
                # mean and standard deviation: center_coord, random_R/3.0, so that 3sigma < random_R
                random_x = np.random.normal(center_coord[1], random_R/3.0) - 0.5*dw*strides
                random_y = np.random.normal(center_coord[0], random_R/3.0) - 0.5*dw*strides
            else:
                random_x = np.random.uniform(low=vgg_min_border + 0.5*dw*strides, high= W - 0.5*dw*strides - vgg_min_border) - 0.5*dw*strides
                random_y = np.random.uniform(low=vgg_min_border + 0.5*dw*strides, high= W - 0.5*dw*strides - vgg_min_border) - 0.5*dw*strides
            
            offset_w = int(min(max(vgg_min_border, random_x), W - dw*strides - vgg_min_border))
            offset_h = int(min(max(vgg_min_border, random_y), H - dw*strides - vgg_min_border))

            coords_crop = coords[offset_h:offset_h+dw*strides:strides,offset_w:offset_w+dw*strides:strides,:]

            select_coords = torch.reshape(coords_crop, [-1, 2]).long()
        else:
            if i < args.precrop_iters:
                dH = int(H//2 * args.precrop_frac)
                dW = int(W//2 * args.precrop_frac)
                coords = torch.stack(
                    torch.meshgrid(
                        torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                        torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                    ), -1)
                if i == start:
                    print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")                
            else:
                coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

            coords = torch.reshape(coords, [-1,2])  # (H * W, 2)

            # select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
            select_inds = torch.randint(
                    0, coords.shape[0], size=(N_rand,), device=coords.device
                    # 0, coords.shape[0], size=(N_rand,), device='cpu'
                )
        
            select_coords = coords[select_inds].long()  # (N_rand, 2)

    rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
    rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
    batch_rays = torch.stack([rays_o, rays_d], 0)
    target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
    if args.use_mask:
        target_mask = target_mask[select_coords[:, 0], select_coords[:, 1]]
    else:
        target_mask = None

    if args.dataset_type == 'dryice':
        bg_color = cam_info_others["bk_img"].copy()
        bg_color = torch.tensor(bg_color, dtype=torch.float32, device=batch_rays.device) ## todo: to cuda in dataloader to speedup
        bg_color = bg_color[select_coords[:, 0], select_coords[:, 1]]
    else:
        if args.use_random_bg:
            bg_color = torch.rand_like(target_s)
            target_s = target_s * target_mask + bg_color * (1 - target_mask)
        else:
            bg_color = 0

    return batch_rays, target_s, dw, target_mask, bg_color, select_coords


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d


def render(H, W, K, model, N_samples = 64, chunk=1024*32, rays=None, c2w=None, netchunk = 1024*64,
                  ndc=False, # only for forward facing scene
                  near=0., far=1.,
                  time_step=None, bkgd_color=None,
                  cuda_ray = False,
                  perturb = 0.
                  ):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays
    sh = rays_d.shape # [..., 3]
    
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()
    
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    
    if time_step != None:
        time_step = time_step.expand(list(rays.shape[0:-1]) + [1])
        # (ray origin, ray direction, min dist, max dist, t)
        rays = torch.cat([rays, time_step], dim=-1)


    all_ret = {}
    for i in range(0, rays.shape[0], chunk):
        if cuda_ray:
            ret = render_rays_cuda(rays[i:i+chunk], model, chunk = netchunk, perturb=perturb)
        else:
            ret = render_rays(rays[i:i+chunk], model, N_samples = N_samples, perturb = perturb, chunk = netchunk)
            
        
        # merge results   
        for k in ret:
            if torch.is_tensor(ret[k]):
                if k not in all_ret:
                    all_ret[k] = []
                    
                if model.training:
                    all_ret[k].append(ret[k])
                else:
                    all_ret[k].append(ret[k].detach())
            else:
                assert(isinstance(ret[k], int))
                if k not in all_ret:
                    all_ret[k] = 0
                all_ret[k] += ret[k]
                
                
    for k in all_ret:
        if isinstance(all_ret[k], list):
            all_ret[k] = torch.cat(all_ret[k], 0)
        
 

    num_rays = None
    for k in all_ret:
       
        # todo:: fix this code..
        if k in ['num_points', 'num_points_static', 'num_points_dynamic', 'raw', 'raw_static', 'raw_dynamic', 'gradients', 'hessians', 'samples_xyz_static', 'samples_xyz_dynamic', 'smoke_vel', 'smoke_weights', 'rays_id']:
            num_rays = rays_d.reshape(-1,1).shape[0]
            continue
        try:
            k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
            all_ret[k] = torch.reshape(all_ret[k], k_sh)
        except:
            import pdb
            pdb.set_trace()

    if num_rays is not None:
        all_ret['num_rays'] = num_rays

    if bkgd_color is not None:
        if torch.is_tensor(bkgd_color):
            torch_bkgd_color = bkgd_color
        else:
            torch_bkgd_color = torch.Tensor(bkgd_color.copy())
        # rgb map for model: fine, coarse, merged, dynamic_fine, dynamic_coarse
        for _i in ['_map', '0', 'h1', 'h10', 'h2', 'h20']: #  add background for synthetic scenes, for image-based supervision
            rgb_i, acc_i = 'rgb'+_i, 'acc'+_i
            if (rgb_i in all_ret) and (acc_i in all_ret):
                # all_ret[rgb_i] = all_ret[rgb_i] + torch_bkgd_color*(1.-all_ret[acc_i][..., None])
                all_ret[rgb_i] = all_ret[rgb_i] + torch_bkgd_color*(1.-all_ret[acc_i].reshape([*all_ret[rgb_i].shape[:-1], 1]))

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]

def _get_minibatch_jacobian(y, x):
    """Computes the Jacobian of y wrt x assuming minibatch-mode.
    Args:
      y: (N, ...) with a total of D_y elements in ...
      x: (N, ...) with a total of D_x elements in ...
    Returns:
      The minibatch Jacobian matrix of shape (N, D_y, D_x)
    """
    assert y.shape[0] == x.shape[0]
    y = y.view(y.shape[0], -1)
    # Compute Jacobian row by row.
    jac = []
    for j in range(y.shape[1]):
        dy_j_dx = torch.autograd.grad(
            y[:, j],
            x,
            torch.ones_like(y[:, j], device=y.get_device()),
            retain_graph=True,
            create_graph=True,
        )[0].view(x.shape[0], -1)
        jac.append(torch.unsqueeze(dy_j_dx, 1))
    return jac

def raw2outputs_hybrid_neus(raw_list, z_vals, rays_d, raw_noise_std=0, cos_anneal_ratio = 1.0, inv_s = None, pytest=False, remove99=False, valid_rays=None):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw_list: a list of tensors in shape [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: a list of [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
        extras: extra information for neus
    """
    neus_render = True if inv_s is not None else False
    extras = {}

    sample_dists = 2.0 / 64

    nerf_z_vals = z_vals[0]

    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = nerf_z_vals[...,1:] - nerf_z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]
    # dists = torch.cat([dists, torch.Tensor([sample_dists]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]


    noise = 0.
    alpha_list = []
    color_list = []
    
    nerf_raw = raw_list[0]
    if raw_noise_std > 0.:
        noise = torch.randn(nerf_raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(42)
            noise = np.random.rand(*list(nerf_raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)
    
    alpha = raw2alpha(nerf_raw[...,3] + noise, dists)  # [N_rays, N_samples]

    if remove99:
        alpha = torch.where(alpha > 0.99, torch.zeros_like(alpha), alpha)
    rgb = torch.sigmoid(nerf_raw[..., :3]) # [N_rays, N_samples, 3]

    if valid_rays is not None:
        alpha = alpha * valid_rays[...,None]
        rgb = rgb * valid_rays[...,None,None]

    alpha_list += [alpha]
    color_list += [rgb]



    assert(neus_render)

    neus_z_vals = z_vals[-1]

    batch_size, n_samples = neus_z_vals.shape

    dists = neus_z_vals[...,1:] - neus_z_vals[...,:-1]
    # dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]
    dists = torch.cat([dists, torch.Tensor([sample_dists]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]


    noise = 0.
    
    neus_raw = raw_list[-1]
    if raw_noise_std > 0.:
        noise = torch.randn(neus_raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(42)
            noise = np.random.rand(*list(neus_raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)
    
    sdf = neus_raw[...,3] + noise
    gradients = neus_raw[...,4:7]
    
    sdf = sdf.reshape(batch_size, n_samples)
    cos_anneal_ratio = cos_anneal_ratio


    # true_cos = (rays_d[...,None,:] * F.normalize(gradients.reshape(batch_size, n_samples,3), dim = -1, p = 2)).sum(-1, keepdim=True)
    true_cos = (rays_d * gradients.reshape(batch_size, n_samples,3)).sum(-1, keepdim=True)

    # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
    # the cos value "not dead" at the beginning training iterations, for better convergence.
    iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

    # Estimate signed distances at section points
    estimated_next_sdf = sdf + iter_cos.squeeze(-1) * dists * 0.5
    estimated_prev_sdf = sdf - iter_cos.squeeze(-1) * dists * 0.5
    prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
    next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

    p = prev_cdf - next_cdf
    c = prev_cdf

    alpha = ((p + 1e-5) / (c + 1e-5)).reshape(batch_size, n_samples).clamp(0.0, 1.0)

    # if remove99:
        # alpha = torch.where(alpha > 0.99, torch.zeros_like(alpha), alpha) #yiming:: remove this for neus

    # rgb = torch.sigmoid(raw[..., :3]) # [N_rays, N_samples, 3] # already done

    rgb = neus_raw[..., :3].reshape(batch_size, n_samples, 3)

    if valid_rays is not None:
        alpha = alpha * valid_rays[...,None]
        rgb = rgb * valid_rays[...,None,None]

    alpha_list += [alpha]
    color_list += [rgb]


    extras['gradients'] = gradients.reshape(batch_size, n_samples,3)

    if neus_raw.shape[-1] > 7:
        hessians = neus_raw[7:]
        extras['hessians'] = hessians

    z_vals_all = torch.cat(z_vals, dim=1)
    alpha_all = torch.cat(alpha_list, dim=1) # [N_rays, N_samples * 2]
    color_all = torch.cat(color_list, dim=1) # [N_rays, N_samples * 2, 3]

    z_vals_all_sorted, index = torch.sort(z_vals_all, dim=-1) 

    # sort alpha and color
    # index = torch.stack([torch.arange(n_samples).reshape(1,n_samples),torch.arange(n_samples).reshape(1,n_samples) + 64], dim = -1).reshape(1,-1).expand(batch_size,-1)



    alpha_all_sorted = torch.gather(alpha_all, dim=1, index=index)
    color_all_sorted = torch.gather(color_all, dim=1, index=index.unsqueeze(-1).expand(-1,-1,3))

    weights_all = alpha_all_sorted * torch.cumprod(
                torch.cat([torch.ones([batch_size, 1]), 1. - alpha_all_sorted + 1e-7], -1), -1)[:, :-1]
    
    weights_nerf_from_all = weights_all[index < nerf_z_vals.shape[1]] ## since original nerf_weights is sorted by depth, todo::check this

    weights_nerf = alpha_list[0] * torch.cumprod(
                torch.cat([torch.ones([batch_size, 1]), 1. - alpha_list[0] + 1e-7], -1), -1)[:, :-1]
    
    weights_neus = alpha_list[1] * torch.cumprod(
                torch.cat([torch.ones([batch_size, 1]), 1. - alpha_list[1] + 1e-7], -1), -1)[:, :-1]

    densTiStack = torch.stack([1.-alpha for alpha in alpha_list], dim=-1) 
    # # [N_rays, N_samples, N_raws]
    # densTi = torch.prod(densTiStack, dim=-1, keepdim=True) 
    # # [N_rays, N_samples]
    # densTi_all = torch.cat([densTiStack, densTi], dim=-1) 
    # [N_rays, N_samples, N_raws + 1] 
    # Ti_all = torch.cumprod(densTi_all + 1e-10, dim=-2) # accu along samples
    # Ti_all = Ti_all / (densTi_all + 1e-10)
    # [N_rays, N_samples, N_raws + 1], exclusive
    # weights_list = [alpha * Ti_all[...,-1] for alpha in alpha_list] # a list of [N_rays, N_samples]
    # self_weights_list = [alpha_list[alpha_i] * Ti_all[...,alpha_i] for alpha_i in range(len(alpha_list))] # a list of [N_rays, N_samples]


    rgb_map  = torch.sum(weights_all[..., None] * color_all_sorted, dim = -2) # [N_rays, 3]
    # Sum of weights along each ray. This value is in [0, 1] up to numerical error.
    acc_map = torch.sum(weights_all, dim = -1) # [N_rays]

    rgb_nerf_map = torch.sum(weights_nerf[..., None] * color_list[0], dim = -2) # [N_rays, 3]
    acc_nerf_map = torch.sum(weights_nerf, dim = -1) # [N_rays]

    rgb_neus_map = torch.sum(weights_neus[..., None] * color_list[1], dim = -2) # [N_rays, 3]
    acc_neus_map = torch.sum(weights_neus, dim = -1) # [N_rays]

    rgb_map_stack = torch.stack([rgb_nerf_map, rgb_neus_map], dim=-1)
    acc_map_stack = torch.stack([acc_nerf_map, acc_neus_map], dim=-1)


    # _, rgb_map_stack = weighted_sum_of_samples(self_weights_list, color_list)
    # _, acc_map_stack = weighted_sum_of_samples(self_weights_list, None, 1)

    # Estimated depth map is expected distance.
    # Disparity map is inverse depth.
    depth_map = torch.sum(weights_all * z_vals_all_sorted, dim = -1) # [N_rays]
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / acc_map)
    # alpha * Ti
    # weights = (1.-densTi)[...,0] * Ti_all[...,-1] # [N_rays, N_samples]
    
    # weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    # rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]
    # depth_map = torch.sum(weights * z_vals, -1)
    # acc_map = torch.sum(weights, -1)

    # weights_nerf: used for nerf up sample 
    return rgb_map, disp_map, acc_map, weights_nerf, depth_map, None, rgb_map_stack, acc_map_stack, extras
    # return rgb_map, disp_map, acc_map, weights_nerf_from_all, depth_map, None, rgb_map_stack, acc_map_stack, extras

def render_rays_cuda(ray_batch,
                model,
                chunk = 1024*64,
                perturb=0.
                ):

    N_rays = ray_batch.shape[0]
    rays_o, rays_d, rays_t = ray_batch[:,0:3], ray_batch[:,3:6], ray_batch[:,-1:] # [N_rays, 3] each


    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]



    ## ray marching
    density_grid_static = model.occupancy_grid_static
    density_grid_dynamic = model.occupancy_grid_dynamic

    t = torch.floor(rays_t * density_grid_dynamic.time_size).clamp(min=0, max=density_grid_dynamic.time_size - 1).long()
    aabb = model.bbox_model.world_bbox
    aabb = aabb.reshape(6).contiguous() # [1, 1, 6]
    aabb_near, aabb_far = raymarching.near_far_from_aabb(rays_o, rays_d, aabb)    
    aabb_near = aabb_near.unsqueeze(-1).clamp(1e-6, 1e6)
    aabb_far = aabb_far.unsqueeze(-1).clamp(1e-6, 1e6)
    
    xyzs_dynamic, dirs_dynamic, ts_dynamic, rays_dynamic, rays_dynamic_id = raymarching.march_rays_train(rays_o, rays_d, density_grid_dynamic.bound, False, density_grid_dynamic.density_bitfield[t], density_grid_dynamic.cascade, density_grid_dynamic.grid_size, aabb_near, aabb_far, perturb, density_grid_dynamic.dt_gamma, density_grid_dynamic.max_steps) # contract = False
    xyzs_static, dirs_static, ts_static, rays_static, rays_static_id = raymarching.march_rays_train(rays_o, rays_d, density_grid_static.bound, False, density_grid_static.density_bitfield, density_grid_static.cascade, density_grid_static.grid_size, aabb_near, aabb_far, perturb, density_grid_static.dt_gamma, density_grid_static.max_steps) # contract = False

    num_points_dynamic = xyzs_dynamic.shape[0]
    num_points_static = xyzs_static.shape[0]

 

    pts_static = torch.cat([xyzs_static, dirs_static], dim = -1) # [num_points_static, 3 + 3]

    rays_t_bc = rays_t[0].reshape(1,1).expand(xyzs_dynamic.shape[0], 1)
    pts_dynamic = torch.cat([xyzs_dynamic, dirs_dynamic, rays_t_bc], dim = -1) # [num_points_dynamic, 3 + 3 + 1]


    def get_raw(fn, pts_dynamic, pts_static):
        static_raw, smoke_raw = None, None

        ## get static raw
        # static_raw = fn.forward_static(pts_static)
        # static_raw = batchify(fn.forward_static, chunk)(pts_static)
        static_raw = batchify_func(fn.forward_static, chunk, not fn.training)(pts_static)


        ## get smoke raw
        ## todo:: not warp pos for rgb
        orig_pos, orig_viewdir, orig_t = torch.split(pts_dynamic, [3, 3, 1], -1)
        pts_dynamic = torch.cat([orig_pos, orig_t], dim = -1)
        # smoke_raw = fn.forward_dynamic(pts_dynamic)
        # smoke_raw = batchify(fn.forward_dynamic, chunk)(pts_dynamic)
        smoke_raw = batchify_func(fn.forward_dynamic, chunk, not fn.training)(pts_dynamic)


        return smoke_raw, static_raw # [N_rays, N_samples, 4], [N_rays, N_samples, 4]

    C_smokeRaw, C_staticRaw = get_raw(model, pts_dynamic, pts_static) 

    smoke_color = C_smokeRaw[..., :3]
    smoke_color = torch.sigmoid(smoke_color)
    smoke_density = C_smokeRaw[..., 3:] 
    smoke_density = torch.relu(smoke_density)
    

    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)
    smoke_alpha = raw2alpha(smoke_density, ts_dynamic[:,1].unsqueeze(-1))  # [N_rays, N_samples]
    

    static_color = C_staticRaw[..., :3]
    static_sdf = C_staticRaw[..., 3:4]
    static_gradients = C_staticRaw[..., 4:7].reshape(-1,3)

   
    ## density/sdf to alpha
    # add variance
    inv_s = model.get_deviation()         # Single parameter
    inv_s = inv_s.expand(num_points_static, 1).squeeze(-1)

    cos_anneal_ratio = model.get_cos_anneal_ratio()

    assert(dirs_static.shape == static_gradients.shape)

    true_cos = (dirs_static * static_gradients).sum(-1, keepdim=True)

    # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
    # the cos value "not dead" at the beginning training iterations, for better convergence.
    iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

    dists = ts_static[:,1]

    # Estimate signed distances at section points
    estimated_next_sdf = static_sdf.squeeze(-1) + iter_cos.squeeze(-1) * dists * 0.5
    estimated_prev_sdf = static_sdf.squeeze(-1) - iter_cos.squeeze(-1) * dists * 0.5
    prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
    next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

    p = prev_cdf - next_cdf
    c = prev_cdf

    static_alpha = ((p + 1e-5) / (c + 1e-5)).reshape(-1, 1).clamp(0.0, 1.0)
    
    weights, weights_sum, depth, image = raymarching.composite_rays_train_hybrid(smoke_alpha, static_alpha, smoke_color, static_color, ts_dynamic, ts_static, rays_dynamic, rays_static)
    
    ## todo:: check depth
    depth_normalized = depth / (weights_sum + 1e-6)
    pred_depth = (depth_normalized.unsqueeze(-1) - near) / (far - near + 1e-6)
    disp_map = pred_depth.clamp(0, 1)

    disp_map = depth

    rgb_map = image

    ret = {'rgb_map' : rgb_map.reshape(-1, 3), 'disp_map' : disp_map.reshape(-1,1), 'acc_map' : weights_sum.reshape(-1,1)}


    ret['gradients'] = static_gradients 

    if C_staticRaw.shape[-1] > 7:
        hessians = C_staticRaw[7:]
        ret['hessians'] = hessians

    dynamic_weights, dynamic_weights_sum, dynamic_depth, dynamic_image = raymarching.composite_rays_train(smoke_alpha, smoke_color, ts_dynamic, rays_dynamic, True) # alpha_mode = True
    static_weights, static_weights_sum, static_depth, static_image = raymarching.composite_rays_train_neus(static_alpha, static_color, ts_static, rays_static)


    # todo: make name more clear
    rgbh2_map = dynamic_image # dynamic
    acch2_map = dynamic_weights_sum # dynamic
    rgbh1_map = static_image # staitc
    acch1_map = static_weights_sum # staitc

    
    ret['rgbh1'] = rgbh1_map
    ret['acch1'] = acch1_map
    ret['rgbh2'] = rgbh2_map
    ret['acch2'] = acch2_map

    ret['raw'] = C_smokeRaw
    ret['raw_static'] = C_staticRaw
    ret['num_points_static'] = num_points_static
    ret['num_points_dynamic'] = num_points_dynamic

    ret['samples_xyz_static'] = xyzs_static
    ret['samples_xyz_dynamic'] = xyzs_dynamic

    # if smoke_vel != None:
    #     ret['smoke_vel'] = smoke_vel
    # ret['smoke_weights'] = dynamic_weights

    # ret['rays_id'] = rays_dynamic_id
    # import pdb
    # pdb.set_trace()
    # ret['rays_id'] = torch.arange(N_rays).unsqueeze(-1).long().expand(N_rays, N_samples).reshape(-1,1)


    ret['num_points'] = num_points_static + num_points_dynamic

    return ret

def render_rays(ray_batch,
                model,
                chunk = 1024*64,
                N_samples=64,
                perturb=0.,
                raw_noise_std=0.
                ):
    N_rays = ray_batch.shape[0]
    rays_o, rays_d, rays_t = ray_batch[:,0:3], ray_batch[:,3:6], ray_batch[:,-1:] # [N_rays, 3] each


    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    valid_rays = None
    
    aabb = model.bbox_model.world_bbox
    aabb = aabb.reshape(6).contiguous() # [1, 1, 6]
    aabb_near, aabb_far = raymarching.near_far_from_aabb(rays_o, rays_d, aabb)

    valid_rays = (aabb_near < aabb_far) & (aabb_near < 1e6) & (aabb_far < 1e6)

    near = aabb_near.unsqueeze(-1).clamp(1e-6, 1e6)
    far = aabb_far.unsqueeze(-1).clamp(1e-6, 1e6)

    t_vals = torch.linspace(0., 1., steps=N_samples)
    z_vals = near * (1.-t_vals) + far * (t_vals)

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        t_rand = torch.rand(z_vals.shape)


        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]
    
    
    rays_t_bc = torch.reshape(rays_t, [-1,1,1]).expand([N_rays, N_samples, 1])
    rays_d = torch.reshape(rays_d, [-1,1,3]).expand([N_rays, N_samples, 3])
    pts = torch.cat([pts, rays_d, rays_t_bc], dim = -1)
    
    def get_raw(fn, pts):
        static_raw, smoke_raw = None, None

     
        ## get static raw
        # static_raw = fn.forward_static(pts[..., :-1])
        # static_raw = batchify(fn.forward_static, chunk)(pts[..., :-1])
        static_raw = batchify_func(fn.forward_static, chunk, not fn.training)(pts[..., :-1])

        ## get smoke raw
        ## todo:: not warp pos for rgb
        orig_pos, orig_viewdir, orig_t = torch.split(pts, [3, 3, 1], -1)
        pts = torch.cat([orig_pos, orig_t], dim = -1)
        # smoke_raw = fn.forward_dynamic(pts)
        # smoke_raw = batchify(fn.forward_dynamic, chunk)(pts)
        smoke_raw = batchify_func(fn.forward_dynamic, chunk, not fn.training)(pts)

        return smoke_raw, static_raw # [N_rays, N_samples, 4], [N_rays, N_samples, 4]

    C_smokeRaw, C_staticRaw = get_raw(model, pts) 

    raw = [C_smokeRaw, C_staticRaw]


    ## density/sdf to alpha
    # add variance
    inv_s = model.get_deviation()         # Single parameter
    inv_s = inv_s.expand(N_rays, N_samples)

    cos_anneal_ratio = model.get_cos_anneal_ratio()
    
    rgb_map, disp_map, acc_map, weights, depth_map, ti_map, rgb_map_stack, acc_map_stack, extras = raw2outputs_hybrid_neus(raw, [z_vals, z_vals], rays_d, raw_noise_std, inv_s = inv_s, cos_anneal_ratio = cos_anneal_ratio, pytest=False, remove99=False, valid_rays=valid_rays)


    if raw[-1] is not None:
        rgbh2_map = rgb_map_stack[...,0] # dynamic
        acch2_map = acc_map_stack[...,0] # dynamic
        rgbh1_map = rgb_map_stack[...,1] # staitc
        acch1_map = acc_map_stack[...,1] # staitc
    
    extras_0 = None

    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map}
    ret['raw'] = raw[0]
    if raw[1] is not None:
        ret['raw_static'] = raw[1]

    if raw[-1] is not None:
        ret['rgbh1'] = rgbh1_map
        ret['acch1'] = acch1_map
        ret['rgbh2'] = rgbh2_map
        ret['acch2'] = acch2_map
        ret['rgbM'] = rgbh1_map * 0.5 + rgbh2_map * 0.5

    if 'gradient_error' in extras:
        ret['gradient_error'] = extras['gradient_error']

    if 'gradients' in extras:
        ret['gradients'] = extras['gradients']

    if 'hessians' in extras:
        ret['hessians'] = extras['hessians']


        
    ret['smoke_weights'] = weights
    ret['samples_xyz_dynamic'] = pts
    ret['rays_id'] = torch.arange(N_rays).unsqueeze(-1).long().expand(N_rays, N_samples).reshape(-1,1)


    if extras_0!= None and 'gradients' in extras_0:
        ret['gradients_coarse'] = extras_0['gradients']



    return ret


def render_path(model, render_poses, hwf, K, chunk, near, far, cuda_ray, netchunk = 1024 * 64, gt_imgs=None, savedir=None, render_factor=0, render_steps=None, bkgd_color=None):

    H, W, focal = hwf



    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []

    t = time.time()
    cur_timestep = None
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        if render_steps is not None:
            cur_timestep = render_steps[i]
        t = time.time()
        rgb, disp, acc, extras = render(H, W, K, model, chunk=chunk, c2w=c2w[:3,:4], netchunk=netchunk, time_step=cur_timestep, bkgd_color=bkgd_color, near = near, far = far, cuda_ray = cuda_ray, perturb=0)
        rgbs.append(rgb.detach().cpu().numpy())
        disps.append(disp.detach().cpu().numpy())
        if i==0:
            print(rgb.shape, disp.shape)

        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

            other_rgbs = []
            if gt_imgs is not None:
                other_rgbs.append(gt_imgs[i])
            for rgb_i in ['rgbh1','rgbh2','rgb0']: 
                if rgb_i in extras:
                    _data = extras[rgb_i].detach().cpu().numpy()
                    other_rgbs.append(_data)
            if len(other_rgbs) >= 1:
                other_rgb8 = np.concatenate(other_rgbs, axis=1)
                other_rgb8 = to8b(other_rgb8)
                filename = os.path.join(savedir, '_{:03d}.png'.format(i))
                imageio.imwrite(filename, other_rgb8)

            filename = os.path.join(savedir, 'disp_{:03d}.png'.format(i))
            imageio.imwrite(filename, to8b(disp.squeeze(-1).detach().cpu().numpy()))

            ## acc map
            filename = os.path.join(savedir, 'acc_{:03d}.png'.format(i))
            imageio.imwrite(filename, to8b(acc.squeeze(-1).detach().cpu().numpy()))


    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)


    return rgbs, disps