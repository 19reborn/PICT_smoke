
import torch
import torch.nn.functional as F
import numpy as np
import math

import raymarching

from src.utils.training_utils import batchify, batchify_func

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _meshgrid3d(
    res, device):
    """Create 3D grid coordinates."""
    assert len(res) == 3
    from packaging import version as pver
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.stack(
            torch.meshgrid(
                [
                    torch.arange(res[0], dtype=torch.long),
                    torch.arange(res[1], dtype=torch.long),
                    torch.arange(res[2], dtype=torch.long),
                ]
            ),
            dim=-1,
        ).to(device)
    else:
        return torch.stack(
            torch.meshgrid(
                [
                    torch.arange(res[0], dtype=torch.long),
                    torch.arange(res[1], dtype=torch.long),
                    torch.arange(res[2], dtype=torch.long),
                ],
                indexing="ij",
            ),
            dim=-1,
        ).to(device)


def custom_meshgrid(*args):
    from packaging import version as pver
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')
    
def inAABB(xyz, aabb_min, aabb_max):
    return torch.all(xyz >= aabb_min, dim=-1) & torch.all(xyz <= aabb_max, dim=-1)

class OccupancyGrid():
    def __init__(self, grid_size = 128, density_thresh = 30.0, bound = 1.0, aabb_min = [-1., -1., -1.], aabb_max = [1., 1., 1.]) -> None:
        self.grid_size = grid_size
        self.dt_gamma = 0
        self.max_steps = 1024
        self.density_thresh = density_thresh
        self.bound = bound 
        assert(bound >= 1.0)
        self.cascade = 1 + math.ceil(math.log2(bound))
        self.density_scale = 1.0

        self.cells_per_lvl = self.grid_size ** 3

        density_grid = torch.zeros([self.cascade, self.cells_per_lvl], device = device) # [CAS, H * H * H]
        density_bitfield = torch.zeros(self.cascade * self.cells_per_lvl // 8, dtype=torch.uint8, device = device) # [CAS * H * H * H // 8]
        self.density_grid = density_grid
        self.density_bitfield = density_bitfield
        self.mean_density = 0
        self.iter_density = 0


        # Grid coords & indices
        grid_coords = _meshgrid3d([grid_size,grid_size,grid_size], device).reshape(
            self.cells_per_lvl, 3
        )
        self.grid_coords = grid_coords

        grid_indices = torch.arange(self.cells_per_lvl, device=device)
        self.grid_indices = grid_indices

        self.morton3D_indices = raymarching.morton3D(grid_coords).long() # [N]

        ## todo:: mark untrained grid using camera information
        if torch.is_tensor(aabb_min):
            self.aabb_min = aabb_min
        else:
            self.aabb_min = torch.tensor(aabb_min, dtype=torch.float32)
        if torch.is_tensor(aabb_min):
            self.aabb_max = aabb_max
        else:
            self.aabb_max = torch.tensor(aabb_max, dtype=torch.float32)
        # aabb: [6]
        self.aabb = torch.cat([self.aabb_min, self.aabb_max], dim=0).reshape(6) # [1, 1, 6]

    ## todo:: mark untrained grid using camera information
    def mark_untrained_grid(self, poses, intrinsics, given_mask = None, S=64):
        # poses: [B, 4, 4]
        # intrinsic: [3, 3]
        
        # self.debug_vis(poses, intrinsics)
        # return

        if isinstance(poses, np.ndarray):
            poses = torch.from_numpy(poses)

        if isinstance(intrinsics, np.ndarray):
            intrinsics = torch.from_numpy(intrinsics)

        assert(len(poses.shape) == 3)
        assert(len(intrinsics.shape) == 3)
        assert(given_mask == None)

        C = poses.shape[0]
        # data read rules:
        # for each view, we are given one video, and we read each frame in the video
        # poses = poses.reshape(T, C, 4, 4)
        # intrinsics = intrinsics.reshape(T, C, 3, 3)
        if given_mask is not None:
            given_mask = given_mask[...,0:1]
            # given_mask = given_mask.reshape(T, C, given_mask.shape[1], given_mask.shape[2], 1)
            given_mask = given_mask.reshape(C, given_mask.shape[1], given_mask.shape[2], 1).transpose(1,0,2,3,4)

        count = torch.zeros_like(self.density_grid)

        poses = poses.to(count.device)
        intrinsics = intrinsics.to(count.device)
        
        # if len(intrinsic.shape) == 2:
        #     intrinsic = intrinsic[0]
        # fx, fy, cx, cy = intrinsic
        
        # X = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_grid.device).split(S)
        # Y = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_grid.device).split(S)
        # Z = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_grid.device).split(S)

        # 5-level loop, forgive me...
        # for xs in X:
        #     for ys in Y:
        #         for zs in Z:
                    
        #             # construct points
        #             xx, yy, zz = custom_meshgrid(xs, ys, zs)
        #             coords = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [N, 3], in [0, 128)
        #             indices = raymarching.morton3D(coords).long() # [N]
        coords = self.grid_coords
        indices = self.morton3D_indices

        world_xyzs = (2 * coords.float() / (self.grid_size - 1) - 1).unsqueeze(0) # [1, N, 3] in [-1, 1]


        # cascading
        for cas in range(self.cascade):
            bound = min(2 ** cas, self.bound)
            half_grid_size = bound / self.grid_size
            # scale to current cascade's resolution
            cas_world_xyzs = world_xyzs * (bound - half_grid_size)

            ## judge if the point is in the aabb
            
            # split batch to avoid OOM
            head = 0
            while head < C:
                tail = min(head + S, C)
                this_intrinsics = intrinsics[head:tail]
                this_poses = poses[head:tail]
                fx, fy, cx, cy = this_intrinsics[:,0,0], this_intrinsics[:,1,1], this_intrinsics[:,0,2], this_intrinsics[:,1,2]
                # world2cam transform (poses is c2w, so we need to transpose it. Another transpose is needed for batched matmul, so the final form is without transpose.)
                # cam_xyzs = cas_world_xyzs - poses[head:tail, :3, 3].unsqueeze(1)
                cam_xyzs = cas_world_xyzs - this_poses[:, :3, 3].unsqueeze(1)
                cam_xyzs = cam_xyzs @ this_poses[:, :3, :3] # [S, N, 3]
                # cam_xyzs = cam_xyzs @ this_poses[:, :3, :3].permute(0,2,1)# [S, N, 3]
                
                ## cam_xyzs coordinates:

                uv = cam_xyzs[:, :, :2] / -cam_xyzs[:, :, 2:] # [S, N, 2]

                uv *= torch.stack([fx, -fy], dim=-1).unsqueeze(1) # [S, N, 2]
                # uv *= torch.stack([fx, fy], dim=-1).unsqueeze(1) # [S, N, 2]
                uv += torch.stack([cx, cy], dim=-1).unsqueeze(1) # [S, N, 2]


                # debug
                # import cv2
                # for view in range(uv.shape[0]):
                #     uv_view = uv[view]
                #     # image_debug = np.zeros((cx[0].int().item()*2,cy[0].int().item()*2,3),dtype=np.uint8)
                #     image_debug = np.zeros((cy[0].int().item()*2,cx[0].int().item()*2,3),dtype=np.uint8)
                # #     # cv2.imwrite(f'debug/real_world_projection/gt_image_{view}.png',image_debug)
                # #     cv2.imwrite(f'debug/na_projection/gt_image_{view}.png',image_debug)
                
                #     # point_color = (0, 0, 255) # BGR
                #     # thickness = 4 #  0 、4、8
                # #     # for coor in uv[view].cpu().numpy():
                #     # cv2.circle(image_debug, (int(coor[0]),int(coor[1])), point_size, point_color, thickness)
                #     # cv2.imwrite(f'debug/projection/outer_vex_projection_{view}.png',image_debug)
                #     point_size = 1
                # #     uv = ori_uv[:,:6890,:]
                #     point_color = (0, 255, 255)
                #     thickness = 2 #  0 、4、8
                #     for coor in uv_view.cpu().numpy():
                #         cv2.circle(image_debug, (int(coor[1]),int(coor[0])), point_size, point_color, thickness)
                # #     # cv2.imwrite(f'debug/real_world_projection/smpl_vex_projection_{view}.png',image_debug)
                #     cv2.imwrite(f'debug/projection_{view}.png',image_debug)
                # # exit(1)
                # import pdb
                # pdb.set_trace()
                if given_mask is not None:
                    gt_masks = given_mask[head:tail]
                    # query uv in gt_masks
                    # uv: views, num, 2
                    # gt_masks: views, H, W
                    ## mask dilation
                    # dilated_masks = []
                    # for view in range(gt_masks.shape[0]):
                    #     this_dilated_mask = (cv2.dilate(gt_masks[view], np.ones((5,5), np.uint8), iterations=10))
                    #     dilated_masks.append(this_dilated_mask)
                    #     # cv2.imwrite(f'debug/gt_masks_{view}.png',gt_masks[view]*255)
                    #     # cv2.imwrite(f'debug/gt_masks_dilated_{view}.png',this_dilated_mask*255)
                    #     # cv2.imwrite(f'debug/gt_masks_dilated_diff_{view}.png',(this_dilated_mask-gt_masks[view][...,0])*255)
                    #     # import pdb
                    #     # pdb.set_trace()
                    # gt_masks = torch.from_numpy(np.stack(dilated_masks, axis=0)).cuda().reshape(gt_masks.shape)
                    gt_masks = torch.from_numpy(gt_masks).cuda()

                    ''' other dilate method
                    def dilation_pytorch(image, strel, origin=(0, 0), border_value=0):
                        # first pad the image to have correct unfolding; here is where the origins is used
                        image_pad = F.pad(image, [origin[0], strel.shape[0] - origin[0] - 1, origin[1], strel.shape[1] - origin[1] - 1], mode='constant', value=border_value)
                        # Unfold the image to be able to perform operation on neighborhoods
                        image_unfold = F.unfold(image_pad, kernel_size=strel.shape)
                        # Flatten the structural element since its two dimensions have been flatten when unfolding
                        strel_flatten = torch.flatten(strel).unsqueeze(0).unsqueeze(-1)
                        # Perform the greyscale operation; sum would be replaced by rest if you want erosion
                        sums = image_unfold + strel_flatten
                        # Take maximum over the neighborhood
                        result, _ = sums.max(dim=1)
                        # Reshape the image to recover initial shape
                        return torch.reshape(result, image.shape)
                    strel_tensor = torch.tensor(np.ones((3, 3)), dtype=torch.float)
                    # dilation_pytorch(torch.from_numpy(given_mask).permute(0,3,1,2).cuda(), strel=strel_tensor, origin=(1,1), border_value=-1000)
                    # dilated_mask = cv2.dilate(given_mask[0], np.ones((3,3),np.uint8), iterations=1)
                    # gt_masks_dilated = dilation_pytorch(torch.from_numpy(gt_masks).permute(0,3,1,2).cuda(), strel=strel_tensor, origin=(1,1), border_value=-1000)
                    # gt_masks = torch.from_numpy(gt_masks).permute(0,3,1,2).cuda()
                    # gt_masks_dilated = e(gt_masks)
                    '''

                    # given_mask = torch.from_numpy(cv2.dilate(given_mask.cpu().numpy(), np.ones((3,3),np.uint8), iterations=1)).cuda()
                    image_board = torch.tensor([gt_masks.shape[2], gt_masks.shape[1]], dtype=torch.float32, device=gt_masks.device)
                    normalized_pixel_locations = uv / (image_board - 1) 
                    normalized_pixel_locations = normalized_pixel_locations * 2 - 1.0
                    normalized_pixel_locations = normalized_pixel_locations.unsqueeze(1)
    
                    mask = F.grid_sample(gt_masks.permute(0,3,1,2).float(), normalized_pixel_locations.float(), align_corners=True).squeeze(2).permute(0,2,1) #[views,num,1]

                    ## debug:
                    # os.makedirs(f'debug/uv_mask_project/time_{t}/', exist_ok = True)
                    # for view in range(uv.shape[0]):
                    #     cv2.imwrite(f'debug/uv_mask_project/time_{t}/gt_mask_{view}.png',gt_masks[view].cpu().numpy()*255)
                    # self.project_uv(uv, gt_masks.shape[2], gt_masks.shape[1], mask, f'debug/uv_mask_project/time_{t}/')

                else:
                    mask = (uv[:, :, 0] > 0) & (uv[:, :, 0] < cx.unsqueeze(1) * 2) & (uv[:, :, 1] > 0) & (uv[:, :, 1] < cy.unsqueeze(1) * 2) # [S, N]
                mask = mask.sum(0).reshape(-1) # [N]
                    # query if point is covered by any camera
                    # mask_z = cam_xyzs[:, :, 2] > 0 # [S, N]
                    # mask_x = torch.abs(cam_xyzs[:, :, 0]) < (cx / fx).unsqueeze(-1) * cam_xyzs[:, :, 2] + half_grid_size * 2
                    # mask_y = torch.abs(cam_xyzs[:, :, 1]) < (cy / fy).unsqueeze(-1) * cam_xyzs[:, :, 2] + half_grid_size * 2
                    # mask = (mask_z & mask_x & mask_y).sum(0).reshape(-1) # [N]
                    

                # update count 
                count[cas, indices] += mask
                head += S
            ## aabb 
            in_aabb = inAABB(cas_world_xyzs, self.aabb_min, self.aabb_max)
            # count[cas, indices][(in_aabb==False)[0]] = 0
            count[cas, indices[(in_aabb==False)[0]]] = 0
        # mark untrained grid as -1
        # self.density_grid[count == 0] = -1
        if given_mask is not None:
            self.density_grid[count < C / 2] = -1
        else:
            self.density_grid[count == 0] = -1

        print(f'[mark untrained grid for static occ grid] {(count == 0).sum()} from {self.grid_size ** 3 * self.cascade}')
        # exit(1)


    @torch.no_grad()
    def update_grid(self, query_fn, decay=0.95, S=128):
        # call before each epoch to update extra states.

        # return
        ### update sdf grid

        tmp_grid = - torch.ones_like(self.density_grid, device = device)
     
        # full update.
      
        if self.iter_density < 16:
        # if self.iter_density < 10000: #debug
        # #if True:
        #     X = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_grid.device).split(S)
        #     Y = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_grid.device).split(S)
        #     Z = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_grid.device).split(S)

        #     for xs in X:
        #         for ys in Y:
        #             for zs in Z:
                        
        #                 # construct points
        #                 xx, yy, zz = custom_meshgrid(xs, ys, zs)
        #                 coords = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [N, 3], in [0, 128)
        #                 indices = raymarching.morton3D(coords).long() # [N]

        #                 xyzs = 2 * coords.float() / (self.grid_size - 1) - 1 # [N, 3] in [-1, 1]

        #                 # cascading
        #                 for cas in range(self.cascade):
        #                     bound = min(2 ** cas, self.bound)
        #                     half_grid_size = bound / self.grid_size
        #                     # scale to current cascade's resolution
        #                     cas_xyzs = xyzs * (bound - half_grid_size)
        #                     # add noise in [-hgs, hgs]
        #                     cas_xyzs += (torch.rand_like(cas_xyzs) * 2 - 1) * half_grid_size
        #                     # query density
        #                     sigmas = query_fn(cas_xyzs).reshape(-1).detach()
        #                     # from `scalbnf(MIN_CONE_STEPSIZE(), 0)`, check `splat_grid_samples_nerf_max_nearest_neighbor`
        #                     # scale == 2 * sqrt(3) / 1024
        #                     # sigmas *= self.density_scale * 0.003383
        #                     sigmas *= self.density_scale
        #                     # assign 
        #                     tmp_grid[cas, indices] = sigmas

        #                     del sigmas


            coords = self.grid_coords
            indices = self.morton3D_indices

            xyzs = 2 * coords.float() / (self.grid_size - 1) - 1 # [N, 3] in [-1, 1]

            # cascading
            for cas in range(self.cascade):
                bound = min(2 ** cas, self.bound)
                half_grid_size = bound / self.grid_size
                # scale to current cascade's resolution
                cas_xyzs = xyzs * (bound - half_grid_size)
                # add noise in [-hgs, hgs]
                cas_xyzs += (torch.rand_like(cas_xyzs) * 2 - 1) * half_grid_size
                # query density
                sigmas = query_fn(cas_xyzs).reshape(-1).detach()
                # from `scalbnf(MIN_CONE_STEPSIZE(), 0)`, check `splat_grid_samples_nerf_max_nearest_neighbor`
                # scale == 2 * sqrt(3) / 1024
                # sigmas *= self.density_scale * 0.003383
                sigmas *= self.density_scale
                # assign 
                tmp_grid[cas, indices] = sigmas

                del sigmas

                # # query sdf
                # sdf = self.sdf(cas_xyzs).reshape(-1).detach()
            
                # if self.opt.use_density_grid:
                # # sdf grid
                #     sdf = 1.0 / (torch.abs(sdf)+1e-6)
                # else:
                # # density grid
                #     inv_s = self.deviation(torch.zeros([1, 3], device=sdf.device))[:, :1].clip(1e-6, 1e6)
                #     sigmoid_sdf = torch.sigmoid(sdf*inv_s)
                #     sdf = inv_s * sigmoid_sdf * (1 - sigmoid_sdf)
            
                # tmp_grid[cas, indices] = sdf.float()

        # partial update (half the computation)
        # TODO: why no need of maxpool ?
        # elif self.iter_density < 100:
        elif self.iter_density < 10000:
            N = self.grid_size ** 3 // 4 # H * H * H / 2
            for cas in range(self.cascade):
                # random sample some positions
                coords = torch.randint(0, self.grid_size, (N, 3), device=self.density_grid.device) # [N, 3], in [0, 128)
                indices = raymarching.morton3D(coords).long() # [N]
                # random sample occupied positions
                occ_indices = torch.nonzero(self.density_grid[cas] > 0).squeeze(-1) # [Nz]
                rand_mask = torch.randint(0, occ_indices.shape[0], [N], dtype=torch.long, device=self.density_grid.device)
                occ_indices = occ_indices[rand_mask] # [Nz] --> [N], allow for duplication
                occ_coords = raymarching.morton3D_invert(occ_indices) # [N, 3]
                # concat
                indices = torch.cat([indices, occ_indices], dim=0)
                coords = torch.cat([coords, occ_coords], dim=0)
                # same below
                xyzs = 2 * coords.float() / (self.grid_size - 1) - 1 # [N, 3] in [-1, 1]
                bound = min(2 ** cas, self.bound)
                half_grid_size = bound / self.grid_size
                # scale to current cascade's resolution
                cas_xyzs = xyzs * (bound - half_grid_size)
                # add noise in [-hgs, hgs]
                cas_xyzs += (torch.rand_like(cas_xyzs) * 2 - 1) * half_grid_size
                # query density
                sigmas = query_fn(cas_xyzs).reshape(-1).detach()
                # from `scalbnf(MIN_CONE_STEPSIZE(), 0)`, check `splat_grid_samples_nerf_max_nearest_neighbor`
                # scale == 2 * sqrt(3) / 1024
                # sigmas *= self.density_scale * 0.003383
                sigmas *= self.density_scale
                # assign 
                tmp_grid[cas, indices] = sigmas

                del sigmas
             

        # ema update
        valid_mask = (self.density_grid >= 0) & (tmp_grid >= 0)
        self.density_grid[valid_mask] = torch.max(self.density_grid[valid_mask] * decay, tmp_grid[valid_mask])
        # self.density_grid[~valid_mask] = -1

        self.mean_density = torch.mean(self.density_grid.clamp(min=0)).item() # -1 non-training regions are viewed as 0 density.
        # self.mean_density = (torch.sum(self.density_grid.clamp(min=0))/ torch.sum(valid_mask)).item() # -1 non-training regions are viewed as 0 density.
        self.iter_density += 1
        
        print("static occ gird mean_density:\n",self.mean_density)
        # convert to bitfield
        # density_thresh = min(self.mean_density, self.density_thresh)
        # if self.iter_density < 20:
        #     density_thresh = min(self.mean_density, self.density_thresh / 4)
        # else:
        density_thresh = min(self.mean_density, self.density_thresh)
        # density_thresh = self.density_thresh
        self.density_bitfield = raymarching.packbits(self.density_grid, density_thresh, self.density_bitfield)

        del tmp_grid
        torch.cuda.empty_cache()
        # print(f'[sdf grid] min={self.density_grid.min().item():.4f}, max={self.density_grid.max().item():.4f}, mean={self.mean_density:.4f}, occ_rate={(self.density_grid > 0.01).sum() / (128**3 * self.cascade):.3f} | [step counter] mean={self.mean_count}')

class OccupancyGridDynamic():
    def __init__(self, cascade = 1, grid_size = 128, density_thresh = 1.0, time_size = 150, bound = 1.0, aabb_min = [-1., -1., -1.], aabb_max = [1., 1., 1.]) -> None:
        self.grid_size = grid_size
        self.time_size = time_size
        self.dt_gamma = 0
        self.max_steps = 1024
        self.density_thresh = density_thresh
        self.bound = bound 
        assert(bound >= 1.0)
        self.cascade = 1 + math.ceil(math.log2(bound))
        self.density_scale = 1.0

        self.cells_per_lvl = self.grid_size ** 3

        density_grid = torch.zeros([self.time_size, self.cascade, self.cells_per_lvl], device = device) # [CAS, H * H * H]
        density_bitfield = torch.zeros(self.time_size, self.cascade * self.cells_per_lvl // 8, dtype=torch.uint8,  device = device) # [CAS * H * H * H // 8]
        self.density_grid = density_grid
        self.density_bitfield = density_bitfield
        
        self.mean_density = torch.zeros(self.time_size, device=self.density_bitfield.device)

        self.iter_density = 0
        # time stamps for density grid
        times = ((torch.arange(self.time_size, dtype=torch.float32) + 0.5) / self.time_size).view(-1, 1, 1) # [T, 1, 1]
        self.times = times

        # Grid coords & indices
        grid_coords = _meshgrid3d([grid_size,grid_size,grid_size], device).reshape(
            self.cells_per_lvl, 3
        )
        self.grid_coords = grid_coords

        grid_indices = torch.arange(self.cells_per_lvl, device=device)
        self.grid_indices = grid_indices

        self.morton3D_indices = raymarching.morton3D(grid_coords).long() # [N]

        ## todo:: mark untrained grid using camera information
        if torch.is_tensor(aabb_min):
            self.aabb_min = aabb_min
        else:
            self.aabb_min = torch.tensor(aabb_min, dtype=torch.float32)
        if torch.is_tensor(aabb_min):
            self.aabb_max = aabb_max
        else:
            self.aabb_max = torch.tensor(aabb_max, dtype=torch.float32)
        # aabb: [6]
        self.aabb = torch.cat([self.aabb_min, self.aabb_max], dim=0).reshape(6) # [1, 1, 6]
        
        
    def project_uv(self, uv, H, W, uv_mask= None, output_dir = None):
        import cv2
        if uv_mask is not None:
            uv_mask = (uv_mask != 0).float().cpu().numpy()
        for view in range(uv.shape[0]):
            uv_view = uv[view]
            if uv_mask is not None:
                uv_mask_view = uv_mask[view]
            # image_debug = np.zeros((cx[0].int().item()*2,cy[0].int().item()*2,3),dtype=np.uint8)
            image_debug = np.zeros((W,H,3),dtype=np.uint8)
        #     # cv2.imwrite(f'debug/real_world_projection/gt_image_{view}.png',image_debug)
            point_size = 1

            point_color = (0, 255, 255)
            
            thickness = 2 #  0 、4、8
            for i, coor in enumerate(uv_view.cpu().numpy()):
                try:
                    if uv_mask is not None:
                # cv2.circle(image_debug, (int(coor[0]),int(coor[1])), point_size, point_color, thickness)
                        cv2.circle(image_debug, (int(coor[0]),int(coor[1])), point_size, point_color*int(uv_mask_view[i,0]), thickness)
                    else:
                        cv2.circle(image_debug, (int(coor[0]),int(coor[1])), point_size, point_color, thickness)
                except:
                    continue
                    
                    import pdb
                    pdb.set_trace()
        #     # cv2.imwrite(f'debug/real_world_projection/smpl_vex_projection_{view}.png',image_debug)
            if output_dir is not None:
                import os
                os.makedirs(output_dir, exist_ok=True)
                cv2.imwrite(f'{output_dir}/projection_{view}.png',image_debug)
            else:
                cv2.imwrite(f'debug/projection_{view}.png',image_debug)

        print("finish projection uv for debug")

    def debug_vis(self, poses, intrinsics, pts = None, pts_mask = None):

        T = self.time_size
        C = poses.shape[0] // T
        poses = poses.reshape(T, C, 4, 4)
        intrinsics = intrinsics.reshape(T, C, 3, 3)

        this_poses = poses[0]
        this_intrinsics = intrinsics[0]
        if pts is not None:
            cas_world_xyzs = pts
        else:
            import trimesh
            cas_world_xyzs = trimesh.load("debug/sample_pts.ply").vertices
            cas_world_xyzs = torch.from_numpy(cas_world_xyzs).float().cuda()

        fx, fy, cx, cy = this_intrinsics[:,0,0], this_intrinsics[:,1,1], this_intrinsics[:,0,2], this_intrinsics[:,1,2]
        # world2cam transform (poses is c2w, so we need to transpose it. Another transpose is needed for batched matmul, so the final form is without transpose.)
        # cam_xyzs = cas_world_xyzs - poses[head:tail, :3, 3].unsqueeze(1)
        cam_xyzs = cas_world_xyzs - this_poses[:, :3, 3].unsqueeze(1)
        cam_xyzs = cam_xyzs @ this_poses[:, :3, :3] # [S, N, 3]
        # cam_xyzs = cam_xyzs @ this_poses[:, :3, :3].permute(0,2,1)# [S, N, 3]
        
        
        ## cam_xyzs coordinates:

        uv = cam_xyzs[:, :, :2] / -cam_xyzs[:, :, 2:] # [S, N, 2]
        # uv = cam_xyzs[:, :, :2] / cam_xyzs[:, :, 2:] # [S, N, 2]

        uv *= torch.stack([fx, -fy], dim=-1).unsqueeze(1) # [S, N, 2]
        # uv *= torch.stack([fx, fy], dim=-1).unsqueeze(1) # [S, N, 2]
        uv += torch.stack([cx, cy], dim=-1).unsqueeze(1) # [S, N, 2]

        self.project_uv(uv, cx[0].int().item()*2,cy[0].int().item()*2)

        exit(1)

    ## todo:: mark untrained grid using camera information
    def mark_untrained_grid(self, poses, intrinsics, given_mask = None, S=64):
        # poses: [B, 4, 4]
        # intrinsic: [3, 3]
        
        # self.debug_vis(poses, intrinsics)
        # return

        if isinstance(poses, np.ndarray):
            poses = torch.from_numpy(poses)

        if isinstance(intrinsics, np.ndarray):
            intrinsics = torch.from_numpy(intrinsics)

        T = self.time_size
        C = poses.shape[0] // T
        # data read rules:
        # for each view, we are given one video, and we read each frame in the video
        # poses = poses.reshape(T, C, 4, 4)
        # intrinsics = intrinsics.reshape(T, C, 3, 3)
        poses = poses.reshape(C, T, 4, 4).permute(1,0,2,3)
        intrinsics = intrinsics.reshape(C, T, 3, 3).permute(1,0,2,3)
        if given_mask is not None:
            given_mask = given_mask[...,0:1]
            # given_mask = given_mask.reshape(T, C, given_mask.shape[1], given_mask.shape[2], 1)
            given_mask = given_mask.reshape(C, T, given_mask.shape[1], given_mask.shape[2], 1).transpose(1,0,2,3,4)

        count = torch.zeros_like(self.density_grid)

        poses = poses.to(count.device)
        intrinsics = intrinsics.to(count.device)
        
        # if len(intrinsic.shape) == 2:
        #     intrinsic = intrinsic[0]
        # fx, fy, cx, cy = intrinsic
        
        # X = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_grid.device).split(S)
        # Y = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_grid.device).split(S)
        # Z = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_grid.device).split(S)

        # # 5-level loop, forgive me...
        # for xs in X:
        #     for ys in Y:
        #         for zs in Z:
                    
        #             # construct points
        #             xx, yy, zz = custom_meshgrid(xs, ys, zs)
        #             coords = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [N, 3], in [0, 128)
        #             indices = raymarching.morton3D(coords).long() # [N]

        coords = self.grid_coords
        indices = self.morton3D_indices
        world_xyzs = (2 * coords.float() / (self.grid_size - 1) - 1).unsqueeze(0) # [1, N, 3] in [-1, 1]


        # cascading
        for cas in range(self.cascade):
            bound = min(2 ** cas, self.bound)
            half_grid_size = bound / self.grid_size
            # scale to current cascade's resolution
            cas_world_xyzs = world_xyzs * (bound - half_grid_size)

            ## judge if the point is in the aabb
            
            for t, time in enumerate(self.times):
                # split batch to avoid OOM
                head = 0
                while head < C:
                    tail = min(head + S, C)
                    this_intrinsics = intrinsics[t][head:tail]
                    this_poses = poses[t][head:tail]
                    fx, fy, cx, cy = this_intrinsics[:,0,0], this_intrinsics[:,1,1], this_intrinsics[:,0,2], this_intrinsics[:,1,2]
                    # world2cam transform (poses is c2w, so we need to transpose it. Another transpose is needed for batched matmul, so the final form is without transpose.)
                    # cam_xyzs = cas_world_xyzs - poses[head:tail, :3, 3].unsqueeze(1)
                    cam_xyzs = cas_world_xyzs - this_poses[:, :3, 3].unsqueeze(1)
                    cam_xyzs = cam_xyzs @ this_poses[:, :3, :3] # [S, N, 3]
                    # cam_xyzs = cam_xyzs @ this_poses[:, :3, :3].permute(0,2,1)# [S, N, 3]
                    
                    ## cam_xyzs coordinates:

                    uv = cam_xyzs[:, :, :2] / -cam_xyzs[:, :, 2:] # [S, N, 2]

                    uv *= torch.stack([fx, -fy], dim=-1).unsqueeze(1) # [S, N, 2]
                    # uv *= torch.stack([fx, fy], dim=-1).unsqueeze(1) # [S, N, 2]
                    uv += torch.stack([cx, cy], dim=-1).unsqueeze(1) # [S, N, 2]


                    # debug
                    # import cv2
                    # for view in range(uv.shape[0]):
                    #     uv_view = uv[view]
                    #     image_debug = np.zeros((cy[0].int().item()*2,cx[0].int().item()*2,3),dtype=np.uint8)
                    
                    #     point_size = 1
                    #     point_color = (0, 255, 255)
                    #     thickness = 2 #  0 、4、8
                    #     for coor in uv_view.cpu().numpy():
                    #         cv2.circle(image_debug, (int(coor[1]),int(coor[0])), point_size, point_color, thickness)

                    #     cv2.imwrite(f'debug/projection_{view}.png',image_debug)
                    # exit(1)
                    # import pdb
                    # pdb.set_trace()
                    if given_mask is not None:
                        gt_masks = given_mask[t, head:tail]
                        # query uv in gt_masks
                        # uv: views, num, 2
                        # gt_masks: views, H, W
                        ## mask dilation
                        # dilated_masks = []
                        # for view in range(gt_masks.shape[0]):
                        #     this_dilated_mask = (cv2.dilate(gt_masks[view], np.ones((5,5), np.uint8), iterations=10))
                        #     dilated_masks.append(this_dilated_mask)
                        #     # cv2.imwrite(f'debug/gt_masks_{view}.png',gt_masks[view]*255)
                        #     # cv2.imwrite(f'debug/gt_masks_dilated_{view}.png',this_dilated_mask*255)
                        #     # cv2.imwrite(f'debug/gt_masks_dilated_diff_{view}.png',(this_dilated_mask-gt_masks[view][...,0])*255)
                        #     # import pdb
                        #     # pdb.set_trace()
                        # gt_masks = torch.from_numpy(np.stack(dilated_masks, axis=0)).cuda().reshape(gt_masks.shape)
                        gt_masks = torch.from_numpy(gt_masks).cuda()

                        ''' other dilate method
                        def dilation_pytorch(image, strel, origin=(0, 0), border_value=0):
                            # first pad the image to have correct unfolding; here is where the origins is used
                            image_pad = F.pad(image, [origin[0], strel.shape[0] - origin[0] - 1, origin[1], strel.shape[1] - origin[1] - 1], mode='constant', value=border_value)
                            # Unfold the image to be able to perform operation on neighborhoods
                            image_unfold = F.unfold(image_pad, kernel_size=strel.shape)
                            # Flatten the structural element since its two dimensions have been flatten when unfolding
                            strel_flatten = torch.flatten(strel).unsqueeze(0).unsqueeze(-1)
                            # Perform the greyscale operation; sum would be replaced by rest if you want erosion
                            sums = image_unfold + strel_flatten
                            # Take maximum over the neighborhood
                            result, _ = sums.max(dim=1)
                            # Reshape the image to recover initial shape
                            return torch.reshape(result, image.shape)
                        strel_tensor = torch.tensor(np.ones((3, 3)), dtype=torch.float)
                        # dilation_pytorch(torch.from_numpy(given_mask).permute(0,3,1,2).cuda(), strel=strel_tensor, origin=(1,1), border_value=-1000)
                        # dilated_mask = cv2.dilate(given_mask[0], np.ones((3,3),np.uint8), iterations=1)
                        # gt_masks_dilated = dilation_pytorch(torch.from_numpy(gt_masks).permute(0,3,1,2).cuda(), strel=strel_tensor, origin=(1,1), border_value=-1000)
                        # gt_masks = torch.from_numpy(gt_masks).permute(0,3,1,2).cuda()
                        # gt_masks_dilated = e(gt_masks)
                        '''

                        # given_mask = torch.from_numpy(cv2.dilate(given_mask.cpu().numpy(), np.ones((3,3),np.uint8), iterations=1)).cuda()
                        image_board = torch.tensor([gt_masks.shape[2], gt_masks.shape[1]], dtype=torch.float32, device=gt_masks.device)
                        normalized_pixel_locations = uv / (image_board - 1) 
                        normalized_pixel_locations = normalized_pixel_locations * 2 - 1.0
                        normalized_pixel_locations = normalized_pixel_locations.unsqueeze(1)
        
                        mask = F.grid_sample(gt_masks.permute(0,3,1,2).float(), normalized_pixel_locations.float(), align_corners=True).squeeze(2).permute(0,2,1) #[views,num,1]

                        ## debug:
                        # os.makedirs(f'debug/uv_mask_project/time_{t}/', exist_ok = True)
                        # for view in range(uv.shape[0]):
                        #     cv2.imwrite(f'debug/uv_mask_project/time_{t}/gt_mask_{view}.png',gt_masks[view].cpu().numpy()*255)
                        # self.project_uv(uv, gt_masks.shape[2], gt_masks.shape[1], mask, f'debug/uv_mask_project/time_{t}/')

                    else:
                        mask = (uv[:, :, 0] > 0) & (uv[:, :, 0] < cx.unsqueeze(1) * 2) & (uv[:, :, 1] > 0) & (uv[:, :, 1] < cy.unsqueeze(1) * 2) # [S, N]
                    mask = mask.sum(0).reshape(-1) # [N]
                    # query if point is covered by any camera
                    # mask_z = cam_xyzs[:, :, 2] > 0 # [S, N]
                    # mask_x = torch.abs(cam_xyzs[:, :, 0]) < (cx / fx).unsqueeze(-1) * cam_xyzs[:, :, 2] + half_grid_size * 2
                    # mask_y = torch.abs(cam_xyzs[:, :, 1]) < (cy / fy).unsqueeze(-1) * cam_xyzs[:, :, 2] + half_grid_size * 2
                    # mask = (mask_z & mask_x & mask_y).sum(0).reshape(-1) # [N]
                    

                    # update count 
                    count[t, cas, indices] += mask
                    head += S
            ## aabb 
            in_aabb = inAABB(cas_world_xyzs, self.aabb_min, self.aabb_max)
            # count[:, cas, indices][(in_aabb==0).expand(count.shape[0],in_aabb.shape[1])] = 0
            count[:, cas, indices[(in_aabb==0)[0]]] = 0
        # mark untrained grid as -1
        # if given_mask is not None:
            # self.density_grid[count < C / 2] = -1
        # else:
            # self.density_grid[count == 0] = -1

        if given_mask is not None:
            self.density_grid[count < C / 2] = -1
        else:
            self.density_grid[count == 0] = -1

        print(f'[mark untrained grid for dynamic occ grid] {(count == 0).sum()} from {self.grid_size ** 3 * self.cascade * self.time_size}')
        # exit(1)

    @torch.no_grad()
    def update_grid(self, query_fn, decay=0.95, S=128):
        # call before each epoch to update extra states.
        
        import time as timer
        start = timer.time()

        ### update sdf grid

        tmp_grid = - torch.ones_like(self.density_grid, device = device)
 
        # # full update.
        if self.iter_density < 16:
        # if self.iter_density < 10000: #debug
        #if True:
            # X = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_grid.device).split(S)
            # Y = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_grid.device).split(S)
            # Z = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_grid.device).split(S)

            for t, time in enumerate(self.times):
                # for xs in X:
                #     for ys in Y:
                #         for zs in Z:
                            
                #             # construct points
                #             xx, yy, zz = custom_meshgrid(xs, ys, zs)
                #             coords = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [N, 3], in [0, 128)
                #             indices = raymarching.morton3D(coords).long() # [N]
                coords = self.grid_coords
                indices = self.morton3D_indices
                xyzs = 2 * coords.float() / (self.grid_size - 1) - 1 # [N, 3] in [-1, 1]

                            # cascading
                for cas in range(self.cascade):
                    bound = min(2 ** cas, self.bound)
                    half_grid_size = bound / self.grid_size
                    half_time_size = 0.5 / self.time_size
                    # scale to current cascade's resolution
                    cas_xyzs = xyzs * (bound - half_grid_size)
                    # add noise in coord [-hgs, hgs]
                    cas_xyzs += (torch.rand_like(cas_xyzs) * 2 - 1) * half_grid_size
                    # add noise in time [-hts, hts]
                    # time_perturb = time + (torch.rand_like(time) * 2 - 1) * half_time_size
                    time_perturb = time 


                    # query density
                    sigmas = query_fn(cas_xyzs, time_perturb).reshape(-1).detach()
                    # from `scalbnf(MIN_CONE_STEPSIZE(), 0)`, check `splat_grid_samples_nerf_max_nearest_neighbor`
                    # scale == 2 * sqrt(3) / 1024
                    # sigmas *= self.density_scale * 0.003383
                    sigmas *= self.density_scale
                    # assign 
                    tmp_grid[t, cas, indices] = sigmas

                    del sigmas

        # partial update (half the computation)
        # TODO: why no need of maxpool ?
        # elif self.iter_density < 100:
        elif self.iter_density < 10000:
            N = self.grid_size ** 3 // 4 # T * C * H * H * H / 4
            for t, time in enumerate(self.times):
                for cas in range(self.cascade):
                    # random sample some positions
                    coords = torch.randint(0, self.grid_size, (N, 3), device=self.density_bitfield.device) # [N, 3], in [0, 128)
                    indices = raymarching.morton3D(coords).long() # [N]
                    # random sample occupied positions
                    occ_indices = torch.nonzero(self.density_grid[t, cas] > 0).squeeze(-1) # [Nz]
                    rand_mask = torch.randint(0, occ_indices.shape[0], [N], dtype=torch.long, device=self.density_bitfield.device)
                    occ_indices = occ_indices[rand_mask] # [Nz] --> [N], allow for duplication
                    occ_coords = raymarching.morton3D_invert(occ_indices) # [N, 3]
                    # concat
                    indices = torch.cat([indices, occ_indices], dim=0)
                    coords = torch.cat([coords, occ_coords], dim=0)
                    # same below
                    xyzs = 2 * coords.float() / (self.grid_size - 1) - 1 # [N, 3] in [-1, 1]
                    bound = min(2 ** cas, self.bound)
                    half_grid_size = bound / self.grid_size
                    half_time_size = 0.5 / self.time_size
                    # scale to current cascade's resolution
                    cas_xyzs = xyzs * (bound - half_grid_size)
                    # add noise in [-hgs, hgs]
                    cas_xyzs += (torch.rand_like(cas_xyzs) * 2 - 1) * half_grid_size
                    # add noise in time [-hts, hts]
                    # time_perturb = time + (torch.rand_like(time) * 2 - 1) * half_time_size
                    time_perturb = time 


                    # query density
                    sigmas = query_fn(cas_xyzs, time_perturb).reshape(-1).detach()
                    # from `scalbnf(MIN_CONE_STEPSIZE(), 0)`, check `splat_grid_samples_nerf_max_nearest_neighbor`
                    # scale == 2 * sqrt(3) / 1024
                    # sigmas *= self.density_scale * 0.003383
                    sigmas *= self.density_scale
                    # assign 
                    tmp_grid[t, cas, indices] = sigmas
        else:
            print("density grid is full, no need to update")
            return 

        # ema update
        valid_mask = (self.density_grid >= 0) & (tmp_grid >= 0)
        self.density_grid[valid_mask] = torch.max(self.density_grid[valid_mask] * decay, tmp_grid[valid_mask])
       
        # self.density_grid[~valid_mask] = -1
        # self.mean_density = torch.mean(self.density_grid.clamp(min=0)).item() # -1 non-training regions are viewed as 0 density.

        # self.mean_density = torch.zeros(self.time_size, device=self.density_bitfield.device)
        ## mean_density for different time
        for t in range(self.time_size):
            self.mean_density[t] = torch.mean(self.density_grid[t].clamp(min=0)).item() # -1 non-training regions are viewed as 0 density.
            # self.mean_density[t] = (torch.sum(self.density_grid[t].clamp(min=0))/ torch.sum(valid_mask[t])).item() # -1 non-training regions are viewed as 0 density.
            # self.mean_density[t] = -100 # -1 non-training regions are viewed as 0 density.
            print(f"\nmean density {t}: ", self.mean_density[t])



        # convert to bitfield
        for t in range(self.time_size):
            # density_thresh = min(self.mean_density[t], self.density_thresh)
            
            # if self.iter_density < 20:
            #     density_thresh = min(self.mean_density[t], self.density_thresh / 4)
            # else:
            density_thresh = min(self.mean_density[t], self.density_thresh)
            raymarching.packbits(self.density_grid[t], density_thresh, self.density_bitfield[t])
        
        ### update step counter
        # total_step = min(16, self.local_step)
        # if total_step > 0:
        #     self.mean_count = int(self.step_counter[:total_step, 0].sum().item() / total_step)
        # self.local_step = 0

        self.iter_density += 1
        del tmp_grid
        torch.cuda.empty_cache()

        end = timer.time()  

        print("[INFO]:: update dynamic occ grid for time: ", end - start)

        #print(f'[sdf grid] min={self.density_grid.min().item():.4f}, max={self.density_grid.max().item():.4f}, mean={self.mean_density:.4f}, occ_rate={(self.density_grid > 0.01).sum() / (128**3 * self.cascade):.3f} | [step counter] mean={self.mean_count}')



def update_occ_grid(args, model, global_step = 0, update_interval = 1000, neus_early_terminated = False):
    if not args.cuda_ray:
        return
    
    if args.test_mode:
      
        if "hybrid_neus" in args.net_model:
            def get_density_dynamic(x, t, chunk=64**3):
                x_t = torch.cat([x, t * torch.ones_like(x[..., :1])], dim = -1)
                
                return batchify(model.density_dynamic, chunk)(x_t)

            for i in range(2):
                model.occupancy_grid_dynamic.update_grid(get_density_dynamic, S = 64)
      
            def get_density_static(x):
                neus = model.static_model
                sdf = model.sdf_static(x)
                sdf = 1.0 / (torch.abs(sdf)+1e-6)
                return sdf

            for i in range(2):
                model.occupancy_grid_static.update_grid(get_density_static, S = 128)
      
      
        else:
            def get_density_dynamic(x, t, chunk=64**3):
                x_t = torch.cat([x, t * torch.ones_like(x[..., :1])], dim = -1)
                
                return batchify(model.density_dynamic, chunk)(x_t)

            for i in range(16):
                model.occupancy_grid_dynamic.update_grid(get_density_dynamic, S = 128)
      
        
    else:

    
        if "hybrid_neus" in args.net_model:
            def get_density_dynamic(x, t, chunk=48**3):
                x_t = torch.cat([x, t * torch.ones_like(x[..., :1])], dim = -1)
                
                
                return batchify(model.density_dynamic, chunk)(x_t)
                # return model.density_dynamic(x_t.reshape(-1, 4))

            if global_step % update_interval == 0 :
                # model.occupancy_grid_dynamic.update_grid(get_density_dynamic, S = 128)
                model.occupancy_grid_dynamic.update_grid(get_density_dynamic, S = 64)
                
            if neus_early_terminated:
                # don't update static occ grid for neus
                return 


            def get_density_static(x):
                neus = model.static_model
                sdf = model.sdf_static(x)
                sdf = 1.0 / (torch.abs(sdf)+1e-6)
                return sdf
         
            if global_step % 100 == 0 :
                model.occupancy_grid_static.update_grid(get_density_static, S = 128)

        else:
            def get_density_dynamic(x, t, chunk=64**3):
                x_t = torch.cat([x, t * torch.ones_like(x[..., :1])], dim = -1)
                
                return batchify(model.density_dynamic, chunk)(x_t)
                # return model.density_dynamic(x_t.reshape(-1, 4))

            if global_step % update_interval == 0 :
                # model.occupancy_grid_dynamic.update_grid(get_density_dynamic, S = 128)
                model.occupancy_grid_dynamic.update_grid(get_density_dynamic, S = 64)
                

def update_static_occ_grid(args, model, times=100):
    if not args.cuda_ray:
        return



    def get_density_static(x):
        neus = model.static_model
        sdf = model.sdf_static(x)
        sdf = 1.0 / (torch.abs(sdf)+1e-6)
        return sdf
    
    for time in range(times):
        model.occupancy_grid_static.update_grid(get_density_static, S = 128)

    


def init_occ_grid(args, model, poses = None, intrinsics = None, given_mask = None):
    # mark untrained grid using camera information
    # poses: [times * num_camera/times, 4, 4]
    # intrinsics: [times * num_camera/times, 3, 3]
    
    if not args.cuda_ray:
        return
    
    if "hybrid_neus" in args.net_model:
        model.occupancy_grid_dynamic.mark_untrained_grid(poses, intrinsics, given_mask, S = 128)
        time_size = model.occupancy_grid_dynamic.time_size
        C = poses.shape[0] // time_size
        poses = poses.reshape(C, time_size, 4, 4)
        intrinsics = intrinsics.reshape(C, time_size, 3, 3)
        model.occupancy_grid_static.mark_untrained_grid(poses[:,0,:,:], intrinsics[:,0,:,:], None, S = 128)
    else:
        AssertionError("Not implemented yet")


def discretize_points(voxel_points, voxel_size):
    # this function turns voxel centers/corners into integer indeices
    # we assume all points are alreay put as voxels (real numbers)

    minimal_voxel_point = voxel_points.min(dim=0, keepdim=True)[0]
    voxel_indices = ((voxel_points - minimal_voxel_point) / voxel_size).round_().long()  # float
    residual = (voxel_points - voxel_indices.type_as(voxel_points) * voxel_size).mean(0, keepdim=True)
    return voxel_indices, residual

def bbox2voxels(bbox, steps):
    vox_min, vox_max = bbox[:3], bbox[3:]
    voxel_size = (vox_max - vox_min) / (steps - 1)
    # steps = ((vox_max - vox_min) / voxel_size).round().astype('int64') + 1
    x, y, z = [c.reshape(-1).astype('float32') for c in np.meshgrid(np.arange(steps), np.arange(steps), np.arange(steps))]
    x, y, z = x * voxel_size[0] + vox_min[0], y * voxel_size[1] + vox_min[1], z * voxel_size[2] + vox_min[2]
    
    return np.stack([x, y, z]).T.astype('float32'), voxel_size
     
# https://github1s.com/facebookresearch/NSVF/blob/HEAD/fairnr/modules/encoder.py#L415-L454
def voxel_mesh(points, keep, voxel_size):

    keep = keep.squeeze(0)
    voxel_pts = points[keep.bool()]

    # generate polygon for voxels
    center_coords, residual = discretize_points(voxel_pts, voxel_size / 2)

    offsets = torch.tensor([[-1,-1,-1],[-1,-1,1],[-1,1,-1],[1,-1,-1],[1,1,-1],[1,-1,1],[-1,1,1],[1,1,1]], device=center_coords.device)
    vertex_coords = center_coords[:, None, :] + offsets[None, :, :]
    vertex_points = vertex_coords.type_as(residual) * voxel_size / 2 + residual
    
    faceidxs = [[1,6,7,5],[7,6,2,4],[5,7,4,3],[1,0,2,6],[1,5,3,0],[0,3,4,2]]
    all_vertex_keys, all_vertex_idxs  = {}, []
    for i in range(vertex_coords.shape[0]):
        for j in range(8):
            key = " ".join(["{}".format(int(p)) for p in vertex_coords[i,j]])
            if key not in all_vertex_keys:
                all_vertex_keys[key] = vertex_points[i,j]
                all_vertex_idxs += [key]
    all_vertex_dicts = {key: u for u, key in enumerate(all_vertex_idxs)}
    all_faces = torch.stack([torch.stack([vertex_coords[:, k] for k in f]) for f in faceidxs]).permute(2,0,1,3).reshape(-1,4,3)

    all_faces_keys = {}
    for l in range(all_faces.size(0)):
        key = " ".join(["{}".format(int(p)) for p in all_faces[l].sum(0) // 4])
        if key not in all_faces_keys:
            all_faces_keys[key] = all_faces[l]

    vertex = np.array([tuple(all_vertex_keys[key].cpu().tolist()) for key in all_vertex_idxs], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    face = np.array([([all_vertex_dicts["{} {} {}".format(*b)] for b in a.cpu().tolist()],) for a in all_faces_keys.values()],
        dtype=[('vertex_indices', 'i4', (4,))])
    return PlyData([PlyElement.describe(vertex, 'vertex'), PlyElement.describe(face, 'face')])

def extract_occ_grid(args, all_models):
    static_grid = None
    dynamic_grid = None
    if args.net_model == "siren":
        dynamic_grid = model.occupancy_grid_dynamic
    elif "hybrid_neus" in args.net_model:
        dynamic_grid = model.occupancy_grid_dynamic
        static_grid = model.occupancy_grid_static

    
    # test:
    grid = static_grid

    grid_points, voxel_size = bbox2voxels(grid.aabb.cpu().numpy(), grid.grid_size)

    density_thresh = min(grid.mean_density, grid.density_thresh)
    grid_keep_morton_3d = torch.where(grid.density_grid > density_thresh)[1]
    grid_keep_morton_3d_inverse = raymarching.morton3D_invert(grid_keep_morton_3d).long()

    grid_points = torch.tensor(grid_points, device = grid_keep_morton_3d.device)
    grid_keep_coord = torch.zeros((grid.grid_size, grid.grid_size, grid.grid_size), device = grid_keep_morton_3d.device)
    grid_keep_coord[grid_keep_morton_3d_inverse[:,0],grid_keep_morton_3d_inverse[:,1],grid_keep_morton_3d_inverse[:,2]] = 1 

    grid_keep = grid_keep_coord.reshape(-1)

    # grid_keep = grid_keep.squeeze()[raymarching.morton3D_invert(grid.morton3D_indices).long()] # coord inverse mapping
    # grid_keep = grid_keep.squeeze()[grid.morton3D_indices.long()] # coord inverse mapping
    
    # grid_keep = grid.density_grid > 1.0

    # grid = dynamic_grid
    # gird_id = dynamic_grid.time_size // 2

    # density_thresh = min(grid.mean_density[gird_id], grid.density_thresh)
    # grid_keep = grid.density_grid[gird_id] > density_thresh

  
    # grid_keep = static_grid.density_grid > -0.1
    voxel = voxel_mesh(grid_points, grid_keep, torch.tensor(voxel_size, device = grid_keep.device))
    # voxel = voxel_mesh(static_grid.grid_coords, grid_keep, static_grid.grid_size)

    return voxel, grid_points[grid_keep.squeeze(0).bool().cpu().numpy()]

def extract_occ_grid_dynamic(args, all_models):
    static_grid = None
    dynamic_grid = None
    if args.net_model == "siren":
        dynamic_grid = model.occupancy_grid_dynamic
    elif "hybrid_neus" in args.net_model:
        dynamic_grid = model.occupancy_grid_dynamic
        static_grid = model.occupancy_grid_static

    
    # test:
    grid = dynamic_grid

    grid_points, voxel_size = bbox2voxels(grid.aabb.cpu().numpy(), grid.grid_size)

    grid_id = 0
    # grid_id = dynamic_grid.time_size // 2

    density_thresh = min(grid.mean_density[grid_id], grid.density_thresh)
    grid_keep_morton_3d = torch.where(grid.density_grid[grid_id] > density_thresh)[1]
    grid_keep_morton_3d_inverse = raymarching.morton3D_invert(grid_keep_morton_3d).long()

    grid_points = torch.tensor(grid_points, device = grid_keep_morton_3d.device)
    grid_keep_coord = torch.zeros((grid.grid_size, grid.grid_size, grid.grid_size), device = grid_keep_morton_3d.device)
    grid_keep_coord[grid_keep_morton_3d_inverse[:,0],grid_keep_morton_3d_inverse[:,1],grid_keep_morton_3d_inverse[:,2]] = 1 

    grid_keep = grid_keep_coord.reshape(-1)

    # grid_keep = static_grid.density_grid > -0.1
    voxel = voxel_mesh(grid_points, grid_keep, torch.tensor(voxel_size, device = grid_keep.device))
    # voxel = voxel_mesh(static_grid.grid_coords, grid_keep, static_grid.grid_size)

    return voxel, grid_points[grid_keep.squeeze(0).bool().cpu().numpy()]

def extract_dynamic_grid_mask_gt(args, gt_densitys):
    grid_masks = []
    total_frame = args.time_size
    for frame in tqdm(range(total_frame)):
        gt_density = torch.tensor(gt_densitys[frame], device = 'cuda')

        mean_density = torch.mean(gt_density.clamp(min=0)).item()
        density_thresh = min(mean_density, 1.0)

        grid_keep = gt_density > density_thresh

        grid_masks.append(grid_keep)

    return grid_masks

def extract_dynamic_grid_mask(args, all_models, poses, intrinsics, target_res = 256):
    init_occ_grid(args, all_models, poses = poses, intrinsics = intrinsics, given_mask=None)
    update_occ_grid(args, all_models, eval_mode = True)

    total_frame = args.time_size
    grids = model.occupancy_grid_dynamic
    grid_masks = []
    for frame in tqdm(range(total_frame)):

        density_thresh = min(grids.mean_density[frame], grids.density_thresh)
        grid_keep_morton_3d = torch.where(grids.density_grid[frame] > density_thresh)[1]
        grid_keep_morton_3d_inverse = raymarching.morton3D_invert(grid_keep_morton_3d).long()

        grid_keep_coord = torch.zeros((grids.grid_size, grids.grid_size, grids.grid_size), device = grid_keep_morton_3d.device)
        grid_keep_coord[grid_keep_morton_3d_inverse[:,0],grid_keep_morton_3d_inverse[:,1],grid_keep_morton_3d_inverse[:,2]] = 1 

        # grid_keep = grid_keep_coord.reshape(-1) # [128, 128, 128]
        grid_keep = grid_keep_coord.squeeze()


        ## upsample to target_res
        grid_keep = grid_keep.unsqueeze(0).unsqueeze(0)
        grid_keep = F.interpolate(grid_keep, size = target_res, mode = "trilinear", align_corners = False)

        # grid_mask = grid_keep.permute(0, 4, 2, 3, 1).bool()
        grid_mask = grid_keep.squeeze().unsqueeze(0).unsqueeze(-1).bool()
        grid_masks.append(grid_mask)

    return grid_masks

