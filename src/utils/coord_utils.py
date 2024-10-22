import numpy as np
import sys, os
import imageio
import torch, torchvision
from torch import optim, nn
import torch.nn.functional as F

from src.utils.training_utils import batchify, batchify_func
from src.utils.visualize_utils import den_scalar2rgb, vel2hsv, vel_uv2hsv, write_ply, draw_mapping


#####################################################################
# Coord Tools (all for torch Tensors)
# Coords:
# 1. resolution space, Frames x Depth x H x W, coord (frame_t, voxel_z, voxel_y, voxel_x),
# 2. simulation space, scale the resolution space to around 0-1, 
#    (FrameLength and Width in [0-1], Height and Depth keep ratios wrt Width)
# 3. target space, 
# 4. world space,
# 5. camera spaces,

# Vworld, Pworld; velocity, position in 4. world coord.
# Vsmoke, Psmoke; velocity, position in 2. simulation coord.
# w2s, 4.world to 3.target matrix (vel transfer uses rotation only; pos transfer includes offsets)
# s2w, 3.target to 4.world matrix (vel transfer uses rotation only; pos transfer includes offsets)
# scale_vector, to scale from 2.simulation space to 3.target space (no rotation, no offset)
#        for synthetic data, scale_vector = openvdb voxel size * [W,H,D] grid resolution (x first, z last), 
#        for e.g., scale_vector = 0.0469 * 256 = 12.0064
# st_factor, spatial temporal resolution ratio, to scale velocity from 2.simulation unit to 1.resolution unit
#        for e.g.,  st_factor = [W/float(max_timestep),H/float(max_timestep),D/float(max_timestep)]

# functions to transfer between 4. world space and 2. simulation space, 
# velocity are further scaled according to resolution as in mantaflow
def vel_world2smoke(Vworld, w2s, scale_vector, st_factor):
    _st_factor = torch.Tensor(st_factor).expand((3, ))
    vel_rot = Vworld[..., None, :] * (w2s[:3,:3])
    vel_rot = torch.sum(vel_rot, -1) # 4.world to 3.target 
    vel_scale = vel_rot / (scale_vector) * _st_factor # 3.target to 2.simulation
    return vel_scale

def vel_smoke2world(Vsmoke, s2w, scale_vector, st_factor):
    _st_factor = torch.Tensor(st_factor).expand((3, ))
    vel_scale = Vsmoke * (scale_vector) / _st_factor # 2.simulation to 3.target
    vel_rot = torch.sum(vel_scale[..., None, :] * (s2w[:3,:3]), -1) # 3.target to 4.world
    return vel_rot
    
def pos_world2smoke(Pworld, w2s, scale_vector):
    pos_rot = torch.sum(Pworld[..., None, :] * (w2s[:3,:3]), -1) # 4.world to 3.target 
    pos_off = (w2s[:3, -1]).expand(pos_rot.shape) # 4.world to 3.target 
    new_pose = pos_rot + pos_off 
    pos_scale = new_pose / (scale_vector) # 3.target to 2.simulation
    return pos_scale

def off_smoke2world(Offsmoke, s2w, scale_vector):
    off_scale = Offsmoke * (scale_vector) # 2.simulation to 3.target
    off_rot = torch.sum(off_scale[..., None, :] * (s2w[:3,:3]), -1)  # 3.target to 4.world
    return off_rot

def pos_smoke2world(Psmoke, s2w, scale_vector):
    pos_scale = Psmoke * (scale_vector) # 2.simulation to 3.target
    pos_rot = torch.sum(pos_scale[..., None, :] * (s2w[:3,:3]), -1)  # 3.target to 4.world
    pos_off = (s2w[:3, -1]).expand(pos_rot.shape) # 3.target to 4.world
    return pos_rot+pos_off



class BBox_Tool(object):
    def __init__(self, smoke_tran_inv, smoke_scale, in_min=0.0, in_max=1.0):
        self.s_w2s = torch.Tensor(smoke_tran_inv).expand([4,4])
        self.s_scale = torch.Tensor(smoke_scale).expand([3])
        self.s_min = torch.Tensor(in_min).expand([3])
        self.s_max = torch.Tensor(in_max).expand([3])

        ## bbox in world coordinate
        bbox_vertices_local = torch.tensor([[in_min,in_min,in_min],[in_min,in_min,in_max],[in_min,in_max,in_min],[in_min,in_max,in_max],[in_max,in_min,in_min],[in_max,in_min,in_max],[in_max,in_max,in_min],[in_max,in_max,in_max]])
        bbox_vertices_world = pos_smoke2world(bbox_vertices_local, torch.inverse(smoke_tran_inv), smoke_scale)

        bbox_world_min = torch.min(bbox_vertices_world, dim=0)[0][0]
        bbox_world_max = torch.max(bbox_vertices_world, dim=0)[0][0]

        self.world_bbox = torch.tensor([[bbox_world_min[0],bbox_world_min[1],bbox_world_min[2]],[bbox_world_max[0],bbox_world_max[1],bbox_world_max[2]]])


    def setMinMax(self, in_min=0.0, in_max=1.0):
        self.s_min = torch.Tensor(in_min).expand([3])
        self.s_max = torch.Tensor(in_max).expand([3])

    def isInside(self, inputs_pts):
        target_pts = pos_world2smoke(inputs_pts, self.s_w2s, self.s_scale)
        above = torch.logical_and(target_pts[...,0] >= self.s_min[0], target_pts[...,1] >= self.s_min[1] ) 
        above = torch.logical_and(above, target_pts[...,2] >= self.s_min[2] ) 
        below = torch.logical_and(target_pts[...,0] <= self.s_max[0], target_pts[...,1] <= self.s_max[1] ) 
        below = torch.logical_and(below, target_pts[...,2] <= self.s_max[2] ) 
        outputs = torch.logical_and(below, above)
        return outputs

    def insideMask(self, inputs_pts):
        return self.isInside(inputs_pts).to(torch.float)
    
def get_voxel_pts(H, W, D, s2w, scale_vector, n_jitter=0, r_jitter=0.8):
    """Get voxel positions."""
    
    i, j, k = torch.meshgrid(torch.linspace(0, D-1, D),
                       torch.linspace(0, H-1, H),
                       torch.linspace(0, W-1, W))
    pts = torch.stack([(k+0.5)/W, (j+0.5)/H, (i+0.5)/D], -1) 
    # shape D*H*W*3, value [(x,y,z)] , range [0,1]

    jitter_r = torch.Tensor([r_jitter/W,r_jitter/H,r_jitter/D]).float().expand(pts.shape)
    for i_jitter in range(n_jitter):
        off_i = torch.rand(pts.shape, dtype=torch.float)-0.5
        # shape D*H*W*3, value [(x,y,z)] , range [-0.5,0.5]

        pts = pts + off_i * jitter_r

    return pos_smoke2world(pts, s2w, scale_vector)


def get_voxel_pts_offset(H, W, D, s2w, scale_vector, r_offset=0.8):
    """Get voxel positions."""
    
    i, j, k = torch.meshgrid(torch.linspace(0, D-1, D),
                       torch.linspace(0, H-1, H),
                       torch.linspace(0, W-1, W))
    pts = torch.stack([(k+0.5)/W, (j+0.5)/H, (i+0.5)/D], -1) 
    # shape D*H*W*3, value [(x,y,z)] , range [0,1]

    jitter_r = torch.Tensor([r_offset/W,r_offset/H,r_offset/D]).expand(pts.shape)
    off_i = torch.rand([1,1,1,3], dtype=torch.float)-0.5
    # shape 1*1*1*3, value [(x,y,z)] , range [-0.5,0.5]
    pts = pts + off_i * jitter_r

    return pos_smoke2world(pts, s2w, scale_vector)

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
    jac = torch.cat(jac, 1)
    return jac

class Voxel_Tool(object):

    def __get_tri_slice(self, _xm, _ym, _zm, _n=1):
        _yz = torch.reshape(self.pts[...,_xm:_xm+_n,:],(-1,3))
        _zx = torch.reshape(self.pts[:,_ym:_ym+_n,...],(-1,3))
        _xy = torch.reshape(self.pts[_zm:_zm+_n,...],(-1,3))
        
        pts_mid = torch.cat([_yz, _zx, _xy], dim=0)
        npMaskXYZ = [np.zeros([self.D,self.H,self.W,1], dtype=np.float32) for _ in range(3)]
        npMaskXYZ[0][...,_xm:_xm+_n,:] = 1.0
        npMaskXYZ[1][:,_ym:_ym+_n,...] = 1.0
        npMaskXYZ[2][_zm:_zm+_n,...] = 1.0
        return pts_mid, torch.tensor(np.clip(npMaskXYZ[0]+npMaskXYZ[1]+npMaskXYZ[2], 1e-6, 3.0))

    def __pad_slice_to_volume(self, _slice, _n, mode=0):
        # mode: 0, x_slice, 1, y_slice, 2, z_slice
        tar_shape = [self.D,self.H,self.W]
        in_shape = tar_shape[:]
        in_shape[-1-mode] = _n
        fron_shape = tar_shape[:]
        fron_shape[-1-mode] = (tar_shape[-1-mode] - _n)//2
        back_shape = tar_shape[:]
        back_shape[-1-mode] = (tar_shape[-1-mode] - _n - fron_shape[-1-mode])

        cur_slice = _slice.view(in_shape+[-1])
        front_0 = torch.zeros(fron_shape + [cur_slice.shape[-1]])
        back_0 = torch.zeros(back_shape + [cur_slice.shape[-1]])

        volume = torch.cat([front_0, cur_slice, back_0], dim=-2-mode)
        return volume


    def __init__(self, smoke_tran, smoke_tran_inv, smoke_scale, D, H, W, middleView=None, hybrid_neus = False):
        self.s_s2w = torch.Tensor(smoke_tran).expand([4,4])
        self.s_w2s = torch.Tensor(smoke_tran_inv).expand([4,4])
        self.s_scale = torch.Tensor(smoke_scale).expand([3])
        self.D = D
        self.H = H
        self.W = W
        self.pts = get_voxel_pts(H, W, D, self.s_s2w, self.s_scale)
        self.pts_mid = None
        self.npMaskXYZ = None
        self.middleView = middleView
        if middleView is not None:
            _n = 1 if self.middleView=="mid" else 3
            _xm,_ym,_zm = (W-_n)//2, (H-_n)//2, (D-_n)//2
            self.pts_mid, self.npMaskXYZ = self.__get_tri_slice(_xm,_ym,_zm,_n)
        
        self.hybrid_neus = hybrid_neus

    def get_raw_geometry_at_pts(self, model, cur_pts, chunk=1024*32):
        input_shape = list(cur_pts.shape[0:-1])

        pts_flat = cur_pts.view(-1, 4)
        pts_N = pts_flat.shape[0]
        # Evaluate model
        all_raw = []
        for i in range(0, pts_N, chunk):
            pts_i = pts_flat[i:i+chunk]
            
            # raw_i = model.forward_geometry(pts_i) 
            if model.use_two_level_density:
                raw_i = model.forward_geometry_all(pts_i) 
            else:
                raw_i = model.forward_geometry(pts_i) 
            if model.training:
                all_raw.append(raw_i)
            else:
                all_raw.append(raw_i.detach())

        raw = torch.cat(all_raw, 0).view(input_shape+[-1])
        return raw

    def get_density_flat(self, model, cur_pts, chunk=1024*32, getSDF = False):
        flat_raw = self.get_raw_geometry_at_pts(model, cur_pts, chunk)

        den_raw = [F.relu(flat_raw[...,-1:]).reshape(-1,1)] if not model.use_two_level_density else [F.relu(flat_raw[...,-2:-1].reshape(-1,1)), F.relu(flat_raw[...,-1:].reshape(-1,1))]
        # lagrangian_density, siren_density
        
        if model.single_scene:
            return den_raw
        
        else:
            static_sdf = flat_raw[...,0:1]
            if not getSDF:
                inv_s = model.get_deviation()
                static_sdf = torch.sigmoid(static_sdf * inv_s).squeeze(-1)
                static_sdf = inv_s * static_sdf * (1. - static_sdf)

            return [*den_raw, static_sdf.reshape(-1,1)]
            # static_normal = flat_raw[...,1:3]
            # return [den_raw, static_sdf, static_normal]
   

    def get_velocity_flat(self, model, cur_pts,chunk=1024*32,):
        pts_N = cur_pts.shape[0]
        world_v = []
        for i in range(0, pts_N, chunk):
            input_i = cur_pts[i:i+chunk]
            # vel_i = batchify(model, chunk)(input_i)
            vel_i = batchify_func(model.dynamic_model_lagrangian.velocity_model, chunk, not model.training)(input_i)
            world_v.append(vel_i)
        world_v = torch.cat(world_v, 0)
        return world_v

    def get_voxel_density_list(self, model, t=None,chunk=1024*32,middle_slice=False):
        model.eval()
        D,H,W = self.D,self.H,self.W
        # middle_slice, only for fast visualization of the middle slice
        pts_flat = self.pts_mid if middle_slice else self.pts.view(-1, 3)
        pts_N = pts_flat.shape[0]
        if t is not None:
            input_t = torch.ones([pts_N, 1])*float(t)
            pts_flat = torch.cat([pts_flat,input_t], dim=-1)


        raw_list = self.get_density_flat(model, pts_flat, chunk)


        # import pdb
        return_list = []
        for den_raw in raw_list:
            if middle_slice:
                # only for fast visualization of the middle slice
                _n = 1 if self.middleView=="mid" else 3
                _yzV, _zxV, _xyV = torch.split(den_raw, [D*H*_n,D*W*_n,H*W*_n], dim=0)
                mixV = self.__pad_slice_to_volume(_yzV, _n, 0) + self.__pad_slice_to_volume(_zxV, _n, 1) +self.__pad_slice_to_volume(_xyV, _n, 2)
                return_list.append(mixV / self.npMaskXYZ)
            else:
                return_list.append(den_raw.view(D,H,W,1))
        model.train()
        return return_list
        
    def get_voxel_velocity(self,model, scale, t, chunk=1024*32, middle_slice=False):
        model.eval()
        # middle_slice, only for fast visualization of the middle slice
        D,H,W = self.D,self.H,self.W
        pts_flat = self.pts_mid if middle_slice else self.pts.view(-1, 3)
        pts_N = pts_flat.shape[0]
        if t is not None:
            input_t = torch.ones([pts_N, 1])*float(t)
            pts_flat = torch.cat([pts_flat,input_t], dim=-1)


        # world_v = self.get_velocity_flat(model.dynamic_model_lagrangian.velocity_model, pts_flat, chunk)
        world_v = self.get_velocity_flat(model, pts_flat, chunk)
        reso_scale = [self.W*scale,self.H*scale,self.D*scale]
        target_v = vel_world2smoke(world_v, self.s_w2s, self.s_scale, reso_scale)

        if middle_slice:
            _n = 1 if self.middleView=="mid" else 3
            _yzV, _zxV, _xyV = torch.split(target_v, [D*H*_n,D*W*_n,H*W*_n], dim=0)
            mixV = self.__pad_slice_to_volume(_yzV, _n, 0) + self.__pad_slice_to_volume(_zxV, _n, 1) +self.__pad_slice_to_volume(_xyV, _n, 2)
            target_v = mixV / self.npMaskXYZ
        else:
            target_v = target_v.view(D,H,W,3) 
        model.train()
        return target_v


    def save_voxel_den_npz(self,model, den_path, t, chunk=1024*32,save_npz=True,save_jpg=False, jpg_mix=True, noStatic=False):
        voxel_den_list = self.get_voxel_density_list(model, t, chunk, middle_slice=not (jpg_mix or save_npz) )
        
        head_tail = os.path.split(den_path)
        if model.use_two_level_density:
            namepre = ["lagrangian_","","static_"]
        else:
            namepre = ["","static_"]
        ret = {}
        for voxel_den, npre in zip(voxel_den_list, namepre):
            voxel_den = voxel_den.detach().cpu().numpy()
            if save_jpg:
                jpg_path = os.path.join(head_tail[0], npre + os.path.splitext(head_tail[1])[0]+".jpg")
                # rgb = den_scalar2rgb(voxel_den, scale=None, is3D=True, logv=False, mix=jpg_mix)
                rgb = den_scalar2rgb(voxel_den, scale=100, is3D=True, logv=False, mix=jpg_mix)
                imageio.imwrite(jpg_path, rgb)
                if not 'static' in npre:
                    ret[npre] = rgb
            if model.args.save_jacobian_den:
                jacobian_den= jacobianDen_np(voxel_den)
                rgb = vel_uv2hsv(jacobian_den[0], scale=160, is3D=True, logv=False)
                jpg_path = os.path.join(head_tail[0], npre + os.path.splitext(head_tail[1])[0]+"_jacobian.jpg")
                imageio.imwrite(jpg_path, rgb)
                ret[npre+"jacobian"] = rgb
            if save_npz:
                # to save some space
                npz_path = os.path.join(head_tail[0], npre + os.path.splitext(head_tail[1])[0]+".npz")
                voxel_den = np.float16(voxel_den)
                np.savez_compressed(npz_path, vel=voxel_den)
                
            if noStatic and 'static' in npre:
                break
        return ret

    @torch.no_grad()
    def vis_cross_feature_error_voxel(self, path, t, dynamic_model_lagrangian, middle_slice = False, chunk = 1024*32):

        dynamic_model_lagrangian.eval_mode = True
        # middle_slice, only for fast visualization of the middle slice
        D,H,W = self.D,self.H,self.W
        pts_flat = self.pts_mid if middle_slice else self.pts.view(-1, 3)
        pts_N = pts_flat.shape[0]


        # get feature0
        feature_0 = []
        for i in range(0, pts_N, chunk):
            input_i = pts_flat[i:i+chunk]
            feature = dynamic_model_lagrangian.velocity_model.forward_feature(input_i, torch.ones([input_i.shape[0], 1])*float(0)).detach()
            feature_0.append(feature)

        feature_0 = torch.cat(feature_0, 0)



        feature_t = []
        for i in range(0, pts_N, chunk):
            input_i = pts_flat[i:i+chunk]
            mapped_xyz = dynamic_model_lagrangian.velocity_model.mapping_forward(torch.cat([input_i, torch.ones([input_i.shape[0], 1])*float(t)], dim = -1), torch.ones([input_i.shape[0], 1])*float(0)).detach()
            feature = dynamic_model_lagrangian.velocity_model.forward_feature(mapped_xyz, torch.ones([mapped_xyz.shape[0], 1])*float(t)).detach()
            feature_t.append(feature)
            
        feature_t = torch.cat(feature_t, 0)

        feature_error = (feature_t - feature_0).abs()

        if middle_slice:
            _n = 1 if self.middleView=="mid" else 3
            # _n = 16
            _yzV, _zxV, _xyV = torch.split(feature_error, [D*H*_n,D*W*_n,H*W*_n], dim=0)
            mixV = self.__pad_slice_to_volume(_yzV, _n, 0) + self.__pad_slice_to_volume(_zxV, _n, 1) +self.__pad_slice_to_volume(_xyV, _n, 2)
            feature_error = mixV / self.npMaskXYZ
        else:
            feature_error = feature_error.view(D,H,W,-1) 
            
        dynamic_model_lagrangian.eval_mode = False

        error_img = vel_uv2hsv(feature_error.cpu(), scale=160, is3D=True, logv=False)

        imageio.imwrite(path, error_img)

        return error_img

    @torch.no_grad()
    def vis_feature_voxel(self, model, path, t = 0.0, middle_slice = False, chunk = 1024*32):
        # t: the time step of this feature mapc

        model.eval()
        # middle_slice, only for fast visualization of the middle slice
        D,H,W = self.D,self.H,self.W
        pts_flat = self.pts_mid if middle_slice else self.pts.view(-1, 3)
        pts_N = pts_flat.shape[0]


        # get feature0

        feature_t = []
        for i in range(0, pts_N, chunk):
            input_i = pts_flat[i:i+chunk]
            feature = model.dynamic_model_lagrangian.velocity_model.forward_feature(input_i, torch.ones([input_i.shape[0], 1])*float(t)).detach()
            feature_t.append(feature)
            
        feature_t = torch.cat(feature_t, 0)

        if middle_slice:
            _n = 1 if self.middleView=="mid" else 3
            # _n = 16
            _yzV, _zxV, _xyV = torch.split(feature_t, [D*H*_n,D*W*_n,H*W*_n], dim=0)
            mixV = self.__pad_slice_to_volume(_yzV, _n, 0) + self.__pad_slice_to_volume(_zxV, _n, 1) +self.__pad_slice_to_volume(_xyV, _n, 2)
            feature_t = mixV / self.npMaskXYZ
        else:
            feature_t = feature_t.view(D,H,W,-1) 
            
        feature_img = feature_t[...,:3].cpu()
        # get three mid slice and concat
        feature_img_x = feature_img[D//2:D//2+1,...].squeeze(0)
        feature_img_y = feature_img[:,H//2:H//2+1,...].squeeze(1)
        feature_img_z = feature_img[...,W//2:W//2+1,:].squeeze(2)
        
        
        def normalize_img(feature_img):
            feature_img = feature_img
            channel_min = feature_img.reshape(-1,3).min(dim=0)[0].reshape(1,1,3)
            channel_max = feature_img.reshape(-1,3).max(dim=0)[0].reshape(1,1,3)
            feature_img = (feature_img - channel_min) / (channel_max - channel_min)
            
            # feature_img = feature_img.abs()
            # feature_img  = feature_img / feature_img.max()
            feature_img = feature_img.cpu().numpy()
            
            return feature_img
        
        normalized_feature_img_x = normalize_img(feature_img_x)
        normalized_feature_img_y = normalize_img(feature_img_y)
        normalized_feature_img_z = normalize_img(feature_img_z)
        
        normalized_feature_img_x = (normalized_feature_img_x * 255).astype(np.uint8)
        normalized_feature_img_y = (normalized_feature_img_y * 255).astype(np.uint8)
        normalized_feature_img_z = (normalized_feature_img_z * 255).astype(np.uint8)
        imageio.imwrite(path+'_yz.png', normalized_feature_img_x)
        imageio.imwrite(path+'_zx.png', normalized_feature_img_y)
        imageio.imwrite(path+'_xy.png', normalized_feature_img_z)
        
        # feature_img = torch.cat([feature_img_x, feature_img_y, feature_img_z], dim=0)
        # feature_img = vel_uv2hsv(feature_t.cpu(), scale=160, is3D=True, logv=False)
        # imageio.imwrite(path, feature_img.numpy().astype(np.int8))
        # imageio.imwrite(path, (feature_img.numpy()* 255).astype(np.uint8) )

        # imageio.imwrite('test_feature_x.png', feature_img_x * 255)
        

        return normalized_feature_img_x, normalized_feature_img_y, normalized_feature_img_z, feature_img_x, feature_img_y, feature_img_z

    @torch.no_grad()
    def vis_mapping_voxel(self, frame_list, t_list, model, sample_pts = 128, change_feature_interval = 1, chunk = 1024*32):

        dynamic_model_lagrangian = model.dynamic_model_lagrangian
        dynamic_model = model.dynamic_model


        pts_flat = self.pts.view(-1, 3)
        def get_density_time(pts_flat, time):
            # only choose density points
            pts_N = pts_flat.shape[0]
            density = []
            for i in range(0, pts_N, chunk):
                input_i = pts_flat[i:i+chunk]
                density_temp = dynamic_model.density(torch.cat([input_i, torch.ones([input_i.shape[0], 1])*float(time)], dim = -1)).detach()
                density.append(density_temp)

            density = torch.cat(density, dim = 0)
            
            return density
        
        time_0 = t_list[frame_list[0]]
        density_0 = get_density_time(pts_flat, time_0)
        # import pdb
        # pdb.set_trace()

        density_mean = density_0.clamp(0.0, 1e5).mean()

        
        pts_num = sample_pts
        pts_flat = pts_flat[density_0.squeeze(-1) >= density_mean]
        if len(pts_flat) == 0:
            # sort the density
            pts_flat = pts_flat[density_0.squeeze(-1).argsort(descending=True)]
            pts_sampled = pts_flat[:sample_pts]
        else:
            sample_id = np.random.randint(0, pts_flat.shape[0], pts_num)
            pts_sampled = pts_flat[sample_id].reshape(-1,3)

        all_xyz = []
        
        feature_sampled = dynamic_model_lagrangian.velocity_model.forward_feature(pts_sampled,  torch.ones([pts_sampled.shape[0], 1])*float(time_0)).detach()
        base_mapped_xyz = dynamic_model_lagrangian.velocity_model.mapping_forward_using_features(feature_sampled, torch.ones([pts_sampled.shape[0], 1])*float(time_0))
        
        mapped_xyz = pts_sampled
        base_world_xyz = pts_sampled
        for idx, frame_i in enumerate(frame_list):
            if idx == 0:
                all_xyz.append(mapped_xyz.detach().cpu().numpy())
                continue
            
            cur_t = t_list[frame_i]
            mapped_xyz_next = dynamic_model_lagrangian.velocity_model.mapping_forward_using_features(feature_sampled, torch.ones([pts_sampled.shape[0], 1])*float(cur_t))
            mapped_xyz = mapped_xyz_next - base_mapped_xyz + base_world_xyz
            if (idx) % change_feature_interval == 0:
                feature_sampled = dynamic_model_lagrangian.velocity_model.forward_feature(mapped_xyz,  torch.ones([mapped_xyz.shape[0], 1])*float(cur_t)).detach()
                base_mapped_xyz = dynamic_model_lagrangian.velocity_model.mapping_forward_using_features(feature_sampled, torch.ones([pts_sampled.shape[0], 1])*float(cur_t))
                base_world_xyz = mapped_xyz

                
            all_xyz.append(mapped_xyz.detach().cpu().numpy())

        all_xyz = np.array(all_xyz)
        write_ply(all_xyz.reshape(-1,3),'vis_mapping.ply')

        return torch.tensor(all_xyz)

    def vis_vel_integration(self, frame_list, t_list, model, sample_pts = 128, change_feature_interval = 1, chunk = 1024*32):

        dynamic_model_lagrangian = model.dynamic_model_lagrangian
        # dynamic_model = model.dynamic_model_siren
        dynamic_model = model.dynamic_model
        # middle_slice, only for fast visualization of the middle slice
        D,H,W = self.D,self.H,self.W

        pts_flat = self.pts.view(-1, 3)
        def get_density_time(pts_flat, time):
            # only choose density points
            pts_N = pts_flat.shape[0]
            density = []
            for i in range(0, pts_N, chunk):
                input_i = pts_flat[i:i+chunk]
                density_temp = dynamic_model.density(torch.cat([input_i, torch.ones([input_i.shape[0], 1])*float(time)], dim = -1)).detach()
                density.append(density_temp)

            density = torch.cat(density, dim = 0)
            
            return density
        
        time_0 = t_list[frame_list[0]]
        density_0 = get_density_time(pts_flat, time_0)
        # import pdb
        # pdb.set_trace()

        # density_mean = density_0.clamp(0.0, 1e5).mean()
        # # pts_flat = pts_flat[density_0.squeeze(-1) >= 8.0]
        # # pts_flat = pts_flat[density_0.squeeze(-1) >= 6.0]
        # # pts_flat = pts_flat[density_0.squeeze(-1) >= 3.0]
        # # pts_flat = pts_flat[density_0.squeeze(-1) >= 3.0]
        # pts_flat = pts_flat[density_0.squeeze(-1) >= density_mean]
            
        # pts_num = sample_pts

        # import random
        # sample_id = np.random.randint(0, pts_flat.shape[0], pts_num)
        # pts_sampled = pts_flat[sample_id].reshape(-1,3)

        pts_flat = pts_flat[density_0.squeeze(-1).argsort(descending=True)]
        pts_sampled = pts_flat[:sample_pts]




        pts_N = pts_sampled.shape[0]
        all_xyz = []

        delta_T = t_list[frame_list[1]] - t_list[frame_list[0]]
        
        mapped_xyz = pts_sampled
        for idx, frame_i in enumerate(frame_list):
         
            if idx == 0:
                # all_xyz.append(pts_sampled.detach().cpu().numpy())
                continue
            
            cur_t = t_list[frame_i]
            input_t = torch.ones([pts_N, 1])*float(cur_t)
            pts_flat = torch.cat([mapped_xyz, input_t], dim=-1)
            world_v = self.get_velocity_flat(model, pts_flat, chunk)
            mapped_xyz = mapped_xyz + world_v * delta_T
            all_xyz.append(mapped_xyz.detach().cpu().numpy())


        return torch.tensor(all_xyz)

    @torch.no_grad()
    def eval_mapping_error(self, frame_list, t_list, model, sample_pts = 128, chunk = 1024*32):

        dynamic_model_lagrangian = model.dynamic_model_lagrangian
        dynamic_model = model.dynamic_model
        # dynamic_model = model.dynamic_model_lagrangian
        # middle_slice, only for fast visualization of the middle slice
        D,H,W = self.D,self.H,self.W


        # pts_flat = self.pts_mid
        # only use xy plane with z = 0.5
        # pts_flat = pts_flat[:pts_flat.shape[0]//3]
   
        # pts_flat = pts_flat[-pts_flat.shape[0]//3:]
        # pts_flat = pts_flat[:pts_flat.shape[0]//3]
        # pts_flat = pts_flat[:-pts_flat.shape[0]//3]

        pts_flat = self.pts.view(-1, 3)

        time_0 = t_list[frame_list[0]]
        
        # only choose density points
        pts_N = pts_flat.shape[0]
        density_0 = []
        for i in range(0, pts_N, chunk):
            input_i = pts_flat[i:i+chunk]
            density = dynamic_model.density(torch.cat([input_i, torch.ones([input_i.shape[0], 1])*time_0], dim = -1)).detach()
            density_0.append(density)

        density_0 = torch.cat(density_0, dim = 0)

        density_mean = density_0.clamp(0.0, 1e5).mean()
        # # pts_flat = pts_flat[density_0.squeeze(-1) >= 0.50]
        # pts_flat = pts_flat[density_0.squeeze(-1) >= 8.0]
        # # pts_flat = pts_flat[density_0.squeeze(-1) >= 0.05]
        # pts_flat = pts_flat[density_0.squeeze(-1) >= 0.05]
        pts_flat = pts_flat[density_0.squeeze(-1) >= density_mean]
            
        pts_num = sample_pts

        sample_id = np.random.randint(0, pts_flat.shape[0], pts_num)
        pts_sampled = pts_flat[sample_id].reshape(-1,3)

        # pts_flat = pts_flat[density_0.squeeze(-1).argsort(descending=True)]
        # pts_sampled = pts_flat[:sample_pts]
        
        gt_all_xyz = []
        
        feature_sampled = dynamic_model_lagrangian.velocity_model.forward_feature(pts_sampled,  torch.ones([pts_sampled.shape[0], 1])*float(time_0)).detach()
        base_mapped_xyz = dynamic_model_lagrangian.velocity_model.mapping_forward_using_features(feature_sampled, torch.ones([pts_sampled.shape[0], 1])*float(time_0))

        mapped_xyz = pts_sampled
        base_world_xyz = pts_sampled
        for idx, frame_i in enumerate(frame_list):
            if idx == 0:
                gt_all_xyz.append(mapped_xyz.detach().cpu().numpy())
                continue
            
            cur_t = t_list[frame_i]
            mapped_xyz_next = dynamic_model_lagrangian.velocity_model.mapping_forward_using_features(feature_sampled, torch.ones([pts_sampled.shape[0], 1])*float(cur_t))
            mapped_xyz = mapped_xyz_next - base_mapped_xyz + base_world_xyz
            if (idx) % 1 == 0:
                feature_sampled = dynamic_model_lagrangian.velocity_model.forward_feature(mapped_xyz,  torch.ones([mapped_xyz.shape[0], 1])*float(cur_t)).detach()
                base_mapped_xyz = dynamic_model_lagrangian.velocity_model.mapping_forward_using_features(feature_sampled, torch.ones([pts_sampled.shape[0], 1])*float(cur_t))
                base_world_xyz = mapped_xyz
                
            gt_all_xyz.append(mapped_xyz)
            # gt_all_xyz.append(mapped_xyz.detach().cpu().numpy())
            
        # pred_all_xyz = []
        all_feature_error = []
        all_mapping_errpr = []
        # pred_all_feature_relative_error = []
        # pred_all_ratio = []
        
        feature_sampled = dynamic_model_lagrangian.velocity_model.forward_feature(pts_sampled,  torch.ones([pts_sampled.shape[0], 1])*float(time_0)).detach()
        base_mapped_xyz = dynamic_model_lagrangian.velocity_model.mapping_forward_using_features(feature_sampled, torch.ones([pts_sampled.shape[0], 1])*float(time_0))

        mapped_xyz = pts_sampled
        base_world_xyz = pts_sampled  
        for frame_i in frame_list[1:]:

            cur_t = t_list[frame_i]
            cur_t = t_list[frame_i]
            mapped_xyz_next = dynamic_model_lagrangian.velocity_model.mapping_forward_using_features(feature_sampled, torch.ones([pts_sampled.shape[0], 1])*float(cur_t))
            mapped_xyz = mapped_xyz_next - base_mapped_xyz + base_world_xyz
  
  
            cur_feature_sampled = dynamic_model_lagrangian.velocity_model.forward_feature(mapped_xyz,  torch.ones([mapped_xyz.shape[0], 1])*float(cur_t)).detach()
            # feature_error = ((feature_sampled - feature_sampled_0) ** 2).mean()
            # pred_all_feature_error.append(feature_error.detach().cpu().numpy())
            # pred_all_feature.append(feature_sampled.detach().cpu().numpy())
            # F.smooth_l1_loss(feature_sampled, feature_sampled_0)
            # F.l1_loss(feature_sampled, feature_sampled_0)
            feature_error = (feature_sampled - cur_feature_sampled).norm(dim=-1).mean()
            mapping_error = (gt_all_xyz[frame_i] - mapped_xyz).norm(dim=-1).mean()
            
            all_mapping_errpr.append(mapping_error.detach().cpu().numpy())
            all_feature_error.append(feature_error.detach().cpu().numpy())
            
            # if (frame_i) % 50 == 0:
            if (frame_i) % 5000 == 0:
                feature_sampled = dynamic_model_lagrangian.velocity_model.forward_feature(mapped_xyz,  torch.ones([mapped_xyz.shape[0], 1])*float(cur_t)).detach()
                base_mapped_xyz = dynamic_model_lagrangian.velocity_model.mapping_forward_using_features(feature_sampled, torch.ones([pts_sampled.shape[0], 1])*float(cur_t))
                base_world_xyz = mapped_xyz
            # (relative_error[relative_error < 0.2]).mean()
            # msre = torch.mean(((feature_sampled - feature_sampled_0) / (feature_sampled_0 + 1e-6)) ** 2)
            # import pdb
            # pdb.set_trace()
            # import pdb
            # pdb.set_trace()

            # ((feature_sampled-feature_sampled_0)**2).mean()
            # (((feature_sampled-feature_sampled_0)**2) / (feature_sampled_0**2)).mean()
            # relative_error = (feature_sampled - feature_sampled_0).norm(dim=-1) / torch.max((feature_sampled_0.norm(dim=-1) + 1e-6), feature_sampled.norm(dim=-1) + 1e-6)
            # relative_error_num = (relative_error < 0.2).sum()
            # relative_error = relative_error.mean()
            # feature_ratio = relative_error_num.item() / sample_pts
            # pred_all_feature_error.append(error.detach().cpu().numpy())
            # pred_all_feature_relative_error.append(relative_error.detach().cpu().numpy())
            # pred_all_ratio.append(feature_ratio)
            # F.l1_loss(feature_sampled, feature_sampled_0)

        
        return np.array(all_feature_error), np.array(all_mapping_errpr)


        # pred_mapped_xyz = np.array(pred_all_xyz)
        # gt_mapped_xyz = np.array(gt_all_xyz) # [frame_N, sample_pts, 3]
        
        # l2_error = np.linalg.norm(pred_mapped_xyz - gt_mapped_xyz, axis=-1).mean(axis=-1)
        # l2_distane = np.linalg.norm(pred_mapped_xyz - gt_mapped_xyz, axis=-1)
        
        # # l2_distane_num = np.linalg.norm(pred_mapped_xyz - gt_mapped_xyz, axis=-1) < 0.1
        # # l2_distane_num = (np.linalg.norm(pred_mapped_xyz - gt_mapped_xyz, axis=-1) < 0.2).sum(axis=-1)
        # l2_distane_num = (np.linalg.norm(pred_mapped_xyz - gt_mapped_xyz, axis=-1) < 0.1).sum(axis=-1)
        # ratio = l2_distane_num / sample_pts
        # # l2_distane_num = l2_distane_num
        # # l2_distane = np.linalg.norm(pred_mapped_xyz - gt_mapped_xyz, axis=-1).min(axis=-1)
        # # np.linalg.norm(pred_mapped_xyz - gt_mapped_xyz, axis=-1).max(axis=-1)

        # # l2_error = np.linalg.norm(pred_mapped_xyz - gt_mapped_xyz, axis=-1).mean(axis=-1)
        # # pred_all_feature_error = np.array(pred_all_feature_error)
        # feature_l2_error =  np.array(pred_all_feature_error)
        # pred_all_feature_relative_error = np.array(pred_all_feature_relative_error)
        # pred_all_ratio = np.array(pred_all_ratio)
        # # feature_relative_l2_error =  np.mean(pred_all_feature_relative_error, axis=1)
        # # feature_relative_l2_error =  np.mean(np.mean(pred_all_feature_relative_error, axis=1), axis=1)



        # return ratio, feature_l2_error, pred_all_feature_relative_error, pred_all_ratio
        # return ratio, pred_all_feature_relative_error
        # return feature_l2_error
        # return feature_l2_error, feature_relative_l2_error


    @torch.no_grad()
    def save_voxel_vel_npz(self,vel_path,deltaT,t,batchify_fn,chunk=1024*32, velocity_model=None,save_npz=True,save_jpg=False,save_vort=False):
        vel_scale = 160
        voxel_vel = self.get_voxel_velocity(deltaT,t,batchify_fn,chunk,velocity_model,middle_slice=not save_npz).detach().cpu().numpy()
        
        if save_jpg:
            jpg_path = os.path.splitext(vel_path)[0]+".jpg"
            imageio.imwrite(jpg_path, vel_uv2hsv(voxel_vel, scale=vel_scale, is3D=True, logv=False))
        if save_npz:
            if save_vort and save_jpg:
                _, NETw = jacobian3D_np(voxel_vel)
                head_tail = os.path.split(vel_path)
                imageio.imwrite( os.path.join(head_tail[0], "vort"+os.path.splitext(head_tail[1])[0]+".jpg"),
                        vel_uv2hsv(NETw[0],scale=vel_scale*5.0,is3D=True) )
            # to save some space
            voxel_vel = np.float16(voxel_vel)
            np.savez_compressed(vel_path, vel=voxel_vel)

    def save_voxel_vel_npz_with_grad(self, model, vel_path, deltaT, t, chunk, save_npz=True, save_jpg=False, save_vort=False):
        # vel_scale = 160
        # vel_scale = 300
        vel_scale = 500
        voxel_vel = self.get_voxel_velocity(model, deltaT, t, chunk, middle_slice=not save_npz).detach().cpu().numpy()
        
        ret = {}
        if save_jpg:
            jpg_path = os.path.splitext(vel_path)[0]+".jpg"
            rgb = vel_uv2hsv(voxel_vel, scale=1000, is3D=True, logv=False)
            imageio.imwrite(jpg_path, rgb)
            ret['vel'] = rgb
        if save_npz:
            if save_vort and save_jpg:
                _, NETw = jacobian3D_np(voxel_vel)
                head_tail = os.path.split(vel_path)
                # rgb = vel_uv2hsv(NETw[0],scale=vel_scale*5.0,is3D=True)
                # rgb = vel_uv2hsv(NETw[0],scale=1500,is3D=True)
                rgb = vel_uv2hsv(NETw[0],scale=1000,is3D=True)
                imageio.imwrite( os.path.join(head_tail[0], "vort"+os.path.splitext(head_tail[1])[0]+".jpg"),
                         rgb)
                ret['vort'] = rgb
            # to save some space
            voxel_vel = np.float16(voxel_vel)
            np.savez_compressed(vel_path, vel=voxel_vel)
        return ret


def jacobian3D(x):
    # x, (b,)d,h,w,ch, pytorch tensor
    # return jacobian and curl

    dudx = x[:,:,:,1:,0] - x[:,:,:,:-1,0]
    dvdx = x[:,:,:,1:,1] - x[:,:,:,:-1,1]
    dwdx = x[:,:,:,1:,2] - x[:,:,:,:-1,2]
    dudy = x[:,:,1:,:,0] - x[:,:,:-1,:,0]
    dvdy = x[:,:,1:,:,1] - x[:,:,:-1,:,1]
    dwdy = x[:,:,1:,:,2] - x[:,:,:-1,:,2]
    dudz = x[:,1:,:,:,0] - x[:,:-1,:,:,0]
    dvdz = x[:,1:,:,:,1] - x[:,:-1,:,:,1]
    dwdz = x[:,1:,:,:,2] - x[:,:-1,:,:,2]

    # u = dwdy[:,:-1,:,:-1] - dvdz[:,:,1:,:-1]
    # v = dudz[:,:,1:,:-1] - dwdx[:,:-1,1:,:]
    # w = dvdx[:,:-1,1:,:] - dudy[:,:-1,:,:-1]

    dudx = torch.cat((dudx, torch.unsqueeze(dudx[:,:,:,-1], 3)), 3)
    dvdx = torch.cat((dvdx, torch.unsqueeze(dvdx[:,:,:,-1], 3)), 3)
    dwdx = torch.cat((dwdx, torch.unsqueeze(dwdx[:,:,:,-1], 3)), 3)

    dudy = torch.cat((dudy, torch.unsqueeze(dudy[:,:,-1,:], 2)), 2)
    dvdy = torch.cat((dvdy, torch.unsqueeze(dvdy[:,:,-1,:], 2)), 2)
    dwdy = torch.cat((dwdy, torch.unsqueeze(dwdy[:,:,-1,:], 2)), 2)

    dudz = torch.cat((dudz, torch.unsqueeze(dudz[:,-1,:,:], 1)), 1)
    dvdz = torch.cat((dvdz, torch.unsqueeze(dvdz[:,-1,:,:], 1)), 1)
    dwdz = torch.cat((dwdz, torch.unsqueeze(dwdz[:,-1,:,:], 1)), 1)

    u = dwdy - dvdz
    v = dudz - dwdx
    w = dvdx - dudy
    
    j = torch.stack([dudx,dudy,dudz,dvdx,dvdy,dvdz,dwdx,dwdy,dwdz], -1)
    c = torch.stack([u,v,w], -1)
    
    return j, c

def jacobian3D_np(x):
    # x, (b,)d,h,w,ch
    # return jacobian and curl

    if len(x.shape) < 5:
        x = np.expand_dims(x, axis=0)
    dudx = x[:,:,:,1:,0] - x[:,:,:,:-1,0]
    dvdx = x[:,:,:,1:,1] - x[:,:,:,:-1,1]
    dwdx = x[:,:,:,1:,2] - x[:,:,:,:-1,2]
    dudy = x[:,:,1:,:,0] - x[:,:,:-1,:,0]
    dvdy = x[:,:,1:,:,1] - x[:,:,:-1,:,1]
    dwdy = x[:,:,1:,:,2] - x[:,:,:-1,:,2]
    dudz = x[:,1:,:,:,0] - x[:,:-1,:,:,0]
    dvdz = x[:,1:,:,:,1] - x[:,:-1,:,:,1]
    dwdz = x[:,1:,:,:,2] - x[:,:-1,:,:,2]

    # u = dwdy[:,:-1,:,:-1] - dvdz[:,:,1:,:-1]
    # v = dudz[:,:,1:,:-1] - dwdx[:,:-1,1:,:]
    # w = dvdx[:,:-1,1:,:] - dudy[:,:-1,:,:-1]

    dudx = np.concatenate((dudx, np.expand_dims(dudx[:,:,:,-1], axis=3)), axis=3)
    dvdx = np.concatenate((dvdx, np.expand_dims(dvdx[:,:,:,-1], axis=3)), axis=3)
    dwdx = np.concatenate((dwdx, np.expand_dims(dwdx[:,:,:,-1], axis=3)), axis=3)

    dudy = np.concatenate((dudy, np.expand_dims(dudy[:,:,-1,:], axis=2)), axis=2)
    dvdy = np.concatenate((dvdy, np.expand_dims(dvdy[:,:,-1,:], axis=2)), axis=2)
    dwdy = np.concatenate((dwdy, np.expand_dims(dwdy[:,:,-1,:], axis=2)), axis=2)

    dudz = np.concatenate((dudz, np.expand_dims(dudz[:,-1,:,:], axis=1)), axis=1)
    dvdz = np.concatenate((dvdz, np.expand_dims(dvdz[:,-1,:,:], axis=1)), axis=1)
    dwdz = np.concatenate((dwdz, np.expand_dims(dwdz[:,-1,:,:], axis=1)), axis=1)

    u = dwdy - dvdz
    v = dudz - dwdx
    w = dvdx - dudy
    
    j = np.stack([dudx,dudy,dudz,dvdx,dvdy,dvdz,dwdx,dwdy,dwdz], axis=-1)
    c = np.stack([u,v,w], axis=-1)
    
    return j, c

def jacobianDen_np(x):
    # x, (b,)d,h,w,ch
    # return jacobian and curl

    if len(x.shape) < 5:
        x = np.expand_dims(x, axis=0)
    dudx = x[:,:,:,1:,0] - x[:,:,:,:-1,0]
    dudy = x[:,:,1:,:,0] - x[:,:,:-1,:,0]
    dudz = x[:,1:,:,:,0] - x[:,:-1,:,:,0]
    
    dudx = np.concatenate((dudx, np.expand_dims(dudx[:,:,:,-1], axis=3)), axis=3)
    dudy = np.concatenate((dudy, np.expand_dims(dudy[:,:,-1,:], axis=2)), axis=2)
    dudz = np.concatenate((dudz, np.expand_dims(dudz[:,-1,:,:], axis=1)), axis=1)

    j = np.stack([dudx,dudy,dudz], axis=-1)
    
    return j
