import numpy as np
import sys, os
import imageio
import torch, torchvision
from torch import optim, nn
import torch.nn.functional as F

from src.utils.training_utils import batchify, batchify_func
from src.utils.visualize_utils import den_scalar2rgb, vel2hsv, vel_uv2hsv


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
            
            raw_i = model.forward_geometry(pts_i) 
            if model.training:
                all_raw.append(raw_i)
            else:
                all_raw.append(raw_i.detach())

        raw = torch.cat(all_raw, 0).view(input_shape+[-1])
        return raw

    def get_density_flat(self, model, cur_pts, chunk=1024*32, getSDF = False):
        flat_raw = self.get_raw_geometry_at_pts(model, cur_pts, chunk)
        den_raw = F.relu(flat_raw[...,-1:])
        static_sdf = flat_raw[...,0:1]
        if not getSDF:
            inv_s = model.get_deviation()
            static_sdf = torch.sigmoid(static_sdf * inv_s).squeeze(-1)
            static_sdf = inv_s * static_sdf * (1. - static_sdf)

        static_normal = flat_raw[...,1:3]
        return [den_raw, static_sdf, static_normal]
   

    def get_velocity_flat(self, model, cur_pts,chunk=1024*32,):
        pts_N = cur_pts.shape[0]
        world_v = []
        for i in range(0, pts_N, chunk):
            input_i = cur_pts[i:i+chunk]
            # vel_i = batchify(model, chunk)(input_i)
            vel_i = batchify_func(model.dynamic_model.vel_model, chunk, not model.training)(input_i)
            world_v.append(vel_i)
        world_v = torch.cat(world_v, 0)
        return world_v

    def get_density_and_derivatives(self, cur_pts, chunk=1024*32, use_viewdirs=False, 
        network_query_fn=None, network_fn=None):
        _den, _ , _ = self.get_density_flat(cur_pts, chunk, use_viewdirs,network_query_fn, network_fn, )[0]
        # requires 1 backward passes 
        # The minibatch Jacobian matrix of shape (N, D_y=1, D_x=4)
        jac = _get_minibatch_jacobian(_den, cur_pts)
        _d_x, _d_y, _d_z, _d_t = [torch.squeeze(_, -1) for _ in jac.split(1, dim=-1)] # (N,1)
        return _den, _d_x, _d_y, _d_z, _d_t

    def get_lagrangian_density_and_derivatives(self, cur_pts, chunk=1024*32, use_viewdirs=False, 
        network_query_fn=None, density_model=None):
        _den, den_middle_output = density_model.forward_with_middle_output(cur_pts, need_jacobian=True)
        # requires 1 backward passes 
        # The minibatch Jacobian matrix of shape (N, D_y=1, D_x=4)
        jac = den_middle_output['jacobian']
        _d_x, _d_y, _d_z, _d_t = [torch.squeeze(_, -1) for _ in jac.split(1, dim=-1)] # (N,3)
        return _den, _d_x, _d_y, _d_z, _d_t

    
    def get_density_and_derivatives_with_sdf(self, model, cur_pts, chunk=1024*32, use_viewdirs=False, 
        network_query_fn=None, network_fn=None):
        _den, _sdf, _normal = self.get_density_flat(model, cur_pts, chunk)
        # requires 1 backward passes 
        # The minibatch Jacobian matrix of shape (N, D_y=1, D_x=4)
        jac = _get_minibatch_jacobian(_den, cur_pts)
        _d_x, _d_y, _d_z, _d_t = [torch.squeeze(_, -1) for _ in jac.split(1, dim=-1)] # (N,1)
        return _den, _d_x, _d_y, _d_z, _d_t, _sdf, _normal

    
    def get_lagrangian_density_and_derivatives_with_sdf(self, cur_pts, chunk=1024*32, use_viewdirs=False, 
        network_query_fn=None, network_fn=None):
        outputs, dynamic_jacobian, Dd_Dt = network_fn.forward_with_jacobian(cur_pts)
        _den = outputs[..., -1:]
        _sdf = outputs[...,3:4]
        _normal = outputs[...,4:7]
        jac = dynamic_jacobian
        _d_x, _d_y, _d_z, _d_t = [torch.squeeze(_, -1) for _ in jac.split(1, dim=-1)] # (N,1)
        return _den, _d_x, _d_y, _d_z, _d_t, Dd_Dt, _sdf, _normal

    def get_velocity_and_derivatives(self, cur_pts, chunk=1024*32, batchify_fn=None, vel_model=None):
        _vel = self.get_velocity_flat(cur_pts, batchify_fn, chunk, vel_model)
        # requires 3 backward passes
        # The minibatch Jacobian matrix of shape (N, D_y=3, D_x=4)
        jac = _get_minibatch_jacobian(_vel, cur_pts)
        _u_x, _u_y, _u_z, _u_t = [torch.squeeze(_, -1) for _ in jac.split(1, dim=-1)] # (N,3)
        return _vel, _u_x, _u_y, _u_z, _u_t

    def get_lagrangian_velocity_and_derivatives(self, cur_pts, chunk=1024*32, batchify_fn=None, vel_model=None):
        predict_vel, middle_output = vel_model.forward_with_middle_output(cur_pts, need_vorticity=True)
        ## supervise using ns equation
        jac = middle_output['jacobian']
        _u_x, _u_y, _u_z, Dv_Dt = [torch.squeeze(_, -1) for _ in jac.split(1, dim=-1)] # (N,3)
        _vel = predict_vel
        return _vel, _u_x, _u_y, _u_z, Dv_Dt, middle_output

    def get_voxel_density_list(self, model, t=None,chunk=1024*32,middle_slice=False):
        model.eval()
        D,H,W = self.D,self.H,self.W
        # middle_slice, only for fast visualization of the middle slice
        pts_flat = self.pts_mid if middle_slice else self.pts.view(-1, 3)
        pts_N = pts_flat.shape[0]
        if t is not None:
            input_t = torch.ones([pts_N, 1])*float(t)
            pts_flat = torch.cat([pts_flat,input_t], dim=-1)


        den_list, sdf_list , _ = self.get_density_flat(model, pts_flat, chunk)

        raw_list = [den_list, sdf_list]

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


        # world_v = self.get_velocity_flat(model.dynamic_model.vel_model, pts_flat, chunk)
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
        namepre = ["","static_"]
        for voxel_den, npre in zip(voxel_den_list, namepre):
            voxel_den = voxel_den.detach().cpu().numpy()
            if save_jpg:
                jpg_path = os.path.join(head_tail[0], npre + os.path.splitext(head_tail[1])[0]+".jpg")
                imageio.imwrite(jpg_path, den_scalar2rgb(voxel_den, scale=None, is3D=True, logv=False, mix=jpg_mix))
            if save_npz:
                # to save some space
                npz_path = os.path.join(head_tail[0], npre + os.path.splitext(head_tail[1])[0]+".npz")
                voxel_den = np.float16(voxel_den)
                np.savez_compressed(npz_path, vel=voxel_den)
            if noStatic:
                break

    @torch.no_grad()
    def vis_cross_feature_error_voxel(self, path, t, dynamic_model, middle_slice = False, chunk = 1024*32):

        dynamic_model.eval_mode = True
        # middle_slice, only for fast visualization of the middle slice
        D,H,W = self.D,self.H,self.W
        pts_flat = self.pts_mid if middle_slice else self.pts.view(-1, 3)
        pts_N = pts_flat.shape[0]


        # get feature0
        feature_0 = []
        for i in range(0, pts_N, chunk):
            input_i = pts_flat[i:i+chunk]
            feature = dynamic_model.feature_map(input_i, torch.ones([input_i.shape[0], 1])*float(0)).detach()
            feature_0.append(feature)

        feature_0 = torch.cat(feature_0, 0)



        feature_t = []
        for i in range(0, pts_N, chunk):
            input_i = pts_flat[i:i+chunk]
            mapped_xyz = dynamic_model.map_model(torch.cat([input_i, torch.ones([input_i.shape[0], 1])*float(t)], dim = -1), torch.ones([input_i.shape[0], 1])*float(0)).detach()
            feature = dynamic_model.feature_map(mapped_xyz, torch.ones([mapped_xyz.shape[0], 1])*float(t)).detach()
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
            
        dynamic_model.eval_mode = False

        error_img = vel_uv2hsv(feature_error.cpu(), scale=160, is3D=True, logv=False)

        imageio.imwrite(path, error_img)

        return error_img

    @torch.no_grad()
    def vis_feature_voxel(self, path, t, dynamic_model, middle_slice = False, chunk = 1024*32):

        dynamic_model.eval_mode = True
        # middle_slice, only for fast visualization of the middle slice
        D,H,W = self.D,self.H,self.W
        pts_flat = self.pts_mid if middle_slice else self.pts.view(-1, 3)
        pts_N = pts_flat.shape[0]


        # get feature0


        feature_t = []
        for i in range(0, pts_N, chunk):
            input_i = pts_flat[i:i+chunk]
            feature = dynamic_model.feature_map(input_i, torch.ones([input_i.shape[0], 1])*float(t)).detach()
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
            
        dynamic_model.eval_mode = False

        feature_img = vel_uv2hsv(feature_t.cpu(), scale=160, is3D=True, logv=False)
        imageio.imwrite(path, feature_img)

        return feature_img

    @torch.no_grad()
    def vis_mapping_voxel(self, frame_list, t_list, dynamic_model, sample_pts = 128, change_feature_interval = 1, chunk = 1024*32):

        dynamic_model.eval_mode = True
        # middle_slice, only for fast visualization of the middle slice
        D,H,W = self.D,self.H,self.W

        pts_flat = self.pts_mid
        # only use xy plane with z = 0.5
   
        # pts_flat = pts_flat[-pts_flat.shape[0]//3:]
        # pts_flat = pts_flat[:pts_flat.shape[0]//3]
        pts_flat = pts_flat[:pts_flat.shape[0]//3]
        # pts_flat = pts_flat[:-pts_flat.shape[0]//3]

        pts_flat = self.pts.view(-1, 3)

        
        # only choose density points
        pts_N = pts_flat.shape[0]
        density_0 = []
        for i in range(0, pts_N, chunk):
            input_i = pts_flat[i:i+chunk]
            density = dynamic_model.density(torch.cat([input_i, torch.ones([input_i.shape[0], 1])*float(0)], dim = -1)).detach()
            density_0.append(density)

        density_0 = torch.cat(density_0, dim = 0)

        pts_flat = pts_flat[density_0.squeeze(-1) >= 0.50]
            
        pts_num = sample_pts
        # pts_num = 32
        # pts_num = 1024
        import random
        sample_id = np.random.randint(0, pts_flat.shape[0], pts_num)
        pts_sampled = pts_flat[sample_id].reshape(-1,3)

        feature_sampled = dynamic_model.feature_map(pts_sampled,  torch.ones([pts_sampled.shape[0], 1])*float(0)).detach()

        all_xyz = []

        for frame_i in frame_list:

            cur_t = t_list[frame_i]
            mapped_xyz = dynamic_model.map_model.forward_with_features(feature_sampled, torch.ones([pts_sampled.shape[0], 1])*float(cur_t))
            if (frame_i - 1) % change_feature_interval == 0:
                feature_sampled = dynamic_model.feature_map(mapped_xyz,  torch.ones([mapped_xyz.shape[0], 1])*float(cur_t)).detach()
            all_xyz.append(mapped_xyz.detach().cpu().numpy())

        dynamic_model.eval_mode = False
        from src.utils import write_ply
        write_ply(np.array(all_xyz).reshape(-1,3),'vis_mapping.ply')
        
    

        return torch.tensor(all_xyz)


    @torch.no_grad()
    def save_voxel_vel_npz(self,vel_path,deltaT,t,batchify_fn,chunk=1024*32, vel_model=None,save_npz=True,save_jpg=False,save_vort=False):
        vel_scale = 160
        voxel_vel = self.get_voxel_velocity(deltaT,t,batchify_fn,chunk,vel_model,middle_slice=not save_npz).detach().cpu().numpy()
        
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
        vel_scale = 160
        voxel_vel = self.get_voxel_velocity(model, deltaT, t, chunk, middle_slice=not save_npz).detach().cpu().numpy()
        
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

