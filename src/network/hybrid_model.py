import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lagrangian_field import Lagrangian_NeRF
from .neus_field import NeuS
from .siren_basic import SIREN_NeRFt
from ..renderer.occupancy_grid import OccupancyGrid, OccupancyGridDynamic

# Model
class Lagrangian_Hybrid_NeuS(nn.Module):
    def __init__(self, args, bbox_model, occupancy_grid_static = None, occupancy_grid_dynamic = None):

        super(Lagrangian_Hybrid_NeuS, self).__init__()

        
        self.bbox_model = bbox_model
        self.single_scene = 'hybrid' in args.net_model

        if self.single_scele:
            self.static_model = None
        else:
            self.static_model = NeuS(args = args, bbox_model=bbox_model)

        self.dynamic_model_lagrangian = Lagrangian_NeRF(args = args, bbox_model = bbox_model)

        self.dynamic_model_siren = SIREN_NeRFt(args = args, bbox_model = bbox_model, density_activation = args.density_activation)

        self.dynamic_model = None
        
        self.occupancy_grid_static = occupancy_grid_static
        self.occupancy_grid_dynamic = occupancy_grid_dynamic
        
        # self.training_stage = 0

        self.iter_step = -1
        self.anneal_end = args.anneal_end
        
    

    def forward(self, x):
 
        inputs_xyz, input_t, input_views = torch.split(x, [3, 1, 3], dim=-1)
        
        dynamic_x = x
        static_x = torch.cat((inputs_xyz, input_views), dim=-1)
        
        static_output = self.static_model.forward(inputs_xyz, input_views)
        dynamic_output = self.dynamic_model.forward(x)
        outputs = torch.cat([static_output, dynamic_output], dim=-1)

        return outputs

    def forward_with_jacobian(self, x):
    
        inputs_xyz, input_t, input_views = torch.split(x, [3, 1, self.input_ch_views], dim=-1)
        
        dynamic_output, dynamic_jacobian, Dd_Dt = self.dynamic_model.forward_density_with_jacobian(x)

        if self.single_scene:
            outputs = dynamic_output
        else:
            static_output = self.static_model.forward(inputs_xyz, input_views)
            outputs = torch.cat([static_output, dynamic_output], dim=-1)
            

        return outputs, dynamic_jacobian, Dd_Dt

    def forward_static(self, x):
        ## suppose x: [-1, 6]
        input_xyz = x[...,:3]
        input_views = x[...,3:6]

        static_output = self.static_model.forward(input_xyz, input_views, xyz_bound=self.occupancy_grid_static.bound if self.occupancy_grid_static is not None else 1.0)
        return static_output

    def forward_dynamic(self, x):
        dynamic_output = self.dynamic_model.forward(x)
        return dynamic_output

    def density_dynamic(self, x):

        density = self.dynamic_model.density(x)
        return density

    def color_dynamic(self, x):
        color = self.dynamic_model.color(x)
        return color

    def sdf_static(self, x):
        sdf = self.static_model.sdf(x, xyz_bound = self.occupancy_grid_static.bound if self.occupancy_grid_static is not None else 1.0)
        return sdf

    def forward_geometry(self, x):
        sdf, gradient = self.static_model.sdf_with_gradient(x[..., :3])
        density = self.density_dynamic(x)
        
        return torch.cat([sdf, gradient, density], dim=-1)  

    def get_deviation(self):
        return self.static_model.deviation_network(torch.zeros([1, 3]))[:, :1].clamp(1e-6, 1e6)

    def get_cos_anneal_ratio(self):
        if self.iter_step == -1:
            # for test mode
            return 1.0
        else:
            return np.min([1.0, self.iter_step / (self.anneal_end + 1e-6)])

    def toDevice(self, device):
        self.static_model = self.static_model.to(device)
        self.dynamic_model = self.dynamic_model.to(device)

    def up_sample(self, rays_o, rays_d, z_vals, n_importance, up_sample_steps, embed_fn):
        return self.static_model.up_sample(rays_o, rays_d, z_vals, n_importance, up_sample_steps, embed_fn)
  
    def update_fading_step(self, global_step):
        self.dynamic_model_siren.fading_step = global_step

    def update_model_type(self, training_stage):

        if training_stage == 1:
            self.dynamic_model = self.dynamic_model_siren
        elif training_stage == 2 or training_stage == 3 or training_stage == 4:
            self.dynamic_model = self.dynamic_model_lagrangian
        else:
            AssertionError("training stage should be set to 1,2,3,4")
            
        self.adjust_grad_type(training_stage)
        

    def adjust_grad_type(self, training_stage):
        if training_stage == 1:
            # fix dynamic lagrangian grad
            for name, p in self.dynamic_model_lagrangian.named_parameters():
                p.requires_grad = False

        elif training_stage == 2:
            
            for name, p in self.dynamic_model_siren.named_parameters():
                p.requires_grad = False
            for name, p in self.static_model.named_parameters():
                p.requires_grad = False

            for name, p in self.dynamic_model_lagrangian.named_parameters():
                # p.requires_grad = True
                if "position_map" in name or "density_map" in name or 'color_model' in name:
                    p.requires_grad = True
                else:
                    p.requires_grad = False


        elif training_stage == 3:
            for name, p in self.static_model.named_parameters():
                p.requires_grad = False
            for name, p in self.dynamic_model_siren.named_parameters():
                p.requires_grad = False
            for name, p in self.dynamic_model_lagrangian.named_parameters():
                p.requires_grad = True


        elif training_stage == 4:
            for name, p in self.static_model.named_parameters():
                p.requires_grad = False
            for name, p in self.dynamic_model_siren.named_parameters():
                p.requires_grad = False
            for name, p in self.dynamic_model_lagrangian.named_parameters():
                p.requires_grad = True
  

        else:
            AssertionError("training stage should be set to 1,2,3,4")



def create_model(args, device, bbox_model):
    
    
    ## create occ grid     
    aabb_min = bbox_model.world_bbox[0]
    aabb_max = bbox_model.world_bbox[1]
    occupancy_grid_static = OccupancyGrid(density_thresh = args.density_thresh_static, bound = args.occ_grid_bound_static, aabb_min=aabb_min, aabb_max=aabb_max)
    occupancy_grid_dynamic = OccupancyGridDynamic(density_thresh = args.density_thresh, bound = args.occ_grid_bound_dynamic, time_size = args.time_size, aabb_min=aabb_min, aabb_max=aabb_max)
    
    
    ## create modelfea
    model = Lagrangian_Hybrid_NeuS(args = args, 
                                    bbox_model = bbox_model,
                                    occupancy_grid_static = occupancy_grid_static,
                                    occupancy_grid_dynamic= occupancy_grid_dynamic).to(device)
    
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lrate, betas=(0.9, 0.999))

    load_model_path = None

    if args.model_path is not None:
        load_model_path = args.model_path
    else:
        basedir = args.basedir
        expname = args.expname
        
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

        print('Found ckpts', ckpts)
        if len(ckpts) > 0:
            load_model_path = ckpts[-1]
            print('Reloading from', load_model_path)
    
    if load_model_path is not None:
        checkpoint = torch.load(load_model_path)
        model.static_model.load_state_dict(checkpoint["static_model_state_dict"])
        model.dynamic_model_lagrangian.load_state_dict(checkpoint["dynamic_model_lagrangian_state_dict"])
        model.dynamic_model_siren.load_state_dict(checkpoint["dynamic_model_siren_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        ## todo::
        # load occ grid
        start_step = checkpoint["global_step"]
    else:
        start_step = 0
        
    return model, optimizer, start_step