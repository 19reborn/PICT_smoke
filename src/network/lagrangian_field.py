
import sys,os
sys.path.append('.')
import torch
import torch.nn as nn
from .siren_basic import SineLayer, trunc_exp
import torch.nn.functional as F

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


class FeatureMapping(nn.Module):
    """
    Lagarian Particle feature mapping
    (x,y,z,t) -> (features)
    """
    def __init__(self, in_channels=4, out_channels=16, D=4, W=128, skips=[]):
        super(FeatureMapping, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.D = D
        self.W = W
        self.skips = skips


        first_omega_0 = 30.0
        hidden_omega_0 = 1.0

        self.linears = nn.ModuleList(
            [SineLayer(in_channels, W, omega_0=first_omega_0)] + 
            [SineLayer(W, W, omega_0=hidden_omega_0) 
                if i not in self.skips else SineLayer(W + in_channels, W, omega_0=hidden_omega_0) for i in range(D-1)] + 
                [nn.Linear(W, out_channels)]
        )

    def forward(self, xyz, t=None):

        # xyz = xyz /  0.33
        if xyz.shape[-1] == 3:
            input_feature = torch.cat([xyz, t], dim=-1)
        elif xyz.shape[-1] == 4:
            input_feature = xyz

        feature = input_feature

        for i, layer in enumerate(self.linears):
            if i in self.skips:
                feature = torch.cat([input_feature, layer(feature)], dim=-1)
            else:
                feature = layer(feature)
        
        return feature


class PositionMapping(nn.Module):
    """
    Lagarian Particle position mapping
    (features, t) -> (x,y,z)
    """
    def __init__(self, in_channels=17, out_channels=3, D=3, W=128, skips=[]):
        super(PositionMapping, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.D = D
        self.W = W
        self.skips = skips

        first_omega_0 = 5.0
        hidden_omega_0 = 1.0

        self.linears = nn.ModuleList(
            [SineLayer(in_channels, W, omega_0=first_omega_0)] +
            [SineLayer(W, W, omega_0=hidden_omega_0) 
                if i not in skips else SineLayer(W + in_channels, W, omega_0=hidden_omega_0) for i in range(D-1)] +
            [nn.Linear(W, out_channels)]
        )



        # linears = []
        # linears += [nn.Linear(in_channels, W)]
        # linears += [nn.ReLU()]
        # for i in range(D-1):
        #     if i not in skips:
        #         linears += [nn.Linear(W, W)]
        #         linears += [nn.ReLU()]
        #     else:
        #         linears += [nn.Linear(W + in_channels, W)]
        #         linears += [nn.ReLU()]
        # linears += [nn.Linear(W, out_channels)]

        # self.linears = nn.Sequential(*linears)


    def forward(self, feature, t=None):
        # x: (features, t)

        input_xyz = torch.cat([feature, t], dim=-1)

        xyz = input_xyz


        for i, layer in enumerate(self.linears):
            if i in self.skips:
                xyz = torch.cat([input_xyz, layer(xyz)], dim=-1)
            else:
                xyz = layer(xyz)
        
        return xyz

class DensityMapping(nn.Module):
    """
    Lagarian Particle density mapping
    (features, t) -> (density)
    """
    # def __init__(self, in_channels=16, out_channels=1, D=2, W=128, skips=[]):
    def __init__(self, in_channels=17, out_channels=1, D=2, W=128, skips=[]):
        super(DensityMapping, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.D = D
        self.W = W
        self.skips = skips

        first_omega_0 = 1.0
        hidden_omega_0 = 1.0

        self.linears = nn.ModuleList(
            [SineLayer(in_channels, W, omega_0=first_omega_0)] +
            [SineLayer(W, W, omega_0=hidden_omega_0) 
                if i not in skips else SineLayer(W + in_channels, W, omega_0=hidden_omega_0) for i in range(D-1)] +
            [nn.Linear(W, out_channels)]
        )


        # linears = []
        # linears += [nn.Linear(in_channels, W)]
        # linears += [nn.ReLU()]
        # for i in range(D-1):
        #     if i not in skips:
        #         linears += [nn.Linear(W, W)]
        #         linears += [nn.ReLU()]
        #     else:
        #         linears += [nn.Linear(W + in_channels, W)]
        #         linears += [nn.ReLU()]
        # linears += [nn.Linear(W, out_channels)]

        # self.linears = nn.Sequential(*linears)

        # self.activation = nn.ReLU()
        # self.activation = nn.LeakyReLU()
        self.activation = trunc_exp

    def forward(self, feature, t=None):
        # x: (features, t)

        input_xyz = torch.cat([feature, t], dim=-1)
        # input_xyz = torch.cat([feature], dim=-1)

        xyz = input_xyz


        for i, layer in enumerate(self.linears):
            if i in self.skips:
                xyz = torch.cat([input_xyz, layer(xyz)], dim=-1)
            else:
                xyz = layer(xyz)
        
        density = self.activation(xyz)


        return density

class MappingNetwork(nn.Module):
    """
    (x.y,z,t,t') -> (x',y',z') by using feature mapping and position mapping
    In detail,
    (x,y,z,t) -> feature_mapping -> (features)
    (features, t') -> position_mapping -> (x,y,z)
    """

    def __init__(self, feature_map, position_map):
        super(MappingNetwork, self).__init__()

        self.feature_map = feature_map
        self.position_map = position_map

        self.eval_mode = False

    def forward(self, xyzt, t1 = None):

        if t1 is None:
            xyz, t, t1 = torch.split(xyzt, (3, 1, 1), dim=-1)
        else:
            xyz, t = torch.split(xyzt, (3, 1), dim=-1)

 
        features = self.feature_map(xyz, t)

        xyz = self.position_map(features, t1)

        return xyz
    
    def forward_with_features(self, features, t1):
        # directly provide features instead of xyzt
        xyz = self.position_map(features, t1)

        return xyz

class VelocityNetwork(nn.Module):
    """
    (x.y,z,t) -> (vx,vy,vz) by using feature mapping and position mapping
    In detail,
    (x,y,z,t) -> feature_mapping -> (features)
    (features, t) -> position_mapping -> (x,y,z)
    (x,y,z) -> auto differentiation -> (vx,vy,vz)
    """

    def __init__(self, feature_map, position_map):
        super(VelocityNetwork, self).__init__()

        self.feature_map = feature_map
        self.position_map = position_map

        self.eval_mode = False

    def fix_feature_map_grad(self):
        for name, p in self.feature_map.named_parameters():
            p.requires_grad = False

    def gradients(self, y, x):
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
        return jac.squeeze(-1)

    def forward_with_middle_output(self, xyzt, need_vorticity = False):
        xyz, t = torch.split(xyzt, (3, 1), dim=-1)

        xyz.requires_grad_(True) ## allow for futhre order derivative
        t.requires_grad_(True) ## todo:: check whether put it after feature_map
        t1 = t.clone().detach()
        t1.requires_grad_(True)

        features = self.feature_map(xyz, t)


        mapped_xyz = self.position_map(features, t1)


        velocity = self.gradients(mapped_xyz, t1)

        middle_output = {}
        middle_output['mapped_features'] = features
        middle_output['mapped_xyz'] = mapped_xyz

        if need_vorticity:
            jaco_xyz = _get_minibatch_jacobian(velocity, xyz)
            jaco_t1 = _get_minibatch_jacobian(velocity, t1)
            jacobian = torch.cat([jaco_xyz, jaco_t1], dim = -1) # [N, 3, 4]
            middle_output['jacobian'] = jacobian
            dfeature_dxyz = _get_minibatch_jacobian(features, xyz)
            dfeature_dt = _get_minibatch_jacobian(features, t)
            middle_output['dfeature_dxyz'] = dfeature_dxyz
            middle_output['dfeature_dt'] = dfeature_dt


        return velocity, middle_output

    def forward(self, xyzt):
        xyz, t = torch.split(xyzt, (3, 1), dim=-1)

        xyz.requires_grad_(True) ## allow for futhre order derivative

        features = self.feature_map(xyz, t)
        t.requires_grad_(True) ## todo:: check whether put it after feature_map

        xyz = self.position_map(features, t)

        velocity = self.gradients(xyz, t)

        return velocity

    def mapping_forward(self, xyzt, t1 = None):

        if t1 is None:
            xyz, t, t1 = torch.split(xyzt, (3, 1, 1), dim=-1)
        else:
            xyz, t = torch.split(xyzt, (3, 1), dim=-1)

 
        features = self.feature_map(xyz, t)

        xyz = self.position_map(features, t1)

        return xyz

    def mapping_forward_with_features(self, features, t1):
        # directly provide features instead of xyzt
        xyz = self.position_map(features, t1)

        return xyz

    def update_fading_step(self, steps):
        return

    def print_fading(self):
        return

class DensityNetwork(nn.Module):
    """
    (x.y,z,t,t') -> (x',y',z') by using feature mapping and position mapping
    In detail,
    (x,y,z,t) -> feature_mapping -> (features)
    (features, t') -> density_mapping -> (density)
    """

    def __init__(self, feature_map, density_map):
        super(DensityNetwork, self).__init__()

        self.feature_map = feature_map
        self.density_map = density_map

        self.eval_mode = False
        
        
    def fix_feature_map_grad(self):
        for name, p in self.feature_map.named_parameters():
            p.requires_grad = False

    def forward(self, xyzt):

        # if t1 is None:
        #     if xyzt.shape[-1] == 5:
        #         xyz, t, t1 = torch.split(xyzt, (3, 1, 1), dim=-1)
        #     else:
        #         xyz, t = torch.split(xyzt, (3, 1), dim=-1)
        #         t1 = t
        #         ## warning:: this may cause careless bug
        # else:
        xyz, t = torch.split(xyzt, (3, 1), dim=-1)

 
        features = self.feature_map(xyz, t)

        density = self.density_map(features, t)
        # density = self.density_map(features)

        return density
    
    def forward_with_features(self, features, t1):
        # directly provide features instead of xyzt
        density = self.density_map(features, t1)
        # density = self.density_map(features)

        return density

    def forward_with_middle_output(self, xyzt, need_jacobian = False):
        # if t1 is None:
        #     if xyzt.shape[-1] == 5:
        #         xyz, t, t1 = torch.split(xyzt, (3, 1, 1), dim=-1)
        #     else:
        #         xyz, t = torch.split(xyzt, (3, 1), dim=-1)
        #         t1 = t
        #         ## warning:: this may cause careless bug
        # else:
        xyz, t = torch.split(xyzt, (3, 1), dim=-1)

        xyz.requires_grad_(True) ## allow for futhre order derivative
        t.requires_grad_(True) ## todo:: check whether put it after feature_map
        t1 = t.clone().detach()
        t1.requires_grad_(True) ## todo:: check whether put it after feature_map

        features = self.feature_map(xyz, t)
        density = self.density_map(features, t1)

        middle_output = {}
        middle_output['mapped_features'] = features

        if need_jacobian:
            # jaco_xyz = _get_minibatch_jacobian(density, xyz)
            # jaco_t = _get_minibatch_jacobian(density, t)
            dfeature_dxyz = _get_minibatch_jacobian(features, xyz)
            dfeature_dt = _get_minibatch_jacobian(features, t)

            ddensity_dxyz = _get_minibatch_jacobian(density, xyz)
            Ddensity_Dt = _get_minibatch_jacobian(density, t1)
            ddensity_dt = _get_minibatch_jacobian(density, t) + Ddensity_Dt
            
            middle_output['dfeature_dxyz'] = dfeature_dxyz
            middle_output['dfeature_dt'] = dfeature_dt
            middle_output['ddensity_dxyz'] = ddensity_dxyz
            middle_output['Ddensity_Dt'] = Ddensity_Dt
            middle_output['ddensity_dt'] = ddensity_dt

        return density, middle_output
    
    def forward_with_Dt(self, xyzt, bbox_mask = None):
        # if t1 is None:
        #     if xyzt.shape[-1] == 5:
        #         xyz, t, t1 = torch.split(xyzt, (3, 1, 1), dim=-1)
        #     else:
        #         xyz, t = torch.split(xyzt, (3, 1), dim=-1)
        #         t1 = t
        #         ## warning:: this may cause careless bug
        # else:
        xyz, t = torch.split(xyzt, (3, 1), dim=-1)

        t1 = t.clone().detach()
        t1.requires_grad_(True) ## todo:: check whether put it after feature_map

        features = self.feature_map(xyz, t)
        density = self.density_map(features, t1)
        if bbox_mask is not None:
            density[bbox_mask==0] = 0

        Ddensity_Dt = _get_minibatch_jacobian(density, t1)
            

        return density, Ddensity_Dt

class ColorNetwork(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, D=3, W=128, skips=[]):
        super(ColorNetwork, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.D = D
        self.W = W
        self.skips = skips


        first_omega_0 = 30.0
        hidden_omega_0 = 1.0

        self.linears = nn.ModuleList(
            [SineLayer(in_channels, W, omega_0=first_omega_0)] + 
            [SineLayer(W, W, omega_0=hidden_omega_0) 
                if i not in self.skips else SineLayer(W + in_channels, W, omega_0=hidden_omega_0) for i in range(D-1)] + 
                [nn.Linear(W, out_channels)]
        )

    def forward(self, xyzt):

        input_feature = torch.cat([xyzt], dim=-1)

        feature = input_feature

        for i, layer in enumerate(self.linears):
            if i in self.skips:
                feature = torch.cat([input_feature, layer(feature)], dim=-1)
            else:
                feature = layer(feature)
        
        color = feature

        return color


class Lagrangian_NeRF(nn.Module):
    def __init__(self, args, bbox_model = None):
        super(Lagrangian_NeRF, self).__init__()

        self.feature_map = FeatureMapping()
        self.position_map = PositionMapping()
        self.density_map = DensityMapping()
        
  

        self.vel_model = VelocityNetwork(self.feature_map, self.position_map)
        self.map_model = MappingNetwork(self.feature_map, self.position_map)

        self.density_model = DensityNetwork(self.feature_map, self.density_map)


        self.bbox_model = bbox_model


    def fix_grad_except_color(self):
        # print(list(self.named_parameters()))
        for name, p in self.named_parameters():
            if "color_model" not in name:
                p.requires_grad = False
            # else:
                # pass

    def free_density_mapping(self):
        for name, p in self.density_map.named_parameters():
            p.requires_grad = True

    def free_all_grad(self):
        for name, p in self.named_parameters():
            p.requires_grad = True

    def print_fading(self):
        print("fading not used in Lagrangian NeRF!")

    def density(self, x):

        density = self.density_model(x)

        if self.bbox_model is not None:
   
            bbox_mask = self.bbox_model.insideMask(x[...,:3])
            density[bbox_mask==0] = 0

        return density
    
    def density_with_Dt(self, x):
        if self.bbox_model is not None:
            bbox_mask = self.bbox_model.insideMask(x[...,:3])
        else:
            bbox_mask = None
        density, Ddensity_Dt = self.density_model.forward_with_Dt(x, bbox_mask)
   
        return density, Ddensity_Dt

    # def color(self, x):

    #     color = self.color_model(x)


    #     if self.bbox_model is not None:
    #         bbox_mask = self.bbox_model.insideMask(x[...,:3])
    #         color[bbox_mask==0] = 0
            
    #     return color

    # def forward(self, x):

    #     density = self.density(x)
        
    #     color = self.color(x)
        

    #     output = torch.cat([color, density], -1)

    #     return output

    # def forward_density_with_jacobian(self, x):

    #     density, middle_output = self.density_model.forward_with_middle_output(x, need_jacobian=True)

    #     jacobian = middle_output['jacobian']
    #     Dd_Dt = middle_output['Dd_Dt']

    #     if self.bbox_model is not None:
   
    #         bbox_mask = self.bbox_model.insideMask(x[...,:3])
    #         density[bbox_mask==0] = 0
    #         jacobian[bbox_mask==0] = 0
    #         Dd_Dt[bbox_mask==0] = 0

    #     return density, jacobian, Dd_Dt