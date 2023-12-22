
import sys,os
sys.path.append('.')
import torch
import torch.nn as nn
from .siren_basic import SineLayer, get_activation
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
    def __init__(self, args, in_channels=4, out_channels=16, D=4, W=128, skips=[]):
        super(FeatureMapping, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.D = D
        self.W = W
        self.skips = skips

        if args.use_scene_scale_before_pe:
            self.scene_scale = args.scene_scale
        else:
            self.scene_scale = 1.0

        first_omega_0 = args.feature_map_first_omega
        hidden_omega_0 = 1.0

        self.linears = nn.ModuleList(
            [SineLayer(in_channels, W, omega_0=first_omega_0)] + 
            [SineLayer(W, W, omega_0=hidden_omega_0) 
                if i not in self.skips else SineLayer(W + in_channels, W, omega_0=hidden_omega_0) for i in range(D-1)] + 
                [nn.Linear(W, out_channels)]
        )

    def forward(self, xyz, t=None):

        xyz = xyz / self.scene_scale

        # xyz = xyz /  0.33
        if xyz.shape[-1] == 3:
            input_feature = torch.cat([xyz, t], dim=-1)
        elif xyz.shape[-1] == 4:
            input_feature = xyz

        feature = input_feature

        for i, layer in enumerate(self.linears):
            if i in self.skips:
                feature_next = torch.cat([input_feature, layer(feature)], dim=-1)
            else:
                feature_next = layer(feature)
                
            feature = feature_next
        
        return feature


class PositionMapping(nn.Module):
    """
    Lagarian Particle position mapping
    (features, t) -> (x,y,z)
    """
    def __init__(self, args, in_channels=17, out_channels=3, D=3, W=128, skips=[]):
        super(PositionMapping, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.D = D
        self.W = W
        self.skips = skips

        first_omega_0 = args.position_map_first_omega
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
    def __init__(self, args, in_channels=16, out_channels=1, D=2, W=128, skips=[]):
        super(DensityMapping, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.D = D
        self.W = W
        self.skips = skips

        first_omega_0 = args.density_map_first_omega
        hidden_omega_0 = 1.0

        self.linears = nn.ModuleList(
            [SineLayer(in_channels, W, omega_0=first_omega_0)] +
            [SineLayer(W, W, omega_0=hidden_omega_0) 
                if i not in skips else SineLayer(W + in_channels, W, omega_0=hidden_omega_0) for i in range(D-1)] +
            [nn.Linear(W, out_channels)]
        )


        self.activation = get_activation(args.lagrangian_density_activation)

    def forward(self, feature, t=None):
        # x: (features, t)

        # input_xyz = torch.cat([feature, t], dim=-1)
        # input_xyz = torch.cat([feature], dim=-1)
        input_xyz = feature

        xyz = input_xyz


        for i, layer in enumerate(self.linears):
            if i in self.skips:
                xyz_next = torch.cat([input_xyz, layer(xyz)], dim=-1)
            else:
                xyz_next = layer(xyz)
                
            xyz = xyz_next
            
        density = self.activation(xyz)
        # density = F.softplus(xyz - 1.)


        return density

class VelocityNetwork(nn.Module):
    """
    (x.y,z,t) -> (vx,vy,vz) by using feature mapping and position mapping
    In detail,
    (x,y,z,t) -> feature_mapping -> (features)
    (features, t) -> position_mapping -> (x,y,z)
    (x,y,z) -> auto differentiation -> (vx,vy,vz)
    """

    def __init__(self, feature_map, position_map, bbox_model = None):
        super(VelocityNetwork, self).__init__()

        self.feature_map = feature_map
        self.position_map = position_map

        self.bbox_model = bbox_model  

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

    def forward_feature(self, xyz, t):
        features = self.feature_map(xyz, t)
        
        if self.bbox_model is not None:
            bbox_mask = self.bbox_model.insideMask(xyz) == False
            features[bbox_mask] = 0.0
            
        return features

    def forward_with_middle_output(self, xyzt, need_vorticity = False):
        xyzt.requires_grad_(True)
        xyz, t = torch.split(xyzt, (3, 1), dim=-1)

        t1 = t.clone().detach()
        t1.requires_grad_(True)
        
        if self.bbox_model is not None:
            bbox_mask = self.bbox_model.insideMask(xyz) == False
        else:
            bbox_mask = torch.zeros_like(xyz[..., 0], dtype=torch.bool, device=xyz.device)
            

        features = self.feature_map(xyz, t)
        features[bbox_mask] = 0.0

        mapped_xyz = self.position_map(features, t1)
        mapped_xyz[bbox_mask] = xyz[bbox_mask]


        velocity = self.gradients(mapped_xyz, t1)
        velocity[bbox_mask] = 0.0


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

    def forward_with_feature_save_middle_output(self, xyzt, features, need_vorticity = False):
        AssertionError(False, "Not implemented!")
        xyz, t = torch.split(xyzt, (3, 1), dim=-1)

        t1 = t.clone().detach()
        t1.requires_grad_(True)


        mapped_xyz = self.position_map(features, t1)

        velocity = self.gradients(mapped_xyz, t1)
        
        if self.bbox_model is not None:
            bbox_mask = self.bbox_model.insideMask(xyz) == False
            features[bbox_mask] = 0.0
            
            mapped_xyz[bbox_mask] = xyz[bbox_mask]
            velocity[bbox_mask] = 0.0


        jaco_t1 = _get_minibatch_jacobian(velocity, t1)


        return velocity, jaco_t1

    def forward_using_feature(self, xyzt, features):
        xyz, t = torch.split(xyzt, (3, 1), dim=-1)

        t1 = t.clone().detach()
        t1.requires_grad_(True)

        if self.bbox_model is not None:
            bbox_mask = self.bbox_model.insideMask(xyz) == False
        else:
            bbox_mask = torch.zeros_like(xyz[..., 0], dtype=torch.bool, device=xyz.device)
            

        mapped_xyz = self.position_map(features, t1)
        mapped_xyz[bbox_mask] = xyz[bbox_mask]

        velocity = self.gradients(mapped_xyz, t1)
        velocity[bbox_mask] = 0.0
        

        return velocity

    def forward(self, xyzt):
        xyz, t = torch.split(xyzt, (3, 1), dim=-1)

        features = self.feature_map(xyz, t)
        t1 = t.clone().detach()
        t1.requires_grad_(True)
        
        if self.bbox_model is not None:
            bbox_mask = self.bbox_model.insideMask(xyz) == False
        else:
            bbox_mask = torch.zeros_like(xyz[..., 0], dtype=torch.bool, device=xyz.device)
            
        mapped_xyz = self.position_map(features, t1)
        mapped_xyz[bbox_mask] = xyz[bbox_mask]


        velocity = self.gradients(mapped_xyz, t1)
        velocity[bbox_mask] = 0.0
            
        return velocity

    def forward_vorticity(self, xyzt):
        xyzt.requires_grad_(True)
        xyz, t = torch.split(xyzt, (3, 1), dim=-1)

        features = self.feature_map(xyz, t)
        t1 = t.clone().detach()
        t1.requires_grad_(True)
        
        if self.bbox_model is not None:
            bbox_mask = self.bbox_model.insideMask(xyz) == False
        else:
            bbox_mask = torch.zeros_like(xyz[..., 0], dtype=torch.bool, device=xyz.device)
            
        mapped_xyz = self.position_map(features, t1)
        mapped_xyz[bbox_mask] = xyz[bbox_mask]


        velocity = self.gradients(mapped_xyz, t1)
        velocity[bbox_mask] = 0.0
        jaco_xyz = _get_minibatch_jacobian(velocity, xyz)

        curl_1 = jaco_xyz[:, 2, 1] - jaco_xyz[:, 1, 2]
        curl_2 = jaco_xyz[:, 0, 2] - jaco_xyz[:, 2, 0]
        curl_3 = jaco_xyz[:, 1, 0] - jaco_xyz[:, 0, 1]
        
        curl = torch.stack([curl_1, curl_2, curl_3], dim = -1)
        
        return velocity, curl        
        

    def mapping_forward(self, xyzt, t1 = None):

        if t1 is None:
            xyz, t, t1 = torch.split(xyzt, (3, 1, 1), dim=-1)
        else:
            xyz, t = torch.split(xyzt, (3, 1), dim=-1)

        if self.bbox_model is not None:
            bbox_mask = self.bbox_model.insideMask(xyz) == False
        else:
            bbox_mask = torch.zeros_like(xyz[..., 0], dtype=torch.bool, device=xyz.device)
            
        features = self.feature_map(xyz, t)

        mapped_xyz = self.position_map(features, t1)
        mapped_xyz[bbox_mask] = xyz[bbox_mask]
            
        return mapped_xyz

    def mapping_forward_using_features(self, features, t1, xyz = None):
        # directly provide features instead of xyzt
        mapped_xyz = self.position_map(features, t1)
        
        if self.bbox_model is not None and xyz is not None:
            bbox_mask = self.bbox_model.insideMask(xyz) == False
        else:
            bbox_mask = torch.zeros_like(mapped_xyz[..., 0], dtype=torch.bool, device=mapped_xyz.device)
        
        if xyz is not None:
            mapped_xyz[bbox_mask] = xyz[bbox_mask]

        return mapped_xyz

    def velocity_mapping_loss(self, x, t, mapped_t):
 
        xyz = x.clone().detach().requires_grad_(True)
        
        features = self.feature_map(xyz, t)

        cross_mapped_xyz = self.position_map(features, mapped_t)

        base_mapped_xyz = self.position_map(features, t).detach()

        mapped_xyz = x + cross_mapped_xyz - base_mapped_xyz

        
        jacobian = _get_minibatch_jacobian(mapped_xyz, xyz).detach()

        # mapped_features = self.feature_map(mapped_xyz, mapped_t).detach()
        t1 = mapped_t.clone().detach()
        t1.requires_grad_(True)


        # mapped_self_xyz = self.position_map(mapped_features, t1)
        mapped_self_xyz = self.position_map(features, t1)

        velocity = self.gradients(mapped_self_xyz, t1)
        
        mapped_velocity = torch.matmul(jacobian.permute(0,2,1), velocity.unsqueeze(-1)).squeeze(-1)
        # mapped_velocity = jacobian.T @ velocity
        
        if self.bbox_model is not None:
            bbox_mask = self.bbox_model.insideMask(x) == False
            mapped_velocity[bbox_mask] = velocity[bbox_mask]
            
        return mapped_velocity
        

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

    def __init__(self, feature_map, density_map, bbox_model = None):
        super(DensityNetwork, self).__init__()

        self.feature_map = feature_map
        self.density_map = density_map

        self.bbox_model = bbox_model
        

    def forward(self, xyzt):

        xyz, t = torch.split(xyzt, (3, 1), dim=-1)
 
        if self.bbox_model is not None:
            bbox_mask = self.bbox_model.insideMask(xyz) == False
        else:
            bbox_mask = torch.zeros_like(xyz[..., 0], dtype=torch.bool, device=xyz.device)
            
        features = self.feature_map(xyz, t)
        # features *= bbox_mask
        features[bbox_mask] = 0.0

        density = self.density_map(features)
        # density *= bbox_mask
        density[bbox_mask] = 0.0
            
        return density, features
    

    def density(self, xyzt):
        
        density, features = self.forward(xyzt)
            
        return density
    

    def density_using_features(self, features):
        AssertionError(False, "Not implemented!")
        # directly provide features instead of xyzt
        density = self.density_map(features)
        
        return density

    def density_with_jacobian(self, xyzt):
        xyzt.requires_grad_(True)
        xyz, t = torch.split(xyzt, (3, 1), dim=-1)

        if self.bbox_model is not None:
            bbox_mask = self.bbox_model.insideMask(xyz) == False
        else:
            bbox_mask = torch.zeros_like(xyz[..., 0], dtype=torch.bool, device=xyz.device)
            
        features = self.feature_map(xyz, t)
        # features *= bbox_mask
        features[bbox_mask] = 0.0

        density = self.density_map(features.clone()) # todo: clone?
        # density *= bbox_mask
        density[bbox_mask] = 0.0

        ddensity_dxyz = _get_minibatch_jacobian(density, xyz)
        ddensity_dt = _get_minibatch_jacobian(density, t)
        
        jacobian = torch.cat([ddensity_dxyz, ddensity_dt], dim = -1) # [N, 3, 4]
           
        return density, features, jacobian

class ColorNetwork(nn.Module):
    def __init__(self, bbox_model = None, in_channels=4, out_channels=3, D=3, W=128, skips=[]):
        super(ColorNetwork, self).__init__()

        self.bbox_model = bbox_model

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

        if self.bbox_model is not None:
            bbox_mask = self.bbox_model.insideMask(xyzt[..., :3]) == False
            color[bbox_mask] = 0.0

        return color

class Lagrangian_NeRF(nn.Module):
    def __init__(self, args, bbox_model = None):
        super(Lagrangian_NeRF, self).__init__()

        feature_dim = args.lagrangian_feature_dim
        self.feature_map = FeatureMapping(args, out_channels = feature_dim)
        self.position_map = PositionMapping(args, in_channels = feature_dim + 1)
        self.density_map = DensityMapping(args, in_channels = feature_dim)
        

        self.velocity_model = VelocityNetwork(self.feature_map, self.position_map, bbox_model)

        self.density_model = DensityNetwork(self.feature_map, self.density_map, bbox_model)

        if not args.use_two_level_density:
            self.color_model = ColorNetwork(bbox_model = bbox_model)


    def print_fading(self):
        print("fading not used in Lagrangian NeRF!")

    def density(self, x):

        density, features = self.density_model(x)

        return density
    
    def density_with_feature_output(self, x):

        density, features = self.density_model(x)

        return density, features
    
    def color(self, x):

        color = self.color_model(x)
            
        return color
    
    def forward(self, x):

        density = self.density(x)
        
        color = self.color(x)
        

        output = torch.cat([color, density], -1)

        return output