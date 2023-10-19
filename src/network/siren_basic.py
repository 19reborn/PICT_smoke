# SIREN (https://github.com/vsitzmann/siren, explore_siren.ipynb)

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os

from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
import skimage
import matplotlib.pyplot as plt

import time

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


class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    
    def forward_with_intermediate(self, input): 
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate
    
    
class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output, coords        

    def forward_with_activations(self, coords, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!'''
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)
                
                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()
                    
                activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else: 
                x = layer(x)
                
                if retain_grad:
                    x.retain_grad()
                    
            activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations

        
def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)


def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd 

class _trunc_exp(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32) # cast to float32
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):
        x = ctx.saved_tensors[0]
        return g * torch.exp(x.clamp(-15, 15))

trunc_exp = _trunc_exp.apply

def get_activation(activation):
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'leakyrelu':
        return nn.LeakyReLU()
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'identity':
        return nn.Identity()
    elif activation == 'exp':
        return trunc_exp
    else:
        raise NotImplementedError

# Model
class SIREN_NeRFt(nn.Module):
    def __init__(self, args, D=8, W=256, input_ch=4, input_ch_views=0, output_ch=4, skips=[4], use_viewdirs=False, fading_fin_step=0, bbox_model=None, density_activation='identity'):
        """ 
        fading_fin_step: >0, to fade in layers one by one, fully faded in when self.fading_step >= fading_fin_step
        """

        super(SIREN_NeRFt, self).__init__()

        self.args = args

        self.scene_scale = args.scene_scale

        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.fading_step = 0
        self.fading_fin_step = fading_fin_step if fading_fin_step>0 else args.fading_layers
        self.bbox_model = bbox_model

        first_omega_0 = 30.0
        hidden_omega_0 = 1.0

        self.pts_linears = nn.ModuleList(
            [SineLayer(input_ch, W, omega_0=first_omega_0)] + 
            [SineLayer(W, W, omega_0=hidden_omega_0) 
                if i not in self.skips else SineLayer(W + input_ch, W, omega_0=hidden_omega_0) for i in range(D-1)]
        )
        
        final_alpha_linear = nn.Linear(W, 1)
        self.alpha_linear = final_alpha_linear
      
        if use_viewdirs:
            self.views_linear = SineLayer(input_ch_views, W//2, omega_0=first_omega_0)
            self.feature_linear = SineLayer(W, W//2, omega_0=hidden_omega_0)
            self.feature_view_linears = nn.ModuleList([SineLayer(W, W, omega_0=hidden_omega_0)])
        
        final_rgb_linear = nn.Linear(W, 3)
        self.rgb_linear = final_rgb_linear

        self.eval_mode = False
        self.occupancy_grid_dynamic = None

        self.density_activation = get_activation(density_activation)


    def update_fading_step(self, fading_step):
        # should be updated with the global step
        # e.g., update_fading_step(global_step - radiance_in_step)
        if fading_step >=0:
            self.fading_step = fading_step

    def fading_wei_list(self):
        # try print(fading_wei_list()) for debug
        step_ratio = np.clip(float(self.fading_step)/float(max(1e-8, self.fading_fin_step)), 0, 1)
        ma = 1+(self.D-2)*step_ratio # in range of 1 to self.D-1
        fading_wei_list = [np.clip(1+ma-m,0,1)*np.clip(1+m-ma,0,1) for m in range(self.D)]
        return fading_wei_list

    def print_fading(self):
        w_list = self.fading_wei_list()
        _str = ["h%d:%0.03f"%(i,w_list[i]) for i in range(len(w_list)) if w_list[i] > 1e-8]
        print("; ".join(_str))

    def density(self, x, xyz_bound = 1.0):
  
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        
        
        # input_pts = input_pts.reshape(-1, self.input_ch)
        # apply xyz bound 
        # scaled_input_pts = input_pts / xyz_bound
        scaled_input_pts = input_pts / self.scene_scale

        h = scaled_input_pts
        h_layers = []
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)

            h_layers += [h]
            if i in self.skips:
                h = torch.cat([scaled_input_pts, h], -1)
        
        # a sliding window (fading_wei_list) to enable deeper layers progressively
        if self.fading_fin_step > self.fading_step: 
            fading_wei_list = self.fading_wei_list()
            h = 0
            for w,y in zip(fading_wei_list, h_layers):
                if w > 1e-8:
                    h = w*y + h

        outputs = self.alpha_linear(h)

        outputs = self.density_activation(outputs)

        if self.bbox_model is not None:
            bbox_mask = self.bbox_model.insideMask(input_pts[...,:3])
            outputs[bbox_mask==0] = 0

        return outputs

    def density_with_jacobian(self, x):
        density = self.density(x)
        jac = _get_minibatch_jacobian(density, x)
        _d_x, _d_y, _d_z, _d_t = [torch.squeeze(_, -1) for _ in jac.split(1, dim=-1)] # (N,1)
        return density, _d_x, _d_y, _d_z, _d_t

    # def density_with_encoding(self, x, encoding):
    #     x = encoding(x)
    #     return self.density(x)[:, :1]

    def color(self, x, xyz_bound = 1.0):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
         # apply xyz bound 
        # scaled_input_pts = input_pts / xyz_bound

        # input_pts = input_pts.reshape(-1, self.input_ch)
        scaled_input_pts = input_pts / self.scene_scale

        h = scaled_input_pts
        h_layers = []
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)

            h_layers += [h]
            if i in self.skips:
                h = torch.cat([scaled_input_pts, h], -1)
        
        # a sliding window (fading_wei_list) to enable deeper layers progressively
        if self.fading_fin_step > self.fading_step: 
            fading_wei_list = self.fading_wei_list()
            h = 0
            for w,y in zip(fading_wei_list, h_layers):
                if w > 1e-8:
                    h = w*y + h


        if self.use_viewdirs:            
            input_pts_feature = self.feature_linear(h)
            input_views_feature = self.views_linear(input_views)

            h = torch.cat([input_pts_feature, input_views_feature], -1)
        
            for i, l in enumerate(self.feature_view_linears):
                h = self.feature_view_linears[i](h)

        rgb = self.rgb_linear(h)

        if self.bbox_model is not None:
            bbox_mask = self.bbox_model.insideMask(input_pts[...,:3]).unsqueeze(-1)
            rgb = bbox_mask * rgb

        return rgb



    def forward(self, x, xyz_bound = 1.0):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
         # apply xyz bound 

        # input_pts = input_pts.reshape(-1, self.input_ch)

        # scaled_input_pts = input_pts / xyz_bound
        scaled_input_pts = input_pts / self.scene_scale

        h = scaled_input_pts
        h_layers = []
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)

            h_layers += [h]
            if i in self.skips:
                h = torch.cat([scaled_input_pts, h], -1)
        
        # a sliding window (fading_wei_list) to enable deeper layers progressively
        if self.fading_fin_step > self.fading_step: 
            fading_wei_list = self.fading_wei_list()
            h = 0
            for w,y in zip(fading_wei_list, h_layers):
                if w > 1e-8:
                    h = w*y + h

        alpha = self.alpha_linear(h)

        alpha = self.density_activation(alpha)

        if self.use_viewdirs:            
            input_pts_feature = self.feature_linear(h)
            input_views_feature = self.views_linear(input_views)

            h = torch.cat([input_pts_feature, input_views_feature], -1)
        
            for i, l in enumerate(self.feature_view_linears):
                h = self.feature_view_linears[i](h)

        rgb = self.rgb_linear(h)
        outputs = torch.cat([rgb, alpha], -1)

        if self.bbox_model is not None:
            bbox_mask = self.bbox_model.insideMask(input_pts[...,:3]).unsqueeze(-1)
            outputs = bbox_mask * outputs

        return outputs

