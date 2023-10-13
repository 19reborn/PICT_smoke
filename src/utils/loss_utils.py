import numpy as np
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F

# Loss Tools (all for torch Tensors)
def fade_in_weight(step, start, duration):
    return min(max((float(step) - start)/(duration+1e-6), 0.0), 1.0)
    

# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

def mean_squared_error(pred, exact):
    if type(pred) is np.ndarray:
        return np.mean(np.square(pred - exact))
    return torch.mean(torch.square(pred - exact))

def cos_loss(x1,x2):
    return F.cosine_similarity(x1,x2).mean()

def smooth_l1_loss(x1,x2):
    return F.smooth_l1_loss(x1,x2)

def l2_loss(x1,x2):
    return ((x1-x2)**2).mean()


# VGG Tool, https://github.com/crowsonkb/style-transfer-pytorch/
class VGGFeatures(nn.Module):
    poolings = {'max': nn.MaxPool2d, 'average': nn.AvgPool2d} #, 'l2': partial(nn.LPPool2d, 2)}
    pooling_scales = {'max': 1., 'average': 2., 'l2': 0.78}

    def __init__(self, layers, pooling='max'):
        super().__init__()
        self.layers = sorted(set(layers))

        # The PyTorch pre-trained VGG-19 expects sRGB inputs in the range [0, 1] which are then
        # normalized according to this transform, unlike Simonyan et al.'s original model.
        self.normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

        # The PyTorch pre-trained VGG-19 has different parameters from Simonyan et al.'s original
        # model.
        self.model = torchvision.models.vgg19(pretrained=True).features[:self.layers[-1] + 1]
        self.devices = [torch.device('cpu')] * len(self.model)

        # Reduces edge artifacts.
        self.model[0] = self._change_padding_mode(self.model[0], 'replicate')

        pool_scale = self.pooling_scales[pooling]
        for i, layer in enumerate(self.model):
            if pooling != 'max' and isinstance(layer, nn.MaxPool2d):
                # Changing the pooling type from max results in the scale of activations
                # changing, so rescale them. Gatys et al. (2015) do not do this.
                self.model[i] = Scale(self.poolings[pooling](2), pool_scale)

        self.model.eval()
        self.model.requires_grad_(False)

    @staticmethod
    def _change_padding_mode(conv, padding_mode):
        new_conv = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size,
                             stride=conv.stride, padding=conv.padding,
                             padding_mode=padding_mode)
        with torch.no_grad():
            new_conv.weight.copy_(conv.weight)
            new_conv.bias.copy_(conv.bias)
        return new_conv

    @staticmethod
    def _get_min_size(layers):
        last_layer = max(layers)
        min_size = 1
        for layer in [4, 9, 18, 27, 36]:
            if last_layer < layer:
                break
            min_size *= 2
        return min_size

    def distribute_layers(self, devices):
        for i, layer in enumerate(self.model):
            if i in devices:
                device = torch.device(devices[i])
            self.model[i] = layer.to(device)
            self.devices[i] = device

    def forward(self, input, layers=None):
        # input shape, b,3,h,w
        layers = self.layers if layers is None else sorted(set(layers))
        h, w = input.shape[2:4]
        min_size = self._get_min_size(layers)
        if min(h, w) < min_size:
            raise ValueError(f'Input is {h}x{w} but must be at least {min_size}x{min_size}')
        feats = {'input': input}
        norm_in = torch.stack([self.normalize(input[_i]) for _i in range(input.shape[0])], dim=0)
        # input = self.normalize(input)
        for i in range(max(layers) + 1):
            norm_in = self.model[i](norm_in.to(self.devices[i]))
            if i in layers:
                feats[i] = norm_in
        return feats

class VGGlossTool(object):
    def __init__(self, device, pooling='max'):
        # The default content and style layers in Gatys et al. (2015):
        #   content_layers = [22], 'relu4_2'
        #   style_layers = [1, 6, 11, 20, 29], relu layers: [ 'relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
        # We use [5, 10, 19, 28], conv layers before relu: [ 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
        self.layer_list = [5, 10, 19, 28]        
        self.layer_names = [
            "block2_conv1",
            "block3_conv1",
            "block4_conv1",
            "block5_conv1",
        ]
        self.device = device

        # Build a VGG19 model loaded with pre-trained ImageNet weights
        self.vggmodel = VGGFeatures(self.layer_list, pooling=pooling)
        device_plan = {0: device}
        self.vggmodel.distribute_layers(device_plan)

    def feature_norm(self, feature):
        # feature: b,h,w,c
        feature_len = torch.sqrt(torch.sum(torch.square(feature), dim=-1, keepdim=True)+1e-12)
        norm = feature / feature_len
        return norm

    def cos_sim(self, a,b):
        cos_sim_ab = torch.sum(a*b, dim=-1)
        # cosine similarity, -1~1, 1 best
        cos_sim_ab_score = 1.0 - torch.mean(cos_sim_ab) # 0 ~ 2, 0 best
        return cos_sim_ab_score

    def compute_cos_loss(self, img, ref):
        # input img, ref should be in range of [0,1]
        input_tensor = torch.stack( [ref, img], dim=0 )
        
        input_tensor = input_tensor.permute((0, 3, 1, 2))
        # print(input_tensor.shape)
        _feats = self.vggmodel(input_tensor, layers=self.layer_list)

        # Initialize the loss
        loss = []
        # Add loss
        for layer_i, layer_name in zip (self.layer_list, self.layer_names):
            cur_feature = _feats[layer_i]
            reference_features = self.feature_norm(cur_feature[0, ...])
            img_features = self.feature_norm(cur_feature[1, ...])

            feature_metric = self.cos_sim(reference_features, img_features)
            loss += [feature_metric]
        return loss



def get_rendering_loss(args, model, rgb, gt_rgb, bg_color, extras, time_locate, global_step, target_mask = None):
    

    #####  core rendering optimization loop  #####
    # allows to take derivative w.r.t. training_samples

    ## Several stages
    # 1. train smoke and static scene
    # 2. stop training, only filter lagrangian velocity
    # 3. start train lagrangian density using nerf, and use it to train lagrian velocity through transport and density mapping loss

    rendering_loss_dict = {}
    rendering_loss = 0.0
    
    # tempo_fading = fade_in_weight(global_step, args.smoke_recon_delay, 10000)
    smoke_recon_fading = fade_in_weight(global_step, args.smoke_recon_delay_start, args.smoke_recon_delay_last) # 
    # smoke_inside_sdf_loss_fading = fade_in_weight(global_step, args.smoke_recon_delay + 5000, 10000)
    smoke_inside_sdf_loss_fading = fade_in_weight(global_step, args.sdf_loss_delay + args.smoke_recon_delay_start + args.smoke_recon_delay_last, 10000)
    
    img_loss = img2mse(rgb, gt_rgb)
    psnr = mse2psnr(img_loss)
    
    if ('rgbh1' in extras) and (smoke_recon_fading < (1.0-1e-8)): # rgbh1: static
        img_loss = img_loss * smoke_recon_fading + img2mse((extras['rgbh1'] - gt_rgb) * (1-extras['acch2']).reshape(-1, 1), 0) * (1.0-smoke_recon_fading) + extras['acch2'].mean() * args.SmokeAlphaReguW

    else:
        # todo::tricky now
        img_loss += (extras['acch2'] * (((gt_rgb - bg_color).abs().sum(-1) < 1e-2)).float()).mean() * args.SmokeAlphaReguW 
        pass

    if args.use_mask:
    # if args.use_mask and global_step <= 20000:
        # img_loss += (extras['acch1'] * (target_mask[:,0].float())).mean() * 0.05
        img_loss += (extras['acch1'] * (target_mask[:,0].float())).mean() * 0.2
        # if provide static scene mask
    
    
    rendering_loss += img_loss
    rendering_loss_dict['img_loss'] = img_loss
    rendering_loss_dict['psnr'] = psnr  

    eikonal_loss = None
    curvature_loss = None

    if 'gradients' in extras:
        gradients = extras['gradients']
        eikonal_loss = (torch.norm(gradients.reshape(-1, 3),
                                            dim=-1) - 1.0) ** 2
        eikonal_loss = eikonal_loss.mean()
        
        rendering_loss += eikonal_loss * args.ekW
        
    rendering_loss_dict['eikonal_loss'] = eikonal_loss


    # if 'hessians' in extras:
    #     hessians = extras['hessians']
    #     laplacian = hessians.sum(dim=-1).abs()
    #     # laplacian = laplacian.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0) 
    #     curvature_loss = laplacian.mean()
    #     curvature_weight = curvature_fading * args.CurvatureW 
    #     loss = loss + curvature_loss * curvature_weight
    rendering_loss_dict['curvature_loss'] = curvature_loss

    smoke_inside_sdf_loss = None
    w_smoke_inside_sdf = args.SmokeInsideSDFW * smoke_inside_sdf_loss_fading
    if w_smoke_inside_sdf > 1e-8:
            
   
        inside_sdf = args.inside_sdf

        if global_step >= args.uniform_sample_step:
          
            samples_xyz_static = extras['samples_xyz_static'].clone().detach()

            samples_xyz_static_t = torch.cat([samples_xyz_static, time_locate * torch.ones_like(samples_xyz_static[..., :1])], dim=-1) # [N, 4]
            samples_xyz_dynamic = extras['samples_xyz_dynamic'].clone().detach()

            ## overlay loss on static_samples, only penalize dynamic part
            static_sdf_on_static = extras['raw_static'][...,3:4]
            smoke_den_on_static = model.density_dynamic(samples_xyz_static_t.detach())

            ## overlay loss on dynamic_samples, only penalize static part
            smoke_den_on_dynamic = extras['raw'][...,3:4]
            static_sdf_on_dynamic = model.sdf_static(samples_xyz_dynamic)

            inside_mask_on_static = static_sdf_on_static.detach() <= - inside_sdf

            smoke_inside_loss_on_static = torch.sum((smoke_den_on_static*inside_mask_on_static) ** 2) / (inside_mask_on_static.sum() + 1e-6)

            inside_mask_on_dynamic = static_sdf_on_dynamic.detach() <= - inside_sdf

            smoke_inside_loss_on_dynamic = torch.sum((smoke_den_on_dynamic*inside_mask_on_dynamic) ** 2) / (inside_mask_on_dynamic.sum() + 1e-6)

            smoke_inside_sdf_loss = smoke_inside_loss_on_static + smoke_inside_loss_on_dynamic
            # smoke_inside_sdf_loss = smoke_inside_loss_on_dynamic
            # smoke_inside_sdf_loss = 0.0

        else:
            smoke_den = extras['raw'][...,3:4]
            static_sdf = extras['raw_static'][...,3:4]

            inside_mask = static_sdf.detach() <= - inside_sdf

            smoke_inside_sdf_loss = torch.sum((smoke_den*inside_mask) ** 2) / (inside_mask.sum() + 1e-5)


        rendering_loss = rendering_loss + smoke_inside_sdf_loss * w_smoke_inside_sdf
        
    rendering_loss_dict['smoke_inside_sdf_loss'] = smoke_inside_sdf_loss
    
    return rendering_loss, rendering_loss_dict


# @profile
def get_velocity_loss(args, model, training_samples, training_stage, global_step):




    #####  core velocity optimization loop  #####
    # allows to take derivative w.r.t. training_samples

    ## Several stages
    # 1. Fix feature, train g,d for lagrangian velocity and lagrangian density, supervised by density transport equation and density
    # 2. start train feature using cycle loss and its material derivatives
    # 3. start train lagrangian density using nerf, and use it to train lagrian velocity through transport and density mapping loss

    vel_model = model.dynamic_model.vel_model
    
    den_model = model.dynamic_model.density_model
    den_model_ref = model.dynamic_model_siren

    _vel, vel_middle_output = vel_model.forward_with_middle_output(training_samples, need_vorticity=True)
    jac = vel_middle_output['jacobian']
    _u_x, _u_y, _u_z, Du_Dt = [torch.squeeze(_, -1) for _ in jac.split(1, dim=-1)] # (N,3)
    
    
    _sdf, _normal = model.static_model.sdf_with_gradient(training_samples[..., :3])
    
    
    _den, den_middle_output = den_model.forward_with_middle_output(training_samples, need_jacobian = True)
    _d_x, _d_y, _d_z = [torch.squeeze(_, -1) for _ in den_middle_output['ddensity_dxyz'].split(1, dim=-1)] # (N,1)
    _f_x, _f_y, _f_z = [torch.squeeze(_, -1) for _ in den_middle_output['dfeature_dxyz'].split(1, dim=-1)] # (N,1)
    Dd_Dt = den_middle_output['Ddensity_Dt'].squeeze(-1)
    _d_t = den_middle_output['ddensity_dt'].squeeze(-1)
    _f_t = den_middle_output['dfeature_dt'].squeeze(-1)
    
    # todo:: ignore some operations unecessary for diff stages
    
    if training_stage == 2 or training_stage == 3:
        # get renference density
        training_samples_ref = training_samples.clone().detach().requires_grad_(True) 
        _den_ref, _d_x_ref, _d_y_ref, _d_z_ref, _d_t_ref = den_model_ref.density_with_jacobian(training_samples_ref)
        
        
    vel_loss_dict = {}
    vel_loss = 0.0

    if training_stage == 2:
        # only train d and v first
        split_nse = PDE_stage2(
            _d_t_ref.detach(), _d_x_ref.detach(), _d_y_ref.detach(), _d_z_ref.detach(),
            _vel, _u_x, _u_y, _u_z, 
            Dd_Dt, Du_Dt)
        
        # density transport, velocitt divergence, scale regularzation, Dd_Dt, Du_Dt
        split_nse_wei = [2.0, 1e-3, 1e-3, 1e-3, 1e-3] 
        
        # loss compared with reference density and color
        
        density_reference_loss = smooth_l1_loss(_den, _den_ref.detach())
        
        _color = model.dynamic_model.color(training_samples)
        _color_ref = model.dynamic_model_siren.color(training_samples)
        
        color_reference_loss = smooth_l1_loss(_color, _color_ref.detach())
         
        vel_loss_dict['density_reference_loss'] = density_reference_loss
        vel_loss_dict['color_reference_loss'] = color_reference_loss       

        vel_loss += density_reference_loss + color_reference_loss

    elif training_stage == 3:
        # start train feature
        split_nse = PDE_stage3(
            _f_t, _f_x, _f_y, _f_z,
            _d_t_ref.detach(), _d_x_ref.detach(), _d_y_ref.detach(), _d_z_ref.detach(),
            _vel, _u_x, _u_y, _u_z, 
            Dd_Dt, Du_Dt)
        
        # density transport, feature continuity, velocitt divergence, scale regularzation, Dd_Dt, Du_Dt
        split_nse_wei = [1.0, 1.0, 1e-3, 1e-3, 1e-3, 1e-3] 
        
        
        # loss compared with reference density and color
        density_reference_loss = smooth_l1_loss(_den, _den_ref.detach())
        
        _color = model.dynamic_model.color(training_samples)
        _color_ref = model.dynamic_model_siren.color(training_samples)
        
        color_reference_loss = smooth_l1_loss(_color, _color_ref.detach())
         
        vel_loss_dict['density_reference_loss'] = density_reference_loss
        vel_loss_dict['color_reference_loss'] = color_reference_loss       

        vel_loss += density_reference_loss + color_reference_loss


    elif training_stage == 4:
        # start train velocity using lagrangian density, and give up siren density
        
        split_nse = PDE_stage3(
            _f_t, _f_x, _f_y, _f_z,
            _d_t.detach(), _d_x.detach(), _d_y.detach(), _d_z.detach(),
            _vel, _u_x, _u_y, _u_z, 
            Dd_Dt, Du_Dt)
        
        # density transport, feature continuity, velocitt divergence, scale regularzation, Dd_Dt, Du_Dt
        split_nse_wei = [1.0, 1.0, 1e-3, 1e-3, 1e-3, 1e-3] 
        

    else:
        
        AssertionError("training stage should be set to 1,2,3,4")
        
  
    nse_errors = [mean_squared_error(x,0.0) for x in split_nse]


    nseloss_fine = 0.0
    for ei,wi in zip (nse_errors, split_nse_wei):
        nseloss_fine = ei*wi + nseloss_fine
    vel_loss += nseloss_fine * args.nseW
    
    vel_loss_dict['nse_errors'] = nse_errors
    vel_loss_dict['nseloss_fine'] = nseloss_fine


    # boundary loss
    # if the sampling point's sdf < 0.05 and > -0.05, we assume it's on the boundary: The velocity along the normal direction must be zero.
    # If the sampling point's sdf < -0.05, we assume it's inside the object : The velocity should be 0.
    _sdf = _sdf.detach()
    _normal = _normal.detach()

    # boundary_sdf = 0.05
    # boundary_sdf = 0.02 * args.scene_scale
    # boundary_sdf = 0.00 * args.scene_scale
    boundary_sdf = args.inside_sdf
    boundary_mask = torch.abs(_sdf) < boundary_sdf
    boundary_vel = _vel * boundary_mask
    
    # boundary_vel_normal = torch.dot(boundary_vel, _normal) 
    boundary_vel_normal = (boundary_vel * _normal).sum(-1)
    _normal_norm_squared = torch.sum(_normal ** 2, dim = -1, keepdim=True)
    boundary_vel_project2normal = boundary_vel_normal[:,None] / (_normal_norm_squared + 1e-6) * boundary_vel
    boundary_loss = torch.sum(boundary_vel_project2normal ** 2) / (boundary_mask.sum() + 1e-6)
    # boundary_loss = mean_squared_error(boundary_vel_project2normal, torch.zeros_like(boundary_vel_project2normal))

    inside_sdf = args.inside_sdf
    inside_mask = _sdf < -inside_sdf
    inside_vel = _vel * inside_mask
    inside_loss = torch.sum(inside_vel ** 2) / (boundary_mask.sum() + 1e-6)

    vel_loss += (boundary_loss + inside_loss) * args.boundaryW

    vel_loss_dict['boundary_loss'] = boundary_loss
    vel_loss_dict['inside_loss'] = inside_loss


    ## cycle loss for lagrangian feature
    if training_stage == 3 or training_stage == 4:
        # add cycle loss for lagrangian mapping
        cycle_loss = None
    
        predict_xyz = vel_middle_output['mapped_xyz']
        cycle_loss = smooth_l1_loss(predict_xyz, training_samples[..., :3])
        vel_loss += 1.0 * cycle_loss

        cross_cycle_loss = None

        mapped_features = vel_middle_output['mapped_features']
        
        mapping_frame_fading = fade_in_weight(global_step, args.stage1_finish_recon + args.stage2_finish_init_lagrangian + args.stage3_finish_init_feature + args.mapping_frame_range_fading_start, args.mapping_frame_range_fading_last) # 

        min_mapping_frame = 3
        max_mapping_frame = args.max_mapping_frame_range
        mapping_frame_range = (max_mapping_frame - min_mapping_frame) * mapping_frame_fading + min_mapping_frame
        random_warpT = torch.rand_like(training_samples[:,0:1]) * mapping_frame_range * 2 - mapping_frame_range # todo:: change to long term frame
        # random_warpT = torch.rand_like(training_samples[:,0:1]) * 6.0 - 3.0 # todo:: change to long term frame

        cross_delta_t =  random_warpT * 1.0 / args.time_size

        cross_training_t = training_samples[...,3:4] + cross_delta_t
  
        cross_training_t = torch.clamp(cross_training_t, 0.0, 1.0) # clamp to (0,1)


        predict_xyz_cross = vel_model.mapping_forward_with_features(mapped_features, cross_training_t)
        cross_features = vel_model.feature_map(predict_xyz_cross.detach(), cross_training_t.detach()) # only train feature mapping

        cross_cycle_loss = smooth_l1_loss(cross_features, mapped_features)
        vel_loss += 0.2 * cross_cycle_loss

        vel_loss_dict['feature_cycle_loss'] = cycle_loss
        vel_loss_dict['feature_cross_cycle_loss'] = cross_cycle_loss

    if training_stage == 4:
        ## density mapping loss to supervise density
        density_mapping_fading = fade_in_weight(global_step, args.stage1_finish_recon + args.stage2_finish_init_lagrangian + args.stage3_finish_init_feature + 20000, 10000) # 

        density_mapping_loss = None
        density_in_xyz = _den

        
        predict_xyzt_cross =  torch.cat([predict_xyz_cross, cross_training_t], dim=-1)
        # density_in_mapped_xyz = den_model(predict_xyzt_cross.detach()) ## todo:: whether detach this
        # density_in_mapped_xyz = den_model(predict_xyzt_cross) ## todo:: whether detach this
        density_in_mapped_xyz = den_model.forward_with_features(cross_features.detach(), cross_training_t) ## todo:: whether detach this
        

        density_mapping_loss = smooth_l1_loss(density_in_xyz, density_in_mapped_xyz)
        
        vel_loss += 0.01 * density_mapping_loss * density_mapping_fading

        
        vel_loss_dict['density_mapping_loss'] = density_mapping_loss



    return vel_loss, vel_loss_dict


def PDE_stage2(d_t, d_x, d_y, d_z, U, U_x, U_y, U_z, Dd_Dt, Du_Dt):
    eqs = []
    u,v,w = U.split(1, dim=-1) # (N,1)

    transport = d_t + (u*d_x + v*d_y + w*d_z) # transport constrain
    
    eqs += [transport]

    eqs += [ U_x[:,0] + U_y[:,1] + U_z[:,2] ] # velocity divergence constrain 
    # todo:: remove velocity divergence in this stage?
    

    if True: # scale regularization
        eqs += [ (u*u + v*v + w*w)* 1e-1]

    eqs += [Dd_Dt]
    
    eqs += [Du_Dt]
    
        
    
    return eqs

def PDE_stage3(f_t, f_x, f_y, f_z,
    d_t, d_x, d_y, d_z, U, U_x, U_y, U_z, Dd_Dt, Du_Dt):
    eqs = []
    u,v,w = U.split(1, dim=-1) # (N,1)

    transport = d_t + (u*d_x + v*d_y + w*d_z) # transport constrain
    
    eqs += [transport]
    
    feature = f_t + (u.detach()*f_x + v.detach()*f_y + w.detach()*f_z) # feature continuous constrain
    
    eqs += [feature]

    eqs += [ U_x[:,0] + U_y[:,1] + U_z[:,2] ] # velocity divergence constrain
    

    if True: # scale regularization
        eqs += [ (u*u + v*v + w*w)* 1e-1]

    eqs += [Dd_Dt]
    
    eqs += [Du_Dt]
    
    
    return eqs
