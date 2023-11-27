import numpy as np
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F

# Loss Tools (all for torch Tensors)
def fade_in_weight(step, start, duration):
    return min(max((float(step) - start)/(duration+1e-6), 0.0), 1.0)
    
def decay_in_weight(step, start, duration, min_decay = 1e-2):
    return max(min_decay, (1.0 - min(max((float(step) - start)/(duration+1e-6), 0.0), 1.0)))

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

def L1_loss(x1,x2):
    return F.l1_loss(x1,x2)

def smooth_l1_loss(x1,x2):
    return F.smooth_l1_loss(x1,x2)

def l2_loss(x1,x2):
    return ((x1-x2)**2).mean()

def my_ghost_loss_func(_rgb, bg, _acc, den_penalty = 0.0):
    _bg = bg.detach()
    _rgb = _rgb.detach()
    ghost_mask = torch.mean(torch.square(_rgb - _bg), -1)
    ghost_mask = torch.sigmoid(ghost_mask*-1.0) + den_penalty # (0 to 0.5) + den_penalty
    ghost_alpha = ghost_mask * _acc
    return torch.mean(torch.square(ghost_alpha))

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



def get_rendering_loss(args, model, rgb, acc, gt_rgb, bg_color, extras, time_locate, global_step, target_mask = None):
    

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
        if not model.single_scene:
            img_loss += (extras['acch2'] * (((gt_rgb - bg_color).abs().sum(-1) < 1e-2)).float()).mean() * args.SmokeAlphaReguW 

    if args.use_mask:
    # if args.use_mask and global_step <= 20000:
        # img_loss += (extras['acch1'] * (target_mask[:,0].float())).mean() * 0.05
        img_loss += (extras['acch1'] * (target_mask[:,0].float())).mean() * 0.2
        # if provide static scene mask
    
    # for car, sperate 
    if args.ColorDivergenceW > 0.0 and not model.single_scene:
        color_divergence_loss = (0.3 -  (extras['rgbh1'].detach() - extras['rgbh2']).abs()).clamp(0.0, 1.0).mean()
        rendering_loss_dict['color_divergence_loss'] = color_divergence_loss
        img_loss += color_divergence_loss* args.ColorDivergenceW
    
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
    if w_smoke_inside_sdf > 1e-8 and not model.single_scene:
            
   
        inside_sdf = args.inside_sdf

        if global_step > args.uniform_sample_step and args.cuda_ray:
          
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

            # smoke_inside_sdf_loss = smoke_inside_loss_on_static + smoke_inside_loss_on_dynamic
            smoke_inside_sdf_loss = smoke_inside_loss_on_dynamic

        else:
            smoke_den = extras['raw'][...,3:4]
            static_sdf = extras['raw_static'][...,3:4]

            inside_mask = static_sdf.detach() <= - inside_sdf

            smoke_inside_sdf_loss = torch.sum((smoke_den*inside_mask) ** 2) / (inside_mask.sum() + 1e-5)


        rendering_loss = rendering_loss + smoke_inside_sdf_loss * w_smoke_inside_sdf
        
    rendering_loss_dict['smoke_inside_sdf_loss'] = smoke_inside_sdf_loss
    
    ghost_loss = None
    if model.single_scene and (args.ghostW > 0.0) and args.white_bkgd is not None:
        ghost_fading = fade_in_weight(global_step, 0, 10000)
        w_ghost = ghost_fading * args.ghostW
        if w_ghost > 1e-8:
            static_back = args.white_bkgd
            # ghost_loss = ghost_loss_func(rgb, static_back, acc, den_penalty=0.0)
            ghost_loss = my_ghost_loss_func(rgb, static_back, acc, den_penalty=0.0)

            rendering_loss += ghost_loss * w_ghost
            
    rendering_loss_dict['ghost_loss'] = ghost_loss
    
    return rendering_loss, rendering_loss_dict


def get_velocity_loss(args, model, training_samples, training_stage, local_step, global_step):


    #####  core velocity optimization loop  #####
    # allows to take derivative w.r.t. training_samples


    velocity_model = model.dynamic_model_lagrangian.velocity_model
    den_model_lagrangian = model.dynamic_model_lagrangian.density_model
    
    if args.use_two_level_density:
        den_model_siren = model.dynamic_model_siren
    
    vel_loss_dict = {}
    vel_loss = 0.0

    # if local_step % args.stage4_train_vel_interval == 0:
    if local_step % 10 == 0:
        
        training_samples = training_samples.requires_grad_(True)
        
        if not model.single_scene:
            _sdf, _normal = model.static_model.sdf_with_gradient(training_samples[..., :3])

        if args.use_two_level_density:
            _den_siren, _d_x_siren, _d_y_siren, _d_z_siren, _d_t_siren = den_model_siren.density_with_jacobian(training_samples)
            _den_lagrangian, features, jacobian = den_model_lagrangian.density_with_jacobian(training_samples)
            _d_x_lagrangian, _d_y_lagrangian, _d_z_lagrangian, _d_t_lagrangian = [torch.squeeze(_, -1) for _ in jacobian.split(1, dim=-1)] # (N,3)
        
            _d_x = _d_x_lagrangian.detach() + 0.01 * _d_x_siren.detach()
            _d_y = _d_y_lagrangian.detach() + 0.01 * _d_y_siren.detach()
            _d_z = _d_z_lagrangian.detach() + 0.01 * _d_z_siren.detach()
            _d_t = _d_t_lagrangian.detach() + 0.01 * _d_t_siren.detach()
        else:
            _den_lagrangian, features, jacobian = den_model_lagrangian.density_with_jacobian(training_samples)
            _d_x, _d_y, _d_z, _d_t = [torch.squeeze(_, -1) for _ in jacobian.split(1, dim=-1)] # (N,3)
            

        if args.use_two_level_density:
            # if training_stage == 4 and not trainVel:
            # # supervise lagrangian density using siren density
            #     density_reference_loss = smooth_l1_loss(F.relu(_den_siren.detach()), F.relu(_den_lagrangian))
                
            #     vel_loss_dict['density_reference_loss'] = density_reference_loss
            #     vel_loss += density_reference_loss
                
            #     return vel_loss, vel_loss_dict
                
            if training_stage == 2:
                # warm up the feature using density
                # warm up the velocity linears using nse equation
        
        
                _vel, Du_Dt = velocity_model.forward_with_feature_save_middle_output(training_samples, features.detach(), need_vorticity=True)
            
                split_nse = PDE_stage2(
                    _d_t.detach(), _d_x.detach(), _d_y.detach(), _d_z.detach(),
                    _vel, 
                    Du_Dt)
                
                split_nse_wei = [1.0, 0.1, 0.1, 1e-2] 


            elif training_stage == 3:
                
                _vel, vel_middle_output = velocity_model.forward_with_middle_output(training_samples, need_vorticity=True)
                
                jac = vel_middle_output['jacobian']
                _u_x, _u_y, _u_z, Du_Dt = [torch.squeeze(_, -1) for _ in jac.split(1, dim=-1)] # (N,3)
                _f_x, _f_y, _f_z = [torch.squeeze(_, -1) for _ in vel_middle_output['dfeature_dxyz'].split(1, dim=-1)] # (N,1)
                _f_t = vel_middle_output['dfeature_dt'].squeeze(-1)
                

                split_nse = PDE_stage3(
                    _f_t, _f_x, _f_y, _f_z,
                    _d_t.detach(), _d_x.detach(), _d_y.detach(), _d_z.detach(),
                    _vel, _u_x, _u_y, _u_z, 
                    Du_Dt)
                
                # density transport, feature continuity, velocity divergence, scale regularzation, Du_Dt,
                # split_nse_wei = [1e-1, 1e-1, 1e-3, 1e-3, 1e-3] 
                # split_nse_wei = [1.0, 1.0, 1e-3, 1e-3, 1e-3] 
                split_nse_wei = [0.1, 0.1, 0.1, 0.1, 1e-3] 
                
                # density_reference_loss = smooth_l1_loss(F.relu(_den_siren.detach()), F.relu(_den_lagrangian))
                        
                # vel_loss_dict['density_reference_loss'] = density_reference_loss 
                
                # vel_loss += density_reference_loss 

            
            elif training_stage == 4:
                
                _vel, vel_middle_output = velocity_model.forward_with_middle_output(training_samples, need_vorticity=True)
                jac = vel_middle_output['jacobian']
                _u_x, _u_y, _u_z, Du_Dt = [torch.squeeze(_, -1) for _ in jac.split(1, dim=-1)] # (N,3)
                _f_x, _f_y, _f_z = [torch.squeeze(_, -1) for _ in vel_middle_output['dfeature_dxyz'].split(1, dim=-1)] # (N,1)
                _f_t = vel_middle_output['dfeature_dt'].squeeze(-1)
                # _vel, Du_Dt = velocity_model.forward_with_feature_save_middle_output(training_samples, features.detach(), need_vorticity=True)
                
                _vel_only_decoder = velocity_model.forward_with_feature(training_samples, vel_middle_output['mapped_features'].detach())

                # split_nse = PDE_stage3(
                #     _f_t, _f_x, _f_y, _f_z,
                #     _d_t.detach(), _d_x.detach(), _d_y.detach(), _d_z.detach(),
                #     _vel, _u_x, _u_y, _u_z, 
                #     Du_Dt)
                split_nse = PDE_stage3(
                    _f_t, _f_x, _f_y, _f_z,
                    _d_t.detach(), _d_x.detach(), _d_y.detach(), _d_z.detach(),
                    _vel_only_decoder, _u_x, _u_y, _u_z, 
                    Du_Dt)
                # split_nse = PDE_stage4(
                    # _f_t, _f_x, _f_y, _f_z,
                    # _d_t.detach(), _d_x.detach(), _d_y.detach(), _d_z.detach(),
                    # _d_t_lagrangian.detach(), _d_x_lagrangian.detach(), _d_y_lagrangian.detach(), _d_z_lagrangian.detach(),
                    # _vel, _u_x, _u_y, _u_z, 
                    # Du_Dt)
                # split_nse_wei = [1.0, 10.0, 1e-1, args.vel_regulization_weight, 1e-1]
                # split_nse_wei = [1.0, 10.0, 1e-1, args.vel_regulization_weight, 0]
                # split_nse_wei = [1.0, 1.0, 1e-2, args.vel_regulization_weight, 0]
                # split_nse_wei = [10.0, 1.0, 1e-2, args.vel_regulization_weight, 0]
                # split_nse_wei = [1.0, 1.0, 1e-2, args.vel_regulization_weight, 10]
                # split_nse_wei = [1.0, 1.0, 1e-1, args.vel_regulization_weight, 0]
                split_nse_wei = [1.0, 1.0, 1e-1, args.vel_regulization_weight, 1e-3]
                
                # split_nse_wei = [1.0, 0.1, 1e-2, args.vel_regulization_weight, 1e-2]
                # split_nse_wei = [1.0, 1.0, 1e-2, args.vel_regulization_weight, 1e-2]
                # split_nse_wei = [1.0, 10.0, 1e-2, args.vel_regulization_weight, 1e-2]
                # split_nse_wei = [1.0, 0.1, 1.0, 1e-2, args.vel_regulization_weight, 1e-2]
                # density transport, feature continuity, velocity divergence, scale regularzation, Du_Dt,
                # split_nse_wei = [0.1, 0.1, 0.1, 0.1, 1e-3] 
                # split_nse_wei = [1.0, 1.0, 1.0, 1e-2, args.vel_regulization_weight, 1e-2]
                # split_nse_wei = [1.0, 1.0, 1e-3, 1e-3, 1e-3] 
                # split_nse_wei = [1.0, 1e-2, 1e-3, 1e-3, 1e-3] 
                # spl0it_nse_wei = [1e-1, 1e-1, 1e-3, 1e-3, 1e-3] 
                
                # density_reference_loss = smooth_l1_loss(F.relu(_den_siren.detach()), F.relu(_den_lagrangian))
                        
                # vel_loss_dict['density_reference_loss'] = density_reference_loss 

                # vel_loss += density_reference_loss 

            else:
                AssertionError("training stage should be set to 1,2,3,4")
                
        else:
            # start train velocity using lagrangian density, and give up siren density


            _vel, vel_middle_output = velocity_model.forward_with_middle_output(training_samples, need_vorticity=True)
            _vel, Du_Dt = velocity_model.forward_with_feature_save_middle_output(training_samples, features.detach(), need_vorticity=True)
            jac = vel_middle_output['jacobian']
            _u_x, _u_y, _u_z, Du_Dt = [torch.squeeze(_, -1) for _ in jac.split(1, dim=-1)] # (N,3)
            _f_x, _f_y, _f_z = [torch.squeeze(_, -1) for _ in vel_middle_output['dfeature_dxyz'].split(1, dim=-1)] # (N,1)
            _f_t = vel_middle_output['dfeature_dt'].squeeze(-1)       
        
            split_nse = PDE_stage3(
                _f_t, _f_x, _f_y, _f_z,
                _d_t.detach(), _d_x.detach(), _d_y.detach(), _d_z.detach(),
                _vel, _u_x, _u_y, _u_z, 
                Du_Dt)
                # Du_Dt, density_mask=_den_lagrangian)
            
            # density transport, feature continuity, velocitt divergence, scale regularzation, Dd_Dt, Du_Dt
            # split_nse_wei = [1.0, 1.0, 1e-3, 1e-3, 1e-3] 
            # split_nse_wei = [1.0, 1.0, 1e-3, 1000000, 1e-3] 
            # split_nse_wei = [1.0, 1.0, 0.1, 100000, 0.1] 
            # split_nse_wei = [1.0, 1.0, 1e-3, 100.0, 1e-3] 
            # split_nse_wei = [10.0, 0.1, 1e-3, 1.0, 1e-3] 
                
            # split_nse_wei = [1.0, 1.0, 1e-3, 1000000, 1e-3] 
            # vel_regulization_weight = 1000 * decay_in_weight(global_step, args.stage1_finish_recon + 3000, 2000, min_decay = 1e-3)
            # split_nse_wei = [1.0, 1.0, 1e-3, vel_regulization_weight, 1e-3]
            # split_nse_wei = [1.0, 1.0, 1e-3, 1000, 1e-3] 
            split_nse_wei = [1.0, 1.0, 1e-1, args.vel_regulization_weight, 1e-1]

                

        # nse_errors = [smooth_l1_loss(x, torch.zeros_like(x)) for x in split_nse]
        # nse_errors = [L1_loss(x, torch.zeros_like(x)) for x in split_nse]
        # nse_errors = [mean_squared_error(x,0.0) for x in split_nse]
        nse_errors = split_nse


        nseloss_fine = 0.0
        for ei,wi in zip (nse_errors, split_nse_wei):
            nseloss_fine = ei*wi + nseloss_fine
        vel_loss += nseloss_fine * args.nseW
        
        vel_loss_dict['nse_errors'] = nse_errors
        vel_loss_dict['nseloss_fine'] = nseloss_fine


        # boundary loss
        if not model.single_scene:
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
            cycle_loss_fading = fade_in_weight(global_step, args.stage1_finish_recon + args.stage2_finish_init_lagrangian + args.stage3_finish_init_feature, 10000) # 
            
            # add cycle loss for lagrangian mapping
            cycle_loss = None
        
            predict_xyz = vel_middle_output['mapped_xyz']
            # cycle_loss = smooth_l1_loss(predict_xyz, training_samples[..., :3])
            cycle_loss = L1_loss(predict_xyz, training_samples[..., :3])
            # vel_loss += 0.1 * cycle_loss
            vel_loss += args.self_cycle_loss_weight * cycle_loss * cycle_loss_fading

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


            # predict_xyz_cross = velocity_model.mapping_forward_with_features(mapped_features, cross_training_t)
            predict_xyz_cross = velocity_model.mapping_forward_with_features(mapped_features, cross_training_t) - predict_xyz + training_samples[..., :3]
            cross_features = velocity_model.forward_feature(predict_xyz_cross.detach(), cross_training_t.detach()) # only train feature mapping

            # cross_cycle_loss = smooth_l1_loss(cross_features, mapped_features)
            cross_cycle_loss = L1_loss(cross_features, mapped_features)
            # vel_loss += 0.05 * cross_cycle_loss * args.nseW
            vel_loss += args.cross_cycle_loss_weight * cross_cycle_loss * cycle_loss_fading
            # vel_loss += 10.0 * cross_cycle_loss

            vel_loss_dict['feature_cycle_loss'] = cycle_loss
            vel_loss_dict['feature_cross_cycle_loss'] = cross_cycle_loss
            
            # advection loss
            # advection_loss = None
            # mapped_xyz_velocity_advect = training_samples[..., :3].detach() + _vel.detach() * 1.0 / args.time_size
            # advect_t = training_samples[..., 3:4] + 1.0 / args.time_size
            # # mapped_xyz_direct = velocity_model.mapping_forward_with_features(mapped_features.detach(), advect_t.detach()) # todo: whether detach features
            # mapped_xyz_direct = velocity_model.mapping_forward_with_features(mapped_features.detach(), advect_t.detach()) - predict_xyz + training_samples[..., :3]
            # advection_loss = smooth_l1_loss(mapped_xyz_velocity_advect.detach(), mapped_xyz_direct)
            # advection_loss = smooth_l1_loss(mapped_xyz_velocity_advect, mapped_xyz_direct)
            
            # advect_t = training_samples[..., 3:4] + 1.0 / args.time_size
            
            # if advect_t <= 1.0:
            # mapped_xyz_direct = velocity_model.mapping_forward_with_features(mapped_features.detach(), advect_t.detach()) # todo: whether detach features
            # # mapped_xyz_direct = velocity_model.mapping_forward_with_features(mapped_features, advect_t.detach()) # todo: whether detach features
            # mapeed_vel = (mapped_xyz_direct - training_samples[..., :3]) * args.time_size
            
            # mask = (_vel.sum(-1) == 0)
            # masked_mapped_vel = mapeed_vel.clone()
            # masked_mapped_vel[mask] = 0

            # mapped_advection_loss = smooth_l1_loss(masked_mapped_vel, _vel.detach())
            # mapped_advection_loss = L1_loss(masked_mapped_vel, _vel.detach())
            # advection_loss = L1_loss(mapeed_vel, _vel.detach())
            # print("mapped_advection_loss: ", mapped_advection_loss)
            # print("advection_loss: ", advection_loss)    
            
            # import pdb
            # pdb.set_trace()
            
            # vel_loss += 1 * advection_loss
            # vel_loss_dict['advection_loss'] = advection_loss
            # vel_loss += 0.0 * mapped_advection_loss * cycle_loss_fading
            # vel_loss_dict['advection_loss'] = advection_loss

        if training_stage == 4:
            # pass
            # density mapping loss to supervise density
            density_mapping_fading = fade_in_weight(global_step, args.stage1_finish_recon + args.stage2_finish_init_lagrangian + args.stage3_finish_init_feature + 50000, 10000) # 

            density_mapping_loss = None
            if args.use_two_level_density:
                _den = _den_siren
                den_model = den_model_siren
            else:
                _den = _den_lagrangian
                den_model = den_model_lagrangian

            density_in_xyz = _den

            predict_xyzt_cross =  torch.cat([predict_xyz_cross, cross_training_t], dim=-1)
            # density_in_mapped_xyz = den_model.density(predict_xyzt_cross.detach()) ## todo:: whether detach this
            density_in_mapped_xyz = den_model.density(predict_xyzt_cross) ## todo:: whether detach this
            # density_in_mapped_xyz = den_model(predict_xyzt_cross) ## todo:: whether detach this
            # density_in_mapped_xyz = den_model.forward_with_features(cross_features.detach(), cross_training_t) ## todo:: whether detach this
            

            density_mapping_loss = smooth_l1_loss(density_in_xyz, density_in_mapped_xyz) # todo:: detach one 
            
            # vel_loss += 0.05 * density_mapping_loss * density_mapping_fading
            vel_loss += args.density_mapping_loss_weight * density_mapping_loss * density_mapping_fading
            vel_loss_dict['density_mapping_loss'] = density_mapping_loss
            

            color_mapping_fading = fade_in_weight(global_step, args.stage1_finish_recon + args.stage2_finish_init_lagrangian + args.stage3_finish_init_feature + 10000, 10000) # 

            color_in_xyz = den_model.color(training_samples.detach())
            # color_in_mapped_xyz = den_model.color(predict_xyzt_cross.detach()) ## todo:: whether detach this
            color_in_mapped_xyz = den_model.color(predict_xyzt_cross) ## todo:: whether detach this
            # color_mapping_loss = smooth_l1_loss(color_in_xyz, color_in_mapped_xyz) # todo:: detach one
            color_mapping_loss = L1_loss(color_in_xyz, color_in_mapped_xyz) # todo:: detach one
            vel_loss += args.color_mapping_loss_weight * color_mapping_loss * color_mapping_fading
            vel_loss_dict['color_mapping_loss'] = color_mapping_loss


            velocity_mapping_fading = fade_in_weight(global_step, args.stage1_finish_recon + args.stage2_finish_init_lagrangian + args.stage3_finish_init_feature + 50000, 10000)
            
            velcotiy_in_xyz = _vel
            velocity_in_mapped_xyz = velocity_model.velocity_mapping_loss(x = training_samples[..., :3], t = training_samples[..., 3:4], mapped_t = cross_training_t) 
            
            velocity_mapping_loss = smooth_l1_loss(velocity_in_mapped_xyz, velcotiy_in_xyz) ## todo:: detach one 
            # vel_loss += 0.001 * velocity_mapping_loss * velocity_mapping_fading
            vel_loss += args.velocity_mapping_loss_weight * velocity_mapping_loss * velocity_mapping_fading
            vel_loss_dict['velocity_mapping_loss'] = velocity_mapping_loss



    else:
        
        cycle_loss_fading = fade_in_weight(global_step, args.stage1_finish_recon + args.stage2_finish_init_lagrangian + args.stage3_finish_init_feature, 10000) # 

        _vel, vel_middle_output = velocity_model.forward_with_middle_output(training_samples, need_vorticity=False)
        
        cycle_loss = None
    
        predict_xyz = vel_middle_output['mapped_xyz']
        # cycle_loss = smooth_l1_loss(predict_xyz, training_samples[..., :3])
        cycle_loss = L1_loss(predict_xyz, training_samples[..., :3])
        # vel_loss += 0.1 * cycle_loss
        vel_loss += args.self_cycle_loss_weight * cycle_loss * cycle_loss_fading

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


        # predict_xyz_cross = velocity_model.mapping_forward_with_features(mapped_features, cross_training_t)
        predict_xyz_cross = velocity_model.mapping_forward_with_features(mapped_features, cross_training_t) - predict_xyz + training_samples[..., :3]
        cross_features = velocity_model.forward_feature(predict_xyz_cross.detach(), cross_training_t.detach()) # only train feature mapping

        # mask = (_vel.sum(-1) == 0)
        # masked_cross_features = cross_features.clone()
        # masked_cross_features[mask] = 0

        # L1_loss(mapped_features, masked_cross_features)
        # import pdb
        # pdb.set_trace()x
        # cross_cycle_loss = smooth_l1_loss(cross_features, mapped_features)
        cross_cycle_loss = L1_loss(cross_features, mapped_features)
        # vel_loss += 0.05 * cross_cycle_loss * args.nseW
        vel_loss += args.cross_cycle_loss_weight * cross_cycle_loss * cycle_loss_fading
        # vel_loss += 10.0 * cross_cycle_loss

        vel_loss_dict['feature_cycle_loss'] = cycle_loss
        vel_loss_dict['feature_cross_cycle_loss'] = cross_cycle_loss
        
        
        # advection loss
        # advection_loss = None
        # mapped_xyz_velocity_advect = training_samples[..., :3].detach() + _vel.detach() * 1.0 / args.time_size
        # advect_t = training_samples[..., 3:4] + 1.0 / args.time_size
        # # mapped_xyz_direct = velocity_model.mapping_forward_with_features(mapped_features.detach(), advect_t.detach()) # todo: whether detach features
        # mapped_xyz_direct = velocity_model.mapping_forward_with_features(mapped_features.detach(), advect_t.detach()) - predict_xyz + training_samples[..., :3]
        # advection_loss = smooth_l1_loss(mapped_xyz_velocity_advect.detach(), mapped_xyz_direct)
        
        # vel_loss_dict['advection_loss'] = advection_loss
        
    return vel_loss, vel_loss_dict


def PDE_stage2(d_t, d_x, d_y, d_z, U, Du_Dt):
    eqs = []
    u,v,w = U.split(1, dim=-1) # (N,1)

    transport = d_t + (u*d_x + v*d_y + w*d_z) # transport constrain
    
    # eqs += [transport]
    eqs += [mean_squared_error(transport,0.0)]

    # if True: # scale regularization
        # eqs += [ (u*u + v*v + w*w)* 1e-1]
    eqs += [mean_squared_error(U, 0.0)]

    
    # eqs += [Du_Dt]
    eqs += [mean_squared_error(Du_Dt,0.0)]
        
    
    return eqs

def PDE_stage3(f_t, f_x, f_y, f_z,
    d_t, d_x, d_y, d_z, U, U_x, U_y, U_z, Du_Dt, density_mask = None):
    eqs = []
    u,v,w = U.split(1, dim=-1) # (N,1)

    transport = d_t + (u*d_x + v*d_y + w*d_z) # transport constrain
    
    # eqs += [transport]
    eqs += [mean_squared_error(transport,0.0)]
    # eqs += [smooth_l1_loss(transport,torch.zeros_like(transport))]
    
    
    feature = f_t + (u.detach()*f_x + v.detach()*f_y + w.detach()*f_z) # feature continuous constrain
    
    # eqs += [feature]
    # eqs += [mean_squared_error(feature,0.0)]
    eqs += [L1_loss(feature,torch.zeros_like(feature))]
    # eqs += [smooth_l1_loss(feature,torch.zeros_like(feature))]

    # eqs += [ U_x[:,0] + U_y[:,1] + U_z[:,2] ] # velocity divergence constrain
    eqs += [mean_squared_error(U_x[:,0] + U_y[:,1] + U_z[:,2],0.0)]
    # eqs += [smooth_l1_loss(U_x[:,0] + U_y[:,1] + U_z[:,2], torch.zeros_like(U_x[:,0] + U_y[:,1] + U_z[:,2]))]
    

    # if True: # scale regularization
    #     eqs += [ (u*u + v*v + w*w)* 1e-1]
    # scale regulization
    # eqs += [U]
    # eqs += [L1_loss(U,torch.zeros_like(U))]
    # eqs += [smooth_l1_loss(U,torch.zeros_like(U))]
    # eqs += [mean_squared_error((u*u + v*v + w*w)* 1e-1, 0.0)]
    # eqs += [mean_squared_error(U, 0.0)]
    if density_mask is not None:
        density_mask = (density_mask < 1e-1).float()
        eqs += [mean_squared_error(U * density_mask, 0.0) + mean_squared_error(U * (1.0 - density_mask), 0.0) * 0.1]
    else:
        eqs += [mean_squared_error(U, 0.0)]
        # eqs += [smooth_l1_loss(U, torch.zeros_like(U))]

    # eqs += [Du_Dt]
    eqs += [mean_squared_error(Du_Dt,0.0)]
    # eqs += [smooth_l1_loss(Du_Dt,torch.zeros_like(Du_Dt))]
    
    # feature norm regulization
    
    return eqs


def PDE_stage4(f_t, f_x, f_y, f_z,
    d_t, d_x, d_y, d_z, 
    d_t_l, d_x_l, d_y_l, d_z_l, 
    U, U_x, U_y, U_z, Du_Dt, density_mask = None):
    eqs = []
    u,v,w = U.split(1, dim=-1) # (N,1)

    transport_lagran = d_t_l + (u*d_x_l + v*d_y_l + w*d_z_l) # transport constrain
    transport = d_t + (u*d_x + v*d_y + w*d_z) # transport constrain
    
    # eqs += [transport]
    eqs += [mean_squared_error(transport_lagran,0.0)]
    eqs += [mean_squared_error(transport,0.0)]
    
    feature = f_t + (u.detach()*f_x + v.detach()*f_y + w.detach()*f_z) # feature continuous constrain
    
    # eqs += [feature]
    eqs += [mean_squared_error(feature,0.0)]

    # eqs += [ U_x[:,0] + U_y[:,1] + U_z[:,2] ] # velocity divergence constrain
    eqs += [mean_squared_error(U_x[:,0] + U_y[:,1] + U_z[:,2],0.0)]
    

    if density_mask is not None:
        density_mask = (density_mask < 1e-1).float()
        eqs += [mean_squared_error(U * density_mask, 0.0) + mean_squared_error(U * (1.0 - density_mask), 0.0) * 0.1]
    else:
        eqs += [mean_squared_error(U, 0.0)]

    # eqs += [Du_Dt]
    eqs += [mean_squared_error(Du_Dt,0.0)]
    
    return eqs


def PDE_constraint(f_t, f_x, f_y, f_z,
    d_t, d_x, d_y, d_z, U, U_x, U_y, U_z, Du_Dt, density_mask = None):
    eqs = []
    u,v,w = U.split(1, dim=-1) # (N,1)

    transport = d_t + (u*d_x + v*d_y + w*d_z) # transport constrain
    
    eqs += [mean_squared_error(transport,0.0)]
    
    feature = f_t + (u.detach()*f_x + v.detach()*f_y + w.detach()*f_z) # feature continuous constrain
    
    eqs += [L1_loss(feature,torch.zeros_like(feature))]

    eqs += [mean_squared_error(U_x[:,0] + U_y[:,1] + U_z[:,2],0.0)]
    
    eqs += [mean_squared_error(U, 0.0)]

    eqs += [mean_squared_error(Du_Dt,0.0)]
    
    
    return eqs