import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0, dim=3):
    if i == -1:
        return nn.Identity(), dim
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : dim,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim


class SDFNetwork(nn.Module):
    def __init__(self,
                args,
                d_in,
                d_out,
                d_hidden,
                n_layers,
                encoding_type = 'PE',
                skip_in=(4,),
                multires=0,
                bias=0.5,
                scale=1,
                geometric_init=True,
                weight_norm=True,
                inside_outside=False):
        super(SDFNetwork, self).__init__()

        self.encoding_type = encoding_type
        self.geometric_init = geometric_init

        dims = [3] + [d_hidden for _ in range(n_layers)] + [d_out]
        if self.encoding_type == "HashGrid":
            import tinycudann as tcnn
            self.bound=1 # for tcnn
            N_max=256 # max_grid_resolution, default 2048
            N_min=16 # base_grid_resolution, default 16
            log2_hashmap_size=19
            n_levels=14
            n_features_per_level=2
            per_level_scale = np.exp2(np.log2(N_max * self.bound / N_min) / (n_levels - 1))
            self.encoder = tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "HashGrid",
                    "n_levels": n_levels,
                    "n_features_per_level": n_features_per_level,
                    "log2_hashmap_size": log2_hashmap_size,
                    "base_resolution": N_min,
                    "per_level_scale": per_level_scale,
                    "interpolation": "Smoothstep" 
                },
            )
            dims[0] = self.encoder.n_output_dims
            if self.geometric_init:
                dims[0] += 3
        elif self.encoding_type == "PE":
            self.encoder = None

            i_embed_neus = args.i_embed_neus
            if i_embed_neus == 0:
                multires = args.multires_neus
            elif i_embed_neus == -1:
                multires = 0
            else:
                AssertionError("i_embed_neus should be 0 or -1")
            if multires > 0:
                embed_fn, input_ch = get_embedder(multires, i_embed_neus, 3)
                self.encoder= embed_fn
                dims[0] = input_ch
        else:
            raise Exception(f"SDFNetwork not support encoding {self.encoding_type}")



        # suppose encoding is finished before passing to SDFNetwork
        # if multires > 0:
        #     embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
        #     self.embed_fn_fine = embed_fn
        #     dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)
                elif (self.encoding_type == "HashGrid" or (self.encoding_type == "PE" and multires > 0)) and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif (self.encoding_type == "HashGrid" or (self.encoding_type == "PE" and multires > 0)) and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.Softplus(beta=100)

    def forward(self, inputs):

        if self.encoding_type == "HashGrid":
        
            scaled_inputs = (inputs + self.bound) / (2 * self.bound) # to [0, 1]
            scaled_inputs = scaled_inputs.clamp(0, 1)
            h = self.encoder(scaled_inputs).float()
            if self.geometric_init:
                # inputs = torch.cat([inputs-0.5, h], dim=-1)
                inputs = torch.cat([inputs, h], dim=-1)
            else:
                inputs = h
        elif self.encoding_type == "PE":
            inputs = inputs * self.scale
            if self.encoder is not None:
                inputs = self.encoder(inputs)
        else:
            raise Exception(f"SDFNetwork not support encoding {self.encoding_type}")

        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], -1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)

        if self.encoding_type == "HashGrid":
            return x
        elif self.encoding_type == "PE":
            return torch.cat([x[..., :1] / self.scale, x[..., 1:]], dim=-1)


    def sdf(self, x):
        return self.forward(x)[..., :1]

    # def sdf_with_encoding(self, x, encoding):
    #     x = encoding(x)
    #     return self.forward(x)[:, :1]


    def sdf_hidden_appearance(self, x):
        return self.forward(x)

    def gradient(self, sdf, x):
        # x.requires_grad_(True)
        # y = self.sdf(x)
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients

# This implementation is borrowed from IDR: https://github.com/lioryariv/idr
class RenderingNetwork(nn.Module):
    def __init__(self,
                args,
                d_feature,
                mode,
                d_in,
                d_out,
                d_hidden,
                n_layers,
                weight_norm=True,
                squeeze_out=True):
        super().__init__()

        self.mode = mode
        self.squeeze_out = squeeze_out

        self.embed_fn = None
        i_embed_neus = args.i_embed_neus
        if i_embed_neus == 0:
            multires = args.multires_neus
        elif i_embed_neus == -1:
            multires = 0
        else:
            AssertionError("i_embed_neus should be 0 or -1")
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, i_embed_neus, 3)
            self.embed_fn = embed_fn
            d_in += input_ch - 3

        # embed_fn, input_ch = get_embedder(6, 0, 3)
        # d_in += input_ch - 3
        # self.embed_fn = embed_fn
   
        dims = [d_in + d_feature] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()

    def forward(self, points, normals, view_dirs, feature_vectors):

        rendering_input = None
        if self.embed_fn is not None:
            points = self.embed_fn(points)

        if self.mode == 'idr':
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_view_dir':
            rendering_input = torch.cat([points, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_normal':
            rendering_input = torch.cat([points, view_dirs, feature_vectors], dim=-1)

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        if self.squeeze_out:
            x = torch.sigmoid(x)
        return x

class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        return torch.ones([len(x), 1]) * torch.exp(self.variance * 10.0)


class NeuS(nn.Module):
# This implementation is borrowed from IDR: https://github.com/lioryariv/idr

    def __init__(self, args, D=4, W=256, input_ch=3, d_feature = 256, input_ch_views=3, output_ch=3, skips=[4], use_viewdirs=False, bbox_model=None,
                    sdf_D = 8, sdf_W = 256, sdf_output_ch = 257, sdf_skips = [4], ):
        """ 
        """
        super(NeuS, self).__init__()

        self.args = args

        # self.scene_scale = args.scene_scale
        self.scene_scale = 1.0

        encoding_type = 'HashGrid' if args.use_neus2_network else 'PE'
        if args.use_neus2_network:
            self.sdf_network = SDFNetwork(args = args, encoding_type = encoding_type, d_in = input_ch, d_out = sdf_output_ch, d_hidden = 64, n_layers = 2, skip_in = ())
            self.color_network = RenderingNetwork(args = args, d_feature = d_feature, mode = 'no_view_dir', d_in = input_ch + 3 , d_out = output_ch, d_hidden = 64, n_layers = 1) # input_ch: additional input like normals
            # self.color_network = RenderingNetwork(args = args, d_feature = d_feature, mode = 'idr', d_in = input_ch + 3 + input_ch_views, d_out = output_ch, d_hidden = 64, n_layers = 1) # input_ch: additional input like normals
        else:
            self.sdf_network = SDFNetwork(args = args, encoding_type = encoding_type, d_in = input_ch, d_out = sdf_output_ch, d_hidden = sdf_W, n_layers = sdf_D, skip_in = sdf_skips)
            self.color_network = RenderingNetwork(args = args, d_feature = d_feature, mode = 'no_view_dir', d_in = input_ch + 3, d_out = output_ch, d_hidden = W, n_layers = D) # input_ch: additional input like normals
            # self.color_network = RenderingNetwork(args = args, d_feature = d_feature, mode = 'idr', d_in = input_ch + 3 + input_ch_views, d_out = output_ch, d_hidden = W, n_layers = D) # input_ch: additional input like normals

        self.deviation_network = SingleVarianceNetwork(init_val = 0.3)

        self.input_ch = input_ch
        self.input_ch_views = input_ch_views

        self.bbox_model = bbox_model
        self.eval_mode = False


    # def forward(self, x):
    def forward(self, pts, views=None, xyz_bound = 1.0):

        # input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        pts.requires_grad_(True)
        # apply bound
        # pts: [-1, 1]
        # scaled_pts = pts / xyz_bound
        scaled_pts = pts / self.scene_scale 

        # input_pts = pts_encoding(scaled_pts)
        # input_views = views_encoding(views)

        sdf_nn_output = self.sdf_network(scaled_pts)
        sdf = sdf_nn_output[..., :1]
        feature_vector = sdf_nn_output[..., 1:]

        gradients = self.sdf_network.gradient(sdf, pts)
        sampled_color = self.color_network(scaled_pts, gradients, views, feature_vector)

        if self.args.CurvatureW != 0:
            hessian = torch.autograd.grad(gradients.sum(), pts, create_graph=True)[0]
            outputs = torch.cat([sampled_color, sdf, gradients, hessian], dim=-1)
        else:
            outputs = torch.cat([sampled_color, sdf, gradients], dim=-1)


        if self.bbox_model is not None:
            bbox_mask = self.bbox_model.insideMask(pts[...,:3]).unsqueeze(-1)
            # outputs = torch.reshape(bbox_mask, [-1,1]) * outputs
            outputs[...,:3] = bbox_mask * outputs[...,:3]
            outputs[...,3:4][(bbox_mask==0)] = 1e6

        return outputs
    
    def fix_params_grad(self):
        for name, p in self.named_parameters():
            p.requires_grad = False


    def sdf(self, pts, xyz_bound = 1.0):

        # sdf = self.sdf_network.sdf_with_encoding(pts, pts_encoding)
        # scaled_pts = pts / xyz_bound
        scaled_pts = pts / self.scene_scale  ## todo:: remove scene_scale?
        sdf = self.sdf_network.sdf(scaled_pts)
        
        if self.bbox_model is not None:
            bbox_mask = self.bbox_model.insideMask(pts[:,:3])
            # sdf = torch.reshape(bbox_mask, [-1,1]) * sdf # buggy!!!

            sdf[bbox_mask.reshape(-1,1)==0] = 1e6

        return sdf


    def sdf_with_gradient(self, pts):
        # input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        pts.requires_grad_(True)
        # apply bound
        # pts: [-1, 1]
        # scaled_pts = pts / xyz_bound
        scaled_pts = pts / self.scene_scale 

        # input_pts = pts_encoding(scaled_pts)
        # input_views = views_encoding(views)

        sdf_nn_output = self.sdf_network(scaled_pts)
        sdf = sdf_nn_output[..., :1]

        gradients = self.sdf_network.gradient(sdf, pts)
        
        return sdf, gradients

    ## used for up sampling
    def cat_z_vals(self, rays_o, rays_d, z_vals, new_z_vals, sdf, embed_fn, last=False):
        batch_size, n_samples = z_vals.shape
        _, n_importance = new_z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
        z_vals, index = torch.sort(z_vals, dim=-1)

        if not last:
            # new_sdf = self.sdf_network.sdf_with_encoding(pts.reshape(-1, 3), embed_fn).reshape(batch_size, n_importance)
            new_sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, n_importance)
            sdf = torch.cat([sdf, new_sdf], dim=-1)
            xx = torch.arange(batch_size)[:, None].expand(batch_size, n_samples + n_importance).reshape(-1)
            index = index.reshape(-1)
            sdf = sdf[(xx, index)].reshape(batch_size, n_samples + n_importance)

        return z_vals, sdf

    def up_sample(self, rays_o, rays_d, z_vals, n_importance, up_sample_steps, embed_fn):

        def sample_pdf(bins, weights, n_samples, det=False):
            # This implementation is from NeRF
            # Get pdf
            weights = weights + 1e-5  # prevent nans
            pdf = weights / torch.sum(weights, -1, keepdim=True)
            cdf = torch.cumsum(pdf, -1)
            cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
            # Take uniform samples
            if det:
                u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples)
                u = u.expand(list(cdf.shape[:-1]) + [n_samples])
            else:
                u = torch.rand(list(cdf.shape[:-1]) + [n_samples])

            # Invert CDF
            u = u.contiguous()
            inds = torch.searchsorted(cdf, u, right=True)
            below = torch.max(torch.zeros_like(inds - 1), inds - 1)
            above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
            inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

            matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
            cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
            bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

            denom = (cdf_g[..., 1] - cdf_g[..., 0])
            denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
            t = (u - cdf_g[..., 0]) / denom
            samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

            return samples

        def up_sample_single_step(rays_o, rays_d, z_vals, sdf, n_importance, inv_s):
            """
            Up sampling give a fixed inv_s
            """
            batch_size, n_samples = z_vals.shape
            pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
            radius = torch.norm(pts, dim=-1, keepdim=False)
            inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)
            sdf = sdf.reshape(batch_size, n_samples)
            prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
            prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
            mid_sdf = (prev_sdf + next_sdf) * 0.5
            cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

            # ----------------------------------------------------------------------------------------------------------
            # Use min value of [ cos, prev_cos ]
            # Though it makes the sampling (not rendering) a little bit biased, this strategy can make the sampling more
            # robust when meeting situations like below:
            #
            # SDF
            # ^
            # |\          -----x----...
            # | \        /
            # |  x      x
            # |---\----/-------------> 0 level
            # |    \  /
            # |     \/
            # |
            # ----------------------------------------------------------------------------------------------------------
            prev_cos_val = torch.cat([torch.zeros([batch_size, 1]), cos_val[:, :-1]], dim=-1)
            cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
            cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
            cos_val = cos_val.clamp(-1e3, 0.0) * inside_sphere

            dist = (next_z_vals - prev_z_vals)
            prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
            next_esti_sdf = mid_sdf + cos_val * dist * 0.5
            prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
            next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
            alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
            weights = alpha * torch.cumprod(
                torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]

            z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()
            return z_samples


        batch_size, n_samples = z_vals.shape
        with torch.no_grad():
            pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
            # sdf = self.sdf_network.sdf_with_encoding(pts.reshape(-1, 3), embed_fn).reshape(batch_size, n_samples)
            sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, n_samples)

            for i in range(up_sample_steps):
                new_z_vals = up_sample_single_step(rays_o,
                                            rays_d,
                                            z_vals,
                                            sdf,
                                            n_importance // up_sample_steps,
                                            64 * 2**i)
                z_vals, sdf = self.cat_z_vals(rays_o,
                                                rays_d,
                                                z_vals,
                                                new_z_vals,
                                                sdf,
                                                embed_fn,
                                                last=(i + 1 == up_sample_steps))

        return z_vals
