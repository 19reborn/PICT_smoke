def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', 
                        help='input data directory')

    # training options
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--fix_seed", type=int, default=42,
                        help='the random seed.')
    parser.add_argument("--model_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--train_vel_grid_size", type=int, default=32,
                        help='the random seed.')

    ## Stage 1
    parser.add_argument("--stage1_finish_recon", type=int, default=50000, help="stage 1 total training steps" )
    parser.add_argument("--uniform_sample_step", type=int, default = 20000, help="stage 1 first uniform sample steps" )
    parser.add_argument("--smoke_recon_delay_start", type=int, default=0,
                        help='for hybrid models, the step to start learning the temporal dynamic component.')
    parser.add_argument("--smoke_recon_delay_last", type=int, default=10000,
                        help='for hybrid models, the step to start learning the temporal dynamic component.')
    parser.add_argument("--sdf_loss_delay", type=int, default=2000,
                        help='for hybrid models, the step to start learning the temporal dynamic component.')
    parser.add_argument("--fading_layers", type=int, default=-1,
                        help='for siren and hybrid models, the step to finish fading model layers one by one during training.')
    parser.add_argument("--density_distillation_delay", type=int, default=2000, help="stage 2 total training steps" )

    ## Stage 2
    parser.add_argument("--stage2_finish_init_lagrangian", type=int, default=20000, help="stage 2 total training steps" )
    parser.add_argument("--mapping_frame_range_fading_start", type=int, default=20000, help="frame_range" )
    parser.add_argument("--mapping_frame_range_fading_last", type=int, default=50000, help="frame_range" )
    parser.add_argument("--max_mapping_frame_range", type=int, default=30, help="frame_range" )
    
    ## Stage 3
    parser.add_argument("--stage3_finish_init_feature", type=int, default=20000, help="stage 2 total training steps" )
    parser.add_argument("--stage4_train_vel_interval", type=int, default=10, help="stage 2 total training steps" )
    parser.add_argument('--neus_early_terminated', action = 'store_true')
    parser.add_argument('--neus_larger_lr_decay', action = 'store_true')


    # network model
    ## lagrangian network
    parser.add_argument('--use_two_level_density', action = 'store_true')
    parser.add_argument("--lagrangian_feature_dim", type=int, default=16, 
                        help='Lagrangian feature dimension')   
    
    parser.add_argument("--feature_map_first_omega", type=int, default=30, 
                        help='Lagrangian feature dimension')   
    parser.add_argument("--position_map_first_omega", type=int, default=30, 
                        help='Lagrangian feature dimension')   
    parser.add_argument("--density_map_first_omega", type=int, default=30, 
                        help='Lagrangian feature dimension')   
    parser.add_argument("--density_activation", type=str,
                        default='identity', help='activation function for density')
    parser.add_argument("--lagrangian_density_activation", type=str,
                        default='exp', help='activation function for density')
    
    ## siren nerf    
    parser.add_argument("--siren_nerf_netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--siren_nerf_first_omega", type=int, default=30, 
                        help='layers in network')
    
    ## neus
    parser.add_argument('--use_scene_scale_before_pe', action = 'store_true')
    parser.add_argument('--neus_progressive_pe', action = 'store_true')
    parser.add_argument('--neus_progressive_pe_start', type=int, default=20000)
    parser.add_argument('--neus_progressive_pe_duration', type=int, default=10000)

    # loss hyper params, negative values means to disable the loss terms
    parser.add_argument("--vgg_strides", type=int, default=4,
                        help='vgg stride, should >= 2')
    parser.add_argument("--ghostW", type=float,
                        default=-0.0, help='weight for the ghost density regularization')
    parser.add_argument("--vggW", type=float,
                        default=-0.0, help='weight for the VGG loss')
    parser.add_argument("--ColorDivergenceW", type=float,
                        default=0.0, help='weight for the VGG loss')
    parser.add_argument("--overlayW", type=float,
                        default=-0.0, help='weight for the overlay regularization')
    parser.add_argument("--d2vW", type=float,
                        default=-0.0, help='weight for the d2v loss')
    parser.add_argument("--nseW", type=float,
                        default=0.001, help='velocity model, training weight for the physical equations')
    parser.add_argument("--ekW", type=float,
                        default=0.0, help='weight for the Ekinoal loss')
    parser.add_argument("--boundaryW", type=float,
                        default=0.0, help='weight for the Boardary constrain loss')
    parser.add_argument("--hardW", type=float,
                        default=0.0, help='weight for the Boardary constrain loss')
    parser.add_argument("--MinusDensityW", type=float,
                        default=0.0, help='weight for the Boardary constrain loss')
    parser.add_argument("--SmokeInsideSDFW", type=float,
                        default=0.0, help='weight for the Boardary constrain loss')
    parser.add_argument("--SmokeAlphaReguW", type=float,
                        default=0.05, help='weight for the Boardary constrain loss')
    parser.add_argument("--CurvatureW", type=float,
                        default=0.00, help='weight for the Boardary constrain loss')
    parser.add_argument("--FlowW", type=float,
                        default=0.00, help='weight for the Boardary constrain loss')
    parser.add_argument("--flow_debug", action='store_true')
    parser.add_argument("--train_vel_within_rendering", action='store_true')
    parser.add_argument("--train_vel_uniform_sample", type=int, default = 2)
    parser.add_argument("--inside_sdf", type=float, default = 0.0)
    parser.add_argument("--vel_regulization_weight", type=float,
                        default=1, help='weight for the Boardary constrain loss')
    ## Lagrangian Feature loss
    parser.add_argument("--self_cycle_loss_weight", type=float, default = 1.0)
    parser.add_argument("--cross_cycle_loss_weight", type=float, default = 0.1)
    
    ## Lagrangian mapping loss
    parser.add_argument("--density_mapping_loss_weight", type=float, default = 0.05)
    parser.add_argument("--velocity_mapping_loss_weight", type=float, default = 0.01)
    parser.add_argument("--color_mapping_loss_weight", type=float, default = 0.0)


    parser.add_argument("--net_model", type=str, default='nerf',
                        help='which model to use, nerf, siren, hybrid..')
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--training_ray_chunk", type=int, default=1024*32, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--test_chunk", type=int, default=1024*4, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
                        
    parser.add_argument("--tempo_delay", type=int, default=0,
                        help='for hybrid models, the step to start learning the temporal dynamic component.')
    parser.add_argument("--vgg_delay", type=int, default=0,
                        help='for hybrid models, the step to start learning the temporal dynamic component.')
    parser.add_argument("--vel_delay", type=int, default=10000,
                        help='for siren and hybrid models, the step to start learning the velocity.')
    parser.add_argument("--boundary_delay", type=int, default=10000,
                        help='for siren and hybrid models, the step to start learning the velocity.')
    parser.add_argument("--N_iter", type=int, default=200000,
                        help='for siren and hybrid models, the step to start learning the velocity.')  
    parser.add_argument("--train_warp", default=False, action='store_true',
                        help='train radiance model with velocity warpping')
    parser.add_argument("--adaptive_num_rays", action='store_true')
    parser.add_argument("--target_batch_size", type=int, default=2**17)
    parser.add_argument("--use_mask", action='store_true')
    parser.add_argument("--use_random_bg", action='store_true')
    parser.add_argument("--use_mask_loss", action='store_true')
    parser.add_argument("--mask_loss_weight", type=float, default = 0.1)
    parser.add_argument("--mask_sample", action='store_true')

    # scene options
    parser.add_argument("--scene_scale", type=float, default = 1.0)
    parser.add_argument("--bbox_min", type=str,
                        default='0.0,0.0,0.0', help='use a boundingbox, the minXYZ')
    parser.add_argument("--bbox_max", type=str,
                        default='1.0,1.0,1.0', help='use a boundingbox, the maxXYZ')
    parser.add_argument("--occ_grid_bound_static", type=float, default = 1.0)
    parser.add_argument("--occ_grid_bound_dynamic", type=float, default = 1.0)
    




    # task params
    parser.add_argument("--test_mode", action='store_true', 
                        help='test mode')
    parser.add_argument("--output_voxel", action='store_true', 
                        help='do not optimize, reload weights and output volumetric density and velocity')
    parser.add_argument("--voxel_video", action='store_true', 
                        help='do not optimize, reload weights and output volumetric density and velocity')
    parser.add_argument("--visualize_mapping", action='store_true', 
                        help='do not optimize, reload weights and output volumetric density and velocity')
    parser.add_argument("--visualize_feature", action='store_true', 
                        help='do not optimize, reload weights and output volumetric density and velocity')
    parser.add_argument("--evaluate_mapping", action='store_true', 
                        help='do not optimize, reload weights and output volumetric density and velocity')
    parser.add_argument("--vol_output_W", type=int, default=256, 
                        help='In output mode: the output resolution along x; In training mode: the sampling resolution for training')
    parser.add_argument("--full_vol_output", action='store_true', 
                        help='do not optimize, reload weights and output volumetric density and velocity')
    
    
    
    parser.add_argument("--vol_output_only", action='store_true', 
                        help='do not optimize, reload weights and output volumetric density and velocity')
    parser.add_argument("--masked_vol_otuput", action='store_true', 
                        help='do not optimize, reload weights and output volumetric density and velocity')
    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--mesh_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_eval", action='store_true', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_train", action='store_true', 
                        help='render the training set instead of render_poses path')
    
    parser.add_argument("--extract_occ_grid", action='store_true', 
                        help='render the training set instead of render_poses path')
    parser.add_argument("--preload_gt_den_vol", action='store_true', 
                        help='render the training set instead of render_poses path')
    parser.add_argument("--vis_feature", action='store_true', 
                        help='render the training set instead of render_poses path')
   
    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--i_embed_neus", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires_neus", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_smoke", type=int, default=6, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views_neus", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    
    # cuda_ray
    parser.add_argument("--cuda_ray", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--time_size", type=int, default=150, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    parser.add_argument("--density_thresh", type=float, default=1.0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    parser.add_argument("--density_thresh_static", type=float, default=30.0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    parser.add_argument("--use_triplane_occ_grid", action='store_true', 
                        help='sampling linearly in disparity rather than depth')

    # NeuS rendering options
    parser.add_argument("--up_sample_steps", type=int, default=4, 
                        help='number of up samples per ray')
    parser.add_argument("--anneal_end", type=int, default=50000, 
                        help='number of up samples per ray')
    
    # Network options
    parser.add_argument('--use_neus2_network', action = 'store_true')
    parser.add_argument('--swish_network', action = 'store_true')
    parser.add_argument('--disentangled_density_color', action = 'store_true')
    parser.add_argument('--density_init_zero', action = 'store_true')


    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 
    parser.add_argument("--neus_early_termination", type=int, default=-1,
                        help='number of steps to train on central crops')
    parser.add_argument("--lagrangian_warmup", type=int, default=10000,
                        help='number of steps to train on central crops')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')
    parser.add_argument("--trainskip", type=int, default=1, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek', 
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a given bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", type=str, default='normal', 
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=400, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=2000, 
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=25000, 
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000, 
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=200000, 
                        help='frequency of render_poses video saving')
    parser.add_argument("--i_visualize",   type=int, default=10000, 
                        help='frequency of render_poses video saving')
                        
    return parser

