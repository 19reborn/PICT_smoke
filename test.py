import os, sys
import imageio
import numpy as np
import torch
import torch.nn.functional as F

from tqdm import trange

from src.dataset.load_dryice import load_dryice_data
from src.dataset.load_pinf import load_pinf_frame_data

from src.network.hybrid_model import create_model

from src.renderer.occupancy_grid import init_occ_grid, update_occ_grid
from src.renderer.render_ray import render_path, render_eval

from src.utils.args import config_parser
from src.utils.training_utils import set_rand_seed, save_log
from src.utils.coord_utils import BBox_Tool, Voxel_Tool, jacobian3D, get_voxel_pts
from src.utils.loss_utils import get_rendering_loss, get_velocity_loss, fade_in_weight, to8b
from src.utils.visualize_utils import draw_mapping, draw_mapping_3d, draw_mapping_3d_animation, vel_uv2hsv, den_scalar2rgb
from src.utils.evaluate_utils import evaluate_mapping

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def visualize_mapping(args, model, testsavedir, voxel_writer, t_info):

    model.eval()
    print('vis_mapping ONLY')


    t_list = list(np.arange(t_info[0],t_info[1],t_info[-1]))
    frame_N = len(t_list)
    

    frame_N = args.time_size
    delta_T = 1.0 / frame_N
    
    # if args.full_vol_output:
    #     frame_list = range(0,frame_N, 1)
    #     testsavedir += "_full_frame"
    # else:
    frame_list = range(0,frame_N, 1)
    # frame_list = range(0,120, 1)
    # frame_list = range(0,150, 1)
        
    
    # frame_list = range(30,frame_N, 1)
    
    os.makedirs(testsavedir, exist_ok=True)

    
    # change_feature_interval = 50
    # sample_pts = 32
    change_feature_interval = 50
    sample_pts = 128
    
    mapping_xyz = voxel_writer.vis_mapping_voxel(frame_list, t_list, model, change_feature_interval = change_feature_interval, sample_pts = sample_pts)

    
    draw_mapping_3d_animation(os.path.join(testsavedir, f'vis_map_3d_animation_interval{change_feature_interval}.gif'), mapping_xyz.permute(1,0,2).cpu().numpy())
    
    draw_mapping_3d(os.path.join(testsavedir, f'vis_map_3d_interval{change_feature_interval}.png'), mapping_xyz.permute(1,0,2).cpu().numpy())
    
    # draw grid_xyz on image
    grid_yz = mapping_xyz[..., [1,2]].permute(1,0,2)
    draw_mapping(os.path.join(testsavedir, f'vis_map_yz_interval{change_feature_interval}.png'), grid_yz.cpu().numpy())
    
    grid_xy = mapping_xyz[..., [0,1]].permute(1,0,2)
    draw_mapping(os.path.join(testsavedir, f'vis_map_xy_interval{change_feature_interval}.png'), grid_xy.cpu().numpy())
    
    grid_xz = mapping_xyz[..., [0,2]].permute(1,0,2)
    draw_mapping(os.path.join(testsavedir, f'vis_map_xz_interval{change_feature_interval}.png'), grid_xz.cpu().numpy())


    change_feature_interval = 1000
    sample_pts = 128
    
    mapping_xyz = voxel_writer.vis_mapping_voxel(frame_list, t_list, model, change_feature_interval = change_feature_interval, sample_pts = sample_pts)

    
    draw_mapping_3d_animation(os.path.join(testsavedir, f'vis_map_3d_animation_interval{change_feature_interval}.gif'), mapping_xyz.permute(1,0,2).cpu().numpy())
    
    draw_mapping_3d(os.path.join(testsavedir, f'vis_map_3d_interval{change_feature_interval}.png'), mapping_xyz.permute(1,0,2).cpu().numpy())
    
    # draw grid_xyz on image
    grid_yz = mapping_xyz[..., [1,2]].permute(1,0,2)
    draw_mapping(os.path.join(testsavedir, f'vis_map_yz_interval{change_feature_interval}.png'), grid_yz.cpu().numpy())
    
    grid_xy = mapping_xyz[..., [0,1]].permute(1,0,2)
    draw_mapping(os.path.join(testsavedir, f'vis_map_xy_interval{change_feature_interval}.png'), grid_xy.cpu().numpy())
    
    grid_xz = mapping_xyz[..., [0,2]].permute(1,0,2)
    draw_mapping(os.path.join(testsavedir, f'vis_map_xz_interval{change_feature_interval}.png'), grid_xz.cpu().numpy())



    mapping_xyz = voxel_writer.vis_vel_integration(frame_list, t_list, model, sample_pts = sample_pts)
    
    draw_mapping_3d_animation(os.path.join(testsavedir, f'vis_vel_integration_3d.gif'), mapping_xyz.permute(1,0,2).cpu().numpy())
    
    draw_mapping_3d(os.path.join(testsavedir, f'vis_vel_integration_3d.png'), mapping_xyz.permute(1,0,2).cpu().numpy())
    
    # draw grid_xyz on image
    grid_yz = mapping_xyz[..., [1,2]].permute(1,0,2)
    draw_mapping(os.path.join(testsavedir, f'vis_vel_integration_yz.png'), grid_yz.cpu().numpy())
    
    grid_xy = mapping_xyz[..., [0,1]].permute(1,0,2)
    draw_mapping(os.path.join(testsavedir, f'vis_vel_integration_xy.png'), grid_xy.cpu().numpy())
    
    grid_xz = mapping_xyz[..., [0,2]].permute(1,0,2)
    draw_mapping(os.path.join(testsavedir, f'vis_vel_integration_xz.png'), grid_xz.cpu().numpy())



    print('Done output', testsavedir)




    # exit(0)

def visualize_feature(args, model, testsavedir, voxel_writer, t_info):

    model.eval()
    print('vis_features ONLY')
    

    t_list = list(np.arange(t_info[0],t_info[1],t_info[-1]))
    frame_N = len(t_list)
    

    frame_N = args.time_size
    delta_T = 1.0 / frame_N
    
    if args.full_vol_output:
        frame_list = range(0,frame_N, 1)
        testsavedir += "_full_frame"
    else:
        frame_list = range(frame_N//10,frame_N, 10)
        
    
    os.makedirs(testsavedir, exist_ok=True)
    
    featureX_imgs = []
    featureY_imgs = []
    featureZ_imgs = []
    
    normalized_featureX_imgs = []
    normalized_featureY_imgs = []
    normalized_featureZ_imgs = []
    
    for frame_i in frame_list:
        cur_t = t_list[frame_i]
        normalized_feature_img_x, normalized_feature_img_y, normalized_feature_img_z, feature_img_x, feature_img_y, feature_img_z = voxel_writer.vis_feature_voxel(model, testsavedir + '/time_%d'%frame_i, cur_t)
        
        featureX_imgs.append(feature_img_x)
        featureY_imgs.append(feature_img_y)
        featureZ_imgs.append(feature_img_z)
        
        normalized_featureX_imgs.append(normalized_feature_img_x)
        normalized_featureY_imgs.append(normalized_feature_img_y)
        normalized_featureZ_imgs.append(normalized_feature_img_z)
    os.makedirs(testsavedir + "/feature_videos", exist_ok=True)
    imageio.mimwrite(testsavedir + "/feature_videos/yz.mp4", np.stack(normalized_featureX_imgs,axis=0).astype(np.uint8), fps=20, quality=8)
    imageio.mimwrite(testsavedir + "/feature_videos/xz.mp4", np.stack(normalized_featureY_imgs,axis=0).astype(np.uint8), fps=20, quality=8)
    imageio.mimwrite(testsavedir + "/feature_videos/xy.mp4", np.stack(normalized_featureZ_imgs,axis=0).astype(np.uint8), fps=20, quality=8)

        
    # globally normalize
    featureX_imgs = np.stack(featureX_imgs,axis=0)
    featureY_imgs = np.stack(featureY_imgs,axis=0)
    featureZ_imgs = np.stack(featureZ_imgs,axis=0)
    
    featureX_imgs = torch.tensor(featureX_imgs)
    featureY_imgs = torch.tensor(featureY_imgs)
    featureZ_imgs = torch.tensor(featureZ_imgs)
    
    def normalize_img(feature_img):
        feature_img = feature_img
        channel_min = feature_img.reshape(-1,3).min(dim=0)[0].reshape(1, 1,1,3)
        channel_max = feature_img.reshape(-1,3).max(dim=0)[0].reshape(1, 1,1,3)
        feature_img = (feature_img - channel_min) / (channel_max - channel_min)
        
        feature_img = feature_img.cpu().numpy()
        feature_img = (feature_img * 255).astype(np.uint8)
        
        return feature_img
    
    featureX_imgs = normalize_img(featureX_imgs)
    featureY_imgs = normalize_img(featureY_imgs)
    featureZ_imgs = normalize_img(featureZ_imgs)
    
    imageio.mimwrite(testsavedir + "/feature_videos/yz_global_normalized.mp4", np.stack(featureX_imgs,axis=0).astype(np.uint8), fps=20, quality=8)
    imageio.mimwrite(testsavedir + "/feature_videos/xz_global_normalized.mp4", np.stack(featureY_imgs,axis=0).astype(np.uint8), fps=20, quality=8)
    imageio.mimwrite(testsavedir + "/feature_videos/xy_global_normalized.mp4", np.stack(featureZ_imgs,axis=0).astype(np.uint8), fps=20, quality=8)
    
    
        
    print('Done output', testsavedir)
    
    # exit(0)

def visualize_all(args, model, voxel_writer, t_info, global_step):
    args.full_vol_output = True
    print('visualize_all')
    basedir = args.basedir
    expname = args.expname
    testsavedir = os.path.join(basedir, expname, 'vis_summary_{:06d}'.format(global_step+1), 'vis_mapping')
    os.makedirs(testsavedir, exist_ok=True)
    visualize_mapping(args, model, testsavedir, voxel_writer, t_info)
    
    testsavedir = os.path.join(basedir, expname, 'vis_summary_{:06d}'.format(global_step+1), 'vis_feature')
    # os.makedirs(testsavedir, exist_ok=True)
    visualize_feature(args, model, testsavedir, voxel_writer, t_info)
    
    testsavedir = os.path.join(basedir, expname, 'vis_summary_{:06d}'.format(global_step+1), 'vis_velocity/')
    os.makedirs(testsavedir, exist_ok=True)
    output_voxel(args, model, testsavedir, voxel_writer, t_info, voxel_video = True)
    
    testsavedir = os.path.join(basedir, expname, 'vis_summary_{:06d}'.format(global_step+1), 'eval_mapping')
    os.makedirs(testsavedir, exist_ok=True)
    evaluate_mapping(args, model, testsavedir, voxel_writer, t_info=t_info)

def render_only(args, model, testsavedir, render_poses, render_timesteps, test_bkg_color, hwf, K, near, far, cuda_ray, gt_images):
    model.eval()
    os.makedirs(testsavedir, exist_ok=True)
    update_occ_grid(args, model)
    print('RENDER ONLY')

    print('test poses shape', render_poses.shape)
    if args.render_eval:
        rgbs, _ = render_eval(model, render_poses, hwf, K, args.test_chunk, near, far, netchunk = args.netchunk, cuda_ray = cuda_ray, gt_imgs=gt_images, savedir=testsavedir, render_factor=args.render_factor, render_steps=render_timesteps, bkgd_color=test_bkg_color)
        imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)
    else:
        rgbs, _ = render_path(model, render_poses, hwf, K, args.test_chunk, near, far, netchunk = args.netchunk, cuda_ray = cuda_ray, gt_imgs=gt_images, savedir=testsavedir, render_factor=args.render_factor, render_steps=render_timesteps, bkgd_color=test_bkg_color)
        imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)
    print('Done rendering', testsavedir)

def output_voxel(args, model, testsavedir, voxel_writer, t_info, voxel_video = False):
    model.eval()

    print('OUTPUT VOLUME ONLY')
    if voxel_video:
        v_deltaT = 0.025
        # with torch.no_grad():
        vel_rgbs = []
        for _t in range(int(1.0/v_deltaT)):
            frame_rgb = []
            voxel_den_list = voxel_writer.get_voxel_density_list(model, _t*v_deltaT, args.chunk, 
                    middle_slice=True)
            for voxel in voxel_den_list:
                frame_rgb.append(den_scalar2rgb(voxel.detach().cpu().numpy(), scale=None, is3D=True, logv=False, mix=False))
            # middle_slice, True: only sample middle slices for visualization, very fast, but cannot save as npz
            #               False: sample whole volume, can be saved as npz, but very slow
            voxel_vel = voxel_writer.get_voxel_velocity(model, t_info[-1], _t*v_deltaT, middle_slice=True)
            voxel_vel = voxel_vel.view([-1]+list(voxel_vel.shape))
            _, voxel_vort = jacobian3D(voxel_vel)
            # frame_rgb.append(vel_uv2hsv(np.squeeze(voxel_vel.detach().cpu().numpy()), scale=300, is3D=True, logv=False))
            # frame_rgb.append(vel_uv2hsv(np.squeeze(voxel_vort.detach().cpu().numpy()), scale=1500, is3D=True, logv=False))
            frame_rgb.append(vel_uv2hsv(np.squeeze(voxel_vel.detach().cpu().numpy()), scale=300, is3D=True, logv=False, mix=False))
            frame_rgb.append(vel_uv2hsv(np.squeeze(voxel_vort.detach().cpu().numpy()), scale=1500, is3D=True, logv=False, mix=False))
            vel_rgbs.append(np.concatenate(frame_rgb, axis=0))
            # vel_rgbs.append(np.concatenate([_vel, _vort], axis=0))
        # moviebase = os.path.join(basedir, expname, '{}_volume_{:06d}_'.format(expname, global_step))
        imageio.mimwrite(testsavedir + 'volume_video.mp4', np.stack(vel_rgbs,axis=0).astype(np.uint8), fps=30, quality=8)
    else:
        savenpz = True # need a large space
        savejpg = True 
        save_vort = True # (velocity_model is not None) and (savenpz) and (savejpg)
        # with torch.no_grad():



        t_list = list(np.arange(t_info[0],t_info[1],t_info[-1]))
        frame_N = len(t_list)
        noStatic = False
        if args.full_vol_output:
            frame_list = range(0,frame_N, 1)
            testsavedir + "_full_frame"
        else:
            frame_list = range(frame_N//10,frame_N, 10)
            
        os.makedirs(testsavedir, exist_ok=True)
        
        for frame_i in frame_list:

            print(frame_i, frame_N)
            cur_t = t_list[frame_i]
            voxel_writer.save_voxel_den_npz(model, os.path.join(testsavedir,"d_%04d.npz"%frame_i), cur_t,  chunk=args.chunk, save_npz=savenpz, save_jpg=savejpg, noStatic=noStatic)
            noStatic = True
            
            voxel_writer.save_voxel_vel_npz_with_grad(model, os.path.join(testsavedir,"v_%04d.npz"%frame_i), t_info[-1], cur_t, args.chunk, savenpz, savejpg, save_vort)
        
    print('Done output', testsavedir)

    # return

def test(args):
    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    logdir, writer = save_log(args)

    # Load data
    cam_info_others = None
    ## todo:: organize dataloader
    if args.dataset_type == "dryice":
        images, masks, poses, time_steps, hwfs, render_poses, render_timesteps, i_split, t_info, voxel_tran, voxel_scale, bkg_color, near, far, cam_info_others = load_dryice_data(args, args.datadir, args.half_res, args.testskip, args.trainskip)
        Ks = []
        for img_i in range(len(images)):
            _cam_id = cam_info_others["cam_ids"][img_i]
            _cam_info = cam_info_others["cam_%d"%_cam_id]
            focals = _cam_info["focal"]
            principles = _cam_info["princpt"]
            K = [
                [focals[0], 0, principles[0]],
                [0, focals[1], principles[1]],
                [0, 0, 1]
            ]
            Ks.append(K)
    else:
        images, masks, poses, time_steps, hwfs, render_poses, render_timesteps, i_split, t_info, voxel_tran, voxel_scale, bkg_color, near, far, data_extras = load_pinf_frame_data(args, args.datadir, args.half_res, args.testskip, args.trainskip)
        Ks = [
            [
            [hwf[-1], 0, 0.5*hwf[1]],
            [0, hwf[-1], 0.5*hwf[0]],
            [0, 0, 1]
            ] for hwf in hwfs
        ]
    voxel_tran_inv = np.linalg.inv(voxel_tran)
    print('Loaded pinf frame data', images.shape, render_poses.shape, hwfs[0], args.datadir)
    print('Loaded voxel matrix', voxel_tran, 'voxel scale',  voxel_scale)


        
    args.time_size = len(list(np.arange(t_info[0],t_info[1],t_info[-1])))

    voxel_tran_inv = torch.Tensor(voxel_tran_inv)
    voxel_tran = torch.Tensor(voxel_tran)
    voxel_scale = torch.Tensor(voxel_scale)
    i_train, i_val, i_test = i_split
    if bkg_color is not None:
        args.white_bkgd = torch.Tensor(bkg_color).to(device)
        print('Scene has background color', bkg_color, args.white_bkgd)
        
    if args.render_test or args.render_eval:
        render_poses = np.array(poses[i_test])
        render_timesteps = np.array(time_steps[i_test])
    
    if args.render_train:
        render_poses = np.array(poses[i_train])
        render_timesteps = np.array(time_steps[i_train])

    # Create Bbox model from smoke perspective
    bbox_model = None

    # this bbox in in the smoke simulation coordinate
    in_min = [float(_) for _ in args.bbox_min.split(",")]
    in_max = [float(_) for _ in args.bbox_max.split(",")]
    bbox_model = BBox_Tool(voxel_tran_inv, voxel_scale, in_min, in_max)


    # Create model
    model, optimizer, start = create_model(args = args, bbox_model = bbox_model, device=device)

    global_step = start


    

    # tempoInStep = max(0,args.tempo_delay) if "hybrid" in args.net_model else 0
    # velInStep = max(0,args.vel_delay) if args.nseW > 1e-8 else 0 # after tempoInStep
    # BoundaryInStep = max(0,args.boundary_delay) if args.boundaryW > 1e-8 else 0 # after velInStep
    
    # if args.net_model != "nerf":
    #     model_fading_update(all_models, start, tempoInStep, velInStep, "hybrid" in args.net_model)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)
    render_timesteps = torch.Tensor(render_timesteps).to(device)

    test_bkg_color = bkg_color
    # test_bkg_color = np.float32([0.0, 0.0, 0.3])
    # test_bkg_color = np.float32([1.0, 1.0, 1.0])

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching
    if (use_batching) or (N_rand is None):
        print('Not supported!')
        return

    # Prepare Loss Tools (VGG, Den2Vel)
    ###############################################
    # vggTool = VGGlossTool(device)

    # Move to GPU, except images
    poses = torch.Tensor(poses).to(device)
    time_steps = torch.Tensor(time_steps).to(device)

    N_iters = args.N_iter

    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    # Prepare Voxel Sampling Tools for Image Summary (voxel_writer), Physical Priors (training_voxel), Data Priors Represented by D2V (den_p_all)
    # voxel_writer: to sample low resolution data for for image summary 
    resX = args.vol_output_W
    resY = int(args.vol_output_W*float(voxel_scale[1])/voxel_scale[0]+0.5)
    resZ = int(args.vol_output_W*float(voxel_scale[2])/voxel_scale[0]+0.5)
    voxel_writer = Voxel_Tool(voxel_tran,voxel_tran_inv,voxel_scale,resZ,resY,resX,middleView='mid3', hybrid_neus='hybrid_neus' in args.net_model)

    model.voxel_writer = voxel_writer


    testimgdir = os.path.join(basedir, expname, "imgs_"+logdir)
    os.makedirs(testimgdir, exist_ok=True)
    # some loss terms 
    

    # init_occ_grid(args, model, poses = poses[i_train], intrinsics = torch.tensor(Ks)[i_train], given_mask=None if not args.use_mask else masks[i_train])
    init_occ_grid(args, model, poses = poses[i_train], intrinsics = torch.tensor(Ks)[i_train], given_mask=None)

    model.iter_step = global_step
    model.update_model(4, global_step)
    
    if args.mesh_only:
        print('mesh ONLY')

        N = 256

        # min_rec, max_rec = -1.2, 1.2
        # min_rec, max_rec = -2., 2
        aabb_min = bbox_model.world_bbox[0]
        aabb_max = bbox_model.world_bbox[1]
        min_rec = aabb_min.min().item()
        max_rec = aabb_max.max().item()
        # min_rec, max_rec = bbox_model.s_min.min().item(), bbox_model.s_max.max().item()
        t = np.linspace(min_rec, max_rec, N+1)

        query_pts = np.stack(np.meshgrid(t, t, t), -1).astype(np.float32)
        sh = query_pts.shape
        pts = torch.Tensor(query_pts.reshape([-1,3]))

        def reconstruct(points):
            # embed_fn = render_kwargs_test['embed_fn_neus']
                                        
            chunk = 1024*64
            # raw = np.concatenate([net_fn.static_model.sdf_network.sdf_with_encoding(points[i:i+chunk, :], embed_fn).detach().cpu().numpy() for i in range(0, points.shape[0], chunk)], 0)
            raw = np.concatenate([model.sdf_static(points[i:i+chunk, :]).detach().cpu().numpy() for i in range(0, points.shape[0], chunk)], 0)
            raw = np.reshape(raw, list(sh[:-1]) + [-1])
            # sigma = np.maximum(raw[...,-1], 0.)
            sigma = raw
            return sigma

        threshold = 0  # this is just a randomly found threshold
        
        sigma = reconstruct(pts)[...,0]
        
        import mcubes
        import trimesh
       

        vertices, triangles = mcubes.marching_cubes(sigma, threshold)
        
        testsavedir = os.path.join(basedir, expname, 'meshonly_{:06d}'.format(start+1))
        # display
        mesh = trimesh.Trimesh(vertices / N * (max_rec - min_rec) + min_rec , triangles)
        os.makedirs(f"{testsavedir}", exist_ok=True)
        mesh.export(f"{testsavedir}/static_object.obj")
        print('Done output', f"{testsavedir}/static_object.obj")

        return

    
    elif args.output_voxel:
        testsavedir = os.path.join(basedir, expname, 'volumeout_{:06d}'.format(start+1))
        output_voxel(args, model, testsavedir, voxel_writer, t_info, voxel_video = args.voxel_video)
    elif args.visualize_feature:
        testsavedir = os.path.join(basedir, expname, 'vis_feature_{:06d}'.format(start+1))
        os.makedirs(testsavedir, exist_ok=True)
        visualize_feature(args, model, testsavedir, voxel_writer, t_info)
        
    elif args.visualize_mapping:
        testsavedir = os.path.join(basedir, expname, 'vis_mapping_{:06d}'.format(start+1))
        visualize_mapping(args, model, testsavedir, voxel_writer, t_info=t_info)
    elif args.evaluate_mapping:
        testsavedir = os.path.join(basedir, expname, 'eval_mapping_{:06d}'.format(start+1))
        evaluate_mapping(args, model, testsavedir, voxel_writer, t_info=t_info)
    elif args.render_only:
        testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start+1))
        if args.render_eval:
            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('eval', start+1))
            
        
        hwf = hwfs[0]
        hwf = [int(hwf[0]), int(hwf[1]), float(hwf[2])]
        K = Ks[0]
            
        # with torch.no_grad():
        if args.render_test or args.render_eval:
            # render_test switches to test poses
            images = images[i_test]
            hwf = hwfs[i_test[0]]
            hwf = [int(hwf[0]), int(hwf[1]), float(hwf[2])] # todo:: support multi-view testset
            K = Ks[i_test[0]]
      
        elif args.render_train:
            # render_train switches to train poses
            images = images[i_train]
            hwf = hwfs[i_train[0]]
            hwf = [int(hwf[0]), int(hwf[1]), float(hwf[2])]
            K = Ks[i_train[0]]
        else:
            # Default is smoother render_poses path
            images = None
        render_only(args, model, testsavedir, render_poses, render_timesteps, test_bkg_color, hwf, K, near, far, global_step >= args.uniform_sample_step, gt_images=images)
    else:
        AssertionError("test mode not defined.")


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')    
    
    parser = config_parser()
    args = parser.parse_args()
    set_rand_seed(args.fix_seed)

    bkg_flag = args.white_bkgd
    args.white_bkgd = np.ones([3], dtype=np.float32) if bkg_flag else None
    args.test_mode = True

    test(args) # call train in run_nerf
    