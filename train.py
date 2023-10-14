import os, sys
import imageio
import numpy as np
import torch
import torch.nn.functional as F

from tqdm import trange

from src.dataset.load_dryice import load_dryice_data
from src.dataset.load_pinf import load_pinf_frame_data

from src.network.hybrid_model import create_model

from src.renderer.occupancy_grid import init_occ_grid, update_occ_grid, update_static_occ_grid
from src.renderer.render_ray import render, render_path, prepare_rays

from src.utils.args import config_parser
from src.utils.training_utils import set_rand_seed, save_log
from src.utils.coord_utils import BBox_Tool, Voxel_Tool, jacobian3D
from src.utils.loss_utils import get_rendering_loss, get_velocity_loss, fade_in_weight, to8b
from src.utils.visualize_utils import den_scalar2rgb, vel2hsv, vel_uv2hsv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(args):
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
    resX = 64 # complexity O(N^3)
    resY = int(resX*float(voxel_scale[1])/voxel_scale[0]+0.5)
    resZ = int(resX*float(voxel_scale[2])/voxel_scale[0]+0.5)
    voxel_writer = Voxel_Tool(voxel_tran,voxel_tran_inv,voxel_scale,resZ,resY,resX,middleView='mid3', hybrid_neus='hybrid_neus' in args.net_model)

    # training_voxel: to sample data for for velocity NSE training
    # training_voxel should have a larger resolution than voxel_writer
    # note that training voxel is also used for visualization in testing
    min_ratio = float(64+4*2)/min(voxel_scale[0],voxel_scale[1],voxel_scale[2])
    minX = int(min_ratio*voxel_scale[0]+0.5)
    trainX = max(args.vol_output_W,minX) # a minimal resolution of 64^3
    trainY = int(trainX*float(voxel_scale[1])/voxel_scale[0]+0.5)
    trainZ = int(trainX*float(voxel_scale[2])/voxel_scale[0]+0.5)
    training_voxel = Voxel_Tool(voxel_tran,voxel_tran_inv,voxel_scale,trainZ,trainY,trainX,middleView='mid3', hybrid_neus='hybrid_neus' in args.net_model)
    training_pts = torch.reshape(training_voxel.pts, (-1,3)) 

    ## spatial alignment from wolrd coord to simulation coord
    train_reso_scale = torch.Tensor([256*t_info[-1],256*t_info[-1],256*t_info[-1]])

  
    # start = start + 1

    testimgdir = os.path.join(basedir, expname, "imgs_"+logdir)
    os.makedirs(testimgdir, exist_ok=True)
    # some loss terms 
    

    # init_occ_grid(args, model, poses = poses[i_train], intrinsics = torch.tensor(Ks)[i_train], given_mask=None if not args.use_mask else masks[i_train])
    init_occ_grid(args, model, poses = poses[i_train], intrinsics = torch.tensor(Ks)[i_train], given_mask=None)


    ## debug
    # voxel_den_list = voxel_writer.get_voxel_density_list(model, 0.5, args.chunk, 
    #         middle_slice=False)[::-1]
        
       
    # voxel_img = []
    # for voxel in voxel_den_list:
    #     voxel = voxel.detach().cpu().numpy()
    #     if voxel.shape[-1] == 1:
    #         voxel_img.append(den_scalar2rgb(voxel, scale=None, is3D=True, logv=False, mix=True))
    #     else:
    #         voxel_img.append(vel_uv2hsv(voxel, scale=300, is3D=True, logv=False))
    # voxel_img = np.concatenate(voxel_img, axis=0) # 128,64*3,3
    # imageio.imwrite( os.path.join(testimgdir, 'vox_{:06d}.png'.format(0)), voxel_img)
    


    trainVGG = False
    trainVel = False
    trainVel_using_rendering_samples = False
    trainImg = False

    total_loss_fading = 1.0
    
    if not model.single_scene and global_step > args.uniform_sample_step:
        update_static_occ_grid(args, model, 100)

    for global_step in trange(start, N_iters + 1):
        local_step = 0
        
        training_stage = 0
        
        if global_step <= args.stage1_finish_recon:
            # smoke and obstacle reconstruction
            training_stage = 1 
            trainImg = True
            trainVel = False
            trainVel_using_rendering_samples = False

        elif global_step <= args.stage1_finish_recon + args.stage2_finish_init_lagrangian:
            # init d,g, not learn feature
            training_stage = 2
            trainImg = False
            trainVel = True
            trainVel_using_rendering_samples = False
        
        elif global_step <= args.stage1_finish_recon + args.stage2_finish_init_lagrangian + args.stage3_finish_init_feature:
            # start learn feature, add its relevant constrain
            # but still only learn from reference density and color , do not use image
            # total_loss_fading = fade_in_weight(global_step, args.stage1_finish_recon + args.stage2_finish_init_lagrangian, 10000) # 
            total_loss_fading = 1.0 # 
            training_stage = 3
            trainImg = False
            trainVel = True
            trainVel_using_rendering_samples = False

        else:
            # start learn feature, add its relevant constrain
            # but still only learn from reference density and color , do not use image
            # total_loss_fading = fade_in_weight(global_step, args.stage1_finish_recon + args.stage2_finish_init_lagrangian + args.stage3_finish_init_feature, 10000) # 
            total_loss_fading = 1.0
            training_stage = 4
            trainImg = True
            # trainVel = True
            trainVel = global_step % 5 == 0
            # trainVel_using_rendering_samples = False # todo:: use this
            trainVel_using_rendering_samples = args.train_vel_within_rendering and not (global_step // 5) % args.train_vel_uniform_sample == 0# todo:: use this

        model.iter_step = global_step
        model.update_model_type(training_stage)
    


        # if training_stage == 1:
        # model.update_fading_step(min(args.stage1_finish_recon, global_step)) # progressive training for siren smoke
        model.update_fading_step(min(args.fading_layers, global_step)) # progressive training for siren smoke
        
        if trainImg and global_step >= args.uniform_sample_step:
            update_occ_grid(args, model, global_step, update_interval = 1000, neus_early_terminated = training_stage is not 1)
              
 
        optimizer.zero_grad()
        loss = 0
        rendering_loss_dict = None
        vel_loss_dict = None
        

        # Random from one frame
        img_i = np.random.choice(i_train)
        target = images[img_i]
        target = torch.Tensor(target).to(device)
        
        if args.use_mask:
            target_mask = masks[img_i]
            target_mask = torch.Tensor(target_mask).to(device)
        else:
            target_mask = None

        pose = poses[img_i, :3,:4]
        time_locate = time_steps[img_i].to(device) 
          
        if trainImg:


            # time1 = time.time()
            # Cast intrinsics to right types
            H, W, focal = hwfs[img_i]
            H, W = int(H), int(W)
            focal = float(focal)
            hwf = [H, W, focal]

            _cam_info = None

            if args.dataset_type == 'dryice':
                _cam_id = cam_info_others["cam_ids"][img_i]
                _cam_info = cam_info_others["cam_%d"%_cam_id]
                focals = _cam_info["focal"]
                principles = _cam_info["princpt"]
                K = np.array([
                    [focals[0], 0, principles[0]],
                    [0, focals[1], principles[1]],
                    [0, 0, 1]
                ])
            else:
                K = np.array([
                    [focal, 0, 0.5*W],
                    [0, focal, 0.5*H],
                    [0, 0, 1]
                ])

            # batch_rays: training rays
            # target_s: target image
            # dw: get a cropped img (dw,dw) to train vgg

            batch_rays, target_s, dw, target_mask, bg_color, select_coords = prepare_rays(args, H, W, K, pose, target, trainVGG, global_step, start, N_rand, target_mask, _cam_info)

            if args.dataset_type == 'dryice':
                pass
            else:
                bg_color = bg_color + args.white_bkgd

            rgb, disp, acc, extras = render(H, W, K, model, N_samples = args.N_samples, chunk=args.training_ray_chunk, rays=batch_rays, netchunk=args.netchunk,
                                        time_step=time_locate,
                                        near = near,
                                        far = far,
                                        bkgd_color=bg_color,
                                        # cuda_ray = trainImg and global_step >= args.uniform_sample_step,
                                        cuda_ray = global_step >= args.uniform_sample_step,
                                        perturb = args.perturb
                                        )
            


            if "num_points" in extras and extras["num_points"] == 0:
                print(f"no points in the ray, skip iteration {global_step}")
                torch.cuda.empty_cache()
                local_step += 1
                continue
                

            rendering_loss, rendering_loss_dict = get_rendering_loss(args, model, rgb, acc, target_s, bg_color, extras, time_locate, global_step, target_mask)
            loss += rendering_loss



        if trainVel:
            if trainVel_using_rendering_samples:

    
                smoke_samples_xyz = extras['samples_xyz_dynamic']

                static_samples_xyz = extras['samples_xyz_static']

                samples_xyz = torch.cat([static_samples_xyz,smoke_samples_xyz], dim = 0)

                max_samples = 32**3
                if samples_xyz.shape[0] > max_samples:
                    print("[DEBUG]: train vel samples_xyz.shape[0] > max_samples", samples_xyz.shape[0], max_samples)
                samples_xyz = samples_xyz[:max_samples] ## todo:: random choose
                samples_xyzt = torch.cat([samples_xyz, time_locate * torch.ones_like(samples_xyz[..., :1])], dim=-1) # [N, 4]
                training_samples = samples_xyzt
                
            else:
                train_random = np.random.choice(trainZ*trainY*trainX, 32*32*32)
                training_samples = training_pts[train_random]

                training_samples = training_samples.view(-1,3)
                training_t = torch.ones([training_samples.shape[0], 1])*time_locate
                training_samples = torch.cat([training_samples,training_t], dim=-1)

            vel_loss, vel_loss_dict = get_velocity_loss(args, model, training_samples, training_stage, global_step = global_step)

            loss += vel_loss

        loss = loss * total_loss_fading         

        loss.backward()
        ## grad clip
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
    


        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate


        if args.adaptive_num_rays and args.cuda_ray == True and global_step >= args.uniform_sample_step:
            samples_per_ray = extras["num_points"] / (extras["num_rays"] + 1e-6)
            num_rays = extras["num_rays"]
            cur_batch_size = num_rays * samples_per_ray
            N_rand = int(round((args.target_batch_size / cur_batch_size) * N_rand))

        # Rest is logging
        if global_step%args.i_weights==0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(global_step))
            save_dic = {
                'global_step': global_step,
                'static_model_state_dict': model.static_model.state_dict() if not model.single_scene else None,
                'dynamic_model_lagrangian_state_dict': model.dynamic_model_lagrangian.state_dict(),
                'dynamic_model_siren_state_dict': model.dynamic_model_siren.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
     
            torch.save(save_dic, path)
            print('Saved checkpoints at', path)

        if global_step%args.i_print==0:
            
            
            print(f"[TRAIN] Training stage: {training_stage} Iter: {global_step} Loss: {loss.item()}")
            print(f"CUDA memory allocated: {torch.cuda.max_memory_allocated() / 1024.0 / 1024.0 / 1024.0} GB\n")
            writer.add_scalar('Loss/loss', loss.item(), global_step)
            
            if rendering_loss_dict is not None:
                
             
                img_loss = rendering_loss_dict['img_loss']
                psnr = rendering_loss_dict['psnr']
                eikonal_loss = rendering_loss_dict['eikonal_loss']
                curvature_loss = rendering_loss_dict['curvature_loss']
                smoke_inside_sdf_loss = rendering_loss_dict['smoke_inside_sdf_loss']
                ghost_loss = rendering_loss_dict['ghost_loss']
                
                print("img_loss: ", img_loss.item())
                writer.add_scalar('Loss/img_loss', img_loss.item(), global_step)
        
                print("PSNR: ", psnr.item())
                writer.add_scalar('Statistics/PSNR', psnr.item(), global_step)
                    
                if smoke_inside_sdf_loss is not None:
                    print("smoke_inside_sdf_loss: ", smoke_inside_sdf_loss.item())
                    writer.add_scalar('Loss/smoke_inside_sdf_loss', smoke_inside_sdf_loss.item(), global_step)
            
                if eikonal_loss is not None:
                    print("eikonal_loss: ", eikonal_loss.item())
                    writer.add_scalar('Loss/eikonal_loss', eikonal_loss.item(), global_step)

                if curvature_loss is not None:
                    print("curvature_loss: ", curvature_loss.item())
                    writer.add_scalar('Loss/curvature_loss', curvature_loss.item(), global_step)
                    
                if ghost_loss is not None:
                    print("ghost_loss: ", ghost_loss.item())
                    writer.add_scalar('Loss/ghost_loss', ghost_loss.item(), global_step)

                if "num_points" in extras:
                    samples_per_ray = extras["num_points"] / (extras["num_rays"] + 1e-6)
                    print("samples_per_ray: ", samples_per_ray)
                    writer.add_scalar('Statistics/samples_per_ray', samples_per_ray, global_step)

                if "num_points_static" in extras:
                    samples_per_ray_static = extras["num_points_static"] / (extras["num_rays"] + 1e-6)
                    print("samples_per_ray_static: ", samples_per_ray_static)
                    writer.add_scalar('Statistics/samples_per_ray_static', samples_per_ray_static, global_step)

                if "num_points_dynamic" in extras:
                    num_points_dynamic = extras["num_points_dynamic"] / (extras["num_rays"] + 1e-6)
                    print("num_points_dynamic: ", num_points_dynamic)
                    writer.add_scalar('Statistics/num_points_dynamic', num_points_dynamic, global_step)

                if args.adaptive_num_rays:
                    writer.add_scalar('Statistics/cur_batch_size', cur_batch_size, global_step)
                    writer.add_scalar('Statistics/batch_size_ratio', cur_batch_size/args.target_batch_size, global_step)
                    writer.add_scalar('Statistics/num_rays', N_rand, global_step)


                if not model.single_scene:
                    with torch.no_grad():
                        inv_s = model.get_deviation()         # Single parameter
                        print("s_val: ",  1.0 / inv_s.item())
                        writer.add_scalar('Statistics/s_val', 1.0 / inv_s.item(), global_step)



   
            if trainVel:
           
                print("vel_loss: ", vel_loss.item())
                writer.add_scalar('Loss/vel_loss', vel_loss.item(), global_step)
            
                nseloss_fine = vel_loss_dict['nseloss_fine']
                nse_errors = vel_loss_dict['nse_errors']
                if nseloss_fine is not None:
                    print(" ".join(["nse(e1-e6):"]+[str(ei.item()) for ei in nse_errors]))
                    print("NSE loss sum = ", nseloss_fine.item(), "* w_nse(%0.4f)"%(args.nseW))
                    writer.add_scalar('Loss/NSE_loss', nseloss_fine.item(), global_step)

                    
                    
                    
                boundary_loss = vel_loss_dict['boundary_loss'] if 'boundary_loss' in vel_loss_dict.keys() else None
                inside_loss = vel_loss_dict['inside_loss'] if 'inside_loss' in vel_loss_dict.keys() else None

                if boundary_loss is not None:
                    print("boundary_loss = ", boundary_loss.item())
                    writer.add_scalar('Loss/boundary_loss', boundary_loss.item(), global_step)
                    
                if inside_loss is not None:
                    print("inside_loss = ", inside_loss.item())
                    writer.add_scalar('Loss/inside_loss', inside_loss.item(), global_step)


                density_reference_loss = vel_loss_dict['density_reference_loss'] if 'density_reference_loss' in vel_loss_dict.keys() else None
                color_reference_loss = vel_loss_dict['color_reference_loss'] if 'color_reference_loss' in vel_loss_dict.keys() else None
                
                if density_reference_loss is not None:
                    print("density_reference_loss = ", density_reference_loss.item())
                    writer.add_scalar('Loss/density_reference_loss', density_reference_loss.item(), global_step)
                    
                if color_reference_loss is not None:
                    print("color_reference_loss = ", color_reference_loss.item())
                    writer.add_scalar('Loss/color_reference_loss', color_reference_loss.item(), global_step)

                

                cycle_loss = vel_loss_dict['feature_cycle_loss'] if "feature_cycle_loss" in vel_loss_dict else None
                cross_cycle_loss = vel_loss_dict['feature_cross_cycle_loss'] if "feature_cross_cycle_loss" in vel_loss_dict else None
                density_mapping_loss = vel_loss_dict['density_mapping_loss'] if "density_mapping_loss" in vel_loss_dict else None

                if cycle_loss is not None:
                    print("cycle_loss = ", cycle_loss.item())
                    writer.add_scalar('Loss/feature_cycle_loss', cycle_loss.item(), global_step)
                    
                if cross_cycle_loss is not None:
                    print("cross_cycle_loss = ", cross_cycle_loss.item())
                    writer.add_scalar('Loss/feature_cross_cycle_loss', cross_cycle_loss.item(), global_step)

                if density_mapping_loss is not None:
                    print("density_mapping_loss = ", density_mapping_loss.item())
                    writer.add_scalar('Loss/density_mapping_loss', density_mapping_loss.item(), global_step)


        if (global_step) % args.i_img==0:
          
                voxel_den_list = voxel_writer.get_voxel_density_list(model, 0.5, args.chunk, 
                    middle_slice=False)[::-1]
               
                if trainVel:
                    voxel_den_list.append(
                        voxel_writer.get_voxel_velocity(model, t_info[-1]*float(args.vol_output_W)/resX, 0.5, middle_slice=True)
                    )
                    
                voxel_img = []
                for voxel in voxel_den_list:
                    voxel = voxel.detach().cpu().numpy()
                    if voxel.shape[-1] == 1:
                        voxel_img.append(den_scalar2rgb(voxel, scale=None, is3D=True, logv=False, mix=True))
                    else:
                        voxel_img.append(vel_uv2hsv(voxel, scale=300, is3D=True, logv=False))
                voxel_img = np.concatenate(voxel_img, axis=0) # 128,64*3,3
                imageio.imwrite( os.path.join(testimgdir, 'vox_{:06d}.png'.format(global_step)), voxel_img)
            

        if global_step % args.i_video==0 and local_step is not 0:
            model.eval()
            if trainImg:
                # Turn on testing mode
                hwf = hwfs[0]
                hwf = [int(hwf[0]), int(hwf[1]), float(hwf[2])]
                # the path rendering can be very slow.
                rgbs, disps = render_path(model, render_poses, hwf, Ks[0], args.test_chunk, near, far, netchunk=args.netchunk, cuda_ray = trainImg and global_step >= args.uniform_sample_step, render_steps=render_timesteps, bkgd_color=test_bkg_color)
                print('Done, saving', rgbs.shape, disps.shape)
                # moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, global_step))
                moviebase = os.path.join(basedir, expname, 'spiral_{:06d}_'.format(global_step))
                imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
                # imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)
                

            if trainVel:
                v_deltaT = 0.025
                # with torch.no_grad():
                vel_rgbs = []
                for _t in range(int(1.0/v_deltaT)):
                    # middle_slice, True: only sample middle slices for visualization, very fast, but cannot save as npz
                    #               False: sample whole volume, can be saved as npz, but very slow
                    voxel_vel = training_voxel.get_voxel_velocity(model, t_info[-1], _t*v_deltaT, middle_slice=True)
                    voxel_vel = voxel_vel.view([-1]+list(voxel_vel.shape))
                    _, voxel_vort = jacobian3D(voxel_vel)
                    _vel = vel_uv2hsv(np.squeeze(voxel_vel.detach().cpu().numpy()), scale=300, is3D=True, logv=False)
                    _vort = vel_uv2hsv(np.squeeze(voxel_vort.detach().cpu().numpy()), scale=1500, is3D=True, logv=False)
                    vel_rgbs.append(np.concatenate([_vel, _vort], axis=0))
                # moviebase = os.path.join(basedir, expname, '{}_volume_{:06d}_'.format(expname, global_step))
                moviebase = os.path.join(basedir, expname, 'volume_{:06d}_'.format(global_step))
                imageio.mimwrite(moviebase + 'velrgb.mp4', np.stack(vel_rgbs,axis=0).astype(np.uint8), fps=30, quality=8)
            model.train()

        if global_step % args.i_testset==0 and global_step > 0:
            model.eval()
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(global_step))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            hwf = hwfs[i_test[0]]
            hwf = [int(hwf[0]), int(hwf[1]), float(hwf[2])]
            render_path(model, torch.Tensor(poses[i_test]).to(device), hwf, Ks[i_test[0]], args.test_chunk, near, far, netchunk = args.netchunk, cuda_ray = trainImg and global_step >= args.uniform_sample_step, gt_imgs=images[i_test], savedir=testsavedir, render_steps=time_steps[i_test], bkgd_color=test_bkg_color)
            print('Saved test set')
            model.train()
    
        sys.stdout.flush()
        torch.cuda.empty_cache()
        local_step += 1


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')    
    
    parser = config_parser()
    args = parser.parse_args()
    set_rand_seed(args.fix_seed)

    bkg_flag = args.white_bkgd
    args.white_bkgd = np.ones([3], dtype=np.float32) if bkg_flag else None

    train(args) # call train in run_nerf
    