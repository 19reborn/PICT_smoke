import os
import torch
import numpy as np
import imageio 
import json
import torch.nn.functional as F
import cv2

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius, rotZ=True, wx=0.0, wy=0.0, wz=0.0):
    # spherical, rotZ=True: theta rotate around Z; rotZ=False: theta rotate around Y
    # wx,wy,wz, additional translation, normally the center coord.
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    if rotZ: # swap yz, and keep right-hand
        c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w

    ct = torch.Tensor([
        [1,0,0,wx],
        [0,1,0,wy],
        [0,0,1,wz],
        [0,0,0,1]]).float()
    c2w = ct @ c2w
    
    return c2w



def visualize_poses(poses, size=0.1):
    # poses: [B, 4, 4]
    import trimesh

    axes = trimesh.creation.axis(axis_length=4)
    sphere = trimesh.creation.icosphere(radius=1)
    objects = [axes, sphere]

    for pose in poses:
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]

        # segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c], [c, d], [d, a]])
        # segs = trimesh.load_path(segs)
        # objects.append(segs)

        segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c], [c, d], [d, a]])

        # Create a line object for each segment
        lines = [trimesh.creation.cylinder(radius=0.01, segment=seg) for seg in segs]
        objects.extend(lines)

    scene = trimesh.Scene(objects)
    # trimesh.Scene(objects).show()
    scene.export('camera_pose_original.ply')
    exit(1)



def load_pinf_frame_data(args, basedir, half_res='normal', testskip=1, train_skip=1):
    # frame data
    all_imgs = []
    all_msks = []
    all_poses = []
    all_hwf = []
    all_time_steps = []
    counts = [0]
    merge_counts = [0]
    t_info = [0.0,0.0,0.0,0.0]

    # render params
    near, far, radius, phi, rotZ, r_center = 0.0, 1.0, 0.5, 20, False, np.float32([0.0]*3)

    # scene data
    voxel_tran, voxel_scale, bkg_color = None, None, None

    scene_scale = args.scene_scale

    with open(os.path.join(basedir, 'info.json'), 'r') as fp:
        # read render settings
        meta = json.load(fp)
        near = float(meta['near'])
        far = float(meta['far'])
        radius = (near + far) * 0.5
        phi = float(meta['phi'])
        rotZ = (meta['rot'] == 'Z')
        r_center = np.float32(meta['render_center'])
        bkg_color = np.float32(meta['frame_bkg_color'])

        # read scene data
        voxel_tran = np.float32(meta['voxel_matrix'])
        voxel_tran = np.stack([voxel_tran[:,2],voxel_tran[:,1],voxel_tran[:,0],voxel_tran[:,3]],axis=1) # swap_zx
        voxel_scale = np.broadcast_to(meta['voxel_scale'],[3])
        
        ## apply manual scaling
        voxel_scale = voxel_scale.copy() * scene_scale
        voxel_tran[:3,3] *= scene_scale
        near *= scene_scale
        far *= scene_scale
        radius *= scene_scale
        r_center *= scene_scale

        # read video frames
        # all videos should be synchronized, having the same frame_rate and frame_num
        for s in 'train,val,test'.split(','):
            if s=='train' or testskip==0:
                skip = train_skip
            else:
                skip = testskip

            video_list = meta[s+'_videos'] if (s+'_videos') in meta else meta['train_videos'][0:1]

            for train_video in video_list:
                imgs = []
                msks = []
                poses = []
                time_steps = []
                H, W, Focal = 0, 0, 0

                f_name = os.path.join(basedir, train_video['file_name'])
                reader = imageio.get_reader(f_name, "ffmpeg")
                if s=='train':
                    frame_num = train_video['frame_num']
                    delta_t = 1.0/train_video['frame_num']
                    video_name = train_video['file_name']

                    

                for frame_i in range(0, train_video['frame_num'], skip):
                    reader.set_image_index(frame_i)
                    frame = reader.get_next_data()

                    if H == 0:
                        H, W = frame.shape[:2]
                        camera_angle_x = float(train_video['camera_angle_x'])
                        Focal = .5 * W / np.tan(.5 * camera_angle_x)

                    cur_timestep = frame_i
                    time_steps.append([frame_i*delta_t])
                    poses.append(np.array(
                        train_video['transform_matrix_list'][frame_i] 
                        if 'transform_matrix_list' in train_video else train_video['transform_matrix']
                    ))
                    
                    imgs.append(frame)


                reader.close()

                imgs = (np.float32(imgs) / 255.)
                poses = np.array(poses).astype(np.float32)
                time_steps = np.array(time_steps).astype(np.float32)

                if half_res !='normal':
                    if half_res =='half': # errors if H or W is not dividable by 2
                        H = H//2
                        W = W//2
                        Focal = Focal/2.
                    elif half_res=='quater': # errors if H or W is not dividable by 4
                        H = H//4
                        W = W//4
                        Focal = Focal/4.
                    elif half_res=='double':
                        H = H*2
                        W = W*2
                        focal = focal*2.

                    imgs_half_res = np.zeros((imgs.shape[0], H, W, imgs.shape[-1]))
                    for i, img in enumerate(imgs):
                        imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
                    imgs = imgs_half_res


                counts.append(counts[-1] + imgs.shape[0])
                all_imgs.append(imgs)
                all_poses.append(poses)
                all_time_steps.append(time_steps)
                all_hwf.append(np.float32([[H,W,Focal]]*imgs.shape[0]))
                all_msks.append(msks)
            merge_counts.append(counts[-1])
        
    t_info = np.float32([0.0, 1.0, 0.5, delta_t]) # min t, max t, mean t, delta_t
    i_split = [np.arange(merge_counts[i], merge_counts[i+1]) for i in range(3)]
    imgs = np.concatenate(all_imgs, 0) # n, H, W
    poses = np.concatenate(all_poses, 0) # n, 4, 4
    time_steps = np.concatenate(all_time_steps, 0) # n, 1
    hwfs = np.concatenate(all_hwf, 0) # n, 3

    ## apply manul scaling
    poses[:,:3,3] *= scene_scale

    # visualize_poses(poses=poses)

    # set render settings:
    render_focal = float(hwfs[0][-1])
    # sp_n = 40 # an even number!
    # sp_poses = [
    #     pose_spherical(angle, phi, radius, rotZ, r_center[0], r_center[1], r_center[2]) 
    #     for angle in np.linspace(-180,180,sp_n+1)[:-1]
    # ]
    # sp_n = 80 # an even number!
    # sp_poses = [
    #     pose_spherical(angle, phi, radius, rotZ, r_center[0], r_center[1], r_center[2]) 
    #     for angle in np.linspace(90,180,sp_n+1)[:-1] # for game scene
    # ]
    sp_n = 120 # an even number! # scalar opposite
    sp_poses = [
        # pose_spherical(-117, phi, radius, rotZ, r_center[0], r_center[1], r_center[2]) 
        pose_spherical(angle, phi, radius, rotZ, r_center[0], r_center[1], r_center[2]) 
        for angle in np.linspace(-180,180,sp_n+1)[:-1] # for scalar scene
    ]
    sp_steps = np.linspace(t_info[0],t_info[1], num=sp_n) # [ float(ct) ]*sp_n, for testing a frozen t
    render_poses = torch.stack(sp_poses,0) # [sp_poses[36]]*sp_n, for testing a single pose
    render_timesteps =  np.reshape(sp_steps,(-1,1))


    extras = {}
   
    return imgs, msks, poses, time_steps, hwfs, render_poses, render_timesteps, i_split, t_info, voxel_tran, voxel_scale, bkg_color, near, far, extras





