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

def resize_flow(flow, H_new, W_new):
    H_old, W_old = flow.shape[0:2]
    flow_resized = cv2.resize(flow, (W_new, H_new), interpolation=cv2.INTER_LINEAR)
    flow_resized[:, :, 0] *= H_new / H_old
    flow_resized[:, :, 1] *= W_new / W_old
    return flow_resized

def get_grid(H, W, num_img, flows_b, flow_masks_b):

    # |--------------------|  |--------------------|
    # |       j            |  |       v            |
    # |   i   *            |  |   u   *            |
    # |                    |  |                    |
    # |--------------------|  |--------------------|

    # i, j = np.meshgrid(np.arange(W, dtype=np.float32),
    #                    np.arange(H, dtype=np.float32), indexing='xy')

    # grid = np.empty((0, H, W, 5), np.float32)
    # for idx in range(num_img):
    #     grid = np.concatenate((grid, np.stack([i,
    #                                            j,
    #                                            flows_b[idx, :, :, 0],
    #                                            flows_b[idx, :, :, 1],
    #                                            flow_masks_b[idx, :, :]], -1)[None, ...]))
    
    # 创建网格坐标
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                    np.arange(H, dtype=np.float32), indexing='xy')


    # 扩展维度，方便拼接
    i = i[np.newaxis, ..., np.newaxis]
    j = j[np.newaxis, ..., np.newaxis]
    i = i.repeat(num_img, axis=0)
    j = j.repeat(num_img, axis=0)
    flows = np.stack([flows_b[..., 0],
                    flows_b[..., 1],
                    flow_masks_b], axis=-1)

    # 合并网格坐标和 flows
    grid = np.concatenate((i, j, flows), axis=-1)

    ## checked
    # uv_grid = grid[0,:,:,:2]
    # uv_grid_gt = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W)),dim=-1)

    # (torch.tensor(uv_grid) - uv_grid_gt).sum()

    return grid




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
    # 提取点云数据
    # point_cloud = scene.export(file_type="ply", return_vertex_colors=False)

    # 保存点云数据为PLY文件
    # trimesh.points.export_point_cloud("camera_pose.ply", point_cloud.vertices)



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
    all_flows_b = []
    all_flow_masks_b = []   
    use_optical_flow = args.FlowW > 0

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
                flows_b = []
                flow_masks_b = []   
                H, W, Focal = 0, 0, 0

                f_name = os.path.join(basedir, train_video['file_name'])
                reader = imageio.get_reader(f_name, "ffmpeg")
                if s=='train':
                    frame_num = train_video['frame_num']
                    delta_t = 1.0/train_video['frame_num']
                    video_name = train_video['file_name']
                    # extract idx from video name in format of 'train{idx}.mp4'
                    camera_idx = int(video_name[5:-4]) ## todo: use re

                    flow_dir = os.path.join(basedir, 'flow', f'view{camera_idx}')  
                    

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

                    if s == 'train' and use_optical_flow:
                        # add flow
                        if frame_i == 0:
                            bwd_flow, bwd_mask = np.zeros((H, W, 2)), np.zeros((H, W))
                        else:
                            bwd_flow_path = os.path.join(flow_dir, '%03d_bwd.npz'%frame_i)
                            bwd_data = np.load(bwd_flow_path)
                            bwd_flow, bwd_mask = bwd_data['flow'], bwd_data['mask']
                            bwd_flow = resize_flow(bwd_flow, H, W)
                            bwd_mask = np.float32(bwd_mask)
                            bwd_mask = cv2.resize(bwd_mask, (W, H),
                                                interpolation=cv2.INTER_NEAREST)
                            
                        flows_b.append(bwd_flow)
                        flow_masks_b.append(bwd_mask)  

                if s == 'train' and use_optical_flow:
                    flows_b = np.stack(flows_b, -1)
                    flow_masks_b = np.stack(flow_masks_b, -1)     

                    flows_b = np.moveaxis(flows_b, -1, 0).astype(np.float32)
                    flow_masks_b = np.moveaxis(flow_masks_b, -1, 0).astype(np.float32)       

                    all_flow_masks_b.append(flow_masks_b)
                    all_flows_b.append(flows_b)


                reader.close()
                if args.use_mask:
                    msk_name = f_name[:-4] + '_mask.mp4'
                    no_reader = False
                    try:
                        reader = imageio.get_reader(msk_name, "ffmpeg")
                    except:
                        no_reader = True
                        print(f'No mask found for {f_name}, use zeros 1s instead')
                        # msks = np.zeros((len(imgs), H, W, 1))
                    for frame_i in range(0, train_video['frame_num'], skip):
                        if no_reader:
                            frame = np.zeros((H, W, 3))
                        else:
                            reader.set_image_index(frame_i)
                            frame = reader.get_next_data()

                        # import cv2
                        # cv2.imwrite(f'debug/read_mask/{frame_i}.png', frame)
                        msks.append(frame)
                    reader.close()
                    # exit(1)
                    msks = np.array(msks).astype(np.float32) / 255.

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

                    if args.use_mask:
                        msks_half_res = np.zeros((msks.shape[0], H, W, msks.shape[-1]))
                        for i, msk in enumerate(msks):
                            msks_half_res[i] = cv2.resize(msk, (W, H), interpolation=cv2.INTER_AREA)
                        msks = msks_half_res

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
    if args.use_mask:
        msks = np.concatenate(all_msks, 0)
    else:
        msks = None
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
        pose_spherical(-117, phi, radius, rotZ, r_center[0], r_center[1], r_center[2]) 
        for angle in np.linspace(-180,180,sp_n+1)[:-1] # for game scene
    ]
    sp_steps = np.linspace(t_info[0],t_info[1], num=sp_n) # [ float(ct) ]*sp_n, for testing a frozen t
    render_poses = torch.stack(sp_poses,0) # [sp_poses[36]]*sp_n, for testing a single pose
    render_timesteps =  np.reshape(sp_steps,(-1,1))


    extras = {}
    if use_optical_flow and len(all_flow_masks_b) != 0:
        flow_masks_b = np.concatenate(all_flow_masks_b, 0)
        flows_b = np.concatenate(all_flows_b, 0)

        # flows_b = np.moveaxis(flows_b, -1, 0).astype(np.float32)
        # flow_masks_b = np.moveaxis(flow_masks_b, -1, 0).astype(np.float32)       

        extras['flow_b'] = flows_b
        extras['flow_mask_b'] = flow_masks_b
        flow_grids = get_grid(int(H), int(W), flows_b.shape[0], flows_b, flow_masks_b) # [N, H, W, 5]

        extras['flow_grids'] = flow_grids
        extras['frame_num'] = frame_num


    return imgs, msks, poses, time_steps, hwfs, render_poses, render_timesteps, i_split, t_info, voxel_tran, voxel_scale, bkg_color, near, far, extras


# if __name__=='__main__':
#     # allres = load_pinf_frame_data("./data/ScalarReal", "quater", testskip=20)
#     # allres = load_pinf_frame_data("./data/Sphere", "normal", testskip=20)
#     allres = load_pinf_frame_data("./data/Game", "half", testskip=20)
#     for a in allres:
#         if isinstance(a, np.ndarray):
#             print(a.shape)
#         else:
#             print(a)



