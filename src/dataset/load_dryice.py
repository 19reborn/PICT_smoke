import os
import torch
import numpy as np
import imageio 
import json
import cv2 as cv

max_timestep = 240
scale_factor = 1.2/0.4909 #
worldscale = 1/256.0
xyzscale = 1.0
zf = 1.0
yf = 1.0
xf = 1.0

trans_t = lambda t : torch.tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1],
], dtype=torch.float32)

rot_phi = lambda phi : torch.tensor([
    [1, 0, 0, 0],
    [0, torch.cos(phi), -torch.sin(phi), 0],
    [0, torch.sin(phi), torch.cos(phi), 0],
    [0, 0, 0, 1],
], dtype=torch.float32)

rot_theta = lambda th : torch.tensor([
    [torch.cos(th), 0, -torch.sin(th), 0],
    [0, 1, 0, 0],
    [torch.sin(th), 0, torch.cos(th), 0],
    [0, 0, 0, 1],
], dtype=torch.float32)

dryice_center = np.array([0.0, 0.0, 0.0])

def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w

    ct = torch.tensor([
        [1, 0, 0, dryice_center[0]],
        [0, 1, 0, dryice_center[1]],
        [0, 0, 1, dryice_center[2]],
        [0, 0, 0, 1],
    ], dtype=torch.float32)

    c2w = ct @ c2w
    return c2w

def readCamMatrixTest(idx, period, R, W, H):
    t = - float(idx) / period
    t_start = -0.25
    x = np.cos(t * 2.0 * np.pi + t_start * np.pi) * R
    y = 0.5
    z = np.sin(t * 2.0 * np.pi + t_start * np.pi) * R
    campos = np.array([x, y, z], dtype=np.float32)

    lookat = np.array([0., 0., 0.], dtype=np.float32)
    up = np.array([0., -1., 0.], dtype=np.float32)
    forward = lookat - campos
    forward /= np.linalg.norm(forward)
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    up /= np.linalg.norm(up)

    xyzf = np.float32([xf, yf, zf])
    pos = xyzf * campos/xyzscale
    fwd = -1*forward

    meta_pose = np.float32([xyzf * right, xyzf * up, xyzf * fwd, pos]) # 4x3
    meta_pose = np.concatenate([meta_pose, np.float32([0.,0.,0.,1.])[:,None]], axis=-1).T # 4x4

    fx = 2000.0 
    fy = 2000.0 
    camera_angle_x = float(np.arctan(.5 * W /fx) + np.arctan(.5 * H /fy))
    
    return meta_pose, camera_angle_x

def load_dryice_smokeTrans():
    smoke_scale = np.float32([0.4909, 0.73635, 0.4909])
    smoke_tran = [ 
         [1.0, 0.0, 0.0, dryice_center[0]],
         [0.0, 1.0, 0.0, dryice_center[1]],
         [0.0, 0.0, 1.0, dryice_center[2]],
         [0.0, 0.0, 0.0, 1.0],
    ]

    smoke_tran = np.float32(smoke_tran)
    smoke_tran[:3, :3] *= scale_factor
    offset = 0.5 * smoke_scale * scale_factor
    offset = np.concatenate([offset, [0.]], axis=0)
    smokepos = smoke_tran[:, 3] - offset
    
    smoke_matrix = np.stack(
        [zf * smoke_tran[:, 2], yf * smoke_tran[:, 1], xf * smoke_tran[:, 0], smokepos], axis=1)
    
    return np.float32(smoke_matrix), smoke_scale

def load_krt(path):
    """Load KRT file containing intrinsic and extrinsic parameters."""
    cameras = {}

    with open(path, "r") as f:
        while True:
            name = f.readline()
            if name == "":
                break

            intrin = [[float(x) for x in f.readline().split()] for i in range(3)]
            dist = [float(x) for x in f.readline().split()]
            extrin = [[float(x) for x in f.readline().split()] for i in range(3)]
            f.readline()

            cameras[name[:-1]] = {
                    "intrin": np.array(intrin),
                    "dist": np.array(dist),
                    "extrin": np.array(extrin)}

    return cameras

def load_dryice_data(args, basedir, half_res='normal', testskip=1, train_skip=1, testid = "400018", skipTrain=False):

    # frameN = -1 ## todo:: hard coded, support load all training image
    frameN = 50 ## todo:: hard coded, support load all training image
    data_dir = basedir
    all_cams = load_krt(data_dir+"KRT")

    others = {}
    others["cam_ids"]  = []
    all_imgs, all_poses, all_time_steps= [],[],[]
    t_info = [0.0,0.0,0.0,0.0]
    H, W = 0, 0
    subN = len(all_cams)

    transf = [
        [0.990026, 0.00956428, 0.140555, 5.72638],
        [0.02397, -0.994582, -0.10116, 22.948],
        [0.138825, 0.10352, -0.984891, 1002.17],
        [0.0, 0.0, 0.0, 1],
    ]
    transf = np.float32(transf)
    transf[:3, :3] *= worldscale
    xyzf = np.float32([xf,yf,zf])
    
    train_imgs = 0

    for s_idx, s in enumerate(all_cams):
        # print(s)

        if skipTrain and s_idx > 4 and s_idx < (subN-4) and s!=testid: ## skipTrain == False
            continue

        imgs,time_steps = [],[]
        skip = 1
        subdir = data_dir + "cam"+s
        cam_obj_matrix = all_cams[s]["extrin"]
        cam_intrin = all_cams[s]['intrin'] 
        campos = (-np.dot(cam_obj_matrix[:3, :3].T, cam_obj_matrix[:3, 3])).astype(np.float32)
        camrot = (cam_obj_matrix[:3, :3]).astype(np.float32)
        focal = (np.diag(cam_intrin[:2, :2]) / 4.).astype(np.float32)
        princpt = (cam_intrin[:2, 2] / 4.).astype(np.float32)

        camrot = np.dot(transf[:3, :3].T, camrot.T).T
        campos = np.dot(transf[:3, :3].T, campos - transf[:3, 3])
        # somehow have to invert fwd
        right, up, fwd = [xyzf * camrot[i] for i in range(3)]
        pos = xyzf * campos/xyzscale
        fwd = -1*fwd
        # dist = pos / fwd
        # print(dist)
        meta_pose = np.float32([right, up, fwd, pos]) # 4x3
        meta_pose = np.concatenate([meta_pose, np.float32([0.,0.,0.,1.])[:,None]], axis=-1).T # 4x4
        # print(meta_pose)

        framelist = os.listdir(subdir)
        framelist = [_ for _ in framelist if _.startswith("image")] 
        framelist = sorted(framelist) # first sort according to abc, then sort according to 123
        framelist.sort(key=lambda f: int(''.join(list(filter(str.isdigit, f))) or -1))
        if frameN > 0 and frameN < 300:
            mid = 156+12
            framelist = framelist[mid-frameN//2:mid-frameN//2+frameN] # too many frames, take 24 for now

        frameI = 0
        for frame in framelist[::skip]:
            fname = os.path.join(subdir, frame)
            # print(frameI, fname)
            imgs.append(imageio.imread(fname))
            if H == 0:
                H, W = imgs[0].shape[:2]
            if imgs[-1].shape[0] != H or imgs[-1].shape[1] != W:
                print("!! resize ", frame)
                imgs[-1] = cv.resize(imgs[-1], (W,H))

            time_steps.append(float(frameI*skip) / max_timestep)
            frameI += 1
            
        # exit()
        imgs = (np.float32(imgs) / 255.) # keep all 4 channels (RGBA)
        imgs = imgs[:,::-1,...] # flip y
        time_steps = np.array(time_steps).astype(np.float32)
        if s_idx == 0:
            mint, maxt = time_steps.min(), time_steps.max()
            delta_t = (maxt - mint) / frameI
            t_info = [mint, maxt, time_steps.mean(), delta_t]
        camIDlist = np.array([s_idx] * frameI).astype(np.int32)
        poses = np.array([meta_pose] * frameI).astype(np.float32)
        train_imgs += frameI
        all_imgs.append(imgs)
        all_poses.append(poses)
        all_time_steps.append(time_steps)
        others["cam_ids"].append(camIDlist)

        frame_min = np.amin(imgs, axis=0)
        # frame_mean = np.mean(imgs, axis=0)
        # frame_max = np.amax(imgs, axis=0)

        # imageio.imwrite( os.path.join(subdir, "fr_amin.jpg"), frame_min)
        # imageio.imwrite( os.path.join(subdir, "fr_mean.jpg"), frame_mean)
        # imageio.imwrite( os.path.join(subdir, "fr_amax.jpg"), frame_max)
        
        bkframe = os.path.join(subdir, "bg.jpg")
        if not os.path.exists(bkframe): 
            print("failed to load ", bkframe)
            bkframe = np.zeros([H,W,3], dtype=np.float32)
        else:
            bkframe = imageio.imread(bkframe)
            bkframe = (np.float32(bkframe) / 255.) # keep all 4 channels (RGBA)
            bkframe = bkframe[::-1,...]

        if half_res=="half":
            bkframe = cv.resize(bkframe, (W//2,H//2))
            frame_min = cv.resize(frame_min, (W//2,H//2))
        elif half_res=="double":
            bkframe = cv.resize(bkframe, (W*2, H*2)) 
            frame_min = cv.resize(bkframe, (W*2, H*2)) 

        # 2500 = .5 * 667 / np.tan(.5 * camera_angle_x)
        
        flip_princpt = np.float32([princpt[0], H-princpt[1]])
        others["cam_%d"%s_idx] = {
            "bk_img": bkframe,
            "min_img": frame_min,
            "bk_img_fname": os.path.join(subdir, "bg.jpg"),
            "focal": focal,
            "princpt":flip_princpt, # princpt,
        }

        if s == testid:
            skip = max(1, testskip)
            testimgs = imgs[::skip]
            testtstep = time_steps[::skip]
            testposes = poses[::skip]
            testcam_id= camIDlist[::skip]
    # exit()

    counts = [train_imgs]
    # val,  subN Frames, one from each cam
    subN0= len(all_imgs)
    imgs = [all_imgs[i][frameN//2-subN//2+i]        for i in range(subN0)]
    tstep = [all_time_steps[i][frameN//2-subN//2+i] for i in range(subN0)]
    poses = [all_poses[i][frameN//2-subN//2+i]      for i in range(subN0)]
    cam_id=[others["cam_ids"][i][frameN//2-subN//2+i] for i in range(subN0)]
    
    all_imgs.append(np.float32(imgs))
    all_poses.append(np.float32(poses))
    all_time_steps.append(np.float32(tstep))
    others["cam_ids"].append(np.array(cam_id).astype(np.int32) )

    counts += [len(imgs) + counts[-1]]
    # test
    all_imgs.append(np.float32(testimgs))
    all_poses.append(np.float32(testposes))
    all_time_steps.append(np.float32(testtstep))
    others["cam_ids"].append( np.array(testcam_id).astype(np.int32) )

    counts += [testimgs.shape[0] + counts[-1]]
    merge_counts = [0] + counts
    i_split = [np.arange(merge_counts[i], merge_counts[i+1]) for i in range(3)]
    
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    time_steps = np.concatenate(all_time_steps, 0)
    others["cam_ids"] = np.concatenate(others["cam_ids"], 0)
    
    avgfocal = np.float32([others["cam_%d"%s_idx]["focal"] for s_idx in range(subN) if "cam_%d"%s_idx in others ]).mean()


    # rotate pos:
    # sp_n = 40 # an even number!
    sp_n = 128 // max(1, testskip) # an even number!
    sp_dist = 3.0
    # sp_steps = [ float(ct) / max_timestep for ct in range(90-sp_n//2,90+sp_n//2) ]
    sp_steps = np.linspace(t_info[0], t_info[1], num=sp_n) # test full time
    render_timesteps = torch.tensor(sp_steps, dtype=torch.float32)
    # render_timesteps = torch.tensor([sp_steps[36]]*40, dtype=torch.float32) # just for testing

    if False:
        phi = 0.0 # -20.0
        sp_poses = [pose_spherical(angle, phi, sp_dist) for angle in np.linspace(-180, 180, sp_n+1)[:-1]]
        render_poses = torch.stack(sp_poses, 0)
        # render_poses = torch.stack([sp_poses[36]]*40, 0)  # just for testing
        # sp_steps = np.linspace(t_info[0], t_info[1], num=sp_n)
    else:
        sp_poses = [readCamMatrixTest(_idx, sp_n, sp_dist, W, H)[0] for _idx in range(sp_n)]
        render_poses = torch.tensor(sp_poses)
        # sp_focal = 1000.0

    if half_res=="half":
        imgs = torch.tensor(cv.resize(imgs, (H//2, W//2)), dtype=torch.float32)
        H = H//2
        W = W//2
        xyFactor = torch.tensor([float(W//2)/W, float(H//2)/H], dtype=torch.float32)
        avgfocal *= xyFactor.mean()
        for s_idx in range(subN):
            if "cam_%d"%s_idx in others:
                others["cam_%d"%s_idx]["focal"] *= xyFactor
                others["cam_%d"%s_idx]["princpt"] *= xyFactor


    elif half_res=="double":
        imgs = torch.tensor(cv.resize(imgs, (H*2, W*2)), dtype=torch.float32)
        H = H*2
        W = W*2
        avgfocal *= 2
        for s_idx in range(subN):
            if "cam_%d"%s_idx in others:
                others["cam_%d"%s_idx]["focal"] *= 2
                others["cam_%d"%s_idx]["princpt"] *= 2
        

    msks = None

    testT = int(t_info[2] * max_timestep) # the mean frame
    voxel_tran, voxel_scale = load_dryice_smokeTrans()

    # hard-coded
    near = 800 # args.blenderNear
    far = 1400 # args.blenderFar
    denscale = 0.1 ## todo:: use this to scale the density


    hwf = [[H, W, avgfocal]]
    hwfs = np.concatenate([hwf]*imgs.shape[0], 0)

    print('Fixed hwf', hwf)
    bkg_color = None ## since bk is different for each cam
    return imgs, msks, poses, time_steps, hwfs, render_poses, render_timesteps, i_split, t_info, voxel_tran, voxel_scale, bkg_color, near, far, others
    # return imgs, msks, poses, time_steps, hwf, render_poses, render_timesteps, i_split, t_info, voxel_tran, voxel_scale, bkg_color, near, far, others
    # return imgs, msks, poses, time_steps, render_poses, render_timesteps, [H, W, avgfocal], i_split, t_info, others


