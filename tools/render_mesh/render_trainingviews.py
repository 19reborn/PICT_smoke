import os
# os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
import pyrender
import os
from pyrender import PerspectiveCamera, IntrinsicsCamera,\
                     DirectionalLight, SpotLight, PointLight,\
                     MetallicRoughnessMaterial,\
                     Primitive, Mesh, Node, Scene,\
                     Viewer, OffscreenRenderer, RenderFlags
from pyrender.constants import RenderFlags

import trimesh
import trimesh.creation
import numpy as np
import cv2 as cv
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp


def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K/K[2,2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3,3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


def inter_pose(cameras, cam_id, cam_id_new, ratio):
    intrinsics, pose_0 = load_K_Rt_from_P(None, (cameras['world_mat_{}'.format(cam_id)] @ cameras['scale_mat_0'.format(cam_id)])[:3, :4])
    intrinsics, pose_1 = load_K_Rt_from_P(None, (cameras['world_mat_{}'.format(cam_id_new)] @ cameras['scale_mat_0'.format(cam_id)])[:3, :4])
    pose_0 = np.linalg.inv(pose_0)
    pose_1 = np.linalg.inv(pose_1)
    rot_0 = pose_0[:3, :3]
    rot_1 = pose_1[:3, :3]
    rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
    key_times = [0, 1]
    key_rots = [rot_0, rot_1]
    slerp = Slerp(key_times, rots)
    rot = slerp(ratio)
    pose = np.diag([1.0, 1.0, 1.0, 1.0])
    pose = pose.astype(np.float32)
    pose[:3, :3] = rot.as_matrix()
    # pose[:3, 3] = (pose_0 * (1.0 - 0.5) + pose_1 * 0.5)[:3, 3]
    pose[:3, 3] = (pose_0 * (1.0 - ratio) + pose_1 * ratio)[:3, 3]
    pose = np.linalg.inv(pose @ np.linalg.inv(cameras['scale_mat_0'.format(cam_id)]))
    return pose

import json

def parse_data_transfrom(data_transform_dir, camera_view):

    with open(data_transform_dir) as f:
        data_transform = json.load(f)
        
    # data_transform = data_transform['test_videos'][0]
    data_transform = data_transform['train_videos'][1]

    pose = np.array(data_transform['transform_matrix']).astype('float32')

   
    H, W = data_transform['camera_hw']

    # H, W = data_transform['H'], data_transform['W']

    camera_angle_x = float(data_transform['camera_angle_x'])
    Focal = .5 * W / np.tan(.5 * camera_angle_x) * 1.0
    
    fl_x, fl_y = Focal, Focal
    cx, cy = W / 2., H / 2.

    ixt = np.array([[fl_x, 0, cx , 0],
                        [0, fl_y, cy, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])

    return ixt, pose, H, W




def visualize(all_data_transform, cam_id, cam_id_new=-1, ratio=-1):
    model_trimesh.visual.vertex_colors = np.ones_like(model_trimesh.vertices) * (np.array([255, 255, 255]) / 255.0)
    model_center = model_trimesh.vertices.mean(axis=0)


    model = Mesh.from_trimesh(model_trimesh)

    direc_l = DirectionalLight(color=np.ones(3), intensity=3.0)
    direc_l_2 = DirectionalLight(color=np.ones(3), intensity=1.0)

    # intrinsics, pose = load_K_Rt_from_P(None, (cameras['world_mat_{}'.format(cam_id)])[:3, :4])

    intrinsics, pose, H, W = parse_data_transfrom(all_data_transform, cam_id)

    # pose = pose @ np.diag([1.0, -1.0, -1.0, 1.0])

    if cam_id_new >= 0:
         pose = inter_pose(cameras, cam_id, cam_id_new, ratio)

    print(intrinsics)

    # fake_H = intrinsics[1, 2] * 2.0
    # fake_W = intrinsics[0, 2] * 2.0
    fake_H = H
    fake_W = W

    # yfov = np.arctan((fake_H / 2.0) / intrinsics[1, 1] * ext) * 2.0
    # print(yfov)
    print(pose)

    # cam = PerspectiveCamera(yfov=yfov)
    cam = IntrinsicsCamera(fx=intrinsics[0, 0], fy=intrinsics[1, 1], cx=intrinsics[0, 2], cy=intrinsics[1, 2], znear=0.1, zfar=1000.0)
    cam_pose = pose
    # cam_pose = np.linalg.inv(cam_pose)
    # cam_pose = np.linalg.inv(cam_pose)
    cam_pose = cam_pose @ np.diag([1.0, 1.0, 1.0, 1.0])

    scene = Scene(ambient_light=np.array([0.02, 0.02, 0.02, 1.0]), bg_color=np.array([1.0, 1.0, 1.0]))
    node = Node(mesh=model, translation=np.array([0.0, 0.0, 0.0]))
    scene.add_node(node)

    # light pose
    model_pos = model_center[:]
    cam_pos = pose[:3, 3]
    offset = (model_pos - cam_pos)
    dis = np.sqrt((offset**2).sum())
    pro = np.linalg.inv(pose[:3, :3])
    # pro = pose[:3, :3]
    cam_x_vector, cam_up_vector, cam_z_vector = pro[0], pro[1], pro[2]

    light_pos = model_pos - cam_up_vector * dis
    light_z_vector = -cam_up_vector
    light_x_vector = cam_x_vector
    light_y_vector = cam_z_vector
    light_pro = np.stack([light_x_vector, light_y_vector, light_z_vector], axis=0)
    light_pose = np.diag([1.0, 1.0, 1.0, 1.0])
    light_pose[:3, :3] = np.linalg.inv(light_pro)
    light_pose[:3, 3] = light_pos

    direc_l_node = scene.add(direc_l, pose=light_pose[:])
    direc_l_node_2 = scene.add(direc_l_2, pose=cam_pose)

    cam_node = scene.add(cam, pose=cam_pose)
    # v = Viewer(scene)

    # r = OffscreenRenderer(viewport_width=fake_W * ext, viewport_height=fake_H * ext)
    r = OffscreenRenderer(viewport_width=W, viewport_height=H)
    # color, depth = r.render(scene, flags=RenderFlags.OFFSCREEN)
    color, depth = r.render(scene)
    return color, depth, fake_H, fake_W




data_transform_dir = "/home/yiming/Documents/workspace/Project_PINF/pinf_clean/data/Game/info.json"



with open(data_transform_dir) as f:
    data_transform = json.load(f)
        
data_transform = data_transform['train_videos'][1]
frame_num = data_transform['frame_num']

# from src.dataset.load_pinf import load_pinf_frame_data
# images, masks, poses, time_steps, hwfs, render_poses, render_timesteps, i_split, t_info, voxel_tran, voxel_scale, bkg_color, near, far, data_extras = load_pinf_frame_data(args, args.datadir, args.half_res, args.testskip, args.trainskip)

model_trimesh = trimesh.load("./game_ours.obj")
out_name = "ours_training_view1"

# model_trimesh = trimesh.load("./game_pinf.obj")
# out_name = "pinf"

trimesh.repair.fix_normals(model_trimesh)

# rotate mesh in y axis for 90 degree
model_trimesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [0, 0, 1]))

model_trimesh.vertices[:, 0] = -model_trimesh.vertices[:, 0]
trimesh.repair.fix_normals(model_trimesh)
# model_trimesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [0, 0, 1]))
# model_trimesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi/4, [0, 0, 1]))

print(model_trimesh.vertices.shape)
print(model_trimesh.vertices.max(), model_trimesh.vertices.min())

for i in range(0, frame_num, 1):
    image, depth, fake_H, fake_W = visualize(data_transform_dir, cam_id=i)
    # image, depth, fake_H, fake_W = visualize()
    # cv.imwrite("test_pyrender.png", image)
    os.makedirs(f'./{out_name}/', exist_ok=True)
    cv.imwrite(f"./{out_name}/image_{i:04}.png", image)

os.system(f"ffmpeg -y -framerate 20 -i {out_name}/image_%04d.png -c:v libx264 -pix_fmt yuv420p ./{out_name}.mp4")


    