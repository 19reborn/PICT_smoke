# merge to videos
import cv2
import os
from tqdm import tqdm
from PIL import Image





# base_dir = '/cluster/project/tang/yiming/project/pinf_clean/log/scalar/1208_v1/volumeout_375001/'
# base_dir = '/cluster/project/tang/yiming/project/pinf_clean/log/scalar/0115_v1_no_reg/volumeout_260001'
# base_dir = '/cluster/project/tang/yiming/project/pinf_clean/log/scalar/0115_v1_no_reg/volumeout_415001'
# base_dir = '/cluster/project/tang/yiming/project/pinf_clean/log/scalar/0115_v1_no_reg/volumeout_445001'
base_dir = '/cluster/project/tang/yiming/project/pinf_clean/log/scalar/0115_v1_no_reg/volumeout_600001'
# base_dir = '/cluster/project/tang/yiming/project/pinf_clean/log/scalar/0116_v4_larger_ghost_loss/volumeout_600001'
# base_dir = '/cluster/project/tang/yiming/project/pinf_clean/log/scalar/1219_ablation_v1_single_layer_less_mapping_constraints/volumeout_600001'
# base_dir = '/cluster/project/tang/yiming/project/pinf_clean/log/scalar/1212_ablation_v3_no_coarse_density_fixed/volumeout_600001'



def write_videos(files, video):
    import cv2
    from tqdm import tqdm
    for i in tqdm(range(len(files))):
        frame = cv2.imread(files[i])
        video.write(frame)
    return video

import glob
path = base_dir + '/d_*.jpg'

den_files = glob.glob(path)
print(den_files)
print('Number of density images: ' + str(len(den_files)))

path = base_dir + '/lagrangian_d_*.jpg'
lag_den_files = glob.glob(path)

path = base_dir + '/v_*.jpg'
vel_files = glob.glob(path)

path = base_dir + '/vortv_*.jpg'
vort_files = glob.glob(path)

den_files.sort()
vel_files.sort()
vort_files.sort()



# create video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_dir = base_dir + '/videos/'
masked_output_dir = base_dir + '/masked_images/'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(masked_output_dir, exist_ok=True)



img = Image.open(den_files[0])
width, height = img.size


den_video = cv2.VideoWriter(output_dir + 'density.mp4', fourcc, 20, (width, height))
# lag_den_video = cv2.VideoWriter(output_dir + 'lagrangian_density.mp4', fourcc, 20, (width, height))
vel_video = cv2.VideoWriter(output_dir + 'velocity.mp4', fourcc, 20, (width, height))
vort_video = cv2.VideoWriter(output_dir + 'vorticity.mp4', fourcc, 20, (width, height))

masked_vel_video = cv2.VideoWriter(output_dir + 'masked_velocity.mp4', fourcc, 20, (width, height))
masked_vort_video = cv2.VideoWriter(output_dir + 'masked_vorticity.mp4', fourcc, 20, (width, height))

mask_thresh = 0.15 * 255
# mask_thresh = 0.075 * 255
# mask_thresh = 0.05 * 255

for i in range(len(den_files)):
    print('Processing ' + str(i) + '...')
    # get size
    img = Image.open(den_files[i])
    width, height = img.size
    
    # read den
    den_frame = cv2.imread(den_files[i])
    vel_frame = cv2.imread(vel_files[i])
    vort_frame = cv2.imread(vort_files[i])

    # read lag den
    # lag_den_frame = cv2.imread(lag_den_files[i])

    den_mask = den_frame < mask_thresh

    masked_vel_frame = vel_frame.copy()
    masked_vort_frame = vort_frame.copy()
    masked_vel_frame[den_mask] = vel_frame[den_mask] * 0.35
    masked_vort_frame[den_mask] = vort_frame[den_mask] * 0.35

    # masked_vel_frame[~den_mask] = vel_frame[~den_mask] * 0.65
    # masked_vort_frame[~den_mask] = vort_frame[~den_mask] * 0.65




    # masked_vel_frame[den_mask] = 0
    # masked_vort_frame[den_mask] = 0

    masked_vel_frame = masked_vel_frame.reshape((height, width, 3))
    masked_vort_frame = masked_vort_frame.reshape((height, width, 3))

    # write masked frames
    cv2.imwrite(masked_output_dir + 'masked_vel_' + str(i) + '.png', masked_vel_frame)
    cv2.imwrite(masked_output_dir + 'masked_vort_' + str(i) + '.png', masked_vort_frame)

    # write frames
    den_video.write(den_frame)
    # lag_den_video.write(lag_den_frame)
    vel_video.write(vel_frame)
    vort_video.write(vort_frame)
    masked_vort_video.write(masked_vort_frame)
    masked_vel_video.write(masked_vel_frame)

den_video.release()
# lag_den_video.release()
vel_video.release()
vort_video.release()
masked_vel_video.release()
masked_vort_video.release()

print('Done!')



