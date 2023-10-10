import os
import glob
import cv2

input_dir = '/home/wym/workspace/my_pinerf/tools/get_mask/output/test_mask'
output_dir = '/home/wym/workspace/my_pinerf/tools/get_mask/output/scalar_mask/view_0'
os.makedirs(output_dir, exist_ok=True)

all_frames = os.listdir(input_dir)

start, end = 0, 120
first_choose_mask_id = 1
last_img = None

import imageio
mask_list = []

for frame_idx, frame in enumerate(all_frames):
    if frame_idx < start or frame_idx >= end:
        continue
    all_mask_in_this_frame = glob.glob(os.path.join(input_dir, frame, '*.png'))
    all_mask_in_this_frame.sort()
    if frame_idx == start:
        choose_mask_id = first_choose_mask_id
    else:
        for i in range(len(all_mask_in_this_frame)):
            ## find the most similar img
            img = cv2.imread(all_mask_in_this_frame[i])
            if img.shape != last_img.shape:
                continue
            diff = cv2.absdiff(img, last_img)
            diff = diff.mean()
            if diff < 2:
                choose_mask_id = i
                break
    chosen_mask = all_mask_in_this_frame[choose_mask_id]
    last_img = cv2.imread(chosen_mask)
   
    ## copy mask to output_dir
    mask = cv2.imread(chosen_mask)
    cv2.imwrite(os.path.join(output_dir, f'{frame_idx:06}.png'), mask)
    mask_list.append(mask)
    print(f'frame_idx: {frame_idx}, mask_id: {choose_mask_id}')

imageio.mimsave(os.path.join(output_dir, 'mask.gif'), mask_list, duration=0.1)
