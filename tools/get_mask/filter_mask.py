import os
import glob
import cv2

input_dir = '/home/wym/workspace/my_pinerf/tools/get_mask/output/test_mask'
output_dir = '/home/wym/workspace/my_pinerf/tools/get_mask/output/scalar_mask/view_0'
os.makedirs(output_dir, exist_ok=True)

all_frames = os.listdir(input_dir)

start, end = 95, 120
choose_mask_id = 0

for frame_idx, frame in enumerate(all_frames):
    if frame_idx < start or frame_idx >= end:
        continue
    all_mask_in_this_frame = glob.glob(os.path.join(input_dir, frame, '*.png'))
    all_mask_in_this_frame.sort()
    chosen_mask = all_mask_in_this_frame[choose_mask_id]
   
    ## copy mask to output_dir
    mask = cv2.imread(chosen_mask)
    cv2.imwrite(os.path.join(output_dir, f'{frame_idx}.png'), mask)
    print(f'frame_idx: {frame_idx}, mask_id: {choose_mask_id}')
