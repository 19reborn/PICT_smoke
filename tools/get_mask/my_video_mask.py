import os
import glob
import cv2

import imageio

import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import json
import argparse
from typing import Any, Dict, List


def main(args: argparse.Namespace) -> None:
    ## input dir
    input_dir = args.input_dir
    all_video = glob.glob(os.path.join(input_dir, '*.mp4'))
    ## remove mask video
    all_video = [video for video in all_video if 'mask' not in video]


    ## load json to know frame num
    with open(os.path.join(input_dir, 'info.json'), 'r') as fp:
        meta = json.load(fp)
        frame_num = meta['train_videos'][0]['frame_num']
        frame_rate = meta['train_videos'][0]['frame_rate']

    ## load segmentation model
    print("Loading model...")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    _ = sam.to(device=args.device)
    output_mode = "coco_rle" if args.convert_to_rle else "binary_mask"
    amg_kwargs = get_amg_kwargs(args)
    generator = SamAutomaticMaskGenerator(sam, output_mode=output_mode, **amg_kwargs)


    ## tracking parameter
    if args.first_mask_index is not None:
        # convert str to list
        all_first_choose_mask_id = [int(i) for i in args.first_mask_index.split(',')]
    else:
        all_first_choose_mask_id = [1 for i in range(len(all_video))]



    ## for each video
    for i in range(len(all_video)):
        if all_first_choose_mask_id[i] == -1:
            continue
        video = all_video[i]
        f_name = os.path.join(input_dir, video)
        reader = imageio.get_reader(f_name, "ffmpeg")
        
        output_dir = video[:-4] + "_mask.mp4"

        first_choose_mask_id = all_first_choose_mask_id[i]

        last_img = None

        mask_list = []

        mask_output_dir = os.path.join(input_dir, 'tmp_mask', video.split('/')[-1][:-4])
        os.makedirs(mask_output_dir, exist_ok=True)
        # for each frame
        for frame_idx, frame in enumerate(range(frame_num)):
            # if frame_idx < start or frame_idx >= end:
            #     continue
            reader.set_image_index(frame)
            image = reader.get_next_data()
            
            # # Convert RGB to BGR 
            # image = frame[:, :, ::-1].copy() 

            ## generate all mask
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            save_base = os.path.join(mask_output_dir, f'{frame_idx:06}')
            if os.path.exists(save_base):
                all_mask_in_this_frame = glob.glob(os.path.join(save_base, '*.png'))
                # all_mask_in_this_frame.sort() ## todo:: sort need the name to be 000000.png
                # read mask
                all_mask_in_this_frame = [cv2.imread(mask)[:,:,0] for mask in all_mask_in_this_frame]
                if len(all_mask_in_this_frame) == 0:
                    # save mask to tmp
                    masks = generator.generate(image)
                    os.makedirs(save_base, exist_ok=True)
                    write_masks_to_folder(masks, save_base)

                    all_mask_in_this_frame = [mask_data["segmentation"].reshape(mask.shape[0],mask.shape[1],-1)*255 for mask_data in masks]

                
            else:
                # save mask to tmp
                masks = generator.generate(image)
                os.makedirs(save_base, exist_ok=True)
                write_masks_to_folder(masks, save_base)

                all_mask_in_this_frame = [mask_data["segmentation"]*255 for mask_data in masks]

            # all_mask_in_this_frame = glob.glob(os.path.join(input_dir, frame, '*.png'))
            # all_mask_in_this_frame.sort()
            if frame_idx == 0:
                choose_mask_id = first_choose_mask_id
            else:
                choose_mask_id = 0
                min_diff = 100000
                for i in range(len(all_mask_in_this_frame)):
                    ## find the most similar img
                    # img = cv2.imread(all_mask_in_this_frame[i])
                    # change to np.float32
                    img = all_mask_in_this_frame[i].astype(np.float32)
                    if img.shape != last_img.shape:
                        continue
                    diff = cv2.absdiff(img, last_img)
                    diff = diff.mean()
                    # if diff < 1:
                        # choose_mask_id = i
                        # break
                    if diff < min_diff:
                        min_diff = diff
                        choose_mask_id = i
            chosen_mask = all_mask_in_this_frame[choose_mask_id].astype(np.float32)
            last_img = chosen_mask
            # last_img = cv2.imread(chosen_mask)
        
            ## copy mask to output_dir
            # mask = cv2.imread(chosen_mask)
            mask = chosen_mask
            # cv2.imwrite(os.path.join(output_dir, f'{frame_idx:06}.png'), mask)
            mask_list.append(mask)
            print(f'frame_idx: {frame_idx}, mask_id: {choose_mask_id}')
            # if frame_idx == 10:
                # break
        # imageio.mimsave(os.path.join(output_dir, 'mask.gif'), mask_list, duration=0.1)
        
        imageio.mimsave(output_dir, mask_list, fps=frame_rate)

parser = argparse.ArgumentParser(
    description=(
        "Runs automatic mask generation on an input image or directory of images, "
        "and outputs masks as either PNGs or COCO-style RLEs. Requires open-cv, "
        "as well as pycocotools if saving in RLE format."
    )
)

parser.add_argument(
    "--input_dir",
    type=str,
    default='/root/data/wym/workspace/my_pinerf/data/ScalarReal',
    help="Path to either a single input image or folder of images.",
)

parser.add_argument(
    "--first_mask_index",
    type=str,
    default=None,
)

parser.add_argument(
    "--model-type",
    type=str,
    default='default',
    help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b']",
)

parser.add_argument(
    "--checkpoint",
    type=str,
    default='./tools/get_mask/sam_vit_h_4b8939.pth',
    help="The path to the SAM checkpoint to use for mask generation.",
)

parser.add_argument("--device", type=str, default="cuda", help="The device to run generation on.")

parser.add_argument(
    "--convert-to-rle",
    action="store_true",
    help=(
        "Save masks as COCO RLEs in a single json instead of as a folder of PNGs. "
        "Requires pycocotools."
    ),
)

amg_settings = parser.add_argument_group("AMG Settings")

amg_settings.add_argument(
    "--points-per-side",
    type=int,
    default=None,
    help="Generate masks by sampling a grid over the image with this many points to a side.",
)

amg_settings.add_argument(
    "--points-per-batch",
    type=int,
    default=None,
    help="How many input points to process simultaneously in one batch.",
)

amg_settings.add_argument(
    "--pred-iou-thresh",
    type=float,
    default=None,
    help="Exclude masks with a predicted score from the model that is lower than this threshold.",
)

amg_settings.add_argument(
    "--stability-score-thresh",
    type=float,
    default=None,
    help="Exclude masks with a stability score lower than this threshold.",
)

amg_settings.add_argument(
    "--stability-score-offset",
    type=float,
    default=None,
    help="Larger values perturb the mask more when measuring stability score.",
)

amg_settings.add_argument(
    "--box-nms-thresh",
    type=float,
    default=None,
    help="The overlap threshold for excluding a duplicate mask.",
)

amg_settings.add_argument(
    "--crop-n-layers",
    type=int,
    default=None,
    help=(
        "If >0, mask generation is run on smaller crops of the image to generate more masks. "
        "The value sets how many different scales to crop at."
    ),
)

amg_settings.add_argument(
    "--crop-nms-thresh",
    type=float,
    default=None,
    help="The overlap threshold for excluding duplicate masks across different crops.",
)

amg_settings.add_argument(
    "--crop-overlap-ratio",
    type=int,
    default=None,
    help="Larger numbers mean image crops will overlap more.",
)

amg_settings.add_argument(
    "--crop-n-points-downscale-factor",
    type=int,
    default=None,
    help="The number of points-per-side in each layer of crop is reduced by this factor.",
)

amg_settings.add_argument(
    "--min-mask-region-area",
    type=int,
    default=None,
    help=(
        "Disconnected mask regions or holes with area smaller than this value "
        "in pixels are removed by postprocessing."
    ),
)


def write_masks_to_folder(masks: List[Dict[str, Any]], path: str) -> None:
    header = "id,area,bbox_x0,bbox_y0,bbox_w,bbox_h,point_input_x,point_input_y,predicted_iou,stability_score,crop_box_x0,crop_box_y0,crop_box_w,crop_box_h"  # noqa
    metadata = [header]
    for i, mask_data in enumerate(masks):
        mask = mask_data["segmentation"]
        filename = f"{i}.png"
        cv2.imwrite(os.path.join(path, filename), mask * 255)
        mask_metadata = [
            str(i),
            str(mask_data["area"]),
            *[str(x) for x in mask_data["bbox"]],
            *[str(x) for x in mask_data["point_coords"][0]],
            str(mask_data["predicted_iou"]),
            str(mask_data["stability_score"]),
            *[str(x) for x in mask_data["crop_box"]],
        ]
        row = ",".join(mask_metadata)
        metadata.append(row)
    metadata_path = os.path.join(path, "metadata.csv")
    with open(metadata_path, "w") as f:
        f.write("\n".join(metadata))

    return


def get_amg_kwargs(args):
    amg_kwargs = {
        "points_per_side": args.points_per_side,
        "points_per_batch": args.points_per_batch,
        "pred_iou_thresh": args.pred_iou_thresh,
        "stability_score_thresh": args.stability_score_thresh,
        "stability_score_offset": args.stability_score_offset,
        "box_nms_thresh": args.box_nms_thresh,
        "crop_n_layers": args.crop_n_layers,
        "crop_nms_thresh": args.crop_nms_thresh,
        "crop_overlap_ratio": args.crop_overlap_ratio,
        "crop_n_points_downscale_factor": args.crop_n_points_downscale_factor,
        "min_mask_region_area": args.min_mask_region_area,
    }
    amg_kwargs = {k: v for k, v in amg_kwargs.items() if v is not None}
    return amg_kwargs

def write_masks_to_folder(masks: List[Dict[str, Any]], path: str) -> None:
    header = "id,area,bbox_x0,bbox_y0,bbox_w,bbox_h,point_input_x,point_input_y,predicted_iou,stability_score,crop_box_x0,crop_box_y0,crop_box_w,crop_box_h"  # noqa
    metadata = [header]
    for i, mask_data in enumerate(masks):
        mask = mask_data["segmentation"]
        filename = f"{i}.png"
        cv2.imwrite(os.path.join(path, filename), mask * 255)
        mask_metadata = [
            str(i),
            str(mask_data["area"]),
            *[str(x) for x in mask_data["bbox"]],
            *[str(x) for x in mask_data["point_coords"][0]],
            str(mask_data["predicted_iou"]),
            str(mask_data["stability_score"]),
            *[str(x) for x in mask_data["crop_box"]],
        ]
        row = ",".join(mask_metadata)
        metadata.append(row)
    metadata_path = os.path.join(path, "metadata.csv")
    with open(metadata_path, "w") as f:
        f.write("\n".join(metadata))

    return


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)