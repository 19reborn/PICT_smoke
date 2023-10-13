## Setup
```
pip install git+https://github.com/facebookresearch/segment-anything.git
// then manually change the code in segment-anything
// return torch.true_divide(intersections, unions)
// download model in https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
pip install argparse 
```

## Run
python tools/get_mask/my_video_mask.py --input_dir /root/data/wym/workspace/my_pinerf/data/ScalarReal
python tools/get_mask/my_video_mask.py --input_dir ./data/Cyl
python tools/get_mask/my_video_mask.py --input_dir ./data/Car
