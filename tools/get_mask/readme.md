## Setup
```
pip install git+https://github.com/facebookresearch/segment-anything.git
// then manually change the code in segment-anything
// return torch.true_divide(intersections, unions)
pip install argparse 
```

## Run
python tools/get_mask/my_video_mask.py --input_dir /root/data/wym/workspace/my_pinerf/data/ScalarReal
