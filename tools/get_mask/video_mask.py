from metaseg import SegAutoMaskPredictor, SegManualMaskPredictor

# If gpu memory is not enough, reduce the points_per_side and points_per_batch.

# For video
results = SegAutoMaskPredictor().video_predict(
    source="/home/wym/workspace/my_pinerf/data/ScalarReal/train00.mp4",
    model_type="vit_h", # vit_l, vit_h, vit_b
    points_per_side=16, 
    points_per_batch=64,
    min_area=1000,
    output_path="output_h.mp4",
)