import os, sys
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def evaluate_mapping(args, model, testsavedir, voxel_writer, t_info):
    model.eval()
    print('evaluate mapping error')
    os.makedirs(testsavedir, exist_ok=True)
    
    t_list = list(np.arange(t_info[0],t_info[1],t_info[-1]))
    frame_N = len(t_list)
    frame_N = args.time_size
    frame_list = range(0,frame_N, 1)
    
    sample_pts = 128
    
    l2_error, feature_l2_error = voxel_writer.eval_mapping_error(frame_list, t_list, model, sample_pts = sample_pts)

    
    plt.plot(l2_error, label='L2 error')
    plt.plot(feature_l2_error, label='Feature L2 error')
    plt.legend()  # Add a legend to the plot
    # plt.show()
    plt.savefig(os.path.join(testsavedir, f'mapping_error.png'))