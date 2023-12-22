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
    # frame_list = range(0,frame_N, 1)
    frame_list = range(0,frame_N, 1)
    
    # sample_pts = 1024
    sample_pts = 32*32*32
    # sample_pts = 32*32*32
    # sample_pts = 32*32*32
    
    # l2_error, feature_l2_error = voxel_writer.eval_mapping_error(frame_list, t_list, model, sample_pts = sample_pts)
    # ratio, feature_l2_error, feature_relative_error, feature_ratio = voxel_writer.eval_mapping_error(frame_list, t_list, model, sample_pts = sample_pts)

    
    # # plt.plot(l2_error, label='L2 error')
    # plt.plot(feature_l2_error, label='Feature L2 error')
    # plt.plot(feature_relative_error, label='Feature Relative error')
    # plt.legend()  # Add a legend to the plot
    # # plt.show()
    # plt.savefig(os.path.join(testsavedir, f'feature_mapping_error.png'))
    # plt.close()
    
    # # plt.plot(l2_error, label='L2 error')
    # plt.plot(ratio, label='distance < 20cm ratio')
    # plt.plot(feature_ratio, label='feature relative error < 20%')
    # plt.legend()  # Add a legend to the plot
    # # plt.show()
    # plt.savefig(os.path.join(testsavedir, f'mapping_distance_error.png'))
    feature_l2_error, mapping_l2_error = voxel_writer.eval_mapping_error(frame_list, t_list, model, sample_pts = sample_pts)

    
    # plt.plot(l2_error, label='L2 error')
    plt.plot(feature_l2_error, label='Feature L2 error')
    # plt.plot(feature_relative_error, label='Feature Relative error')
    plt.legend()  # Add a legend to the plot
    # plt.show()
    plt.savefig(os.path.join(testsavedir, f'mapped_feature_error.png'))
    plt.close()
    
    # plt.plot(l2_error, label='L2 error')
    plt.plot(mapping_l2_error, label='Mapping L2 error')
    # plt.plot(feature_ratio, label='feature relative error < 20%')
    plt.legend()  # Add a legend to the plot
    # plt.show()
    plt.savefig(os.path.join(testsavedir, f'mapping_distance_error.png'))
    
    # save the error
    np.save(os.path.join(testsavedir, f'mapping_error.npy'), mapping_l2_error)
    np.save(os.path.join(testsavedir, f'feature_error.npy'), feature_l2_error)