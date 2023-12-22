import numpy as np


ours_dir = '/cluster/project/tang/yiming/project/pinf_clean/log/cyl/1214_v2_150k_iters/eval_mapping_150001'
ablation_dir = '/cluster/project/tang/yiming/project/pinf_clean/log/cyl/1217_ablation_no_all_cycle_loss_no_vel_reg/eval_mapping_150001'


mapping_error_ours = np.load(ours_dir + '/mapping_error.npy')

mapping_error_ablation = np.load(ablation_dir + '/mapping_error.npy')

import matplotlib.pyplot as plt

plt.plot(mapping_error_ours, label='With Intrinsic Constraints')
plt.plot(mapping_error_ablation, label='W/o Intrinsic Constraints')

# x_axis legend 
plt.xlabel('Mapping frame interval')
plt.ylabel('Mapping error (m)')

plt.legend()  # Add a legend to the plot

plt.savefig('comp_mapping_distance_error.png')
plt.close()