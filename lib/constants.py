import os

import numpy as np


# Mean and standard deviation for normalizing input image
IMG_NORM_MEAN = [0.485, 0.456, 0.406]
IMG_NORM_STD = [0.229, 0.224, 0.225]


# Joint selectors
# Indices to get the 14 LSP joints from the 17 H36M joints
H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]
H36M_TO_J14 = H36M_TO_J17[:14]


DATA_DIR = 'data'
SMPL_MODEL_DIR = os.path.join(DATA_DIR, 'smpl')
SMPL_MEAN_PARAMS = os.path.join(DATA_DIR, 'smpl_mean_params.npz')
JOINT_REGRESSOR_TRAIN_EXTRA = os.path.join(DATA_DIR, 'J_regressor_extra.npy')
JOINT_REGRESSOR_H36M = os.path.join(DATA_DIR, 'J_regressor_h36m.npy')