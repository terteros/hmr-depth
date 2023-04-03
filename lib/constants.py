import os

import numpy as np


# Mean and standard deviation for normalizing input image
IMG_NORM_MEAN = [0.485, 0.456, 0.406]
IMG_NORM_STD = [0.229, 0.224, 0.225]
FOCAL_LENGTH = 5000.
IMG_RES = 224


GRPH_COL_MAP = {'background': [0,0,0], 'hat': [128,0,0], 'hair': [255,0,0], 'glove': [0,85,0],
                  'sunglasses': [170,0,51], 'upperclothes': [255,85,0], 'dress': [0,0,85], 'coat': [0,119,221],
                  'socks': [85,85,0], 'pants': [0,85,85], 'jumpsuits': [85,51,0], 'scarf': [52,86,128],
                  'skirt': [0,128,0], 'face': [0,0,255], 'leftArm': [51,170,221], 'rightArm': [0,255,255],
                  'leftLeg': [85,255,170], 'rightLeg': [170,255,85], 'leftShoe': [255,255,0], 'rightShoe': [255,170,0]}
GRPH_LABEL = ['background', 'hat', 'hair', 'glove', 'sunglasses', 'upperclothes', 'dress', 'coat', 'socks',
              'pants', 'jumpsuits', 'scarf', 'skirt', 'face', 'leftArm', 'rightArm', 'leftLeg', 'rightLeg',
              'leftShoe', 'rightShoe']
GRPH_COL_MAP_NORM = {k: np.array(v, dtype=np.float32)/255. for k, v in GRPH_COL_MAP.items()}

# Merging Graphonomy Colors to upperbodyclothes, pants, left/right arms, left/right legs, left/right shoes, face, hai
# REMAP_GRPH_COL is for mapping original graphonomy images obtained from fprop to new scheme
REMAP_GRPH_COL = {'background': [0,0,0], 'hat': [255,0,0], 'hair': [255,0,0], 'glove': [255,85,0],
                  'sunglasses': [0,0,255], 'upperclothes': [255,85,0], 'dress': [255,85,0], 'coat': [255,85,0],
                  'socks': [0,85,85], 'pants': [0,85,85], 'jumpsuits': [255,85,0], 'scarf': [255,0,0],
                  'skirt': [0,85,85], 'face': [0,0,255], 'leftArm': [51,170,221], 'rightArm': [0,255,255],
                  'leftLeg': [85,255,170], 'rightLeg': [170,255,85], 'leftShoe': [255,255,0], 'rightShoe': [255,170,0]}
NEW_GRPH_COL_MAP = {'background': [0,0,0], 'hair': [255,0,0], 'face': [0,0,255], 'upperclothes': [255,85,0],
                      'pants': [0,85,85], 'leftArm': [51,170,221], 'rightArm': [0,255,255], 'leftLeg': [85,255,170],
                      'rightLeg': [170,255,85], 'leftShoe': [255,255,0], 'rightShoe': [255,170,0]}
NEW_GRPH_LABEL = ['background', 'hair', 'face', 'upperclothes', 'pants', 'leftArm', 'rightArm', 'leftLeg',
                  'rightLeg', 'leftShoe', 'rightShoe']
NEW_GRPH_COL_MAP_NORM = {k: np.array(v, dtype=np.float32)/255. for k, v in NEW_GRPH_COL_MAP.items()}

"""
We create a superset of joints containing the OpenPose joints together with the ones that each dataset provides.
We keep a superset of 24 joints such that we include all joints from every dataset.
If a dataset doesn't provide annotations for a specific joint, we simply ignore it.
The joints used here are the following:
"""
JOINT_NAMES = [
# 25 OpenPose joints (in the order provided by OpenPose)
'OP Nose',
'OP Neck',
'OP RShoulder',
'OP RElbow',
'OP RWrist',
'OP LShoulder',
'OP LElbow',
'OP LWrist',
'OP MidHip',
'OP RHip',
'OP RKnee',
'OP RAnkle',
'OP LHip',
'OP LKnee',
'OP LAnkle',
'OP REye',
'OP LEye',
'OP REar',
'OP LEar',
'OP LBigToe',
'OP LSmallToe',
'OP LHeel',
'OP RBigToe',
'OP RSmallToe',
'OP RHeel',
# 24 Ground Truth joints (superset of joints from different datasets)
'Right Ankle',
'Right Knee',
'Right Hip',
'Left Hip',
'Left Knee',
'Left Ankle',
'Right Wrist',
'Right Elbow',
'Right Shoulder',
'Left Shoulder',
'Left Elbow',
'Left Wrist',
'Neck (LSP)',
'Top of Head (LSP)',
'Pelvis (MPII)',
'Thorax (MPII)',
'Spine (H36M)',
'Jaw (H36M)',
'Head (H36M)',
'Nose',
'Left Eye',
'Right Eye',
'Left Ear',
'Right Ear'
]

# Dict containing the joints in numerical order
JOINT_IDS = {JOINT_NAMES[i]: i for i in range(len(JOINT_NAMES))}

# Map joints to SMPL joints
JOINT_MAP = {
'OP Nose': 24, 'OP Neck': 12, 'OP RShoulder': 17,
'OP RElbow': 19, 'OP RWrist': 21, 'OP LShoulder': 16,
'OP LElbow': 18, 'OP LWrist': 20, 'OP MidHip': 0,
'OP RHip': 2, 'OP RKnee': 5, 'OP RAnkle': 8,
'OP LHip': 1, 'OP LKnee': 4, 'OP LAnkle': 7,
'OP REye': 25, 'OP LEye': 26, 'OP REar': 27,
'OP LEar': 28, 'OP LBigToe': 29, 'OP LSmallToe': 30,
'OP LHeel': 31, 'OP RBigToe': 32, 'OP RSmallToe': 33, 'OP RHeel': 34,
'Right Ankle': 8, 'Right Knee': 5, 'Right Hip': 45,
'Left Hip': 46, 'Left Knee': 4, 'Left Ankle': 7,
'Right Wrist': 21, 'Right Elbow': 19, 'Right Shoulder': 17,
'Left Shoulder': 16, 'Left Elbow': 18, 'Left Wrist': 20,
'Neck (LSP)': 47, 'Top of Head (LSP)': 48,
'Pelvis (MPII)': 49, 'Thorax (MPII)': 50,
'Spine (H36M)': 51, 'Jaw (H36M)': 52,
'Head (H36M)': 53, 'Nose': 24, 'Left Eye': 26,
'Right Eye': 25, 'Left Ear': 28, 'Right Ear': 27
}

# Joint selectors
# Indices to get the 14 LSP joints from the 17 H36M joints
H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]
H36M_TO_J14 = H36M_TO_J17[:14]
# Indices to get the 14 LSP joints from the ground truth joints
J24_TO_J17 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 14, 16, 17]
J24_TO_J14 = J24_TO_J17[:14]

# Permutation of SMPL pose parameters when flipping the shape
SMPL_JOINTS_FLIP_PERM = [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20, 23, 22]
SMPL_POSE_FLIP_PERM = []
for i in SMPL_JOINTS_FLIP_PERM:
    SMPL_POSE_FLIP_PERM.append(3*i)
    SMPL_POSE_FLIP_PERM.append(3*i+1)
    SMPL_POSE_FLIP_PERM.append(3*i+2)
# Permutation indices for the 24 ground truth joints
J24_FLIP_PERM = [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13, 14, 15, 16, 17, 18, 19, 21, 20, 23, 22]
# Permutation indices for the full set of 49 joints
J49_FLIP_PERM = [0, 1, 5, 6, 7, 2, 3, 4, 8, 12, 13, 14, 9, 10, 11, 16, 15, 18, 17, 22, 23, 24, 19, 20, 21]\
              + [25+i for i in J24_FLIP_PERM]

pw3d_occluded_sequences = [
    'courtyard_backpack',
    'courtyard_basketball',
    'courtyard_bodyScannerMotions',
    'courtyard_box',
    'courtyard_golf',
    'courtyard_jacket',
    'courtyard_laceShoe',
    'downtown_stairs',
    'flat_guitar',
    'flat_packBags',
    'outdoors_climbing',
    'outdoors_crosscountry',
    'outdoors_fencing',
    'outdoors_freestyle',
    'outdoors_golf',
    'outdoors_parcours',
    'outdoors_slalom',
]

pw3d_test_sequences = [
    'flat_packBags_00',
    'downtown_weeklyMarket_00',
    'outdoors_fencing_01',
    'downtown_walkBridge_01',
    'downtown_enterShop_00',
    'downtown_rampAndStairs_00',
    'downtown_bar_00',
    'downtown_runForBus_01',
    'downtown_cafe_00',
    'flat_guitar_01',
    'downtown_runForBus_00',
    'downtown_sitOnStairs_00',
    'downtown_bus_00',
    'downtown_arguing_00',
    'downtown_crossStreets_00',
    'downtown_walkUphill_00',
    'downtown_walking_00',
    'downtown_car_00',
    'downtown_warmWelcome_00',
    'downtown_upstairs_00',
    'downtown_stairs_00',
    'downtown_windowShopping_00',
    'office_phoneCall_00',
    'downtown_downstairs_00'
]
DATA_DIR = 'data'
H36M_DIR = os.path.join(DATA_DIR, 'h36m_densepose_keypoints/')
H36M_HDF_DIR = os.path.join(DATA_DIR, 'h36m/')
H36M_EXTRACTED_DATA = os.path.join(DATA_DIR, 'h36m_extracted_data.pkl')
H36M_BBOX_TEST = os.path.join(DATA_DIR, 'h36m_bboxes_test.pkl')
VIBE_H36M_TRAIN_DB = os.path.join(DATA_DIR, 'h36m_train_db.pt')
VIBE_H36M_TEST_DB = os.path.join(DATA_DIR, 'h36m_test_db.pt')
SMPL_MODEL_DIR = os.path.join(DATA_DIR, 'smpl')
SMPL_MEAN_PARAMS = os.path.join(DATA_DIR, 'smpl_mean_params.npz')
JOINT_REGRESSOR_TRAIN_EXTRA = os.path.join(DATA_DIR, 'J_regressor_extra.npy')
JOINT_REGRESSOR_H36M = os.path.join(DATA_DIR, 'J_regressor_h36m.npy')
COCOPLUS_REGRESSOR = os.path.join(DATA_DIR, 'cocoplus_regressor.npy')
BODY_MODELS = os.path.join(DATA_DIR, 'body_models')
SMPL_RIGID_PARTS_FILE = os.path.join(DATA_DIR, 'smpl_rigid_parts_w_faces.pkl')
IMG_SIZES = {  # height, width
    '54138969': [1002, 1000],
    '55011271': [1000, 1000],
    '58860488': [1000, 1000],
    '60457274': [1002, 1000]
}
DENSEPOSE_PARTS = {
    1: 'Torso',
    2: 'Torso',
    3: 'Right Hand',
    4: 'Left Hand',
    5: 'Left Foot',
    6: 'Right Foot',
    7: 'Right Upper Leg',
    9: 'Right Upper Leg',
    8: 'Left Upper Leg',
    10: 'Left Upper Leg',
    11: 'Right Lower Leg',
    13: 'Right Lower Leg',
    12: 'Left Lower Leg',
    14: 'Left Lower Leg',
    15: 'Left Upper Arm',
    17: 'Left Upper Arm',
    16: 'Right Upper Arm',
    18: 'Right Upper Arm',
    19: 'Left Lower Arm',
    21: 'Left Lower Arm',
    20: 'Right Lower Arm',
    22: 'Right Lower Arm',
    23: 'Head',
    24: 'Head',
}
SMPL_PARTS = {
    'Torso': 1,
    'Right Hand': 2,
    'Left Hand': 3,
    'Right Foot': 4,
    'Left Foot': 5,
    'Right Upper Leg': 6,
    'Left Upper Leg': 7,
    'Right Lower Leg': 8,
    'Left Lower Leg': 9,
    'Right Upper Arm': 10,
    'Left Upper Arm': 11,
    'Right Lower Arm': 12,
    'Left Lower Arm': 13,
    'Head': 14
}
CODEBOOK = {}
for k, v in DENSEPOSE_PARTS.items():
    CODEBOOK[k] = SMPL_PARTS[v]