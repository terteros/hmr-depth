import argparse
from yacs.config import CfgNode as CN


# Configuration variables
cfg = CN()

cfg.OUTPUT_DIR = 'results'
cfg.EXP_NAME = 'default'
cfg.DEVICE = 'cuda'
cfg.DEBUG = False
cfg.LOGDIR = ''
cfg.NUM_WORKERS = 8
cfg.DEBUG_FREQ = 10
cfg.SEED_VALUE = -1
cfg.PIN_MEMORY = False
cfg.FOCAL_LENGTH = 5000.
cfg.MESH_COLOR = 'blue'

cfg.CUDNN = CN()
cfg.CUDNN.BENCHMARK = True
cfg.CUDNN.DETERMINISTIC = False
cfg.CUDNN.ENABLED = True

# DATASET
cfg.DATASET = CN()
cfg.DATASET.DB_FILE_PATH = ''
cfg.DATASET.EVAL_DB_FILE_PATH = ''
cfg.DATASET.TEST_DB_FILE_PATH = ''
cfg.DATASET.HDF5 = True
cfg.DATASET.IMG_RES = 224
cfg.DATASET.SHUFFLE_TRAIN = False
cfg.DATASET.FRAME_SKIP = 1

# TRAIN
cfg.TRAIN = CN()
cfg.TRAIN.BATCH_SIZE = 32
cfg.TRAIN.MAX_EPOCH = 30
cfg.TRAIN.VAL_BATCH_SIZE = 32
cfg.TRAIN.RESUME = ''
cfg.TRAIN.PRETRAINED = None
cfg.TRAIN.LOG_FREQ_TB_IMAGES = 200
cfg.TRAIN.SAVE_IMAGES = True

# OPTIMIZER
cfg.OPTIMIZER = CN()
cfg.OPTIMIZER.TYPE = 'adam'
cfg.OPTIMIZER.LR = 0.00003
cfg.OPTIMIZER.WD = 0.0

# MODEL
cfg.MODEL = CN()
cfg.MODEL.NAME = ''
cfg.MODEL.BACKBONE = 'resnet50'

# TEST
cfg.TEST = CN()
cfg.TEST.EXPERIMENT_FOLDER = ''
cfg.TEST.BATCH_SIZE = 32
cfg.TEST.SIDEVIEW = True
cfg.TEST.LOG_FREQ_TB_IMAGES = 50
cfg.TEST.SAVE_IMAGES = True
cfg.TEST.SAVE_RESULTS = True

# LOSS
cfg.LOSS = CN()
cfg.LOSS.SHAPE_LOSS_WEIGHT = 0.
cfg.LOSS.KEYPOINT_3D_LOSS_WEIGHT = 0.
cfg.LOSS.KEYPOINT_2D_LOSS_WEIGHT = 5.
cfg.LOSS.POSE_LOSS_WEIGHT = 0.
cfg.LOSS.BETA_LOSS_WEIGHT = 0.0
cfg.LOSS.OPENPOSE_TRAIN_WEIGHT = 0.
cfg.LOSS.GT_TRAIN_WEIGHT = 1.
cfg.LOSS.CAM_LOSS_WEIGHT = 0.
cfg.LOSS.LOSS_WEIGHT = 60.
cfg.LOSS.DEPTH_LOSS_WEIGHT = 0.1
cfg.LOSS.DEPTH_ALIGNMENT = 'N_DEPTH'
cfg.LOSS.ONLY_TORSO = 0
# AUG
cfg.NOISE_FACTOR = 0
cfg.ROT_FACTOR = 30
cfg.SCALE_FACTOR = 0.25

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return cfg.clone()


def update_cfg(cfg_file):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_file)
    return cfg.clone()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='cfg file path')
    args = parser.parse_args()
    print(args, end='\n\n')

    cfg_file = args.cfg
    if args.cfg is not None:
        cfg = update_cfg(args.cfg)
    else:
        cfg = get_cfg_defaults()

    return cfg, cfg_file
