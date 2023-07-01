import argparse
from yacs.config import CfgNode as CN


# Configuration variables
cfg = CN()

cfg.DEVICE = 'cuda'
cfg.NUM_WORKERS = 8
cfg.SEED_VALUE = -1
cfg.PIN_MEMORY = False
cfg.FOCAL_LENGTH = 5000.

cfg.CUDNN = CN()
cfg.CUDNN.BENCHMARK = True
cfg.CUDNN.DETERMINISTIC = False
cfg.CUDNN.ENABLED = True

# DATASET
cfg.DATASET = CN()
cfg.DATASET.TRAIN = CN()
cfg.DATASET.TRAIN.SET = ['h36m', '3dpw']
cfg.DATASET.TRAIN.SHUFFLE = True
cfg.DATASET.TRAIN.AUG = True
cfg.DATASET.TRAIN.FRAME_SKIP = 1
cfg.DATASET.TRAIN.BATCH_SIZE = 32
cfg.DATASET.VAL = CN()
cfg.DATASET.VAL.SET = ['3dpw', 'h36m']
cfg.DATASET.VAL.SHUFFLE = False
cfg.DATASET.VAL.AUG = False
cfg.DATASET.VAL.FRAME_SKIP = 1
cfg.DATASET.VAL.BATCH_SIZE = 32
cfg.DATASET.TEST = CN()
cfg.DATASET.TEST.SET = ['3dpw', 'h36m']
cfg.DATASET.TEST.SHUFFLE = False
cfg.DATASET.TEST.AUG = False
cfg.DATASET.TEST.FRAME_SKIP = 1
cfg.DATASET.TEST.BATCH_SIZE = 32

cfg.DATASET.IMG_RES = 224

# TRAIN
cfg.TRAIN = CN()

cfg.TRAIN.MAX_EPOCH = 30
cfg.TRAIN.VAL_BATCH_SIZE = 32
cfg.TRAIN.PRETRAINED = None
cfg.TRAIN.PHASES = [list(range(24))]

# OPTIMIZER
cfg.OPTIMIZER = CN()
cfg.OPTIMIZER.TYPE = 'adam'
cfg.OPTIMIZER.LR = 0.00003
cfg.OPTIMIZER.WD = 0.0



# TEST
cfg.TEST = CN()
cfg.TEST.BATCH_SIZE = 32

# LOSS
cfg.LOSS = CN()
cfg.LOSS.SHAPE_LOSS_WEIGHT = 0.
cfg.LOSS.KEYPOINT_3D_LOSS_WEIGHT = 0.
cfg.LOSS.KEYPOINT_2D_LOSS_WEIGHT = 5.
cfg.LOSS.POSE_LOSS_WEIGHT = 0.
cfg.LOSS.BETA_LOSS_WEIGHT = 0.0
cfg.LOSS.CAM_LOSS_WEIGHT = 0.
cfg.LOSS.LOSS_WEIGHT = 60.
cfg.LOSS.DEPTH_LOSS_WEIGHT = 0.1
cfg.LOSS.DEPTH_ALIGNMENT = 'N_DEPTH'

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

    cfg_file = args.cfg
    if args.cfg is not None:
        cfg = update_cfg(args.cfg)
    else:
        cfg = get_cfg_defaults()

    return cfg, cfg_file
