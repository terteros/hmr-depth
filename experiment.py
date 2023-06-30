import argparse

import cv2
from pytorch3d.transforms import rotation_6d_to_matrix
from tqdm import tqdm

from lib.cfg import parse_args, update_cfg
from lib.misc import to_numpy
from lib.models.hmr import HMR
from lib.models.hmr_head import HmrHead
from lib.models.resnet import resnet50
from lib.models.smpl_head import SmplHead
from lib.constants import SMPL_MEAN_PARAMS
from lib.dataset.dataset import DatasetDepth
import numpy as np
import torch

ORIGINAL_CHKPT = '/home/batuhan/ssl-part-render/results/cviu/3dpwn_l3d_drl_cp/DEPTH_LOSS_9.0_SEED_VALUE_4_' \
                 '/3dpwn_l3d_drl_cp/a0f9ee9a629e4aba95ac83adef5b9b12/checkpoints/epoch=4-val_loss=198.6940.ckpt'


def update_hmr_checkpoint(in_file, out_file):
    chkpt = torch.load(in_file)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for name, param in chkpt['state_dict'].items():
        if 'model' in name:
            name = name.replace('model.', '')
        new_state_dict[name] = param
    chkpt['state_dict'] = new_state_dict
    torch.save(chkpt, out_file)


parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, help='cfg file path')
args = parser.parse_args()
cfg = update_cfg(args.cfg)

model = HMR.load_from_checkpoint('epoch=4-val_loss=198.6940.ckpt', strict=False, hparams=cfg)
model.eval()

dataset = DatasetDepth(f'./data/h36m/h36m_test.pt', f'./data/h36m', cfg, frame_skip=10, use_augmentation=False)
dataloader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0
)

pbar = tqdm(len(dataset))
for idx, batch in enumerate(dataloader):
    pbar.update(cfg.TRAIN.BATCH_SIZE)

    img_cv2 = dataset.get_debug_image(batch)
    cv2.imwrite('test_gt.png', img_cv2)

    pred = model(batch['img'])
    batch['kp_3d'] = pred['smpl_joints3d']
    batch['kp_2d'] = pred['smpl_joints2d']

    img_cv2 = dataset.get_debug_image(batch)
    cv2.imwrite('test_pred.png', img_cv2)
    # breakpoint()
    exit(0)
