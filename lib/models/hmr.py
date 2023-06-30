import numpy as np
import pytorch_lightning as pl
import torch

from lib.models.hmr_head import HmrHead
from lib.models.resnet import resnet50
from lib.models.smpl_head import SmplHead


class HMR(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        # TODO: change it to torch.lit save hyperparam method
        self._hparams = hparams
        self.backbone = resnet50(pretrained=True)
        self.head = HmrHead(num_input_features=2048)
        self.smpl = SmplHead(img_res=self._hparams.DATASET.IMG_RES)
        self.losses = []

    def forward(self, images, valid_joints=list(range(24)), infer_shape=True):
        features = self.backbone(images)
        hmr_output = self.head(features, valid_joints=valid_joints)
        smpl_output = self.smpl(
            rotmat=hmr_output['pred_pose'],
            shape=hmr_output['pred_shape'],
            cam=hmr_output['pred_cam'],
            normalize_joints2d=True,
        )
        smpl_output.update(hmr_output)
        return smpl_output




