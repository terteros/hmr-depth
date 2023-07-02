import pytorch_lightning as pl
import torch

from lib import metrics
from lib.dataset.dataset import build_dataloaders
from lib.models.hmr_head import HmrHead
from lib.models.resnet import resnet50
from lib.models.smpl_head import SmplHead
from lib.losses import *


class HMR(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        # TODO: change it to torch.lit save hyperparam method
        self._hparams = hparams
        self.backbone = resnet50(pretrained=True)
        self.head = HmrHead(num_input_features=2048)
        self.smpl = SmplHead(img_res=self._hparams.DATASET.IMG_RES)
        self.losses = {
            '2d': {'fun': Keypoints2DLoss(), 'weight': self._hparams.LOSS.KEYPOINT_2D_LOSS_WEIGHT},
            '3d': {'fun': Keypoints3DLoss(), 'weight': self._hparams.LOSS.KEYPOINT_3D_LOSS_WEIGHT},
            'cam': {'fun': CamRegularizationLoss(), 'weight': self._hparams.LOSS.CAM_LOSS_WEIGHT}
        }

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

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self._hparams.OPTIMIZER.LR,
            weight_decay=self._hparams.OPTIMIZER.WD
        )

    def training_step(self, batch, batch_nb):
        valid_joints = self._hparams.TRAIN.PHASES[:int(self.current_epoch + 1)]
        valid_joints = sum(valid_joints, [])
        pred = self(batch['img'], valid_joints=valid_joints)

        self.losses['2d']['current_val'] = self.losses['2d']['fun'](pred['smpl_joints2d'], batch['kp_2d'][:, 25:39, :])
        self.losses['3d']['current_val'] = self.losses['3d']['fun'](pred['smpl_joints3d'], batch['kp_3d'][:, 25:39, :])
        self.losses['cam']['current_val'] = self.losses['cam']['fun'](pred['pred_cam'])

        loss_log = {f'loss/{loss}': self.losses[loss]['weight'] * self.losses[loss]['current_val'] for loss in
                    self.losses.keys()}
        self.log_dict(loss_log, on_step=True, prog_bar=True)
        return sum(weighted_loss for weighted_loss in loss_log.values())

    def validation_step(self, batch, batch_nb, dataloader_idx=0):
        # TODO: make scheduled training more flexible (10 epochs only root, 20 epochs all etc.).
        valid_joints = self._hparams.TRAIN.PHASES[:int(self.current_epoch + 1)]
        valid_joints = sum(valid_joints, [])
        pred = self(batch['img'], valid_joints=valid_joints)

        # For 3DPW get the 14 common joints from the rendered shape
        gt_keypoints_3d = batch['kp_3d'][:, 25:39, :]
        pred_keypoints_3d = pred['smpl_joints3d']

        mpjpe = metrics.mpjpe(pred_keypoints_3d, gt_keypoints_3d)
        pampjpe, r_error_per_joint = metrics.pampjpe(pred_keypoints_3d, gt_keypoints_3d)
        results = {'mpjpe': mpjpe, 'pampjpe': pampjpe, 'd_idx': dataloader_idx}
        return results

    def _val_epoch_end(self, outputs, is_test=False):
        if isinstance(outputs[0], dict):
            outputs = [outputs]
        for output in outputs:
            if is_test:
                dataset_name = self._hparams.DATASET.TEST.SET[output[0]['d_idx']]
                prefix = f'test_{dataset_name}'
            else:
                dataset_name = self._hparams.DATASET.VAL.SET[output[0]['d_idx']]
                prefix = f'val_{dataset_name}'

            metrics_dict = {}
            for metric in ['mpjpe', 'pampjpe']:
                metrics_dict[metric] = torch.cat([errs[metric] for errs in output]).mean()
            metrics_dict['val_loss'] = 1.5 * metrics_dict['pampjpe'] + 1. * metrics_dict['mpjpe']
            for metric, value in metrics_dict.items():
                self.log(f'{prefix}/{metric}', value)

    def validation_epoch_end(self, outputs):
        self._val_epoch_end(outputs)

    def test_step(self, batch, batch_nb, dataloader_idx=0):
        return self.validation_step(batch, batch_nb, dataloader_idx)

    def test_epoch_end(self, outputs):
        self._val_epoch_end(outputs, is_test=True)

    def train_dataloader(self):
        return build_dataloaders(self._hparams, 'train')

    def val_dataloader(self):
        return build_dataloaders(self._hparams, 'val')

    def test_dataloader(self):
        return build_dataloaders(self._hparams, 'test')
