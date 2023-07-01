import os

import numpy as np
import pytorch_lightning as pl
import torch

from lib.constants import DATA_DIR, H36M_TO_J14, SMPL_MODEL_DIR
from lib.dataset.dataset import Dataset3D
from lib.models.hmr_head import HmrHead
from lib.models.resnet import resnet50
from lib.models.smpl_head import SmplHead


def _similarity_transform_params(S1, S2):
    """
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    """
    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1 ** 2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale * (R.dot(mu1))

    return scale, R, t


def compute_similarity_transform(S1, S2):
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert (S2.shape[1] == S1.shape[1])
    # 7. Error:
    scale, R, t = _similarity_transform_params(S1, S2)
    S1_hat = scale * R.dot(S1) + t
    if transposed:
        S1_hat = S1_hat.T
    return S1_hat


def compute_similarity_transform_batch(S1, S2):
    """Batched version of compute_similarity_transform."""
    S1_hat = np.zeros_like(S1)
    for i in range(S1.shape[0]):
        S1_hat[i] = compute_similarity_transform(S1[i], S2[i])
    return S1_hat


def reconstruction_error(S1, S2, reduction='mean'):
    """Do Procrustes alignment and compute reconstruction error."""
    S1_hat = compute_similarity_transform_batch(S1, S2)

    re_per_joint = np.sqrt(((S1_hat - S2) ** 2).sum(axis=-1))
    re = re_per_joint.mean(axis=-1)
    if reduction == 'mean':
        re = re.mean()
    elif reduction == 'sum':
        re = re.sum()
    return re, re_per_joint


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

    def validation_step(self, batch, batch_nb, dataloader_idx=0):
        valid_joints = self._hparams.TRAIN.PHASES[:int(self.current_epoch + 1)]
        valid_joints = sum(valid_joints, [])
        pred = self(batch['img'], valid_joints=valid_joints)

        # For 3DPW get the 14 common joints from the rendered shape
        gt_keypoints_3d = batch['kp_3d'][:, 25:39, :]
        gt_pelvis = (gt_keypoints_3d[:, 2, :] + gt_keypoints_3d[:, 3, :]) / 2
        gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]
        pred_keypoints_3d = pred['smpl_joints3d']


        pred_pelvis = (pred_keypoints_3d[:, 2, :] + pred_keypoints_3d[:, 3, :]) / 2
        pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, None, :]

        # Absolute error (MPJPE)
        mpjpe = torch.sqrt(((pred_keypoints_3d - gt_keypoints_3d) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()

        # Reconstruction_error
        pampjpe, r_error_per_joint = reconstruction_error(
            pred_keypoints_3d.cpu().numpy(),
            gt_keypoints_3d.cpu().numpy(),
            reduction=None,
        )
        results = {'mpjpe': mpjpe, 'pampjpe': pampjpe, 'd_idx': dataloader_idx}
        pred = {k: v.cpu().numpy() for k, v in pred.items()}
        results.update(pred)
        return results

    def _val_epoch_end(self, outputs, is_test=False):
        if isinstance(outputs[0], dict):
            outputs = [outputs]
        for output in outputs:
            idx = np.array([batch['d_idx'] for batch in output]).mean()
            val_mpjpe = np.concatenate([errs['mpjpe'] for errs in output])
            val_pampjpe = np.concatenate([errs['pampjpe'] for errs in output])

            avg_mpjpe, avg_pampjpe = 1000 * val_mpjpe.mean(), 1000 * val_pampjpe.mean()
            avg_mpjpe, avg_pampjpe = torch.tensor(avg_mpjpe), torch.tensor(avg_pampjpe)
            # Best model selection criterion - 1.5 * PAMPJPE + MPJPE
            best_result = 1.5 * avg_pampjpe.clone().cpu().numpy() + avg_mpjpe.clone().cpu().numpy()

            postfix = f'_{idx}' if idx > 0 else ''
            prefix = 'test/' if is_test else ''

            self.log(f'{prefix}mpjpe{postfix}', avg_mpjpe)
            self.log(f'{prefix}pampjpe{postfix}', avg_pampjpe)
            self.log(f'{prefix}val_loss{postfix}', best_result)

    def validation_epoch_end(self, outputs):
        self._val_epoch_end(outputs)

    def val_dataloader(self):
        dataset = Dataset3D(f'./data/3dpw/3dpw_val_db.pt', f'./data/3dpw', self._hparams)
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self._hparams.TRAIN.VAL_BATCH_SIZE,
            shuffle=False,
            num_workers=self._hparams.NUM_WORKERS
        )
        return dataloader
