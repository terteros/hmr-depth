import torch
import numpy as np

def root_align(joints):
    '''
    joints: Bx14x3(2nd and 3rd joints should be lhip and rhip.
    '''
    pelvis = (joints[:, 2, :] + joints[:, 3, :]) / 2
    return joints - pelvis[:, None, :]


def mpjpe(pred, gt):
    gt, pred = root_align(gt), root_align(pred)
    return torch.sqrt(((gt - pred) ** 2).sum(dim=-1)).mean(dim=-1) * 1000


def compute_similarity_transform(source_points: torch.Tensor,
                                 target_points: torch.Tensor,
                                 return_tform=False):

    assert target_points.shape[0] == source_points.shape[0]
    assert target_points.shape[1] == 3 and source_points.shape[1] == 3

    source_points = source_points.T
    target_points = target_points.T

    # 1. Remove mean.
    mu1 = source_points.mean(dim=1, keepdims=True)
    mu2 = target_points.mean(dim=1, keepdims=True)
    X1 = source_points - mu1
    X2 = target_points - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = (X1**2).sum()

    # 3. The outer product of X1 and X2.
    K = X1 @ X2.T

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, _, Vh = torch.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[0], device=U.device)
    Z[-1, -1] *= torch.sign(torch.linalg.det(U @ V.T))
    # Construct R.
    R = V @ Z @ U.T

    # 5. Recover scale.
    scale = torch.trace(R @ K) / var1

    # 6. Recover translation.
    t = mu2 - scale * R @ mu1

    # 7. Transform the source points:
    source_points_hat = scale * R @ source_points + t

    source_points_hat = source_points_hat.T

    if return_tform:
        return source_points_hat, {
            'rotation': R,
            'scale': scale,
            'translation': t
        }

    return source_points_hat


def pampjpe(pred, gt):
    pred_aligned = torch.stack([
        compute_similarity_transform(pred_i, gt_i)
        for pred_i, gt_i in zip(pred, gt)])
    re_per_joint = torch.sqrt(((pred_aligned - gt) ** 2).sum(dim=-1)) * 1000
    re = re_per_joint.mean(axis=-1)
    return re, re_per_joint


