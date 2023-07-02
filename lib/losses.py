import torch
from torch import nn


class Keypoints2DLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss(reduction='none')

    def forward(self, pred, target):
        conf = target[:, :, -1].unsqueeze(-1).clone()
        loss = (conf * self.criterion(pred, target[:, :, :-1])).mean()
        return loss


def root_align(joints):
    '''
    joints: Bx14x3(2nd and 3rd joints should be lhip and rhip.
    '''
    pelvis = (joints[:, 2, :] + joints[:, 3, :]) / 2
    return joints - pelvis[:, None, :]


class Keypoints3DLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss(reduction='none')

    def forward(self, pred, target, valid_mask=None):
        if valid_mask is None:
            valid_mask = torch.ones(pred.shape[0], dtype=torch.bool, device=pred.device)
        if valid_mask.count_nonzero() == 0:
            return torch.FloatTensor(1).fill_(0.).to(pred.device)
        pred, target = root_align(pred[valid_mask, :, :]), root_align(target[valid_mask, :, :])
        return self.criterion(pred, target).mean()


class CamRegularizationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_cam):
        return torch.exp((-pred_cam[:, 0]) * 10 ** 2).mean()

