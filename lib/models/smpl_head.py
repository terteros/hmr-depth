import torch
import numpy as np
from pytorch_lightning import LightningModule

from smplx import SMPL as _SMPL
from smplx.utils import SMPLOutput
from smplx.lbs import vertices2joints
from torch import nn

from lib import constants
from lib.constants import H36M_TO_J14


class JointRegressor(nn.Module):
    def __init__(self, regressor_type):
        super(JointRegressor, self).__init__()
        self.regressor_type = regressor_type
        if regressor_type == 'smpl_54':
            joint_regressor = np.load(constants.JOINT_REGRESSOR_TRAIN_EXTRA)
            self.register_buffer('joint_regressor', torch.tensor(joint_regressor, dtype=torch.float32))
        elif regressor_type == 'h36m':
            joint_regressor = np.load(constants.JOINT_REGRESSOR_H36M)
            self.register_buffer('joint_regressor', torch.tensor(joint_regressor, dtype=torch.float32))
        else:
            raise NotImplementedError()

    def forward(self, smpl_output):
        extra_joints = vertices2joints(self.joint_regressor, smpl_output.vertices)
        if self.regressor_type == 'smpl_54':
            joints = [constants.JOINT_MAP[i] for i in constants.JOINT_NAMES]
            joint_map = torch.tensor(joints, dtype=torch.long)
            joints = torch.cat([smpl_output.joints, extra_joints], dim=1)
            joints = joints[:, joint_map, :]
            return joints[:, 25:39, :]
        if self.regressor_type == 'h36m':
            return extra_joints[:, H36M_TO_J14, :]


class SMPL(_SMPL):
    """ Extension of the official SMPL implementation to support more joints """
    def __init__(self, *args, regressor_type='h36m', **kwargs):
        super(SMPL, self).__init__(*args, **kwargs)
        self.joint_regressor = JointRegressor(regressor_type)

    def forward(self, *args, **kwargs):
        kwargs['get_skin'] = True
        smpl_output = super(SMPL, self).forward(*args, **kwargs)
        joints = self.joint_regressor(smpl_output)
        output = SMPLOutput(vertices=smpl_output.vertices,
                            global_orient=smpl_output.global_orient,
                            body_pose=smpl_output.body_pose,
                            joints=joints,
                            betas=smpl_output.betas,
                            full_pose=smpl_output.full_pose)
        return output


def convert_weak_perspective_to_perspective(
        weak_perspective_camera,
        focal_length=5000.,
        img_res=224,
):
    # Convert Weak Perspective Camera [s, tx, ty] to camera translation [tx, ty, tz]
    # in 3D given the bounding box size
    # This camera translation can be used in a full perspective projection
    perspective_camera = torch.stack(
        [
            weak_perspective_camera[:, 1],
            weak_perspective_camera[:, 2],
            2 * focal_length / (img_res * weak_perspective_camera[:, 0] + 1e-9)
        ],
        dim=-1
    )
    return perspective_camera


def perspective_projection(points, rotation, translation,
                           focal_length, camera_center):
    """
    This function computes the perspective projection of a set of points.
    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs,) or scalar: Focal length
        camera_center (bs, 2): Camera center
    """
    batch_size = points.shape[0]
    K = torch.zeros([batch_size, 3, 3], device=points.device)
    K[:, 0, 0] = focal_length
    K[:, 1, 1] = focal_length
    K[:, 2, 2] = 1.
    K[:, :-1, -1] = camera_center

    # Transform points
    points = torch.einsum('bij,bkj->bki', rotation, points)
    points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / points[:, :, -1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)

    return projected_points[:, :, :-1]


class SmplHead(LightningModule):
    def __init__(self, focal_length=5000., img_res=224, regressor_type='h36m'):
        super(SmplHead, self).__init__()
        self.smpl = SMPL(constants.SMPL_MODEL_DIR, create_transl=False, regressor_type='h36m')
        self.faces = torch.from_numpy(self.smpl.faces.astype('int32')[None])
        self.focal_length = focal_length
        self.img_res = img_res

    def forward(self, rotmat, shape, cam=None, normalize_joints2d=False):
        smpl_output = self.smpl(
            betas=shape,
            body_pose=rotmat[:, 1:].contiguous(),
            global_orient=rotmat[:, 0].unsqueeze(1).contiguous(),
            pose2rot=False,
        )

        output = {
            'smpl_vertices': smpl_output.vertices,
            'smpl_joints3d': smpl_output.joints,
        }

        if cam is not None:
            # CHANGE -14 to 25:39 to switch J regressor extra keypoints
            joints3d = smpl_output.joints
            batch_size = joints3d.shape[0]
            cam_t = convert_weak_perspective_to_perspective(cam)
            joints2d = perspective_projection(
                joints3d,
                rotation=torch.eye(3).unsqueeze(0).expand(batch_size, -1, -1).to(joints3d.device),
                translation=cam_t,
                focal_length=self.focal_length,
                camera_center=torch.zeros(batch_size, 2).to(joints3d.device)
            )
            if normalize_joints2d:
                # Normalize keypoints to [-1,1]
                joints2d = joints2d / (self.img_res / 2.)

            output['smpl_joints2d'] = joints2d
            output['pred_cam_t'] = cam_t

        return output
