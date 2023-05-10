from lib.models.smpl_head import SmplHead
from lib.constants import SMPL_MEAN_PARAMS
import numpy as np
import torch

device = 'cuda'
smpl = SmplHead(img_res=224).to(device)

batch_size = 1
mean_params = np.load(SMPL_MEAN_PARAMS)
init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0).expand(batch_size, -1)
init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0).expand(batch_size, -1)
init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0).expand(batch_size, -1)

def rot6d_to_rotmat(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Input:
        (B,6) Batch of 6-D rotation representations
    Output:
        (B,3,3) Batch of corresponding rotation matrices
    """
    import torch.nn.functional as F
    x = x.reshape(-1, 3, 2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)

init_pose = rot6d_to_rotmat(init_pose).view(batch_size, 24, 3, 3)
smpl_output = smpl(
    rotmat=init_pose.to(device),
    shape=init_shape.to(device),
    cam=init_cam.to(device),
    normalize_joints2d=True,
)

breakpoint()