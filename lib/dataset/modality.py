import io
import os.path

import cv2
import h5py
import joblib
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Normalize
import kp_utils
from .image_utils import crop_cv2, flip_img, transform, rot_aa, flip_kp, flip_pose
from lib import constants


def to_cv2(img: torch.Tensor):
    img = img * torch.tensor([0.229, 0.224, 0.225], device=img.device).reshape(3, 1, 1)
    img = img + torch.tensor([0.485, 0.456, 0.406], device=img.device).reshape(3, 1, 1)
    cv2_im = (img.permute((1, 2, 0)).detach().cpu().numpy() * 255).astype(np.uint8)
    cv2_im = cv2.cvtColor(cv2_im, cv2.COLOR_RGB2BGR)
    return cv2_im


class Modality:
    def __init__(self, index_record, data_root=None):
        self.index_record = index_record
        self.data_root = data_root
    def getitem(self, index, augmentation_params) -> dict:
        raise NotImplementedError

    def get_image(self, input_img, data):
        return to_cv2(input_img).copy()


class ImageModality(Modality):
    def __init__(self, index_record, data_root, options):
        super().__init__(index_record, data_root)
        self.options = options
        self.normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)

    def rgb_processing(self, rgb_img, center, scale, rot, flip, pn):
        """Process rgb image and do augmentation."""

        rgb_img = crop_cv2(rgb_img, center, scale,
                           [self.options.DATASET.IMG_RES, self.options.DATASET.IMG_RES], rot=rot)

        # flip the image
        if flip:
            rgb_img = flip_img(rgb_img)

        # in the rgb image we add pixel noise in a channel-wise manner
        rgb_img[:, :, 0] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, 0] * pn[0]))
        rgb_img[:, :, 1] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, 1] * pn[1]))
        rgb_img[:, :, 2] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, 2] * pn[2]))
        # (3,224,224),float,[0,1]

        rgb_img = np.transpose(rgb_img.astype('float32'), (2, 0, 1)) / 255.0
        return rgb_img

    def getitem(self, index, augmentation_params):
        scale, center, flip, pn, rot, sc = augmentation_params
        img_path = str(self.index_record[index])
        if img_path.startswith('hdf5'):
            hdf5_path, img_path_in_hdf5 = img_path.split('@')
            hdf5_path = f'{self.data_root}/{hdf5_path}'
            # TODO: remove this!
            try:
                hf = h5py.File(hdf5_path,'r')
                binary = np.array(hf[img_path_in_hdf5])
                img = np.asarray(Image.open(io.BytesIO(binary)))
            except:
                return {'image_path': img_path, 'img': torch.zeros(3,224,224)}
        else:
            img = np.asarray(Image.open(f'{self.data_root}/{img_path}'))



        img = self.rgb_processing(img, center, sc * scale, rot, flip, pn)
        img = torch.from_numpy(img).float()
        img = self.normalize_img(img)
        return {'image_path': img_path, 'img': img}


class Keypoints2D(Modality):
    def __init__(self, index_record, options):
        self.index_record = index_record
        self.options = options

    def j2d_processing(self, kp, center, scale, r, f):
        """Process gt 2D keypoints and apply all augmentation transforms."""
        nparts = kp.shape[0]

        for i in range(nparts):
            kp[i, 0:2] = transform(kp[i, 0:2], center, scale,
                                   [self.options.DATASET.IMG_RES, self.options.DATASET.IMG_RES], rot=r)
        # convert to normalized coordinates
        kp[:, :-1] = 2. * kp[:, :-1] / self.options.DATASET.IMG_RES - 1.
        # flip the x coordinates
        if f:
            kp = flip_kp(kp)
        kp = kp.astype('float32')
        return kp

    def getitem(self, index, augmentation_params):
        scale, center, flip, pn, rot, sc = augmentation_params
        kp_2d = self.index_record['joints2D'][index].copy()
        kp_2d = self.j2d_processing(kp_2d, center, sc * scale, rot, flip)
        kp_3d = torch.from_numpy(kp_2d).float()
        return {'kp_2d': kp_2d}

    def get_image(self, input_img, kp_2d):
        kp_img = super().get_image(input_img, None)
        kp_2d = kp_2d.cpu().numpy()
        kp_2d[:, :2] = 0.5 * 224 * (kp_2d[:, :2] + 1)  # normalize_2d_kp(kp_2d[:,:2], 224, inv=True)
        #kp_2d = np.hstack([kp_2d, np.ones((kp_2d.shape[0], 1))])
        kp_2d[:, 2] = kp_2d[:, 2] > 0.3
        kp_2d = np.array(kp_2d, dtype=int)
        kp_2d = kp_2d[25:39]

        rcolor = [255, 0, 0]
        pcolor = [0, 255, 0]
        lcolor = [0, 0, 255]

        skeleton = kp_utils.get_common_skeleton()
        joints = kp_utils.get_common_joint_names()

        # common_lr = [0,0,1,1,0,0,0,0,1,0,0,1,1,1,0]
        for idx, pt in enumerate(kp_2d):
            if pt[2] > 0: # if visible
                cv2.circle(kp_img, (pt[0], pt[1]), 4, pcolor, -1)
                cv2.putText(kp_img, f'{idx}', (pt[0]+1, pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

        for i, (j1, j2) in enumerate(skeleton):
            if kp_2d[j1, 2] > 0 and kp_2d[j2, 2] > 0: # if visible
            # if dataset == 'common':
            #     color = rcolor if common_lr[i] == 0 else lcolor
            # else:
                color = lcolor if i % 2 == 0 else rcolor
                pt1, pt2 = (kp_2d[j1, 0], kp_2d[j1, 1]), (kp_2d[j2, 0], kp_2d[j2, 1])

                cv2.line(kp_img, pt1=pt1, pt2=pt2, color=color, thickness=2)
        return kp_img


class Keypoints3D(Modality):
    def __init__(self, index_record, options):
        self.index_record = index_record
        self.options = options

    @staticmethod
    def j3d_processing(self, S, r, f):
        """Process gt 3D keypoints and apply all augmentation transforms."""
        # in-plane rotation
        rot_mat = np.eye(3)
        if not r == 0:
            rot_rad = -r * np.pi / 180
            sn, cs = np.sin(rot_rad), np.cos(rot_rad)
            rot_mat[0, :2] = [cs, -sn]
            rot_mat[1, :2] = [sn, cs]
        S[:, :-1] = np.einsum('ij,kj->ki', rot_mat, S[:, :-1])
        # flip the x coordinates
        if f:
            S = flip_kp(S)
        S = S.astype('float32')
        return S

    def getitem(self, index, augmentation_params):
        scale, center, flip, pn, rot, sc = augmentation_params
        kp_3d = self.index_record['joints3D'][index].copy()
        kp_3d = torch.from_numpy(kp_3d).float()
        return {'kp_3d': kp_3d}

    def get_image(self, input_img, kp_3d):
        kp_img = super().get_image(input_img, None)
        kp_img = np.ones_like(kp_img) * 255
        kp_2d = kp_3d[25:39]
        print("MEAN KP DEPTH: ", kp_2d[:, 2].mean())
        kp_2d = kp_2d.cpu().numpy()
        kp_2d[:, 0] = kp_2d[:,0] - kp_2d[:,0].min()
        kp_2d[:, 1] = kp_2d[:, 1] - kp_2d[:, 1].min()
        kp_2d *= 224. / kp_2d[:,:2].max()
        kp_2d[:,2] = 1
        kp_2d = kp_2d.astype(np.int)
        #breakpoint()

        #breakpoint()
        rcolor = [255, 0, 0]
        pcolor = [0, 255, 0]
        lcolor = [0, 0, 255]

        skeleton = kp_utils.get_common_skeleton()
        joints = kp_utils.get_common_joint_names()

        # common_lr = [0,0,1,1,0,0,0,0,1,0,0,1,1,1,0]
        for idx, pt in enumerate(kp_2d):
            if pt[2] > 0: # if visible
                cv2.circle(kp_img, (pt[0], pt[1]), 4, pcolor, -1)
                cv2.putText(kp_img, f'{idx}', (pt[0]+1, pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

        for i, (j1, j2) in enumerate(skeleton):
            if kp_2d[j1, 2] > 0 and kp_2d[j2, 2] > 0: # if visible
            # if dataset == 'common':
            #     color = rcolor if common_lr[i] == 0 else lcolor
            # else:
                color = lcolor if i % 2 == 0 else rcolor
                pt1, pt2 = (kp_2d[j1, 0], kp_2d[j1, 1]), (kp_2d[j2, 0], kp_2d[j2, 1])

                cv2.line(kp_img, pt1=pt1, pt2=pt2, color=color, thickness=2)
        return kp_img


class SmplModality(Modality):
    def __init__(self, index_record):
        self.index_record = index_record

    @staticmethod
    def pose_processing(pose, r, f):
        """Process SMPL theta parameters  and apply all augmentation transforms."""
        # rotation or the pose parameters
        pose[:3] = rot_aa(pose[:3], r)
        # flip the pose parameters
        if f:
            pose = flip_pose(pose)
        # (72),float
        pose = pose.astype('float32')
        return pose

    def getitem(self, index, augmentation_params):
        scale, center, flip, pn, rot, sc = augmentation_params
        pose = self.index_record['pose'][index].copy()
        pose = self.pose_processing(pose, rot, flip)
        shape = self.index_record['shape'][index].copy()
        item = {}
        item['pose'] = torch.from_numpy(pose).float()
        item['betas'] = torch.from_numpy(shape).float()
        return item


class DensePoseModality(Modality):
    def __init__(self, index_record, data_root, dp_size=(256, 256)):
        super().__init__(index_record, data_root)
        self.dp_size = dp_size

    def getitem(self, index, augmentation_params):
        img_path = self.index_record[index]
        img_densepose = np.zeros(self.dp_size + (3,), dtype='uint8')
        valid = False
        if img_path.startswith('hdf5'):
            hdf5_path, dp_path_in_hdf5 = img_path.split('@')
            hdf5_path = f'{self.data_root}/{hdf5_path}'
            hf = h5py.File(hdf5_path, 'r')
            if dp_path_in_hdf5 in hf:
                binary = np.array(hf[dp_path_in_hdf5])
                img_densepose = np.asarray(Image.open(io.BytesIO(binary)))[...,::-1]
                if np.count_nonzero(img_densepose[..., 0]) > 1000:
                    valid = True
        else:
            # TODO: support imgfile read for densepose
            pass
        img_densepose = torch.from_numpy(img_densepose.copy()).permute(2, 0, 1)
        return {'dp_valid': valid, 'dp': img_densepose}

    def get_image(self, input_img, data):
        input_img = super().get_image(input_img, None)
        densepose_img = np.transpose(data.cpu().detach().numpy().copy(), (1, 2, 0)).astype('uint8')
        densepose_img[densepose_img[..., 0] == 0] = 255
        densepose_img = cv2.resize(densepose_img, (input_img.shape[1], input_img.shape[0]))
        return densepose_img


class DepthModality(Modality):
    def __init__(self, index_record, data_root, depth_size=(256, 256)):
        super().__init__(index_record, data_root)
        self.depth_size = depth_size

    def getitem(self, index, augmentation_params):
        img_path = self.index_record[index]
        valid = False
        if img_path.startswith('hdf5'):
            hdf5_path, depth_path_in_hdf5 = img_path.split('@')
        hdf5_path = f'{self.data_root}/{hdf5_path}'
        # breakpoint()
        hf = h5py.File(hdf5_path, 'r')
        if depth_path_in_hdf5 in hf:
            binary = np.array(hf[depth_path_in_hdf5])
            depth_data = np.load(io.BytesIO(binary))
            depth_tensor = cv2.resize(depth_data['depth'], self.depth_size)
            f = depth_data.get('f', np.zeros(1) + self.depth_size[0] * 2).astype(np.float32)
            c = depth_data.get('c', np.zeros(2) + self.depth_size[0] / 2).astype(np.float32)
            valid = True
        else:
            depth_tensor = np.zeros(self.depth_size, dtype=np.float32)
            f = np.zeros(1, dtype=np.float32) + self.depth_size[0] * 2
            c = np.zeros(2, dtype=np.float32) + self.depth_size[0] / 2
            valid = False
        depth_tensor = torch.from_numpy(depth_tensor)
        f = torch.from_numpy(f)
        c = torch.from_numpy(c)
        return {'depth_valid': valid,
                'depth': depth_tensor,
                'depth_f': f,
                'depth_c': c}

    def get_image(self, input_img, data):
        input_img = super().get_image(input_img, None)
        depth_img = data.cpu().detach().numpy().copy()
        bg_mask = depth_img == 0
        depth_img[bg_mask] = depth_img.max()
        depth_img = depth_img - depth_img.min()
        depth_img = (depth_img / depth_img.max() * 255.).astype(np.uint8)
        depth_img = cv2.applyColorMap(depth_img, cv2.COLORMAP_JET)
        depth_img[bg_mask] = 255
        heatmap = cv2.resize(depth_img, (input_img.shape[1], input_img.shape[0]))
        return heatmap
