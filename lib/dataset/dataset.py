import io

import cv2
import h5py
import joblib
import numpy as np
import torch
import tqdm
from PIL import Image
from torch.utils.data import Dataset

from lib.cfg import parse_args
from lib.misc import select_batches
from lib.dataset.modality import *


def parse_image_path(img_path: str):
    split = img_path.split('/')
    subject = split[-3]
    action, cam_id = split[-2].split('.')
    frame_id = split[-1].split('.')[0]
    return subject, action, cam_id, frame_id


class ModularBaseDataset(Dataset):
    def __init__(self, db_file_path, options, frame_skip=1):
        self.options = options
        self.db = joblib.load(db_file_path)
        if frame_skip is None:
            frame_skip = options.DATASET.FRAME_SKIP
        def not_skip(img_name):
            id = int(str(img_name).split('/')[-1].split('.')[0].replace('image_',''))
            return (id-1) % frame_skip == 0
        frame_skip_mask = np.vectorize(not_skip)(self.db['image_path'])
        self.db = select_batches(self.db, frame_skip_mask)

        self.modalities = {}

    def generate_post_processing_params(self, item_idx):
        return None

    def __len__(self):
        return self.db['image_path'].shape[0]

    def __getitem__(self, item_idx):
        post_processing_params = self.generate_post_processing_params(item_idx)
        item = {}
        for modality in self.modalities.values():
            item.update(modality.getitem(item_idx, post_processing_params))
        return item

    def get_debug_image(self, batch):
        batch_imgs = []
        for i in range(batch['img'].shape[0]):
            imgs = [mod.get_image(batch['img'][i], batch[mod_key][i] if mod_key in batch else None) for mod_key, mod in
                    self.modalities.items()]
            batch_imgs.append(np.hstack(imgs))
        return np.vstack(batch_imgs)


class Dataset3D(ModularBaseDataset):
    def __init__(self, db_file_path, data_root, options, hdf5=True, frame_skip=1, use_augmentation=False):
        super().__init__(db_file_path, options, frame_skip=frame_skip)
        self.use_augmentation = use_augmentation
        self.modalities = {
            'img': ImageModality(self.db['image_path_hdf'] if hdf5 else self.db['image_path'], data_root, options),
            'kp_2d': Keypoints2D(self.db, options),
            'kp_3d': Keypoints3D(self.db, options),
        }
        if 'pose' in self.db.keys():
            self.modalities['smpl'] = SmplModality(self.db)

    def generate_augmentation_params(self):
        """Get augmentation parameters."""
        flip = 0  # flipping
        pn = np.ones(3)  # per channel pixel-noise
        rot = 0  # rotation
        sc = 1  # scaling
        if self.use_augmentation:
            # We flip with probability 1/2
            if np.random.uniform() <= 0.5:
                flip = 1

            # Each channel is multiplied with a number
            # in the area [1-opt.noiseFactor,1+opt.noiseFactor]
            pn = np.random.uniform(1 - self.options.NOISE_FACTOR, 1 + self.options.NOISE_FACTOR, 3)

            # The rotation is a number in the area [-2*rotFactor, 2*rotFactor]
            rot = min(2 * self.options.ROT_FACTOR,
                      max(-2 * self.options.ROT_FACTOR, np.random.randn() * self.options.ROT_FACTOR))

            # The scale is multiplied with a number
            # in the area [1-scaleFactor,1+scaleFactor]
            sc = min(1 + self.options.SCALE_FACTOR,
                     max(1 - self.options.SCALE_FACTOR, np.random.randn() * self.options.SCALE_FACTOR + 1))
            # but it is zero with probability 3/5
            if np.random.uniform() <= 0.6:
                rot = 0

        return flip, pn, rot, sc

    def generate_post_processing_params(self, item_idx):
        scale = max(self.db['bbox_scale'][item_idx].copy())
        center = self.db['bbox_center'][item_idx].copy()
        return (scale, center) + self.generate_augmentation_params()


class DatasetDepth(Dataset3D):
    def __init__(self, db_file_path, data_root, options, hdf5=True, frame_skip=1, use_augmentation=False):
        super().__init__(db_file_path, data_root, options, hdf5, frame_skip=frame_skip, use_augmentation=use_augmentation)
        self.modalities['dp'] = DensePoseModality(self.db['dp_path_hdf'], data_root, dp_size=(256, 256))
        self.modalities['depth'] = DepthModality(self.db['depth_path_hdf'], data_root, depth_size=(256, 256))


if __name__ == '__main__':
    cfg, cfg_file = parse_args()

    dataset = DatasetDepth(f'./data_new/h36m/h36m_test.pt',f'./data_new/h36m', cfg, frame_skip=10, use_augmentation=False)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0
    )

    pbar = tqdm.tqdm(len(dataset))
    for idx, batch in enumerate(dataloader):
        pbar.update(cfg.TRAIN.BATCH_SIZE)
        img_cv2 = dataset.get_debug_image(batch)
        cv2.imwrite('test.png', img_cv2)
        # breakpoint()
        exit(0)
