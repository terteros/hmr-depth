import logging
import os
import shutil
import time
from typing import Union

import numpy
import numpy as np
import torch
import yaml
from distinctipy import distinctipy


def create_logger(logdir, phase='train'):
    os.makedirs(logdir, exist_ok=True)

    log_file = os.path.join(logdir, f'{phase}_log.txt')

    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=log_file,
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_dict_to_yaml(obj, filename, mode='w'):
    with open(filename, mode) as f:
        yaml.dump(obj, f, default_flow_style=False)


def prepare_output_dir(cfg, cfg_file):
    # ==== create logdir
    logtime = time.strftime('%d-%m-%Y_%H-%M-%S')
    logdir = f'{logtime}_{cfg.EXP_NAME}'

    logdir = os.path.join(cfg.OUTPUT_DIR, logdir)
    os.makedirs(logdir, exist_ok=True)
    shutil.copy(src=cfg_file, dst=os.path.join(cfg.OUTPUT_DIR, 'config.yaml'))

    cfg.LOGDIR = logdir

    # save config
    save_dict_to_yaml(cfg, os.path.join(cfg.LOGDIR, 'config.yaml'))

    return cfg


def create_part_img(part_silhouettes):
    batch_size = part_silhouettes.shape[1]
    part_count = part_silhouettes.shape[0]
    part_colors = distinctipy.get_colors(part_count)

    img_res = part_silhouettes.shape[-1]

    imgs = np.zeros((batch_size, img_res, img_res, 3))
    for batch_id in range(batch_size):
        for part_idx in range(part_count):
            part_sil=part_silhouettes[part_idx][batch_id].repeat(3, 1, 1).cpu().numpy().transpose((1, 2, 0))
            imgs[batch_id][part_sil > 0] = 0
            breakpoint()


def to_numpy(arr: Union[numpy.ndarray, torch.Tensor, dict[Union[numpy.ndarray, torch.Tensor]]]) -> numpy.ndarray:
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    elif isinstance(arr, numpy.ndarray):
        return arr
    elif isinstance(arr, dict):
        return {k:to_numpy(v) for k,v in arr.items()}
    else:
        raise TypeError()


def select_batches(batch, mask):
    if isinstance(mask, int):
        new_batch = {k: v[mask] for k, v in batch.items()}
    else:
        def select_data(data):
            if isinstance(data, list):
                return [data[i] for i in range(len(data)) if mask[i]]
            else:
                return data[mask]
        new_batch = {k:select_data(v) for k,v in batch.items()}
    return new_batch

def numpy_imgrid(imgs: list, ncols=3):
    if not isinstance(imgs, np.ndarray):
        imgs = np.array(imgs)
    nindex, height, width, intensity = imgs.shape
    nrows = nindex//ncols
    assert nindex == nrows*ncols
    # want result.shape = (height*nrows, width*ncols, intensity)
    result = (imgs.reshape(nrows, ncols, height, width, intensity)
              .swapaxes(1,2)
              .reshape(height*nrows, width*ncols, intensity))
    return result