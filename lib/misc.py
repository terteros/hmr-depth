from typing import Union
import numpy
import numpy as np
import torch


def to_numpy(arr: Union[numpy.ndarray, torch.Tensor, dict[Union[numpy.ndarray, torch.Tensor]]]) -> numpy.ndarray:
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    elif isinstance(arr, numpy.ndarray):
        return arr
    elif isinstance(arr, dict):
        return {k: to_numpy(v) for k, v in arr.items()}
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

        new_batch = {k: select_data(v) for k, v in batch.items()}
    return new_batch


def numpy_imgrid(imgs: list, ncols=3):
    if not isinstance(imgs, np.ndarray):
        imgs = np.array(imgs)
    nindex, height, width, intensity = imgs.shape
    nrows = nindex // ncols
    assert nindex == nrows * ncols
    # want result.shape = (height*nrows, width*ncols, intensity)
    result = (imgs.reshape(nrows, ncols, height, width, intensity)
              .swapaxes(1, 2)
              .reshape(height * nrows, width * ncols, intensity))
    return result
