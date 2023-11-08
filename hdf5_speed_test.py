import h5py
import time
import numpy as np
import joblib
import torch
from torch.utils.data import Dataset, DataLoader
class HdfTest(Dataset):
    def __init__(self):
        self.hdf5_path = 'data/3dpw/hdf5/3dpw_curated.hdf5'
        self.db = joblib.load('data/3dpw/3dpw_train.pt')

    def __len__(self):
        return self.db['image_path'].shape[0]

    def __getitem__(self, item_idx):
        img_densepose = np.zeros((256,256,3), dtype='uint8')
        with h5py.File(self.hdf5_path, 'r') as hf:
            if 'x' in hf:
                pass
        return torch.from_numpy(img_densepose.copy()).permute(2, 0, 1)

loader = DataLoader(
                dataset=HdfTest(),
                batch_size=1,
                shuffle=True,
                num_workers=0 #cfg.NUM_WORKERS
            )

start = time.time()
for i, b in enumerate(loader):
    if i % 50 == 0:
        print(i)  # wandb.log({'batch_nb': i})
    if i > 500:
        break
print(time.time() - start)


hdf5_path = 'data/h36m/hdf5/h36m_curated.hdf5'
start = time.time()
for i in range(2000):
    hf = h5py.File(hdf5_path, 'r')
    if 'x' in hf:
        pass

print(time.time() - start)