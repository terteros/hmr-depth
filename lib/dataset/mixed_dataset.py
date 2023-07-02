import numpy as np
from torch.utils.data import ConcatDataset, Dataset, WeightedRandomSampler


class MixedDataset(Dataset):
    def __init__(self,
                 datasets: list,
                 partition: list,
                 num_data = None):
        """Load data from multiple datasets."""
        assert min(partition) >= 0
        self.dataset = ConcatDataset(datasets)
        if num_data is not None:
            self.length = num_data
        else:
            self.length = max(len(ds) for ds in datasets)
        weights = [
            np.ones(len(ds)) * p / len(ds)
            for (p, ds) in zip(partition, datasets)
        ]
        weights = np.concatenate(weights, axis=0)
        self.sampler = WeightedRandomSampler(weights, 1)
        self.source_datasets = datasets

    def __len__(self):
        """Get the size of the dataset."""
        return self.length

    def __getitem__(self, idx):
        """Given index, sample the data from multiple datasets with the given
        proportion."""
        idx_new = list(self.sampler)[0]
        return self.dataset[idx_new]

    def get_debug_image(self, batch):
        return self.source_datasets[0].get_debug_image(batch)