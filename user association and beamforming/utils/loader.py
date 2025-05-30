import torch
from torch.utils.data import Dataset
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset, dataloader, distributed

class CustomDataset(Dataset):
    def __init__(self, txt_file, seed=None, cache=False, device='cpu'):
        # Set seed if provided
        if seed is not None:
            self.set_seed(seed)

        with open(txt_file, 'r') as file:
            self.file_paths = [path.strip() for path in file.readlines()]
        self.commun_paths = [
            file_path.replace('camera_tensor', 'commun').replace('.pth', '.npy')
            for file_path in self.file_paths
        ]

        # Initialize cache
        self.cache = cache
        self.data_cache = {}
        self.commun_cache = {}
        self.device = torch.device(device)  # Specify the device (default is CPU)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        if idx in self.data_cache:
            # Load from cache if already cached
            data = self.data_cache[idx]
            commun = self.commun_cache[idx]
        else:
            # Load from file system if not cached
            file_path = self.file_paths[idx]
            commun_path = self.commun_paths[idx]
            data = torch.load(file_path).to(self.device)  # Load and move to device
            commun = torch.from_numpy(np.load(commun_path, allow_pickle=True)).to(self.device)  # Load, convert to tensor, and move to device

            # Cache the loaded data if caching is enabled
            if self.cache:
                self.data_cache[idx] = data
                self.commun_cache[idx] = commun

        return data, commun, self.file_paths[idx]

    def set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)

class CustomDataset_RF(Dataset):
    def __init__(self, txt_file, seed=None, cache=False, device='cpu'):
        # Set seed if provided
        if seed is not None:
            self.set_seed(seed)

        with open(txt_file, 'r') as file:
            self.file_paths = [path.strip() for path in file.readlines()]
        self.commun_paths = [
            file_path.replace('camera_tensor', 'commun').replace('.pth', '.npy')
            for file_path in self.file_paths
        ]

        # Initialize cache
        self.cache = cache
        self.data_cache = {}
        self.commun_cache = {}
        self.device = torch.device(device)  # Specify the device (default is CPU)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        if idx in self.commun_cache:
            # Load from cache if already cached
            commun = self.commun_cache[idx]
        else:
            # Load from file system if not cached

            commun_path = self.commun_paths[idx]
            commun = torch.from_numpy(np.load(commun_path)).to(self.device)  # Load, convert to tensor, and move to device
            # Cache the loaded data if caching is enabled
            if self.cache:
                self.commun_cache[idx] = commun

        return commun, self.file_paths[idx]

    def set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)

class InfiniteDataLoader(dataloader.DataLoader):
    """
    Dataloader that reuses workers.

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self.iterator)

class _RepeatSampler:
    """
    Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)
