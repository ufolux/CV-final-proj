from itertools import repeat
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

class ImageDataset(Dataset):
    def __init__(self, dataset, num=0, batch_size=64, size=64):
        super(Dataset, self).__init__()
        self.size = size
        self.batch_size = batch_size

        self.label = num
        self.image_paths = list(Path('../datasets/mnist_png/').resolve().glob(f'*/{num}/*.png'))
        self.dataset = dataset

    def __len__(self):
        return self.batch_size * 20

    def __getitem__(self, idx):
        image_id = np.random.randint(low=0, high=len(self.image_paths))
        # image_id = 46
        image = self.image_paths[image_id]
        image = cv2.imread(str(image), cv2.IMREAD_COLOR)

        image = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_AREA)

        return 1 - (image / 255.0), self.label


def inf_loader_wrapper(data_loader):
    for loader in repeat(data_loader):
        for data in loader:
            yield data

def get_image_loader_dict(dataset, batch_size=64):
    loader_dict = {}
    for i in range(1, 6):
        loader = DataLoader(
            ImageDataset(dataset=dataset, num=i, batch_size=batch_size),
            batch_size=batch_size,
            num_workers=1,
            shuffle=True,
            worker_init_fn=lambda id: np.random.seed(torch.initial_seed() // 2 ** 32 + id),
        )
        loader = inf_loader_wrapper(loader)
        loader_dict[f'{i}'] = loader

    return loader_dict

