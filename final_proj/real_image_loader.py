from itertools import repeat
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

class RealImageDataset(Dataset):
    def __init__(self, batch_size=64, size=64):
        super(Dataset, self).__init__()
        self.size = size
        self.batch_size = batch_size

        self.image_paths = list(Path('../datasets/mnist_png/').resolve().glob('*/*/*.png'))

    def __len__(self):
        return self.batch_size * 20

    def __getitem__(self, idx):
        image_id = np.random.randint(low=0, high=len(self.image_paths))
        # image_id = 46
        image = self.image_paths[image_id]
        label = float(image.parent.name)
        image = cv2.imread(str(image), cv2.IMREAD_COLOR)
        image = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_AREA)

        return 1 - (image / 255.0), label


def inf_loader_wrapper(data_loader):
    for loader in repeat(data_loader):
        for data in loader:
            yield data

def get_real_image_loader(batch_size=64):
    loader = DataLoader(
        RealImageDataset(batch_size=batch_size),
        batch_size=batch_size,
        num_workers=1,
        shuffle=True,
        worker_init_fn=lambda id: np.random.seed(torch.initial_seed() // 2**32 + id),
    )
    loader = inf_loader_wrapper(loader)
    return loader


if __name__ == '__main__':

    dataset = RealImageDataset()
    img, label = dataset[1]
    print()