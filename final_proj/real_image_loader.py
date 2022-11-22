from pathlib import Path
import random
from itertools import repeat
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
     
        
def get_loader_iterator(data_loader):
    for loader in repeat(data_loader):
        for data in loader:
            yield data

def get_real_image_loader(dataset, batch_size=64):

    class RealImageDataset(Dataset):
        def __init__(self, dataset, batch_size=64, size=64):
            super(Dataset, self).__init__()

            self.batch_size = batch_size
            if dataset == 'mnist':
                self.size = size
                dataset_root = Path('../datasets/mnist_png/')
                sub_dirs = dataset_root.resolve().glob('*/4/*.png')
                self.image_paths = list(sub_dirs)
                self.image_num = len(self.image_paths)
            else:
                assert False, 'unknown dataset'

            self.dataset = dataset

        def __len__(self):
            return self.batch_size * 20

        def __getitem__(self, idx):
            ## choose a random image
            random_id = random.randrange(0, self.image_num)
            image = self.image_paths[random_id]
            image = cv2.imread(str(image), cv2.IMREAD_COLOR)
            return image / 255.0

    loader = DataLoader(
        RealImageDataset(dataset=dataset, batch_size=batch_size),
        batch_size=batch_size,
        num_workers=1,
        shuffle=True,
        worker_init_fn=lambda id: np.random.seed(torch.initial_seed() // 2**32 + id),
    )
    return get_loader_iterator(loader)

if __name__ == '__main__':
    # seed_loader = DataLoader(
    #     SeedDataset(pad_target=pad_target, seed_shape=(72,72,16)),
    #     batch_size=8,
    #     num_workers=4,
    #     shuffle=True,
    #     worker_init_fn=lambda id: np.random.seed(torch.initial_seed() // 2**32 + id),
    # )
    # seed_loader = inf_loader_wrapper(seed_loader)
    pass
    