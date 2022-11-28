import torch
import torch.nn.functional as F

label = F.one_hot(torch.randint(0, 10, (1,)), num_classes=10)
print()