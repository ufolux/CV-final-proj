import torch
from torchvision.utils import make_grid, save_image
from torchvision.io import read_image
import os
import matplotlib.pyplot as plt
import cv2

sample_dir = os.getcwd() + '\\samples'
images = []
for i in range(8):
    for j in range(1, 6):
        #img = read_image(sample_dir+'\\'+str(j)+'\\'+str(i)+'.png')
        img = cv2.imread(sample_dir+'\\'+str(j)+'\\'+str(i)+'.png', cv2.IMREAD_COLOR)
        images.append(torch.FloatTensor(img).permute(2,0,1))
grid = make_grid(torch.stack(images, 0), nrow=5)
#plt.imshow(grid.permute(1, 2, 0))
save_image(grid, sample_dir+"\\"+'demo.png', normalize=True)
