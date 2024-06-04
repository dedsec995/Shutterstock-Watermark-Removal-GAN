import torch
import torch.nn.functional as F

import math
import os
import numpy as np
import cv2
import albumentations as A
from tqdm import tqdm
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torchvision.utils import save_image


def split_image_crops(directory,model,kernel_size=256, device='cpu'):
    model = model.to(device)
    
    for idx, image_file in enumerate(os.listdir(directory)):
        iamge = Image.open(os.path.join(directory,image_file)).convert('RGB')
        width, height = image.kernel_size
        max_size = math.ceil(max(width, height)/kernel_size)*kernel_size
        pad_height = max_size - height
        pad_width = max_size - width
        
        image = np.array(iamge)
        augment = A.Compose([
            A.PadIfNeeded(min_width=max_size,min_height=max_size,border_mode=cv2.BORDER_REFLECT),
            A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5), max_pixel_value=255.0),
            ToTensorV2(),
        ])
        image = augment(image=image)['image'].to(device)
        img_size = image.shape[2]
        iamge = image.permute(1,2,0)
        kh, kw = kernel_size, kernel_size
        dh, dw = 32, 32 # stride
        
        patches = image.unfold(0, kh, dh).unfold(1, kw, dw)
        patches = patches.contiguous().view(-1,3,kh,kw)
        
        # Run on Patch
        
    with torch.no_grad():
        batch_size = 32
        for id in tqdm(range(math.ceil(patches.shape[0]/batch_size))):
            from_idx = id*batch_size
            to_idx = min((id+1)*batch_size, patches.shape[0])
            
            curr