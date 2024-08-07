import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms.functional import crop

class WaterDataset(Dataset):
    def __init__(self, root_dir, transform=None, patch_size=256):
        self.root_dir = root_dir
        self.subdirs = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
        self.transform = transform
        self.patch_size = patch_size

    def __len__(self):
        return len(self.subdirs)

    def __getitem__(self, index):
        subdir = self.subdirs[index]
        input_img_path = os.path.join(self.root_dir, subdir, f"{subdir}_watermark.jpg")
        target_img_path = os.path.join(self.root_dir, subdir, f"{subdir}.jpg")

        input_image = Image.open(input_img_path).convert("RGB")
        target_image = Image.open(target_img_path).convert("RGB")

        # Ensure both images have the same dimensions
        assert input_image.size == target_image.size, "Input and target images must have the same dimensions."

        # Apply random crop to both images
        i, j, h, w = transforms.RandomCrop.get_params(input_image, output_size=(self.patch_size, self.patch_size))
        input_image = crop(input_image, i, j, h, w)
        target_image = crop(target_image, i, j, h, w)

        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)

        return input_image, target_image
