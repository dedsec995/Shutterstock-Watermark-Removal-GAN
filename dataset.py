import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class WaterDataset(Dataset):
    def __init__(self, root_dir, image_size=(256, 256)):
        self.root_dir = root_dir
        self.image_size = image_size
        self.list_files = [
            f
            for f in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, f))
        ]
        self.transform = transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        folder_name = self.list_files[index]
        folder_path = os.path.join(self.root_dir, folder_name)

        input_image_path = os.path.join(folder_path, f"{folder_name}_watermark.jpg")
        target_image_path = os.path.join(folder_path, f"{folder_name}.jpg")

        input_image = Image.open(input_image_path).convert("RGB")
        target_image = Image.open(target_image_path).convert("RGB")

        input_image = self.transform(input_image)
        target_image = self.transform(target_image)

        return input_image, target_image
