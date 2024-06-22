import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class WatermarkedDataset(Dataset):
    def __init__(self, watermarked_dir, clean_dir, transform=None):
        self.watermarked_dir = watermarked_dir
        self.clean_dir = clean_dir
        self.transform = transform

        self.watermarked_images = sorted(os.listdir(watermarked_dir))
        self.clean_images = sorted(os.listdir(clean_dir))

    def __len__(self):
        return len(self.watermarked_images)

    def __getitem__(self, idx):
        watermarked_image_path = os.path.join(
            self.watermarked_dir, self.watermarked_images[idx]
        )
        clean_image_path = os.path.join(self.clean_dir, self.clean_images[idx])

        watermarked_image = Image.open(watermarked_image_path).convert("RGB")
        clean_image = Image.open(clean_image_path).convert("RGB")

        if self.transform:
            watermarked_image = self.transform(watermarked_image)
            clean_image = self.transform(clean_image)

        return watermarked_image, clean_image

# Example Usage
# transform = transforms.Compose(
#     [
#         transforms.ToTensor(),
#         transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
#     ]
# )

# dataset = WatermarkedDataset(
#     watermarked_dir="path_to_watermarked_images",
#     clean_dir="path_to_clean_images",
#     transform=transform,
# )
