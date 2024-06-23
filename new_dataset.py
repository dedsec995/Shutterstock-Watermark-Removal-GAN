import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class WatermarkedDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.image_pairs = []
        for subdir in os.listdir(root_dir):
            subdir_path = os.path.join(root_dir, subdir)
            if os.path.isdir(subdir_path):
                base_name = subdir
                clean_image_path = os.path.join(subdir_path, f"{base_name}.jpg")
                watermarked_image_path = os.path.join(
                    subdir_path, f"{base_name}_watermark.jpg"
                )
                if os.path.exists(clean_image_path) and os.path.exists(
                    watermarked_image_path
                ):
                    self.image_pairs.append((clean_image_path, watermarked_image_path))

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        clean_image_path, watermarked_image_path = self.image_pairs[idx]

        clean_image = Image.open(clean_image_path).convert("RGB")
        watermarked_image = Image.open(watermarked_image_path).convert("RGB")

        if self.transform:
            clean_image = self.transform(clean_image)
            watermarked_image = self.transform(watermarked_image)

        return watermarked_image, clean_image


# Example usage:
# transform = transforms.Compose(
#     [
#         transforms.ToTensor(),
#         transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
#     ]
# )

# dataset = WatermarkedDataset(root_dir="path_to_root_directory", transform=transform)
# loader = DataLoader(
#     dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True
# )
