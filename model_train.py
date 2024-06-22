import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torchvision.transforms as transforms
from tqdm import tqdm


# Hyperparameters
lr = 0.0002
batch_size = 16
num_epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
patch_size = 256
stride = 32

# Initialize models
generator = Generator(in_channels=3, out_channels=3).to(device)
discriminator = Discriminator().to(device)
vgg_loss = VGGLoss().to(device)

# Optimizers
opt_gen = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
opt_disc = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# Loss functions
bce_loss = nn.BCEWithLogitsLoss()
l1_loss = nn.L1Loss()

# Dataset and DataLoader
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
)

dataset = WatermarkedDataset(
    watermarked_dir="path_to_watermarked_images",
    clean_dir="path_to_clean_images",
    transform=transform,
)
loader = DataLoader(
    dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
)


# Helper function to split image into patches
def extract_patches(image, patch_size, stride):
    b, c, h, w = image.size()
    patches = image.unfold(2, patch_size, stride).unfold(3, patch_size, stride)
    patches = patches.contiguous().view(b, c, -1, patch_size, patch_size)
    patches = (
        patches.permute(0, 2, 1, 3, 4).contiguous().view(-1, c, patch_size, patch_size)
    )
    return patches


# Helper function to reassemble patches into an image
def reassemble_patches(patches, image_size, patch_size, stride):
    b, c, h, w = image_size
    patches = patches.view(b, -1, c, patch_size, patch_size).permute(0, 2, 1, 3, 4)
    output = F.fold(
        patches.view(b * c, -1, patch_size * patch_size),
        output_size=(h, w),
        kernel_size=patch_size,
        stride=stride,
    )
    recovery_mask = F.fold(
        torch.ones_like(patches).view(b * c, -1, patch_size * patch_size),
        output_size=(h, w),
        kernel_size=patch_size,
        stride=stride,
    )
    output = output / recovery_mask
    output = output.view(b, c, h, w)
    return output


# Training loop
for epoch in range(num_epochs):
    for idx, (watermarked_imgs, clean_imgs) in enumerate(
        tqdm(loader, desc=f"Epoch {epoch}/{num_epochs}")
    ):
        watermarked_imgs = watermarked_imgs.to(device)
        clean_imgs = clean_imgs.to(device)

        # Extract patches
        watermarked_patches = extract_patches(watermarked_imgs, patch_size, stride)
        clean_patches = extract_patches(clean_imgs, patch_size, stride)

        # Train Discriminator
        fake_patches = generator(watermarked_patches)
        disc_real = discriminator(clean_patches, watermarked_patches)
        disc_fake = discriminator(fake_patches.detach(), watermarked_patches)

        loss_disc_real = bce_loss(disc_real, torch.ones_like(disc_real))
        loss_disc_fake = bce_loss(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real + loss_disc_fake) / 2

        opt_disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        # Train Generator
        disc_fake = discriminator(fake_patches, watermarked_patches)
        loss_gen_gan = bce_loss(disc_fake, torch.ones_like(disc_fake))
        loss_gen_l1 = l1_loss(fake_patches, clean_patches) * 100
        loss_gen_vgg = vgg_loss(fake_patches, clean_patches) * 10
        loss_gen = loss_gen_gan + loss_gen_l1 + loss_gen_vgg

        opt_gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        if idx % 100 == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch [{idx}/{len(loader)}] \
                  Loss D: {loss_disc.item():.4f}, loss G: {loss_gen.item():.4f}"
            )

    # Save some example images
    if epoch % 10 == 0:
        # Reassemble patches into images for visualization
        fake_imgs = reassemble_patches(
            fake_patches, watermarked_imgs.size(), patch_size, stride
        )
        save_image(
            fake_imgs[:25], f"saved_images/fake_{epoch}.png", nrow=5, normalize=True
        )
        save_image(
            clean_imgs[:25], f"saved_images/real_{epoch}.png", nrow=5, normalize=True
        )
