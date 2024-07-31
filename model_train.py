import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from dataset import WaterDataset
from generator_model import Generator
from discriminator_model import Discriminator
from vgg_loss import VGGLoss

# Initialize models
device = "cuda" if torch.cuda.is_available() else "cpu"
gen = Generator(in_channels=3, out_channels=3).to(device)
disc = Discriminator(in_channels=3).to(device)
vgg_loss = VGGLoss().to(device)

# Optimizers
opt_gen = optim.Adam(gen.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=2e-4, betas=(0.5, 0.999))

# DataLoader
dataset = WaterDataset(root_dir="data")
loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

# TensorBoard
writer = SummaryWriter("runs/watermark_removal")

# Training loop
num_epochs = 100
kernel_size = 256
stride = 32
batch_size = 8

for epoch in range(num_epochs):
    loop = tqdm(loader, leave=True)
    for idx, (input_image, target_image) in enumerate(loop):
        input_image = input_image.to(device)
        target_image = target_image.to(device)

        # Patch processing
        patches_input = input_image.unfold(2, kernel_size, stride).unfold(
            3, kernel_size, stride
        )
        patches_input = patches_input.contiguous().view(-1, 3, kernel_size, kernel_size)

        patches_target = target_image.unfold(2, kernel_size, stride).unfold(
            3, kernel_size, stride
        )
        patches_target = patches_target.contiguous().view(
            -1, 3, kernel_size, kernel_size
        )

        # Discriminator training
        disc_real = disc(patches_target, patches_input)
        disc_fake = disc(gen(patches_input), patches_input.detach())
        loss_disc_real = F.binary_cross_entropy_with_logits(
            disc_real, torch.ones_like(disc_real)
        )
        loss_disc_fake = F.binary_cross_entropy_with_logits(
            disc_fake, torch.zeros_like(disc_fake)
        )
        loss_disc = (loss_disc_real + loss_disc_fake) / 2

        opt_disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        # Generator training
        disc_fake = disc(gen(patches_input), patches_input)
        loss_gen_adv = F.binary_cross_entropy_with_logits(
            disc_fake, torch.ones_like(disc_fake)
        )
        loss_gen_vgg = vgg_loss(gen(patches_input), patches_target)
        loss_gen = loss_gen_adv + 0.1 * loss_gen_vgg

        opt_gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # TensorBoard logging
        writer.add_scalar(
            "Loss/Discriminator", loss_disc.item(), epoch * len(loader) + idx
        )
        writer.add_scalar("Loss/Generator", loss_gen.item(), epoch * len(loader) + idx)

        if idx % 100 == 0:
            with torch.no_grad():
                fake_images = gen(input_image)
                writer.add_images(
                    "Images/Input", input_image, epoch * len(loader) + idx
                )
                writer.add_images(
                    "Images/Target", target_image, epoch * len(loader) + idx
                )
                writer.add_images("Images/Fake", fake_images, epoch * len(loader) + idx)

        loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
        loop.set_postfix(loss_disc=loss_disc.item(), loss_gen=loss_gen.item())

# Save models
torch.save(gen.state_dict(), "generator.pth")
torch.save(disc.state_dict(), "discriminator.pth")
