import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from dataset import WaterDataset
from generator_model import Generator
from discriminator_model import Discriminator
from vgg_loss import VGGLoss

# Define hyperparameters
num_epochs = 100
batch_size = 8
lr = 0.0002
patch_size = 256

# Transform for dataset
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Initialize dataset and dataloader
dataset = WaterDataset(root_dir="data", transform=transform, patch_size=patch_size)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize models
generator = Generator(in_channels=3, out_channels=3).cuda()
discriminator = Discriminator(in_channels=3).cuda()

# Losses
criterion_GAN = nn.MSELoss().cuda()
criterion_pixelwise = nn.L1Loss().cuda()
vgg_loss = VGGLoss().cuda()

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# TensorBoard
writer = SummaryWriter()

# Training loop
for epoch in range(num_epochs):
    for i, (input_image, target_image) in enumerate(dataloader):
        input_image = input_image.cuda()
        target_image = target_image.cuda()

        # Generate fake image
        fake_image = generator(input_image)

        # Adversarial ground truths
        pred_real = discriminator(input_image, target_image)
        pred_fake = discriminator(input_image, fake_image.detach())

        valid = torch.ones_like(pred_real, requires_grad=False).cuda()
        fake = torch.zeros_like(pred_fake, requires_grad=False).cuda()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Real loss
        loss_real = criterion_GAN(pred_real, valid)

        # Fake loss
        loss_fake = criterion_GAN(pred_fake, fake)

        # Total loss
        loss_D = (loss_real + loss_fake) / 2

        loss_D.backward()
        optimizer_D.step()

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # GAN loss
        pred_fake = discriminator(input_image, fake_image)
        loss_GAN = criterion_GAN(pred_fake, valid)

        # Pixel-wise loss
        loss_pixel = criterion_pixelwise(fake_image, target_image)

        # VGG loss
        loss_vgg = vgg_loss(fake_image, target_image)

        # Total loss
        loss_G = loss_GAN + 100 * loss_pixel + 10 * loss_vgg

        loss_G.backward()
        optimizer_G.step()

        # Print log
        if i % 10 == 0:
            print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(dataloader)}] [D loss: {loss_D.item()}] [G loss: {loss_G.item()}]")

        # TensorBoard log
        writer.add_scalar("Loss/Discriminator", loss_D.item(), epoch * len(dataloader) + i)
        writer.add_scalar("Loss/Generator", loss_G.item(), epoch * len(dataloader) + i)

        # Log images to TensorBoard every 100 batches
        if i % 100 == 0:
            img_grid_input = make_grid(input_image[:4], normalize=True)
            img_grid_target = make_grid(target_image[:4], normalize=True)
            img_grid_fake = make_grid(fake_image[:4], normalize=True)
            writer.add_image("Input Images", img_grid_input, global_step=epoch * len(dataloader) + i)
            writer.add_image("Target Images", img_grid_target, global_step=epoch * len(dataloader) + i)
            writer.add_image("Fake Images", img_grid_fake, global_step=epoch * len(dataloader) + i)

writer.close()
