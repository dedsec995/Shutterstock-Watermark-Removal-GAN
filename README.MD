# Watermark Removal GAN
## Overview
This project features a generative adversarial network (GAN) designed for watermark removal from images. The GAN removes watermarks by reconstructing the image from scratch. This implementation is intended for educational purposes and provides a framework for learning about image inpainting and GANs.

In addition to the watermark removal GAN, the project includes a script to create a custom dataset with 90% accurate watermark creation. The generated dataset can be used to train and evaluate the watermark removal GAN.

## Project Structure
```lua
watermark_removal_gan/
|-- data/
|-- dataset/
|-- dataset.py          # Custom dataset loader for training
|-- discriminator_model.py
|-- generator_model.py
|-- License              # GNU Public License
|-- model_train.py       # Script for training the model
|-- inference.py         # Script for running inference
|-- vgg_loss.py          # Loss function based on VGG network
```

# Installation
## Prerequisites
Ensure you have Python 3.8 or later installed. This project requires the following Python packages:

- torch
- torchvision
- numpy
- PIL
- Tensorboard

You can install the necessary packages using pip:
```bash
pip install -r requirements.txt
```

# Usage
## Preparing the Dataset
Place your images in the data/ directory.
Use the script `download_images.py` in the dataset/ folder to download image from unsplash and using `create_dataset.py` to generate a custom dataset with watermarked images. The script creates a dataset where watermarks are added with 90% accuracy of the original watermark that shutterstock creates. Feel free to change the parameters in `create_dataset.py`. I created custom fonts `shutterstock.tff` just for creating watermark.

## Training the Model
To train the watermark removal GAN, run:
```bash
python model_train.py
```
This script will:

Load the dataset from the data/ directory.
Train the GAN model using the provided dataset.

## Running Inference
To remove watermarks from an image, use:
```bash
python inference.py --input_image path/to/image.jpg --output_image path/to/output.jpg
```

# Code Overview
### generator_model.py
Contains the definition of the Generator class, which is responsible for generating watermark-free images. The generator uses a U-Net-like architecture with residual blocks for image reconstruction.

### discriminator_model.py
Defines the Discriminator class that distinguishes between real and generated images. It uses spectral normalization to stabilize training and improve performance.

### dataset.py
Provides a custom dataset loader for training the GAN. It loads images and applies the watermark removal process.

### model_train.py
Script for training the GAN. It initializes the generator and discriminator, sets up the loss functions and optimizers, and trains the model.

### inference.py
Script for applying the trained model to remove watermarks from input images.

### vgg_loss.py
Defines a VGG-based loss function used to enhance the quality of generated images by comparing them with ground truth images.

# License
This project is licensed under the GNU Public License. See the License file for more details.

# Acknowledgements
The implementation is inspired by various GAN architectures and image processing techniques.
Special thanks to the research community and open-source contributors for their invaluable resources and tools.
