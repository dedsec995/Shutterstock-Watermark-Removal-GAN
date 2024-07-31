import os
import cv2
import numpy as np
import math
from PIL import Image, ImageDraw, ImageFont
import random
import string

# Pro Tip: ~ will give shutterstock logo and ^ also give out the logo with grey background. Both at 40 degree rotation.

text_template = "-----shutterstock-----~-------^---------"
thickness = 0.01
scale = 18  # Size of the one Tile
pad = 45  # Space between two text
angle = -40  # Angle of the text.
blend = 0.40  # Opacity of the imposed Tile
font_path = "shutterstock.ttf"


def generate_random_string():
    length = random.randint(4, 9)
    letters = string.ascii_letters
    return "".join(random.choice(letters) for i in range(length))


def rotate_bound(image, angle):
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    return cv2.warpAffine(image, M, (nW, nH))


def draw_text_with_styles(draw, position, text, font, angle):
    x, y = position
    for char in text:
        char_bbox = font.getbbox(char)
        char_w, char_h = char_bbox[2] - char_bbox[0], char_bbox[3] - char_bbox[1]

        if char == "^":
            # Rotate the rectangle
            rotated_rect = [
                (x, y + 5),
                (x + char_w + 20, y + 5),
                (x + char_w + 20, y + char_h + 20),
                (x, y + char_h + 20),
            ]
            rotated_rect = rotate_rect(
                rotated_rect, angle, (x + char_w / 2, y + char_h / 2)
            )
            draw.polygon(rotated_rect, fill=(111, 111, 111, 255))
            draw.text((x, y), char, font=font, fill=(255, 255, 255, 255))
        elif char == "-":
            draw.text((x, y), char, font=font, fill=(0, 0, 0, 255))
        else:
            draw.text((x, y), char, font=font, fill=(255, 255, 255, 255))

        x += char_w


def rotate_rect(points, angle, center):
    rotated_points = []
    angle_rad = math.radians(angle)
    cos_theta = math.cos(angle_rad)
    sin_theta = math.sin(angle_rad)

    for x, y in points:
        cx, cy = center
        nx = (x - cx) * cos_theta - (y - cy) * sin_theta + cx
        ny = (x - cx) * sin_theta + (y - cy) * cos_theta + cy
        rotated_points.append((nx, ny))

    return rotated_points


def apply_watermark(photo_path, custom_name, text):
    photo = cv2.imread(photo_path)
    ph, pw = photo.shape[:2]

    # Finding out the size for text image using Pillow
    font = ImageFont.truetype(font_path, int(scale * 10))
    bbox = font.getbbox(text)
    wd, ht = bbox[2] - bbox[0], bbox[3] - bbox[1]
    baseLine = 0  # Not used in Pillow

    pad2 = 2 * pad
    text_img_pil = Image.new("RGBA", (wd + pad2, ht + pad2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(text_img_pil)

    # Draw the text with custom styles
    draw_text_with_styles(draw, (pad, pad), text, font, 40)

    # Convert Pillow image to OpenCV format with alpha channel
    text_img = np.array(text_img_pil)

    # Rotate text image
    text_rot = rotate_bound(text_img, angle)
    th, tw = text_rot.shape[:2]

    xrepeats = math.ceil(pw / tw)
    yrepeats = math.ceil(ph / th)
    tiled_text = np.tile(text_rot, (yrepeats, xrepeats, 1))[0:ph, 0:pw]

    alpha_mask = tiled_text[:, :, 3] / 255.0  # Create a mask from the alpha channel
    alpha_mask = np.stack([alpha_mask] * 3, axis=-1)

    # Blend the text with the image using the alpha mask
    photo = photo.astype(float)
    tiled_text = tiled_text[:, :, :3].astype(float)

    result = (
        photo * (1 - blend * alpha_mask) + tiled_text * (blend * alpha_mask)
    ).astype(np.uint8)

    # Save the result
    output_path = os.path.splitext(photo_path)[0] + f"_{custom_name}.jpg"
    cv2.imwrite(output_path, result)
    print(f"Saved {output_path}")


def process_images(folder_path, custom_name):
    for subdir in os.listdir(folder_path):
        subdir_path = os.path.join(folder_path, subdir)
        if os.path.isdir(subdir_path):
            files = os.listdir(subdir_path)
            
            if len(files) == 1:
                random_string = generate_random_string()
                text = text_template + random_string
                file_path = os.path.join(subdir_path, files[0])
                apply_watermark(file_path, custom_name, text)
            else:
                print(f"Skipping {subdir_path}: does not contain exactly one file")


# Example usage
folder_path = "data"
custom_name = "watermark"
process_images(folder_path, custom_name)
