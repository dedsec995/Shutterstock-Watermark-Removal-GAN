import cv2
import numpy as np
import math
from PIL import Image, ImageDraw, ImageFont

text = "shutterstock"
thickness = 0.01
scale = 30
pad = 150
angle = -40
blend = 0.25
font_path = 'mytryd.ttf'

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

def add_distortion(image):
    h, w = image.shape[:2]
    src_points = np.float32([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
    dst_points = np.float32([[0, 0], [w - 1, 0], [int(0.1 * w), h - 1], [int(0.9 * w), h - 1]])
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    distorted = cv2.warpPerspective(image, matrix, (w, h))
    return distorted

# read image
photo = cv2.imread('input/demo.jpg')
ph, pw = photo.shape[:2]

# determine size for text image using Pillow
font = ImageFont.truetype(font_path, int(scale * 10))
bbox = font.getbbox(text)
wd, ht = bbox[2] - bbox[0], bbox[3] - bbox[1]
baseLine = 0  # Not used in Pillow

# add text to black background image padded all around
pad2 = 2 * pad
text_img_pil = Image.new('RGB', (wd + pad2, ht + pad2), (0, 0, 0))
draw = ImageDraw.Draw(text_img_pil)
draw.text((pad, pad), text, font=font, fill=(255, 255, 255))

# convert Pillow image to OpenCV format
text_img = np.array(text_img_pil)

distorted_text_img = add_distortion(text_img)

# rotate text image
text_rot = rotate_bound(distorted_text_img, angle)
th, tw = text_rot.shape[:2]

# tile the rotated text image to the size of the input
xrepeats = math.ceil(pw / tw)
yrepeats = math.ceil(ph / th)
tiled_text = np.tile(text_rot, (yrepeats, xrepeats, 1))[0:ph, 0:pw]

# combine the text with the image
result = cv2.addWeighted(photo, 1, tiled_text, blend, 0)

# save results
cv2.imwrite("output/text_img.png", text_img)
cv2.imwrite("output/text_img_rot.png", text_rot)
cv2.imwrite("output/result.jpg", result)

