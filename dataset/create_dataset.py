import cv2
import numpy as np
import math
from PIL import Image, ImageDraw, ImageFont

text = "-------shutterstock-----^---------"
thickness = 0.01
scale = 10  # Size of the one Tile
pad = 20 # Space between two text
angle = -40  # Angle of the text.
blend = 0.25  # Opacity of the imposed Tile
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


# read image
photo = cv2.imread('input/input.jpg')
ph, pw = photo.shape[:2]

# determine size for text image using Pillow
font = ImageFont.truetype(font_path, int(scale * 10))
bbox = font.getbbox(text)
wd, ht = bbox[2] - bbox[0], bbox[3] - bbox[1]
baseLine = 0  # Not used in Pillow

# add text to transparent background image padded all around
pad2 = 2 * pad
text_img_pil = Image.new('RGBA', (wd + pad2, ht + pad2), (0, 0, 0, 0))
draw = ImageDraw.Draw(text_img_pil)

def draw_rotated_rectangle(draw, xy, angle, fill=None):
    (x0, y0, x1, y1) = xy
    rect_w, rect_h = x1 - x0 + 10, y1 - y0 - 5
    rect_img = Image.new('RGBA', (rect_w, rect_h), (0, 0, 0, 0))
    rect_draw = ImageDraw.Draw(rect_img)
    rect_draw.rectangle([0, -5, rect_w, rect_h], fill=fill)
    
    rect_img = rect_img.rotate(angle, expand=True)
    draw.bitmap((x0 - 17, y0), rect_img, fill=None)



# function to draw individual letters with different styles
def draw_text_with_styles(draw, position, text, font):
    x, y = position
    for char in text:
        char_bbox = font.getbbox(char)
        char_w, char_h = char_bbox[2] - char_bbox[0], char_bbox[3] - char_bbox[1]
        
        if char == '^':
            draw_rotated_rectangle(draw, [x, y, x + char_w, y + char_h ], 50, fill=(0, 0, 0, 255))
            # draw.rectangle([x - 5, y + 10, x + char_w + 5, y + char_h + 10], fill=(0, 0, 0, 255))
            draw.text((x, y), char, font=font, fill=(0, 0, 0, 255))
        elif char == '-':
            draw.text((x, y), char, font=font, fill=(0,0,0, 255))
        else:
            draw.text((x, y), char, font=font, fill=(255, 255, 255, 255))
        
        x += char_w

# draw the text with custom styles
draw_text_with_styles(draw, (pad, pad), text, font)

# convert Pillow image to OpenCV format with alpha channel
text_img = np.array(text_img_pil)

# rotate text image
text_rot = rotate_bound(text_img, angle)
th, tw = text_rot.shape[:2]

# tile the rotated text image to the size of the input
xrepeats = math.ceil(pw / tw)
yrepeats = math.ceil(ph / th)
tiled_text = np.tile(text_rot, (yrepeats, xrepeats, 1))[0:ph, 0:pw]

# create a mask from the alpha channel
alpha_mask = tiled_text[:, :, 3] / 255.0
alpha_mask = np.stack([alpha_mask] * 3, axis=-1)

# blend the text with the image using the alpha mask
photo = photo.astype(float)
tiled_text = tiled_text[:, :, :3].astype(float)

result = (photo * (1 - blend * alpha_mask) + tiled_text * (blend * alpha_mask)).astype(np.uint8)

# save results
cv2.imwrite("output/text_img.png", text_img)
cv2.imwrite("output/text_img_rot.png", text_rot)
cv2.imwrite("output/result.jpg", result)
