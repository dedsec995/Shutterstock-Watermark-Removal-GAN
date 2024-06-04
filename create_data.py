import cv2
import numpy as np
import math

text = "WATERMARK"
thickness = 2
scale = 0.75
pad = 5
angle = -45
blend = 0.25


def rotate_bound(image, angle):
    # function to rotate an image
    # from https://github.com/PyImageSearch/imutils/blob/master/imutils/convenience.py
    # grab the dimensions of the image and then determine the center
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2, h / 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))


# read image
photo = cv2.imread("sample/input/input.jpg")
ph, pw = photo.shape[:2]

# determine size for text image
(wd, ht), baseLine = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
print(wd, ht, baseLine)

# add text to black background image padded all around
pad2 = 2 * pad
text_img = np.zeros((ht + pad2, wd + pad2, 3), dtype=np.uint8)
text_img = cv2.putText(
    text_img,
    text,
    (pad, ht + pad),
    cv2.FONT_HERSHEY_SIMPLEX,
    scale,
    (255, 255, 255),
    thickness,
)

# rotate text image
text_rot = rotate_bound(text_img, angle)
th, tw = text_rot.shape[:2]
# tile the rotated text image to the size of the input
print(f'pw:{pw},tw: {tw}')
print(f'ph:{ph},th: {th}')
xrepeats = math.ceil(pw / tw)
yrepeats = math.ceil(ph / th)
print(yrepeats, xrepeats)
yrepeats //= 2
xrepeats //= 2
print(yrepeats, xrepeats)
tiled_text = np.tile(text_rot, (yrepeats, xrepeats, 1))[0:ph, 0:pw]

# combine the text with the image
result = cv2.addWeighted(photo, 1, tiled_text, blend, 0)

# save results
cv2.imwrite("sample/output/text_img.png", text_img)
cv2.imwrite("sample/output/text_img_rot.png", text_rot)
cv2.imwrite("sample/output/final_output.jpg", result)

# show the results
# cv2.imshow("text_img", text_img)
# cv2.imshow("text_rot", text_rot)
# cv2.imshow("tiled_text", tiled_text)
# cv2.imshow("result", result)
# cv2.waitKey(0)
