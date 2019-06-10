# Python 3.7.3
# Author: Yi-Hsiang Lo (yihsiang.lo@rutgers.edu)
# Usage Example: python ImageCutter.py cloudy-chance-meatballs_0.jpg 200
import cv2
import numpy as np
from argparse import ArgumentParser
import math

parser = ArgumentParser(description='Cut image into left and right parts with a specific overlapping area.')
parser.add_argument("input", help="input image")
parser.add_argument("overlap_width", help="image overlapping width")

args = parser.parse_args()
print("input image:", args.input)

input_image = cv2.imread(args.input)
# cv2.imshow("input image", input_image)
# cv2.waitKey(1)

print(input_image.shape)

h, w, c = input_image.shape

OVERLAP_WIDTH = min(int(args.overlap_width), w)

left_bound = math.floor(w/2 + OVERLAP_WIDTH/2)
right_bound = math.floor(w/2 - OVERLAP_WIDTH/2)
image_filename = ["patch_left.png", "patch_right.png"]
cv2.imwrite(image_filename[0], input_image[:, 0:left_bound, :])
print(f"Saved to file \"{image_filename[0]}\"")
cv2.imwrite(image_filename[1], input_image[:, right_bound:w, :])
print(f"Saved to file \"{image_filename[1]}\"")