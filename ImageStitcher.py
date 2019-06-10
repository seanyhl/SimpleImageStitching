# Python 3.7.3
# Author: Yi-Hsiang Lo (yihsiang.lo@rutgers.edu)
# Usage Example: python ImageStitcher.py patch_left.png patch_right.png
import cv2
import numpy as np
from argparse import ArgumentParser
import math

parser = ArgumentParser(description='Stitch two images with their common corner features into one big image.')
parser.add_argument("input1", help="input image 1")
parser.add_argument("input2", help="input image 2")

args = parser.parse_args()
print("input image 1:", args.input1)
print("input image 2:", args.input2)

input_image1 = cv2.imread(args.input1)
input_image2 = cv2.imread(args.input2)

print(input_image1.shape)
print(input_image2.shape)

h1, w1, c1 = input_image1.shape
h2, w2, c2 = input_image2.shape

G1 = cv2.cvtColor(input_image1, cv2.COLOR_BGR2GRAY)
# gp1 = [G1]
# for i in range(math.floor(math.log2(w1) * 0.5)):
#     G1 = cv2.pyrDown(G1)
#     gp1.append(G1)

G2 = cv2.cvtColor(input_image2, cv2.COLOR_BGR2GRAY)
# gp2 = [G2]
# for i in range(math.floor(math.log2(w1) * 0.5)):
#     G2 = cv2.pyrDown(G2)
#     gp2.append(G2)

hcorner1 = cv2.cornerHarris(G1, 2, 5, 0.01)
corners1 = np.where(hcorner1>0.5*hcorner1.max())
feature_pts1 = []

hcorner2 = cv2.cornerHarris(G2, 2, 5, 0.01)
corners2 = np.where(hcorner2>0.5*hcorner2.max())
feature_pts2 = []

def get_image_patch3x3(image, r, c):
    h = image.shape[0]
    w = image.shape[1]
    r = max(1, min(r, h-2))
    c = max(1, min(c, w-2))
    return image[r-1:r+2,c-1:c+2].reshape(9)

print("features 1:")

for i in range(len(corners1[0])):
    r = corners1[0][i]
    c = corners1[1][i]
    fpt = get_image_patch3x3(G1, r, c)
    feature_pts1.append([(r,c), np.array(fpt)])
    print(r, c, fpt)

print("features 2:")

for i in range(len(corners2[0])):
    r = corners2[0][i]
    c = corners2[1][i]
    fpt = get_image_patch3x3(G2, r, c)
    feature_pts2.append([(r,c), np.array(fpt)])
    print(r, c, fpt)

print("matching features:")
# assuming only translations, no rotations, no scales
offset = None

for i, feature1 in enumerate(feature_pts1):
    closest = None
    distance = None
    pos1 = feature1[0]
    fvec1 = feature1[1]
    for j, feature2 in enumerate(feature_pts2):
        pos2 = feature2[0]
        fvec2 = feature2[1]
        if closest == None:
            closest = j
            distance = np.linalg.norm(fvec1 - fvec2)
        else:
            new_distance = np.linalg.norm(fvec1 - fvec2)
            if new_distance < distance:
                closest = j
                distance = new_distance
    if distance < 100:
        print(i, closest, distance, feature_pts1[i][0], feature_pts2[closest][0])
        offset = np.array(feature_pts1[i][0]) - np.array(feature_pts2[closest][0])

print('offset:', tuple(offset))

r_min, r_max = 0, h1
c_min, c_max = 0, w1

r_min = min(0, 0 + offset[0])
c_min = min(0, 0 + offset[1])

r_max = max(h1, h2 + offset[0])
c_max = max(w1, w2 + offset[1])

print('r_min, r_max:', r_min, r_max)
print('c_min, c_max:', c_min, c_max)

h3, w3, c3 = r_max - r_min, c_max - c_min, c1
stitched_image = np.zeros((h3, w3, c3), np.uint8)
stitched_image[0-r_min : 0-r_min+h1, 0-c_min : 0-c_min+w1] = input_image1[0:h1, 0:w1]
stitched_image[offset[0]-r_min : offset[0]-r_min+h2, offset[1]-c_min : offset[1]-c_min+w2] = input_image2[0:h2, 0:w2]
stitched_image_filename = "stitched_image.png"
cv2.imwrite(stitched_image_filename, stitched_image)
print(f"Saved to file \"{stitched_image_filename}\"")
# cv2.imshow('stitched_image', stitched_image)
# cv2.waitKey()