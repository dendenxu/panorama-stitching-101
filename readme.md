# Panorama Stitching 101

## Introduction

Computer Vision homework #2.

See [Implementation Details](#Implementation-Details) for usage examples.

> If the camera center is fixed, or if the scene is planar, different images of the same scene are related by a homography. In this project, you will implement an algorithm to calculate the homography between two images. Several images of each scene will be provided. Your program should generate image mosaics according to these estimated homographies.

We implement the four basic components of a panorama stitching pipeline:

1. Feature detection, description using SIFT / Concatenated pixel
2. Exhaustive feature matching between image pairs with ambiguity check
3. Homography estimation using DLT and M-estimator with RANSAC
4. The actual panorama stitching using linear alpha blending

Then we go on to propose a new graph shortest path based sequential homography method to perform panorama stitching on a set of unordered images with arbitrary perspective transform (e.g. when some of the image pairs have literally zero content overlap).

We are not sure whether such algorithm exist in literature due to limited knowledge of the author, however we do implement it and it seem to work much better than selecting on pivot image and match every other ones on it.

We evaluate the performance of our algorithm on 4 sets of images. Note that the number of images in the provided dataset is quite small (3-5 images). We also provide a comparison between the SIFT descriptor and concatenated pixel descriptor, analysing their respective influence on the number of RANSAC iterations.

## Method

## Implementation Details

## Experiments

## Conclusion
