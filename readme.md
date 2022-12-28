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

### Feature Detection and Descriptor

For feature detector, we adopted [kornia's implementation of SIFT detector](https://github.com/kornia/kornia-examples/blob/master/image-matching-example.ipynb) by constructing a `ScaleSpaceDetector`. We chunk the images before feeding them to the feature detector to avoid running out of memory by a heuristic batch size. Note that to retain parallelism, we extract a fixed number of features for each image (typically 5000). After feature detection, we perform feature extraction with the user selected feature descriptor.

-   For SIFT, we use the `SIFTDescriptor` class provided by [kornia](https://github.com/kornia/kornia).
-   For pixel concatenation descriptor, we simply aggregate all pixel values of a patch by concatenation to form its descriptor.

It's worth noting that this concatenation is merely for the **descriptor**, both SIFT and pixel concantenation use the same SIFT **detector** thus we don't expect to see any difference in the detected features.

### Feature Matching

We implement an naive exhaustive feature matching technique by utilizing the `cdist` function to densely compute feature distances. Note that we have all images instead of only one pair of them in mind when performing this panorama stitching, thus **exhaustive** is refering to not the way we compute distance between every pair of features, but the way we compute distance between every pair of images.

We also implement a simple ambiguity check to remove false matches. For an image pair and for one of the feature in the first image, we find the best and second best match in the second image. If the distance ratio between the best and second best match is larger than a threshold, we consider the match to be ambiguous and remove it. The threshold of discarding a matching is controlled dynamically by a user parameter `match_ratio`. Specifically, we find the cut-off value of the distance ratio to retain `match_ratio * 100` percent of all matches and use this cut-off value to determine which match to discard. `match_ratio` is set to `0.9` in all experiments if not specified otherwise (meaning, 90% of all matches will be retained).

### Homography Estimation

The transformation between two images can be described with a homography, defined by a $3 \times 3$ matrix with 8 degrees of freedom (after normalizing the last element to 1.0). To obtain this homography, we need at least four pairs of observation. Each of them provides us with 2 linearly independent equations. Thus we 4 pairs of observations and 8 equations in total we can solve the least square homography estimation problem with the DLT (Discrete Linear Transform) algorithm by SVD decomposition on the constructed linear system of equations.

We perform RANSAC on top of the proposed DLT algorithm to obtain a good estimation and discard outliers reliably. During experiments we found that RANSAC essential to the success of the homography estimation. Since one or two outliers would lead to a completely wrong result. We first perform a few random experiments `min_iter` to determine the number of iterations required to achieve a certain `confidence` level. The number of expected iterations `exp_min_iter` is dynamically changed with the estimated ratio of inlier `inlier_ratio`. `exp_min_iter` is determined with the following equation:

```python
def find_min_iter(inlier_ratio, confidence, min_sample): 
    return int(np.ceil(np.log(1 - confidence) / np.log(1 - inlier_ratio**min_sample)))
```

Where `min_sample` is 4 as determined in the first paragraph of this section. This equation computes the number of iteration required to have `confidence` probability of have a random sampling with all inliers, which will lead to the success of the RANSAC algorithm.

> "All good matches are alike; every bad match is bad in its own way." - Leo Tolstoy

After obtaining the largest set of inliers, we perform M-estimation on those pairs. M-estimation convert the linear least square homography estimation problem to a non-linear one with a robust error function ignoring outliers. To optimize this non-linear problem, we define the robust error function and symmetric error function with differentiable PyTorch operators and perform 1000 iterastion with the Adam optimizer of learning rate `1e-2`.

To obtain a more robust estimation, we repeat the M-estimator with the updated inlier set of the first M-estimation.

### Graph-based Sequential Homography



### Panorama Stitching

## Implementation Details

We implemented our pipeline with PyTorch (even the graph shortest path part) for GPU acceleration and further scalability considerations.

With the default arguments, we perform parallel image loading onto the user defined device (RAM or VRAM) and perform feature detection and description on the GPU. We then perform exhaustive feature matching on the CPU and GPU, after which homography estimation are done on the GPU. We then perform panorama stitching on the GPU.

## Experiments

## Conclusion
