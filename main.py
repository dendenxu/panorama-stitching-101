import os
import cv2
import rich
import torch
import kornia
import argparse
import numpy as np
from torch import nn
from glob import glob
from tqdm import tqdm
from os.path import join
from typing import Tuple
from rich import traceback
import matplotlib.pyplot as plt

from kornia.feature.laf import get_laf_pts_to_draw, get_laf_center, get_laf_orientation, get_laf_scale
from kornia.feature.scale_space_detector import ScaleSpaceDetector
from kornia.feature import BlobHessian, BlobDoG, SIFTDescriptor, LAFOrienter, extract_patches_from_pyramid, match_snn
from kornia.geometry import ScalePyramid, ConvQuadInterp3d, transform
from kornia.color import rgb_to_grayscale

import torch.nn.functional as F

from utils import parallel_execution, load_image, load_unchanged, save_image, save_unchanged, list_to_tensor, tensor_to_list, normalize
traceback.install()

# https://github.com/kornia/kornia-examples/blob/master/image-matching-example.ipynb
# Lets define some functions for local feature matching


class FeatureDetector(nn.Module):
    def __init__(self, PS=41, n_features=500) -> None:
        super().__init__()
        self.PS = PS
        self.n_features = n_features

        self.resp = BlobDoG()
        self.scale_pyr = ScalePyramid(3, 1.6, PS, double_image=False)  # if double_image, almost certainly oom
        self.nms = ConvQuadInterp3d(10)
        self.ori = LAFOrienter(19)

        self.detector = ScaleSpaceDetector(n_features,
                                           resp_module=self.resp,
                                           scale_space_response=True,  # We need that, because DoG operates on scale-space
                                           nms_module=self.nms,
                                           scale_pyr_module=self.scale_pyr,
                                           ori_module=self.ori,
                                           mr_size=6.0,
                                           minima_are_also_good=True)
        self.descriptor = SIFTDescriptor(PS, rootsift=True)

    def forward(self, gray: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if gray.shape[1] == 3:  # rgb:
            gray = rgb_to_grayscale(gray)
        # B, C, H, W: gray scale image
        lafs, resps = self.detector(gray)
        patches = extract_patches_from_pyramid(gray, lafs, self.PS)
        B, N, CH, H, W = patches.size()
        # Descriptor accepts standard tensor [B, CH, H, W], while patches are [B, N, CH, H, W] shape
        # So we need to reshape a bit :)
        descs = self.descriptor(patches.view(B * N, CH, H, W)).view(B, N, -1)
        # scores, matches = kornia.feature.match_snn(descs[0], descs[1], 0.9)
        return lafs, descs, resps


def resize_image_tensor(imgs: torch.Tensor, ratio: float):
    B, C, H, W = imgs.shape
    nH, nW = int(H * ratio), int(W * ratio)
    # assert int(nH * ratio) == H and int(nW * ratio) == W, 'resizing ratio should be exact for easy inversion'
    imgs = F.interpolate(imgs, (nH, nW), mode='area', align_corners=False)
    return imgs


def pack_laf_into_opencv_fmt(laf: torch.Tensor, resp: torch.Tensor):
    # return value's last dimension: center x2, scale x1, orientation x1 (in degree)
    xy = get_laf_center(laf)  # B, N, 2
    scale = get_laf_scale(laf)  # B, N, 1, 1
    ori = get_laf_orientation(laf)  # B, N, 1
    packed = torch.cat([xy, scale[..., 0], ori, resp[..., None]], dim=-1)  # B, N, 4
    return packed


def feature_matching(descs: torch.Tensor, match_ratio: float = .9) -> Tuple[int, torch.Tensor]:
    # B, B, N, N -> every image pair, every distance pair -> 36 * 500 * 500 * 128 * 4 / 2**20 MB
    # half of memory and computation wasted
    pvt = descs[:, None, :, None, :]  # B, 1, N, 1, 128
    src = descs[None, :, None, :, :]  # 1, B, 1, N, 128
    ssd = (pvt - src).pow(2).sum(-1)  # BBNN sum of squared error, diagonal values should be ignored

    # Find the one with cloest match to other images as the pivot (# ? not scalable?)
    min2, match = ssd.topk(2, dim=-1, largest=False)  # find closest distance
    match = match[..., 0]  # only the largest value corresponde to dist B, N
    dist = min2[..., 0] / min2[..., 1]  # unambiguous match should have low distance here B, B, N,
    pivot = dist.sum(-1).sum(-1).argmin().item()  # find the pivot image to use (diagonal trivially ignored) # MARK: SYNC

    # Rearranage images and downscaled images according to pivot image selection
    match = torch.cat([match[pivot, :pivot], match[pivot, pivot + 1:]])  # B-1, N
    dist = torch.cat([dist[pivot, :pivot], dist[pivot, pivot + 1:]])  # B-1, N

    # Select valid threshold -> might not be wise to batch since every image pair is different...
    threshold = dist.ravel().topk(int(dist.numel() * (1 - match_ratio))).values.min()  # the ratio used to discard unmatched values
    matched = (dist < threshold).nonzero()  # M, 2 # MARK: SYNC
    match = torch.cat([matched, match[matched.chunk(2, dim=-1)]], dim=-1)  # M, 3 # image id, source feat id, target feat id
    return pivot, match


def discrete_linear_transform(pvt: torch.Tensor, src: torch.Tensor):
    # given feature point pairs, solve them with DLT algorithm
    # pvt: N, 2 # the x prime vector
    # src: N, 2 # the x vector

    assert pvt.shape[0] > 4 and src.shape == pvt.shape, f'Needs at least four points for homography with matching shape: {pvt.shape}, {src.shape}'

    # assemble feature points into big A matrix
    # 0, 0, 0, -x, -y, -1,  y'x,  y'y,  y'
    # x, y, 1,  0,  0,  0, -x'x, -x'y, -x'
    x, y = src.chunk(2, dim=-1)
    xp, yp = pvt.chunk(2, dim=-1)
    r0 = torch.cat([torch.zeros_like(x), -x, -y, -torch.ones_like(x), y * xp, y * yp, yp], dim=-1)  # N, 9
    r1 = torch.cat([x, y, torch.ones_like(x), torch.zeros_like(x), -x * xp, -x * yp, -xp], dim=-1)  # N, 9
    A = torch.stack([r0, r1], dim=1).view(-1, 9)  # interlased

    U, S, Vh = torch.linalg.svd(A, full_matrices=True)
    V = Vh.mH  # Vh is the conjugate transpose of V

    h = V[:, -1]  # 9, the homography
    H = h.view(3, 3)
    return H  # xp = H x


def unique_with_indices(x: torch.Tensor, sorted=False, dim=-1):
    if sorted:
        unique, inverse, counts = torch.unique_consecutive(x, dim=dim, return_inverse=True, return_counts=True)
    else:
        unique, inverse, counts = torch.unique(x, dim=dim, sorted=True, return_inverse=True, return_counts=True)
    _, ind_sorted = torch.sort(inverse)
    cum_sum: torch.Tensor = counts.cumsum(0)
    cum_sum = torch.cat((torch.tensor([0], device=cum_sum.device, dtype=cum_sum.dtype), cum_sum[:-1]))
    indices = ind_sorted[cum_sum]
    return unique, inverse, counts, indices


def transform_and_blending(pvt: torch.Tensor, src: torch.Tensor, homography: torch.Tensor):
    # blends two images with homography transform
    # determine the final size of the blended image to construct a canvas
    # pvt: 3, H, W
    # src: 3, H, W
    # homography: 3, 3
    C, H, W = pvt.shape

    # straight lines will always map to straight lines
    corners = pvt.new_tensor([
        [0, 0, 1],
        [0, H, 1],
        [W, 0, 1],
        [W, H, 1],
    ])  # 4, 3
    corners = corners @ homography.mT  # 4, 3
    corners = corners / corners[..., -1:]  # 4, 3 / 4, 1 -> x, y pixels
    min_corner = corners.min(dim=0)[0]  # x, y for min value
    max_corner = corners.max(dim=0)[0]  # x, y for max value
    src_min_x, src_min_y = int(min_corner[0].floor().item()), int(min_corner[1].floor().item())  # MARK: SYNC
    src_max_x, src_max_y = int(max_corner[0].ceil().item()), int(max_corner[1].ceil().item())  # MARK: SYNC
    src_W, src_H = src_max_x - src_min_x, src_max_y - src_min_y

    # Calculate the pixel coodinates
    i, j = torch.meshgrid(torch.arange(src_H, dtype=pvt.dtype, device=pvt.device),
                          torch.arange(src_W, dtype=pvt.dtype, device=pvt.device),
                          indexing='ij')  # H, W
    # 0->H, 0->W
    xy1 = torch.stack([j, i, torch.ones_like(i)], dim=-1)  # mH, mW, 3
    xy1[..., :2] = xy1[..., :2] + min_corner  # shift to origin
    xy1 = xy1 @ homography.mT  # mH, mW, 3
    xy1 = xy1 / xy1[..., -1:]  # mH, mW, 3 / mH, mW, 1 -> x, y pixels
    xy = xy1[..., :2]  # for xy coordinates, mH, mW (x, y inside)

    src = F.grid_sample(src[None], xy[None], align_corners=False, padding_mode='zeros', mode='bilinear')[0]  # 3, mH, mW

    # Determine canvas size:
    pvt_min_x, pvt_min_y = min(0, src_min_x), min(0, src_min_y)
    pvt_max_x, pvt_max_y = max(W, src_max_x), max(H, src_max_y)
    pvt_W, pvt_H = pvt_max_x - pvt_min_x, pvt_max_y - pvt_min_y

    # Linear blending for now (excess memory usage?)
    pvt_canvas = pvt.new_zeros((C, int(pvt_H), int(pvt_W)))  # 3, pH, pW
    src_canvas = pvt.new_zeros((C, int(pvt_H), int(pvt_W)))  # 3, pH, pW
    pvt_canvas[-pvt_min_y:H, -pvt_min_x:W] = pvt
    src_canvas[-pvt_min_y - src_min_y:src_H, -pvt_min_x - src_min_x:src_W] = src
    blended = (pvt_canvas + src_canvas) / 2
    return blended


def main():
    # Commandline Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data/data1')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--ratio', default=0.5, type=float)  # otherwise, typicall out of memory
    parser.add_argument('--match_ratio', default=0.9, type=float)  # otherwise, typicall out of memory
    args = parser.parse_args()

    # Loading images from disk and downscale them
    imgs = glob(join(args.data_root, '*'))
    imgs = list_to_tensor(parallel_execution(imgs, action=load_image), args.device)  # rgb images: B, C, H, W
    down = resize_image_tensor(imgs, args.ratio)
    B, C, H, W = imgs.shape
    B, C, nH, nW = down.shape

    # Perform feature detection (use kornia implementation)
    feature_detector = FeatureDetector().to(args.device, non_blocking=True)  # the actual feature detector
    with torch.no_grad():
        lafs, descs, resp = feature_detector(down)  # B, N, 128 -> batch size, number of descs, desc dimensionality

    # TODO: visualize feature detection results

    # Perform feature matching
    pivot, match = feature_matching(descs, args.match_ratio)  # M, 3 -> image id, source feat id, target feat id

    # Visualize feature matching results
    resp_pivot = resp[pivot:pivot + 1]  # 1, C, 2, 3
    lafs_pivot = lafs[pivot:pivot + 1]  # 1, C, 2, 3
    imgs_pivot = imgs[pivot:pivot + 1]  # 1, C, H, W
    down_pivot = down[pivot:pivot + 1]  # 1, C, nH, nW
    resp = torch.cat([resp[:pivot], resp[pivot + 1:]])  # B, C, 2, 3
    lafs = torch.cat([lafs[:pivot], lafs[pivot + 1:]])  # B, C, 2, 3
    imgs = torch.cat([imgs[:pivot], imgs[pivot + 1:]])  # B, C, H, W
    down = torch.cat([down[:pivot], down[pivot + 1:]])  # B, C, nH, nW

    # Pack laf representation to xy, scale, orientation, and response
    packed_pivot = pack_laf_into_opencv_fmt(lafs_pivot, resp_pivot)  # 1, N, 4
    packed = pack_laf_into_opencv_fmt(lafs, resp)  # B-1, N, 4

    # TODO: visualize feature matching results & log feature matching results

    # Construct matched feature pairs from matching results
    pvt = packed_pivot[0, match[..., 1], :2]  # M, 2, discarding scale, orientation, and response
    src = packed[match[..., 0], match[..., 2], :2]  # M, 2
    pairs = torch.cat([match, pvt, src], dim=-1)  # M, 7 -> source image id, source feat id, target feat id, pivot xy, source xy
    pairs = pairs[torch.argsort(pairs[..., 0])]  # sort pairs by source image id
    pvt, src = pairs[..., -2 - 2:-2], pairs[..., -2:]  # sorted pairs by source image id

    # Perform RANSAC and homography one by one for target image ids
    uni, inv, cnt, ind = unique_with_indices(pairs[..., 0], sorted=True)
    __import__('ipdb').set_trace()


if __name__ == "__main__":
    main()
