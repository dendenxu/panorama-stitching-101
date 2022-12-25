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
from rich import traceback
from termcolor import colored
from typing import Tuple, List
import matplotlib.pyplot as plt

from kornia.feature.laf import get_laf_pts_to_draw, get_laf_center, get_laf_orientation, get_laf_scale
from kornia.feature.scale_space_detector import ScaleSpaceDetector
from kornia.feature import BlobHessian, BlobDoG, SIFTDescriptor, LAFOrienter, extract_patches_from_pyramid, match_snn
from kornia.geometry import ScalePyramid, ConvQuadInterp3d, transform
from kornia.color import rgb_to_grayscale

import torch.nn.functional as F

from utils import parallel_execution, load_image, load_unchanged, save_image, save_unchanged, list_to_tensor, tensor_to_list, normalize, log
traceback.install()

# https://github.com/kornia/kornia-examples/blob/master/image-matching-example.ipynb
# Lets define some functions for local feature matching


class FeatureDetector(nn.Module):
    def __init__(self, PS=41, n_features=1000) -> None:
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
    imgs = F.interpolate(imgs, (nH, nW), mode='area')
    return imgs


def pack_laf_into_opencv_fmt(lafs: torch.Tensor, resp: torch.Tensor):
    # return value's last dimension: center x2, scale x1, orientation x1 (in degree)
    xy = get_laf_center(lafs)  # B, N, 2
    scale = get_laf_scale(lafs)  # B, N, 1, 1
    ori = get_laf_orientation(lafs)  # B, N, 1
    packed = torch.cat([xy, scale[..., 0], ori, resp[..., None]], dim=-1)  # B, N, 4
    return packed


def feature_matching(desc: torch.Tensor, match_ratio: float = .7) -> Tuple[int, torch.Tensor]:
    # B, B, N, N -> every image pair, every distance pair -> 36 * 500 * 500 * 128 * 4 / 2**20 MB
    # half of memory and computation wasted
    B, N, C = desc.shape
    # pvt = desc[:, None, :, None, :]
    # src =  desc[None, :, None, :, :]
    # ssd = (pvt - src).pow(2).sum(-1).sqrt()  # BBNN sum of squared error, diagonal values should be ignored
    pvt = desc[:, None].expand(B, B, N, C).reshape(B * B, N, C)
    src = desc[None, :].expand(B, B, N, C).reshape(B * B, N, C)
    ssd = torch.cdist(pvt, src, compute_mode='donot_use_mm_for_euclid_dist').view(B, B, N, N)  # some numeric error here?

    # Find the one with cloest match to other images as the pivot (# ? not scalable?)
    min2, match = ssd.topk(2, dim=-1, largest=False)  # find closest distance
    match: torch.Tensor = match[..., 0]  # only the largest value correspond to dist B, N, cloest is on diagonal
    dist: torch.Tensor = min2[..., 0] / min2[..., 1]  # unambiguous match should have low distance here B, B, N,
    # dist = min2[..., 0]  # ambiguous matches also present B, B, N,

    # Find actual two-way matching results
    reversed = (torch.zeros_like(match) - 1).scatter_(dim=-1, index=match.permute(1, 0, 2), src=torch.arange(N, device=match.device, dtype=match.dtype)[None, None].expand(B, B, N))  # valid index, B, B, N
    two_way_match = match == reversed  # B, B, N

    # Select valid threshold -> might not be wise to batch since every image pair is different...
    # threshold = dist.ravel().topk(int(dist.numel() * (1 - match_ratio))).values.min()  # the ratio used to discard unmatched values
    threshold = match_ratio
    log(f'Thresholding matches with: {colored(f"{threshold:.6f}", "yellow")}')

    matched = dist <= threshold  # number of matches per image? B, B, N
    matched = matched & two_way_match  # need a two way matching to make this work?
    # pivot = dist.sum(-1).sum(-1).argmin().item()  # find the pivot image to use (diagonal trivially ignored) # MARK: SYNC
    pivot = matched.sum(-1).sum(-1).argmax().item()  # find the pivot image to use (diagonal trivially ignored) # MARK: SYNC
    # Rearranage images and downscaled images according to pivot image selection
    matched = torch.cat([matched[pivot, :pivot], matched[pivot, pivot + 1:]])  # B-1, N, source feat id
    match = torch.cat([match[pivot, :pivot], match[pivot, pivot + 1:]])  # B-1, N, source feat id
    dist = torch.cat([dist[pivot, :pivot], dist[pivot, pivot + 1:]])  # B-1, N, source feat id

    matched = matched.nonzero()  # M, 2 # MARK: SYNC
    match = torch.cat([matched, match[matched.chunk(2, dim=-1)]], dim=-1)  # M, 3 # image id, pivot feat id, source feat id
    return pivot, match


def discrete_linear_transform(pvt: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    # given feature point pairs, solve them with DLT algorithm
    # pvt: N, 2 # the x prime vector
    # src: N, 2 # the x vector
    dtype = pvt.dtype
    pvt = pvt.double()
    src = src.double()  # otherwise would be numerically instable

    assert pvt.shape[0] >= 4 and src.shape == pvt.shape, f'Needs at least four points for homography with matching shape: {pvt.shape}, {src.shape}'

    # assemble feature points into big A matrix
    # 0, 0, 0, -x, -y, -1,  y'x,  y'y,  y'
    # x, y, 1,  0,  0,  0, -x'x, -x'y, -x'
    x, y = src.chunk(2, dim=-1)
    xp, yp = pvt.chunk(2, dim=-1)
    z = torch.zeros_like(x)
    o = torch.ones_like(x)
    r0 = torch.cat([z, z, z, -x, -y, -o, yp * x, yp * y, yp], dim=-1)  # N, 9
    r1 = torch.cat([x, y, o, z, z, z, -xp * x, -xp * y, -xp], dim=-1)  # N, 9
    A = torch.stack([r0, r1], dim=1).view(-1, 9)  # interlased

    U, S, Vh = torch.linalg.svd(A, full_matrices=True)
    V = Vh.mH  # Vh is the conjugate transpose of V
    # V = Vh  # Vh is the conjugate transpose of V
    log(f'reprojection error: {colored(f"{S[-1].item():.6f}", "yellow")}')

    h = V[:, -1]  # 9, the homography
    H = h.view(3, 3) / h[-1]  # normalize homography to 1
    return H.to(dtype)  # xp = H x


def unique_with_indices(x: torch.Tensor, sorted=False, dim=-1):
    if sorted:
        unique, inverse, counts = torch.unique_consecutive(x, dim=dim, return_inverse=True, return_counts=True)
    else:
        unique, inverse, counts = torch.unique(x, dim=dim, sorted=True, return_inverse=True, return_counts=True)
    _, ind_sorted = torch.sort(inverse, stable=True)
    cum_sum: torch.Tensor = counts.cumsum(0)
    cum_sum = torch.cat((torch.tensor([0], device=cum_sum.device, dtype=cum_sum.dtype), cum_sum[:-1]))
    indices = ind_sorted[cum_sum]
    return unique, inverse, counts, indices


def homography_transform(src: torch.Tensor, homography: torch.Tensor):
    # blends two images with homography transform
    # determine the final size of the blended image to construct a canvas
    # pvt: 3, H, W
    # src: 3, H, W
    # homography: 3, 3
    C, H, W = src.shape
    M = max(H, W)

    # straight lines will always map to straight lines
    corners = src.new_tensor([
        [0, 0, 1],
        [0, H / M, 1],
        [W / M, 0, 1],
        [W / M, H / M, 1],
    ])  # 4, 3
    corners = corners @ homography.mT  # 4, 3
    corners = corners[..., :-1] / corners[..., -1:]  # 4, 3 / 4, 1 -> x, y pixels
    corners = corners * M  # renormalize pixel coordinates
    min_corner = corners.min(dim=0)[0]  # x, y for min value
    max_corner = corners.max(dim=0)[0]  # x, y for max value
    min_x, min_y = int(min_corner[0].floor().item()), int(min_corner[1].floor().item())  # MARK: SYNC
    max_x, max_y = int(max_corner[0].ceil().item()), int(max_corner[1].ceil().item())  # MARK: SYNC
    src_W, src_H = max_x - min_x, max_y - min_y

    # Calculate the pixel coodinates
    i, j = torch.meshgrid(torch.arange(src_H, dtype=src.dtype, device=src.device),
                          torch.arange(src_W, dtype=src.dtype, device=src.device),
                          indexing='ij')  # H, W
    # 0->H, 0->W
    xy1 = torch.stack([j, i, torch.ones_like(i)], dim=-1)  # mH, mW, 3
    xy1[..., :2] = xy1[..., :2] + min_corner  # shift to origin
    xy1[..., :2] = xy1[..., :2] / M
    xy1 = xy1 @ torch.inverse(homography).mT  # mH, mW, 3
    xy = xy1[..., :2] / xy1[..., -1:]  # mH, mW, 3 / mH, mW, 1 -> x, y pixels
    xy = xy * 2 - 1  # normalized to -1, 1

    src = F.grid_sample(src[None], xy[None], align_corners=False, padding_mode='zeros', mode='bilinear')[0]  # 3, mH, mW

    return min_x, min_y, max_x, max_y, src


def visualize_detection(imgs: torch.Tensor, lafs: torch.Tensor, resp: torch.Tensor, paths: List[str], ratio=1.0):
    pack = pack_laf_into_opencv_fmt(lafs, resp)  # B, N, 4
    pack[..., :2] = pack[..., :2] / ratio
    imgs = imgs.permute(0, 2, 3, 1)  # B, H, W, C
    imgs = imgs.detach().cpu().numpy()
    pack = pack.detach().cpu().numpy()
    for path, img, kps in zip(paths, imgs, pack):
        img = (img.clip(0, 1) * 255).astype(np.uint8)[..., ::-1].copy()  # bgr to rgb -> contiguous
        kps = [cv2.KeyPoint(*d) for d in kps]
        img = cv2.drawKeypoints(img, kps, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, img)


def visualize_matches(imgs_pivot: torch.Tensor,
                      lafs_pivot: torch.Tensor,
                      resp_pivot: torch.Tensor,
                      imgs: torch.Tensor,
                      lafs: torch.Tensor,
                      resp: torch.Tensor,
                      match: torch.Tensor,  # M, 3
                      paths: List[str],
                      pivot: int,
                      ratio=1.0):
    pack_pivot = pack_laf_into_opencv_fmt(lafs_pivot, resp_pivot)  # B, N, 4
    pack_pivot[..., :2] = pack_pivot[..., :2] / ratio
    imgs_pivot = imgs_pivot.permute(0, 2, 3, 1)  # B, H, W, C
    imgs_pivot = imgs_pivot.detach().cpu().numpy()
    pack_pivot = pack_pivot.detach().cpu().numpy()

    pack = pack_laf_into_opencv_fmt(lafs, resp)  # B, N, 4
    pack[..., :2] = pack[..., :2] / ratio
    imgs = imgs.permute(0, 2, 3, 1)  # B, H, W, C
    imgs = imgs.detach().cpu().numpy()
    pack = pack.detach().cpu().numpy()
    _, H, W, C = imgs.shape

    img_pivot = imgs_pivot[0]  # H, W, 3
    img_pivot = (img_pivot.clip(0, 1) * 255).astype(np.uint8)[..., ::-1].copy()  # bgr to rgb -> contiguous
    kps_pivot = pack_pivot[0]  # N, 4
    kps_pivot = [cv2.KeyPoint(*d) for d in kps_pivot]

    match = match[match[..., 0].argsort()]  # sort by image id
    uni, _, _, ind = unique_with_indices(match[..., 0], sorted=True)

    # From now on, use numpy instead of tensors since we're writing output
    uni = uni.detach().cpu().numpy()
    ind = ind.detach().cpu().numpy()
    match = match.detach().cpu().numpy()

    # For now, only consider matched points of the first image?
    def get_actual_idx(idx, pivot=pivot): return idx if idx < pivot else idx + 1
    for idx, curr, next in zip(uni, ind, np.concatenate([ind[1:], np.array([len(match)])])):
        idx = int(idx)
        curr = int(curr)
        next = int(next)
        size = next - curr
        log(f'Visualizing image pair: {colored(f"{pivot:02d}-{get_actual_idx(idx):02d}", "magenta")}, matches: {colored(f"{size}", "magenta")}')

        # Prepare source images
        path = paths[idx]
        img = imgs[idx]
        kps = pack[idx]
        img = (img.clip(0, 1) * 255).astype(np.uint8)[..., ::-1].copy()  # bgr to rgb -> contiguous
        kps = [cv2.KeyPoint(*d) for d in kps]

        # Prepare matching parameters
        match_mask = np.zeros(len(kps_pivot), dtype=np.uint8)
        match_mask[match[curr:next, 1]] = 1  # pivot selection
        match_1to2 = np.zeros((len(kps_pivot), 3), dtype=np.uint8)
        match_1to2[match[curr:next, 1]] = match[curr:next]
        matches1to2 = [cv2.DMatch(m[1], m[2], m[0], 0) for m in match_1to2]

        canvas = np.zeros([H, 2 * W, 3])
        canvas = cv2.drawMatches(img_pivot, kps_pivot, img, kps, matches1to2, canvas, matchesMask=match_mask, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)  # weird...
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, canvas)


def ransac_dlt_then_m(pvt: torch.Tensor,
                      src: torch.Tensor,
                      min_sample: int = 4,
                      inlier_iter: int = 100,
                      inlier_crit: float = 0.0001,
                      confidence: float = 0.9999,
                      max_iter: int = 10000,
                      ):
    N, C = pvt.shape
    # rand_ind = np.random.choice(N, size=(inlier_iter, min_sample))
    max_ratio = 0
    best_fit = None
    pvt_inliers = None
    src_inliers = None
    def find_min_iter(inlier_ratio, confidence, min_sample): return int(np.ceil(np.log(1 - confidence) / np.log(1 - inlier_ratio**min_sample)))

    # for i, rand in zip(range(inlier_iter), rand_ind):
    i = 0
    while i < max_iter:
        rand = np.random.choice(N, size=(min_sample), replace=False)
        pvt_rand = pvt[rand]
        src_rand = src[rand]
        homography = discrete_linear_transform(pvt_rand, src_rand)
        # Apply homography for linear error estimation
        pred = torch.cat([src, torch.ones_like(src[..., -1:])], dim=-1) @ homography.mT
        pred = pred[..., :-1] / pred[..., -1:]
        # Find ratio of inlier
        crit = (pred - pvt).pow(2).sum(-1)
        crit = crit < inlier_crit  # SSD critical threshold
        ratio = crit.sum().item() / crit.numel()  # MARK: SYNC
        if ratio > max_ratio:
            max_ratio = ratio
            best_fit = homography
            pvt_inliers = pvt[crit]  # MARK: SYNC
            src_inliers = src[crit]  # MARK: SYNC
        min_iter = find_min_iter(max_ratio, confidence, min_sample)
        min_iter = min(min_iter, inlier_iter)
        log(f'ransac random sampling:')
        log(f'iter: {colored(f"{i}", "magenta")}')
        log(f'min iter: {colored(f"{min_iter}", "magenta")}')
        log(f'remaining iter: {colored(f"{min_iter - i}", "magenta")}')
        log(f'inlier ratio: {colored(f"{ratio:.6f}", "green")}')
        log(f'max inlier ratio: {colored(f"{max_ratio:.6f}", "green")}')
        if i >= min_iter:
            break
        i += 1
    return discrete_linear_transform(pvt_inliers, src_inliers)


def main():
    # Commandline Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data/data1')
    parser.add_argument('--output_dir', default='output')
    parser.add_argument('--ext', default='.JPG')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--n_feat', default=1000, type=int)  # otherwise, typicall out of memory
    parser.add_argument('--ratio', default=1.0, type=float)  # otherwise, typicall out of memory
    parser.add_argument('--match_ratio', default=0.9, type=float)  # otherwise, typicall out of memory
    args = parser.parse_args()

    # Loading images from disk and downscale them
    log(f'Loading images from: {args.data_root}')
    imgs = sorted(glob(join(args.data_root, f'*{args.ext}')))
    imgs = list_to_tensor(parallel_execution(imgs, action=load_image), args.device)  # rgb images: B, C, H, W
    down = resize_image_tensor(imgs, args.ratio)
    B, C, H, W = imgs.shape
    B, C, nH, nW = down.shape

    # Perform feature detection (use kornia implementation)
    log(f'Performing feature detection using: {args.device}')
    feature_detector = FeatureDetector(n_features=args.n_feat).to(args.device, non_blocking=True)  # the actual feature detector
    with torch.no_grad():
        lafs, desc, resp = feature_detector(down)  # B, N, 128 -> batch size, number of descs, desc dimensionality

    # Visualize feature detection results as images
    log(f'Visualizing detected feature')
    visualize_detection(imgs,
                        lafs,
                        resp,
                        [join(args.data_root, args.output_dir, f'detect_{i:02d}.jpg') for i in range(B)],
                        args.ratio)

    # Perform feature matching
    log(f'Performing exhaustive feature matching on: {args.device}')
    pivot, match = feature_matching(desc, args.match_ratio)  # M, 3 -> image id, source feat id, target feat id

    # Visualize feature matching results
    log(f'Visualizing feature matching results')
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

    def get_actual_idx(idx, pivot=pivot): return idx if idx < pivot else idx + 1
    visualize_matches(imgs_pivot,
                      lafs_pivot,
                      resp_pivot,
                      imgs,
                      lafs,
                      resp,
                      match,
                      [join(args.data_root, args.output_dir, f'match_{pivot:02d}-{get_actual_idx(i):02d}.jpg') for i in range(B)],
                      pivot,
                      args.ratio,)

    # Construct matched feature pairs from matching results
    pvt = packed_pivot[0, match[..., 1], :2]  # M, 2, discarding scale, orientation, and response
    src = packed[match[..., 0], match[..., 2], :2]  # M, 2
    pairs = torch.cat([match, pvt, src], dim=-1)  # M, 7 -> source image id, source feat id, target feat id, pivot xy, source xy
    pairs = pairs[torch.argsort(pairs[..., 0])]  # sort pairs by source image id
    pvt, src = pairs[..., -2 - 2:-2], pairs[..., -2:]  # sorted pairs by source image id, M, 2; M, 2

    # Perform RANSAC and homography one by one for target image ids
    uni, inv, cnt, ind = unique_with_indices(pairs[..., 0], sorted=True)

    # For now, only consider matched points of the first image?
    for idx, curr, next in zip(uni, ind, torch.cat([ind[1:], ind.new_tensor([len(pairs)])])):
        idx = int(idx)  # MARK: SYNC
        curr = int(curr)  # MARK: SYNC
        next = int(next)  # MARK: SYNC
        # next_appearance - first_appearance records the size
        size = next - curr
        log(f'Processing image pair: {colored(f"{pivot:02d}-{get_actual_idx(idx):02d}", "magenta")}, matches: {colored(f"{size}", "magenta")}')
        pvt_pair = pvt[curr:next]
        src_pair = src[curr:next]
        pvt_pair = pvt_pair / args.ratio  # use original image when stitching
        src_pair = src_pair / args.ratio  # use original image when stitching

        # Normalize pixel coordinates to roughly 0, 1 for a controlled threshold
        M = max(H, W)
        pvt_pair[..., 0] = pvt_pair[..., 0] / M
        pvt_pair[..., 1] = pvt_pair[..., 1] / M
        src_pair[..., 0] = src_pair[..., 0] / M
        src_pair[..., 1] = src_pair[..., 1] / M

        # Appply RANSAC and M estimator
        homography = ransac_dlt_then_m(pvt_pair, src_pair)

        # homography = discrete_linear_transform(pvt_pair, src_pair)
        min_x, min_y, max_x, max_y, trans = homography_transform(imgs[idx], homography)
        save_image(join(args.data_root, args.output_dir, f'trans_{pivot:02d}-{get_actual_idx(idx):02d}.jpg'), trans.permute(1, 2, 0).detach().cpu().numpy())
        # break  # how do we deal with multiple blending?

    # # Determine canvas size:
    # pvt_min_x, pvt_min_y = min(0, src_min_x), min(0, src_min_y)
    # pvt_max_x, pvt_max_y = max(W, src_max_x), max(H, src_max_y)
    # pvt_W, pvt_H = pvt_max_x - pvt_min_x, pvt_max_y - pvt_min_y

    # # Linear blending for now (excess memory usage?)
    # pvt_canvas = pvt.new_zeros((C, int(pvt_H), int(pvt_W)))  # 3, pH, pW
    # src_canvas = pvt.new_zeros((C, int(pvt_H), int(pvt_W)))  # 3, pH, pW
    # pvt_canvas[-pvt_min_y:H, -pvt_min_x:W] = pvt
    # src_canvas[-pvt_min_y - src_min_y:src_H, -pvt_min_x - src_min_x:src_W] = src
    # blended = (pvt_canvas + src_canvas) / 2
    # return blended

    # save_image(join(args.data_root, args.output_dir, 'merged.jpg'), trans.permute(1, 2, 0).detach().cpu().numpy())


if __name__ == "__main__":
    main()
