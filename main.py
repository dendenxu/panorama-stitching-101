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


def feature_matching(desc: torch.Tensor, match_ratio: float = .7) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
    # B, B, N, N -> every image pair, every distance pair -> 36 * 500 * 500 * 128 * 4 / 2**20 MB
    # half of memory and computation wasted
    B, N, C = desc.shape
    # pvt = desc[:, None, :, None, :]
    # src =  desc[None, :, None, :, :]
    # ssd = (pvt - src).pow(2).sum(-1).sqrt()  # BBNN sum of squared error, diagonal values should be ignored
    pvt = desc[:, None].expand(B, B, N, C).reshape(B * B, N, C)
    src = desc[None, :].expand(B, B, N, C).reshape(B * B, N, C)
    # ssd = torch.cdist(pvt, src, compute_mode='donot_use_mm_for_euclid_dist').view(B, B, N, N)  # some numeric error here?
    device = desc.device
    dtype = desc.dtype
    # pvt = pvt.cpu()
    # src = src.cpu()
    pvt = pvt.double()
    src = src.double()
    ssd = torch.cdist(pvt, src).view(B, B, N, N).to(dtype)  # some numeric error here?
    # ssd = torch.cdist(pvt, src, compute_mode='donot_use_mm_for_euclid_dist').view(B, B, N, N)  # some numeric error here?

    # Find the one with cloest match to other images as the pivot (# ? not scalable?)
    min2, match = ssd.cpu().topk(2, dim=-1, largest=False)  # find closest distance
    min2, match = min2.to(device, non_blocking=True), match.to(device, non_blocking=True)
    match: torch.Tensor = match[..., 0]  # only the largest value correspond to dist B, B, N, cloest is on diagonal
    score: torch.Tensor = min2[..., 0] / min2[..., 1].clip(1e-6)  # unambiguous match should have low distance here B, B, N,
    # dist = min2[..., 0]  # ambiguous matches also present B, B, N,

    # Find actual two-way matching results
    reversed = (torch.zeros_like(match) - 1).scatter_(dim=-1, index=match.permute(1, 0, 2), src=torch.arange(N, device=match.device, dtype=match.dtype)[None, None].expand(B, B, N))  # valid index, B, B, N
    two_way_match = match == reversed  # B, B, N

    # Select valid threshold -> might not be wise to batch since every image pair is different...
    # total: B * B * N
    # dummy: B * N
    # actual ratio: (1 - 1 / B) * match_ratio
    match_ratio = (1 - 1 / B) * match_ratio + 1 / B
    threshold = score.ravel().topk(int(score.numel() * (1 - match_ratio))).values.min()  # the ratio used to discard unmatched values
    # threshold = match_ratio
    log(f'Thresholding matches with: {colored(f"{threshold:.6f}", "yellow")}')

    valid = score <= threshold  # number of matches per image? B, B, N
    valid = valid & two_way_match  # need a two way matching to make this work?
    # pivot = dist.sum(-1).sum(-1).argmin().item()  # find the pivot image to use (diagonal trivially ignored) # MARK: SYNC
    pivot = valid.sum(-1).sum(-1).argmax().item()  # find the pivot image to use (diagonal trivially ignored) # MARK: SYNC

    return pivot, valid, match, score


def discrete_linear_transform(pvt: torch.Tensor, src: torch.Tensor, quite=False) -> torch.Tensor:
    # given feature point pairs, solve them with DLT algorithm
    # pvt: N, 2 # the x prime vector
    # src: N, 2 # the x vector
    dtype = pvt.dtype
    sh = pvt.shape[:-2]
    pvt = pvt.double()
    src = src.double()  # otherwise would be numerically instable

    assert pvt.shape[-2] >= 4 and src.shape == pvt.shape, f'Needs at least four points for homography with matching shape: {pvt.shape}, {src.shape}'

    # assemble feature points into big A matrix
    # 0, 0, 0, -x, -y, -1,  y'x,  y'y,  y'
    # x, y, 1,  0,  0,  0, -x'x, -x'y, -x'
    x, y = src.chunk(2, dim=-1)
    xp, yp = pvt.chunk(2, dim=-1)
    z = torch.zeros_like(x)
    o = torch.ones_like(x)
    r0 = torch.cat([z, z, z, -x, -y, -o, yp * x, yp * y, yp], dim=-1)  # N, 9
    r1 = torch.cat([x, y, o, z, z, z, -xp * x, -xp * y, -xp], dim=-1)  # N, 9
    A = torch.stack([r0, r1], dim=-2).view(*sh, -1, 9)  # interlased

    U, S, Vh = torch.linalg.svd(A, full_matrices=True)
    V = Vh.mH  # Vh is the conjugate transpose of V
    # V = Vh  # Vh is the conjugate transpose of V
    if not quite:
        log(f'reprojection error: {colored(f"{S[-1].item():.6f}", "yellow")}')

    h = V[:, -1]  # 9, the homography
    H = h.view(*sh, 3, 3) / h[-1]  # normalize homography to 1
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


def homography_transform(img: torch.Tensor, homography: torch.Tensor):
    # blends two images with homography transform
    # determine the final size of the blended image to construct a canvas
    # pvt: 3, H, W
    # src: 3, H, W
    # homography: 3, 3
    C, H, W = img.shape
    scale = img.new_tensor([W, H])

    # straight lines will always map to straight lines
    corners = img.new_tensor([
        [0, 0, 1],
        [0, H, 1],
        [W, 0, 1],
        [W, H, 1],
    ])  # 4, 3
    corners[..., :-1] = corners[..., :-1] / scale  # normalization
    corners = corners @ homography.mT  # 4, 3
    corners = corners[..., :-1] / corners[..., -1:]  # 4, 3 / 4, 1 -> x, y pixels
    corners = corners * scale  # renormalize pixel coordinates
    min_corner = corners.min(dim=0)[0]  # x, y for min value
    max_corner = corners.max(dim=0)[0]  # x, y for max value
    min_x, min_y = int(min_corner[0].floor().item()), int(min_corner[1].floor().item())  # MARK: SYNC
    max_x, max_y = int(max_corner[0].ceil().item()), int(max_corner[1].ceil().item())  # MARK: SYNC
    src_W, src_H = max_x - min_x, max_y - min_y

    # Calculate the pixel coodinates
    i, j = torch.meshgrid(torch.arange(src_H, dtype=img.dtype, device=img.device),
                          torch.arange(src_W, dtype=img.dtype, device=img.device),
                          indexing='ij')  # H, W
    # 0->H, 0->W
    xy1 = torch.stack([j, i, torch.ones_like(i)], dim=-1)  # mH, mW, 3
    xy1[..., :2] = xy1[..., :2] + min_corner  # shift to origin
    xy1[..., :2] = xy1[..., :2] / scale
    xy1 = xy1 @ torch.inverse(homography).mT  # mH, mW, 3
    xy = xy1[..., :-1] / xy1[..., -1:]  # mH, mW, 3 / mH, mW, 1 -> x, y pixels
    xy = xy * 2 - 1  # normalized to -1, 1, mH, mW, 2

    # Sampled pixel values
    img = F.grid_sample(img[None], xy[None], align_corners=False, padding_mode='zeros', mode='bilinear')[0]  # 3, mH, mW

    # Valid pixel values
    msk = (xy >= -1).all(-1) & (xy <= 1).all(-1)  # both x and y need to meet crit

    return min_x, min_y, max_x, max_y, img, msk


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


def random_sampling_consensus(pvt: torch.Tensor,
                              src: torch.Tensor,
                              min_sample: int = 4,
                              inlier_iter: int = 100,
                              inlier_crit: float = 1e-5,
                              confidence: float = 1 - 1e-5,
                              max_iter: int = 10000,
                              m_repeat: int = 2,  # repeat m-estimator 10 times
                              quite=False,
                              ):
    N, C = pvt.shape
    # rand_ind = np.random.choice(N, size=(inlier_iter, min_sample))
    dtype = pvt.dtype
    src = src.double()
    pvt = pvt.double()

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
        homography = discrete_linear_transform(pvt_rand, src_rand, quite=quite)
        # Apply homography for linear error estimation
        pred = apply_homography(src, homography)
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
        min_iter = max(min_iter, inlier_iter)
        if not quite:
            log(f'RANSAC random sampling:')
            log(f'Iter: {colored(f"{i}", "magenta")}')
            log(f'Min iter: {colored(f"{min_iter}", "magenta")}')
            log(f'Remaining iter: {colored(f"{min_iter - i}", "magenta")}')
            log(f'Inlier ratio: {colored(f"{ratio:.6f}", "green")}')
            log(f'Max inlier ratio: {colored(f"{max_ratio:.6f}", "green")}')
        if i >= min_iter:
            break
        i += 1

    ransac_iter = i
    inlier_ratio = max_ratio

    # return discrete_linear_transform(pvt_inliers, src_inliers).to(dtype), ransac_iter, inlier_ratio
    for i in range(m_repeat):
        homography = m_estimator(pvt_inliers, src_inliers, quite=quite)

        # Apply homography for linear error estimation
        pred = apply_homography(src, homography)
        # Find ratio of inlier
        crit = (pred - pvt).pow(2).sum(-1)
        crit = crit < inlier_crit  # SSD critical threshold
        ratio = crit.sum().item() / crit.numel()  # MARK: SYNC
        if ratio > max_ratio:
            max_ratio = ratio
            best_fit = homography
            pvt_inliers = pvt[crit]  # MARK: SYNC
            src_inliers = src[crit]  # MARK: SYNC
        if not quite:
            log(f'M-estimator repeatation:')
            log(f'Iter: {colored(f"{i}", "magenta")}')
            log(f'Inlier ratio: {colored(f"{ratio:.6f}", "green")}')
            log(f'Max inlier ratio: {colored(f"{max_ratio:.6f}", "green")}')
    return best_fit.to(dtype), ransac_iter, inlier_ratio


def apply_homography(src: torch.Tensor, homography: torch.Tensor):
    src = torch.cat([src, torch.ones_like(src[..., -1:])], dim=-1)
    pvt = src @ homography.mT
    pvt = pvt[..., :-1] / pvt[..., -1:]
    return pvt


def m_estimator(pvt: torch.Tensor, src: torch.Tensor, iter=1000, lr=1e-2, quite=False):
    def rou(error: torch.Tensor, sigma=1):
        squared = error ** 2
        return squared / (squared + sigma ** 2)

    def sym(pvt: torch.Tensor, src: torch.Tensor, homography: torch.Tensor, homography_inv=None):
        if homography_inv is None:
            homography_inv = homography.inverse()
        loss = (pvt - apply_homography(src, homography)).pow(2).sum(-1).sqrt() + (src - apply_homography(pvt, homography_inv)).pow(2).sum(-1).sqrt()
        loss = loss.mean()
        return loss

    homography = discrete_linear_transform(pvt, src, quite=quite)  # initialization
    homography.requires_grad_()
    optim = torch.optim.Adam([homography], lr=lr)

    if not quite:
        pbar = tqdm(total=iter)
    for i in range(iter):
        loss = rou(sym(pvt, src, homography))  # robust symmetric loss
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        if not quite:
            pbar.desc = f'Loss: {loss.item():.8f}'
            pbar.update(1)
    homography = homography.detach().requires_grad_(False)
    homography = homography / homography[..., -1, -1]
    return homography


def parallel_floyd_warshall(distance: torch.Tensor):
    # The Floyd-Warshall algorithm is a popular algorithm for finding the shortest path for each vertex pair in a weighted directed graph.
    # https://www.baeldung.com/cs/floyd-warshall-shortest-path
    # https://cse.buffalo.edu/faculty/miller/Courses/CSE633/Asmita-Gautam-Spring-2019.pdf
    # https://saadmahmud14.medium.com/parallel-programming-with-cuda-tutorial-part-4-the-floyd-warshall-algorithm-5e1281c46bf6
    assert distance.shape[-1] == distance.shape[-2], 'Graph matrix should be square'
    V = distance.shape[-1]
    connect = distance.new_full(distance.shape, fill_value=-1, dtype=torch.long)
    for k in range(V):
        connect_with_k = distance[:, k:k + 1].expand(-1, V) + distance[k:k + 1, :].expand(V, -1)
        closer_with_k = connect_with_k < distance
        distance = torch.where(closer_with_k, connect_with_k, distance)
        connect = torch.where(closer_with_k, k, connect)  # if 1, go to k, else stay put
    return distance, connect  # geodesic distance (closest distance of every pair of element)


def magenta(x): return colored(str(x), 'magenta')
def cyan(x): return colored(str(x), 'cyan')


def extract_path_from_connect(i, j, connect) -> List[int]:
    # https://stackoverflow.com/questions/64163232/how-to-record-the-path-in-this-critical-path-algo-python-floyd-warshall
    k = connect[i][j]
    if k == -1:
        return [i, j]
    else:
        path = extract_path_from_connect(i, k, connect)
        path.pop()  # remove k to avoid duplicates
        path.extend(extract_path_from_connect(k, j, connect))
        return path


def main():
    # Commandline Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data/data1')
    parser.add_argument('--output_dir', default='output')
    parser.add_argument('--ext', default='.JPG')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--pivot', default=-1, type=int)
    parser.add_argument('--n_feat', default=10000, type=int)  # otherwise, typically out of memory
    parser.add_argument('--ratio', default=1.0, type=float)  # otherwise, typically out of memory
    parser.add_argument('--match_ratio', default=0.9, type=float)  # otherwise, typically out of memory
    parser.add_argument('--verbose', action='store_true')  # otherwise, typically out of memory
    args = parser.parse_args()

    # Loading images from disk and downscale them
    log(f'Loading images from: {cyan(args.data_root)}')
    imgs = sorted(glob(join(args.data_root, f'*{args.ext}')))
    imgs = list_to_tensor(parallel_execution(imgs, action=load_image), args.device)  # rgb images: B, C, H, W
    # imgs = imgs.double()
    down = resize_image_tensor(imgs, args.ratio)
    B, C, H, W = imgs.shape
    B, C, nH, nW = down.shape
    scale = imgs.new_tensor([W, H])

    # Perform feature detection (use kornia implementation)
    log(f'Performing feature detection using: {colored(f"{args.device}", "cyan")}')
    feature_detector = FeatureDetector(n_features=args.n_feat).to(args.device, non_blocking=True)  # the actual feature detector
    with torch.no_grad():
        lafs, desc, resp = feature_detector(down)  # B, N, 128 -> batch size, number of descs, desc dimensionality

    # Visualize feature detection results as images
    log(f'Visualizing detected feature at: {cyan(join(args.data_root, args.output_dir))}')
    visualize_detection(imgs,
                        lafs,
                        resp,
                        [join(args.data_root, args.output_dir, f'detect_{i:02d}.jpg') for i in range(B)],
                        args.ratio)

    # Perform feature matching
    log(f'Performing exhaustive feature matching on: {colored(f"{args.device}", "cyan")}')
    pivot, valid, match, score = feature_matching(desc, args.match_ratio)
    # pivot: int
    # valid: B, B, N indicates whether a match is valid
    # match: B, B, N source image id, target image id, target feature id
    # score: B, B, N matching distance corresponding to matches indicated by match

    log(f'Constructing sequential homography on: {colored(f"{args.device}", "cyan")}')
    pivot = pivot if args.pivot < 0 else args.pivot  # use user pivot if defined
    graph = 1 / valid.sum(-1)  # inverse of number of matches
    graph[torch.arange(B), torch.arange(B)] = 0  # no cost to self
    distance, connect = parallel_floyd_warshall(graph)  # geodesic distance (closest distance of every pair of element) -> B, B; B, B

    # Find location of 2d keypoints
    keypoints2d = get_laf_center(lafs)  # B, N, 2

    # Define the matrix of homography
    homographies = keypoints2d.new_zeros(B, B, 3, 3)  # to be filled with actual homography
    visited = connect.new_zeros(B, B, dtype=torch.bool)  # zero indicateds not visited
    match_map = connect.new_zeros(B, B, dtype=torch.long)
    iter_map = connect.new_zeros(B, B, dtype=torch.long)
    ratio_map = connect.new_zeros(B, B, dtype=torch.float)

    # Things should be moved to cpu first to avoid excessive syncing
    connect = connect.detach().cpu().numpy()
    visited = visited.detach().cpu().numpy()
    match_map = match_map.detach().cpu().numpy()
    iter_map = iter_map.detach().cpu().numpy()
    ratio_map = ratio_map.detach().cpu().numpy()
    paths = [[extract_path_from_connect(i, j, connect) for j in range(B)] for i in range(B)]

    ret = []  # return values for summary and linear blending
    meta = []
    for source in range(B):
        log(f'Processing homography pair: {magenta(f"{source:02d}-{pivot:02d}")}')
        # Iterate through all images
        # If pivot, trivially return the original image
        if source == pivot:
            ret.append([0, 0, W, H, imgs[source],
                        torch.ones_like(imgs[source][0], dtype=torch.bool), ])
            meta.append([pivot, source, -1, -1, -1, -1, []])
            continue

        # If not pivot image, do a connected homography to pivot
        # Find the shortest path between src and pvt
        shortest_path = paths[source][pivot] # number of jumps from index to pivot

        # Find homography transformation from source to pivot of this index
        cnn = []
        cum = 0
        cum_match = 0
        cum_iter = 0
        cum_ratio = 1.0
        prev = source
        cum_homo = torch.eye(3, dtype=homographies.dtype, device=homographies.device)  # 3, 3
        for next in shortest_path:
            if next == prev: continue  # skip self loop
            # Need to find homography if jumping to next node
            if not visited[prev, next]:
                # If not visited, need to find homography
                # Find the matches between prev and next
                valid_prev_next = valid[prev, next]  # N,
                match_prev_next = match[prev, next]  # N, stores matched feature id

                valid_prev_next = valid_prev_next.nonzero()[..., 0]  # M, # MARK: SYNC
                match_prev_next = match_prev_next[valid_prev_next]  # M

                src = keypoints2d[prev][valid_prev_next]  # valid match, source image id
                pvt = keypoints2d[next][match_prev_next]  # valid match, target image id

                src = src / scale  # normalize the images to 0, 1
                pvt = pvt / scale  # normalize the images to 0, 1

                homo, iter, ratio = random_sampling_consensus(pvt, src, quite=not args.verbose)
                homographies[prev, next] = homo
                visited[prev, next] = 1
                match_map[prev, next] = len(src)
                iter_map[prev, next] = iter
                ratio_map[prev, next] = ratio

            # If already visited, just left multiply
            cum_homo = homographies[prev, next] @ cum_homo
            cum_match += match_map[prev, next]
            cum_iter += iter_map[prev, next]
            cum_ratio *= ratio_map[prev, next]
            cum += 1
            cnn.append(next)

            # If jumping, prev should updated
            prev = next

        # Apply the constructed homography transformation to get the actual result
        min_x, min_y, max_x, max_y, img, msk = homography_transform(imgs[source], cum_homo)
        ret.append([min_x, min_y, max_x, max_y, img, msk, ])
        meta.append([pivot, source, cum_match, cum_iter, cum_ratio, cum, cnn])

    ret = list(zip(*ret))  # inverted batching
    can_min_x = min(ret[0])
    can_min_y = min(ret[1])
    can_max_x = max(ret[2])
    can_max_y = max(ret[3])
    can_W = can_max_x - can_min_x  # canvas size
    can_H = can_max_y - can_min_y  # canvas size
    log(f'Canvas size: {colored(f"{can_H}, {can_W}", "magenta")}')

    canvas = torch.zeros((3, can_H, can_W), dtype=torch.float, device=args.device)
    accumu = torch.zeros((1, can_H, can_W), dtype=torch.float, device=args.device)

    # Linear blending for now (excess memory usage?)
    log(f'Performing linear blending image stitching on: {colored(f"{args.device}", "cyan")}')
    ret = list(zip(*ret))
    for min_x, min_y, max_x, max_y, img, msk in ret:
        x = min_x - can_min_x
        y = min_y - can_min_y
        canvas[..., y:y + img.shape[1], x:x + img.shape[2]] += img
        accumu[..., y:y + img.shape[1], x:x + img.shape[2]] += msk
    canvas = canvas / accumu.clip(1e-6) * (accumu > 0)

    # Save the blended image to disk for visualization
    result_path = join(args.data_root, args.output_dir, 'canvas.jpg')
    save_image(result_path, canvas.permute(1, 2, 0).detach().cpu().numpy())
    log(f'Blended result saved to: {cyan(result_path)}')

    # Output some summary for debugging
    log(f'Summary:')
    for source in range(len(meta)):
        pivot, source, cum_match, cum_iter, cum_ratio, cum, cnn = meta[source]
        min_x, min_y, max_x, max_y, img, msk = ret[source]
        log(f'Pair: {magenta(f"{source:02d}-{pivot:02d}")}')
        log(f'-- Number of matches:  {magenta(cum_match)}')
        log(f'-- Upper left corner:  {magenta(f"{min_x-can_min_x}-{min_y-can_min_x}")}')
        log(f'-- Lower right corner: {magenta(f"{max_x-can_min_x}-{max_y-can_min_x}")}')
        log(f'-- Valid pixel region: {magenta(f"{max_x-min_x}-{max_y-min_y}")}')
        log(f'-- Final inlier ratio: {magenta(f"{cum_ratio:.6f}")}')
        log(f'-- RANSAC iteration:   {magenta(cum_iter)}')
        log(f'-- Connections:        {magenta(cnn)}')


if __name__ == "__main__":
    main()
