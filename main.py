import os
import torch
import kornia
import argparse
import numpy as np
from torch import nn
from glob import glob
from tqdm import tqdm
from os.path import join

from utils import parallel_execution, load_image, load_unchanged, save_image, save_unchanged, list_to_tensor, tensor_to_list

from kornia.feature.scale_space_detector import ScaleSpaceDetector
from kornia.feature import BlobHessian, BlobDoG, SIFTDescriptor, LAFOrienter, extract_patches_from_pyramid
from kornia.geometry import ScalePyramid, ConvQuadInterp3d
from kornia.color import rgb_to_grayscale

# https://github.com/kornia/kornia-examples/blob/master/image-matching-example.ipynb


class FeatureDetector(nn.Module):
    def __init__(self, PS=41, n_features=4000) -> None:
        super().__init__()
        self.PS = PS
        self.n_features = n_features

        self.resp = BlobDoG()
        self.scale_pyr = ScalePyramid(3, 1.6, PS, double_image=True)
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

    def forward(self, gray: torch.Tensor):
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
        return descs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data/data1')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    imgs = glob(join(args.data_root, '*'))
    imgs = list_to_tensor(parallel_execution(imgs, action=load_image), args.device)  # rgb images: B, C, H, W

    feature_detector = FeatureDetector().to(args.device, non_blocking=True)  # the actual feature detector
    descs = feature_detector(imgs)
    __import__('ipdb').set_trace()


if __name__ == "__main__":
    main()
