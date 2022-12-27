# experiment runner script
import os
import argparse
from os.path import join

from utils import run

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='data')
args = parser.parse_args()

for data in sorted(os.listdir(args.data_root)):
    data_root = join(args.data_root, data)
    command = f'python main.py --data_root {data_root} --output_dir output_sift --desc_type sift --visualize'
    run(command)
    command = f'python main.py --data_root {data_root} --output_dir output_pixel --desc_type pixel --visualize'
    run(command)
