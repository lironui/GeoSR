import os
import cv2
import shutil
import argparse
import random
import torch
import numpy as np


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def makedir(dir_path):
    dir_path = os.path.dirname(dir_path)
    if os.path.exists(dir_path):
        pass
    else:
        os.makedirs(dir_path)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/uavid/uavid_test")
    parser.add_argument("--output-dir", default="data/UAVid_2x/test")
    parser.add_argument("--scale", type=int, default=2)
    return parser.parse_args()


def test_downsample(basic_path, new_path, scale):
    for subfold in os.listdir(basic_path):
        high_res_directory = os.path.join(basic_path, subfold, 'Images')
        low_res_directory = os.path.join(new_path, subfold, 'images')
        ref_directory = os.path.join(new_path, subfold, 'references')

        for filename in os.listdir(high_res_directory):
            # Read high res images from input directory
            high_res = cv2.imread(os.path.join(high_res_directory, filename))

            # Blur images with gaussian
            high_res = cv2.GaussianBlur(high_res, (0, 0), 1, 1)


            # Resize 1/4
            low_res_4x = cv2.resize(high_res, (0, 0), fx=1 / scale, fy=1 / scale,
                                   interpolation=cv2.INTER_CUBIC)

            if not os.path.exists(os.path.join(low_res_directory)):
                makedir(os.path.join(low_res_directory, filename))
                makedir(os.path.join(ref_directory, filename))

            cv2.imwrite(os.path.join(low_res_directory, filename), low_res_4x)
            shutil.copyfile(os.path.join(high_res_directory, filename),
                                    os.path.join(ref_directory, filename))


if __name__ == "__main__":
    seed_everything(42)
    args = parse_args()
    basic_path = args.input_dir
    new_path = args.output_dir
    scale = args.scale

    test_downsample(basic_path, new_path, scale)