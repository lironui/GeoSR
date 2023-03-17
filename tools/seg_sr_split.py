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
    parser.add_argument("--basic-path", default="data/uavid/uavid_train_val")
    parser.add_argument("--train-seg-path", default="data/uavid/train_seg")
    parser.add_argument("--train-sr-path", default="data/uavid/train_sr")
    parser.add_argument("--val-seg-path", default="data/uavid/val_seg")
    parser.add_argument("--val-sr-path", default="data/uavid/val_sr")
    return parser.parse_args()


def split_seg_sr(basic_path, train_seg_path, train_sr_path, val_seg_path, val_sr_path):
    for subfold in os.listdir(basic_path):
        ori_directory = os.path.join(basic_path, subfold, 'Images')
        lab_directory = os.path.join(basic_path, subfold, 'Labels')

        train_seg_directory = os.path.join(train_seg_path, subfold, 'references')
        train_sr_directory = os.path.join(train_sr_path, subfold, 'references')
        train_seg_directory_label = os.path.join(train_seg_path, subfold, 'labels')
        train_sr_directory_label = os.path.join(train_sr_path, subfold, 'labels')

        val_seg_directory = os.path.join(val_seg_path, subfold, 'references')
        val_sr_directory = os.path.join(val_sr_path, subfold, 'references')
        val_seg_directory_label = os.path.join(val_seg_path, subfold, 'labels')
        val_sr_directory_label = os.path.join(val_sr_path, subfold, 'labels')

        for filename in os.listdir(ori_directory):
            if int(filename[3]) == 8:
                if not os.path.exists(os.path.join(val_sr_directory)):
                    makedir(os.path.join(val_sr_directory, filename))
                    makedir(os.path.join(val_sr_directory_label, filename))

                shutil.copyfile(os.path.join(ori_directory, filename),
                                os.path.join(val_sr_directory, filename))
                shutil.copyfile(os.path.join(lab_directory, filename),
                                os.path.join(val_sr_directory_label, filename))
            elif int(filename[3]) == 9:
                if not os.path.exists(os.path.join(val_seg_directory)):
                    makedir(os.path.join(val_seg_directory, filename))
                    makedir(os.path.join(val_seg_directory_label, filename))
                shutil.copyfile(os.path.join(ori_directory, filename),
                                os.path.join(val_seg_directory, filename))
                shutil.copyfile(os.path.join(lab_directory, filename),
                                os.path.join(val_seg_directory_label, filename))

            else:
                if int(filename[3]) % 2 == 0:
                    if not os.path.exists(os.path.join(train_sr_directory)):
                        makedir(os.path.join(train_sr_directory, filename))
                        makedir(os.path.join(train_sr_directory_label, filename))

                    shutil.copyfile(os.path.join(ori_directory, filename),
                                    os.path.join(train_sr_directory, filename))
                    shutil.copyfile(os.path.join(lab_directory, filename),
                                    os.path.join(train_sr_directory_label, filename))
                else:
                    if not os.path.exists(os.path.join(train_seg_directory)):
                        makedir(os.path.join(train_seg_directory, filename))
                        makedir(os.path.join(train_seg_directory_label, filename))

                    shutil.copyfile(os.path.join(ori_directory, filename),
                                    os.path.join(train_seg_directory, filename))
                    shutil.copyfile(os.path.join(lab_directory, filename),
                                    os.path.join(train_seg_directory_label, filename))


if __name__ == "__main__":
    seed_everything(42)
    args = parse_args()
    basic_path = args.basic_path
    train_seg_path = args.train_seg_path
    train_sr_path = args.train_sr_path
    val_seg_path = args.val_seg_path
    val_sr_path = args.val_sr_path

    split_seg_sr(basic_path, train_seg_path, train_sr_path, val_seg_path, val_sr_path)
