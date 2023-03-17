import argparse
import glob
import time

import ttach as tta
import albumentations as albu
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from train_supervision_SR import *
import random
import os
from tools.metric_sp import Evaluator
from queue import Queue
import torch
from threading import Thread
import shutil
from PIL import Image


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config-path", type=Path, default='./config/uavid_SR/lswin2sr.py', help="Path to config")
    arg("-m", "--prediction-mode", type=int, default=2, help="1 for training set 2 for validation set")
    arg("-t", "--tta", help="Test time augmentation.", default='lr', choices=[None, "d4", "lr"])
    arg("-b", "--batch-size", help="batch size", type=int, default=4)
    return parser.parse_args()


def load_checkpoint(checkpoint_path, model):
    pretrained_dict = torch.load(checkpoint_path)['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    return model


class InferenceDataset(Dataset):
    def __init__(self, tile_list=None, tile_list_ref=None):
        self.tile_list = tile_list
        self.tile_list_ref = tile_list_ref

    def __getitem__(self, index):
        img_name = self.tile_list[index]
        ref_name = self.tile_list_ref[index]

        img = Image.open(img_name).convert('RGB')
        ref = Image.open(ref_name).convert('RGB')
        img, ref = np.array(img), np.array(ref)
        img_id = self.tile_list[index].split('\\')[-1]

        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        ref = torch.from_numpy(ref).permute(2, 0, 1).float() / 255.0
        results = dict(img_id=img_id, img=img, ref=ref)
        return results

    def __len__(self):
        return len(self.tile_list)

def main():
    args = get_args()
    seed_everything(42)
    config = py2cfg(args.config_path)

    if args.prediction_mode == 1:
        image_path = config.image_path_seg_train
        output_path = config.output_path_seg_train
    elif args.prediction_mode == 2:
        image_path = config.image_path_seg_val
        output_path = config.output_path_seg_val

    model = Supervision_Train.load_from_checkpoint(os.path.join(config.weights_path, config.test_weights_name+'.ckpt'), config=config)

    model.cuda(config.gpus[0])
    model.eval()

    metrics = Evaluator(normalization=True)

    if args.tta == "lr":
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.VerticalFlip()
            ]
        )
        model = tta.SegmentationTTAWrapper(model, transforms)
    elif args.tta == "d4":
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                # tta.VerticalFlip(),
                # tta.Rotate90(angles=[0, 90, 180, 270]),
                tta.Scale(scales=[0.75, 1, 1.25, 1.5, 1.75]),
                # tta.Multiply(factors=[0.8, 1, 1.2])
            ]
        )
        model = tta.SegmentationTTAWrapper(model, transforms)

    img_paths = []
    ref_paths = []
    output_img_path = os.path.join(output_path, 'images')
    output_mask_path = os.path.join(output_path, 'masks')
    if not os.path.exists(output_img_path):
        os.makedirs(output_img_path)
    for ext in ('*.tif', '*.png', '*.jpg'):
        img_paths.extend(glob.glob(os.path.join(image_path, 'images', ext)))
        ref_paths.extend(glob.glob(os.path.join(image_path, 'references', ext)))

    dataset = InferenceDataset(tile_list=img_paths, tile_list_ref=ref_paths)
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size,
                            drop_last=False, shuffle=False)
    with torch.no_grad():
        for input in tqdm(dataloader):
            predictions = model(input['img'].cuda(config.gpus[0])).cpu()
            reference = input['ref']
            for i in range(predictions.shape[0]):
                metrics.add_batch(reference[i].numpy(), predictions[i].numpy())
                mask = predictions[i].numpy()

                mask = mask.swapaxes(0, 2)
                mask = mask.swapaxes(0, 1)

                output_mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
                output_mask = (output_mask * 255.0).round().astype(np.uint8)

                cv2.imwrite(os.path.join(output_img_path, input['img_id'][i]), output_mask)

    psnr = metrics.get_psnr()
    ssim = metrics.get_ssim()
    mae = metrics.get_mae()
    test_value = {'psnr': psnr,
                  'ssim': ssim,
                  'mae': mae}
    print('test:', test_value)

    shutil.copytree(os.path.join(image_path, 'masks'),
                    output_mask_path)


if __name__ == "__main__":
    seed_everything(42)
    main()