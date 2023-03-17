import argparse
import glob
import ttach as tta
import albumentations as albu
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from train_supervision import *
import random
import os
from tools.metric_sp import Evaluator
import torch
import cv2


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
    arg("-c", "--config_path", type=Path, default='./config/uavid_SR/lswin2sr.py', help="Path to config")
    arg("-t", "--tta", help="Test time augmentation.", default='lr', choices=[None, "d4", "lr"])
    arg("-ph", "--patch-height", help="height of patch size", type=int, default=128)
    arg("-pw", "--patch-width", help="width of patch size", type=int, default=128)
    arg("-b", "--batch-size", help="batch size", type=int, default=4)
    arg("-d", "--dataset", help="dataset", default="uavid", choices=["pv", "landcoverai", "uavid"])
    return parser.parse_args()


def load_checkpoint(checkpoint_path, model):
    pretrained_dict = torch.load(checkpoint_path)['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    return model


def get_img_padded(image, patch_size):
    oh, ow = image.shape[0], image.shape[1]
    rh, rw = oh % patch_size[0], ow % patch_size[1]

    width_pad = 0 if rw == 0 else patch_size[1] - rw
    height_pad = 0 if rh == 0 else patch_size[0] - rh
    # print(oh, ow, rh, rw, height_pad, width_pad)
    h, w = oh + height_pad, ow + width_pad

    pad = albu.PadIfNeeded(min_height=h, min_width=w, border_mode=0,
                           position='bottom_right', value=[0, 0, 0])(image=image)
    img_pad = pad['image']
    return img_pad, height_pad, width_pad


class InferenceDataset(Dataset):
    def __init__(self, tile_list=None, tile_list_ref=None):
        self.tile_list = tile_list
        self.tile_list_ref = tile_list_ref

    def __getitem__(self, index):
        img = self.tile_list[index]
        img_id = index
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        ref = self.tile_list_ref[index]
        ref = torch.from_numpy(ref).permute(2, 0, 1).float() / 255.0
        results = dict(img_id=img_id, img=img, ref=ref)
        return results

    def __len__(self):
        return len(self.tile_list)


def make_dataset_for_one_huge_image(img_path, patch_size, scale=4):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    ref = cv2.imread(img_path.replace('Images', 'references'), cv2.IMREAD_COLOR)
    ref = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)

    tile_list = []
    image_pad, height_pad, width_pad = get_img_padded(img.copy(), patch_size)

    tile_list_ref = []
    ref_pad, height_pad_ref, width_pad_ref = get_img_padded(ref.copy(), [i * scale for i in patch_size])

    output_height, output_width = image_pad.shape[0], image_pad.shape[1]
    output_height_ref, output_width_ref = ref_pad.shape[0], ref_pad.shape[1]

    for x in range(0, output_height, patch_size[0]):
        for y in range(0, output_width, patch_size[1]):
            image_tile = image_pad[x:x+patch_size[0], y:y+patch_size[1]]
            tile_list.append(image_tile)

    for x in range(0, output_height_ref, patch_size[0] * scale):
        for y in range(0, output_width_ref, patch_size[1] * scale):
            ref_tile = ref_pad[x:x+patch_size[0]*scale, y:y+patch_size[1]*scale]
            tile_list_ref.append(ref_tile)

    dataset = InferenceDataset(tile_list=tile_list, tile_list_ref=tile_list_ref)
    # return dataset, width_pad, height_pad, output_width, output_height, image_pad, img.shape
    return dataset, width_pad_ref, height_pad_ref, output_width_ref, output_height_ref, ref_pad, ref.shape


def main():
    args = get_args()
    seed_everything(42)
    config = py2cfg(args.config_path)
    seqs = os.listdir(config.image_path)

    # print(img_paths)
    patch_size = (args.patch_height, args.patch_width)

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

    for seq in seqs:
        img_paths = []
        output_path = os.path.join(config.output_path, str(seq), 'Images')
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        for ext in ('*.tif', '*.png', '*.jpg'):
            img_paths.extend(glob.glob(os.path.join(config.image_path, str(seq), 'Images', ext)))
        img_paths.sort()
        # print(img_paths)
        for img_path in img_paths:
            img_name = img_path.split('\\')[-1]
            # print('origin mask', original_mask.shape)
            dataset, width_pad, height_pad, output_width, output_height, img_pad, img_shape = \
                make_dataset_for_one_huge_image(img_path, patch_size, config.scale)
            # print('img_padded', img_pad.shape)
            output_tiles = []
            output_mask = np.zeros(shape=(3, output_height, output_width), dtype=np.float32)
            k = 0
            with torch.no_grad():
                dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size,
                                        drop_last=False, shuffle=False)
                for input in tqdm(dataloader):
                    predictions = model(input['img'].cuda(config.gpus[0]))
                    reference = input['ref']

                    image_ids = input['img_id']
                    # print('prediction', predictions.shape)
                    # print(np.unique(predictions))

                    for i in range(predictions.shape[0]):
                        metrics.add_batch(reference[i].cpu().detach().numpy(),
                                                     predictions[i].cpu().detach().numpy())
                        mask = predictions[i].cpu().numpy()
                        output_tiles.append((mask, image_ids[i].cpu().numpy()))

            for m in range(0, output_height, patch_size[0] * config.scale):
                for n in range(0, output_width, patch_size[1] * config.scale):
                    output_mask[:, m:m + patch_size[0] * config.scale, n:n + patch_size[1] * config.scale] = \
                        output_tiles[k][0]
                    k = k + 1

            output_mask = output_mask[:, -img_shape[0]:, -img_shape[1]:]
            output_mask = (output_mask * 255.0).round().astype(np.uint8)

            output_mask = output_mask.swapaxes(0, 2)
            output_mask = output_mask.swapaxes(0, 1)

            # output_mask = cv2.cvtColor(output_mask, cv2.COLOR_RGB2BGR)
            output_mask = cv2.cvtColor(output_mask, cv2.COLOR_RGB2BGR)

            cv2.imwrite(os.path.join(output_path, img_name), output_mask)

    psnr = metrics.get_psnr()
    ssim = metrics.get_ssim()
    mae = metrics.get_mae()
    test_value = {'psnr': psnr,
                  'ssim': ssim,
                  'mae': mae}
    print('test:', test_value)


if __name__ == "__main__":
    seed_everything(42)
    main()