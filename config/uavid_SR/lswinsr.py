"""
UnetFormer for uavid datasets with supervision training
Rui Li, 17.03.2023
"""
import os.path

from torch.utils.data import DataLoader
from geoseg.losses import *
from geoseg.datasets.uavid_dataset_sr import *
from geoseg.models.LSwinSR import LSwinSR
from catalyst.contrib.nn import Lookahead
from catalyst import utils
from queue import Queue
import torch
from threading import Thread
import torch.nn as nn

# training hparam
max_epoch = 75
ignore_index = 255
train_batch_size = 4
val_batch_size = 4
lr = 1e-3
backbone_lr = 1e-4
weight_decay = 0.0003
backbone_weight_decay = 0.01
accumulate_n = 1
visualization_n = 10
num_classes = 3
classes = CLASSES
scale = 8

net = LSwinSR(upscale=scale)
weights_name = net.name
weights_path = "model_weights/uavid_" + str(scale) + "x/{}".format(weights_name)
test_weights_name = net.name
log_name = 'uavid/{}'.format(weights_name)
monitor = 'val_ssim'
monitor_mode = 'max'
save_top_k = 3
save_last = True
check_val_every_n_epoch = 1
gpus = [0]
strategy = None
pretrained_ckpt_path = None
resume_ckpt_path = None
loss = nn.L1Loss()

# inference path for test set
image_path = 'data/UAVid_{}x/test'.format(scale)
output_path = os.path.join('data/UAVid_{}x'.format(scale), net.name, 'test')

# inference path for training and validation sets of segmentation
image_path_seg_train = 'data/UAVid_{}x/train_seg'.format(scale)
output_path_seg_train = os.path.join('data/UAVid_{}x'.format(scale), net.name, 'train')
image_path_seg_val = 'data/UAVid_{}x/val_seg'.format(scale)
output_path_seg_val = os.path.join('data/UAVid_{}x'.format(scale), net.name, 'val')


# define the dataloader
class CudaDataLoader:
    def __init__(self, loader, device, queue_size=2):
        self.device = device
        self.queue_size = queue_size
        self.loader = loader

        self.load_stream = torch.cuda.Stream(device=device)
        self.queue = Queue(maxsize=self.queue_size)

        self.idx = 0
        self.worker = Thread(target=self.load_loop)
        self.worker.setDaemon(True)
        self.worker.start()

    def load_loop(self):
        # The loop that will load into the queue in the background
        torch.cuda.set_device(self.device)
        while True:
            for i, sample in enumerate(self.loader):
                self.queue.put(self.load_instance(sample))

    def load_instance(self, sample):
        if torch.is_tensor(sample):
            with torch.cuda.stream(self.load_stream):
                return sample.to(self.device, non_blocking=True)
        elif sample is None or type(sample) == str:
            return sample
        elif isinstance(sample, dict):
            return {k: self.load_instance(v) for k, v in sample.items()}
        else:
            return [self.load_instance(s) for s in sample]

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        # 加载线程挂了
        if not self.worker.is_alive() and self.queue.empty():
            self.idx = 0
            self.queue.join()
            self.worker.join()
            raise StopIteration
        # 一个epoch加载完了
        elif self.idx >= len(self.loader):
            self.idx = 0
            raise StopIteration
        # 下一个batch
        else:
            out = self.queue.get()
            self.queue.task_done()
            self.idx += 1
        return out

    def next(self):
        return self.__next__()

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset


class _RepeatSampler(object):
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class MultiEpochsDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


train_dataset = UAVIDDataset(data_root='data/UAVid_{}x/train_sr'.format(scale), img_dir='images',
                             mask_dir='references', mode='train', mosaic_ratio=0.25, transform=train_aug,
                             img_size=(512, 512))

val_dataset = UAVIDDataset(data_root='data/UAVid_{}x/val_sr'.format(scale), img_dir='images',
                           mask_dir='references', mode='val', mosaic_ratio=0.0, transform=val_aug, img_size=(512, 512))

train_loader = MultiEpochsDataLoader(dataset=train_dataset,
                                     batch_size=train_batch_size,
                                     num_workers=0,
                                     pin_memory=True,
                                     shuffle=True,
                                     drop_last=True)

val_loader = MultiEpochsDataLoader(dataset=val_dataset,
                                   batch_size=val_batch_size,
                                   num_workers=0,
                                   shuffle=False,
                                   pin_memory=True,
                                   drop_last=False)

if gpus is not None:
    train_loader = CudaDataLoader(train_loader, 'cuda:' + str(gpus[0]))
    val_loader = CudaDataLoader(val_loader, 'cuda:' + str(gpus[0]))

# define the optimizer
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = utils.process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
# lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)
