import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from tools.cfg import py2cfg
import os
import torch
from torch import nn
import cv2
# import visualize
import numpy as np
import argparse
from pathlib import Path
from tools.metric_sr import Evaluator
from pytorch_lightning.loggers import CSVLogger
import random


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
    arg("-c", "--config_path", type=Path, help="Path to the config.",
        default='./config/uavid_sr/ttst.py')
    return parser.parse_args()


class Supervision_Train(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.net = config.net
        self.automatic_optimization = False

        self.loss = config.loss

        self.metrics_train = Evaluator()
        self.metrics_val = Evaluator()

    def forward(self, x):
        # only net is used in the prediction/inference
        seg_pre = self.net(x)
        return seg_pre

    def training_step(self, batch, batch_idx):
        img, mask = batch['img'], batch['gt_semantic_seg']

        prediction = self.net(img)
        loss = self.loss(prediction, mask)

        for i in range(mask.shape[0]):
            self.metrics_train.add_batch(mask[i].cpu().detach().numpy(), prediction[i].cpu().detach().numpy())

        # supervision stage
        opt = self.optimizers(use_pl_optimizer=False)
        self.manual_backward(loss)
        if (batch_idx + 1) % self.config.accumulate_n == 0:
            opt.step()
            opt.zero_grad()

        sch = self.lr_schedulers()
        if self.trainer.is_last_batch and (self.trainer.current_epoch + 1) % 1 == 0:
            sch.step()

        # if batch_idx % self.config.visualization_n == 0:
        #     vis.img_many({"Reference": mask[-1].cpu().detach(), "Prediction": prediction[-1].cpu().detach()})
        # vis.plot("loss", loss.item())
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        psnr = self.metrics_train.get_psnr()
        ssim = self.metrics_train.get_ssim()
        mae = self.metrics_train.get_mae()

        eval_value = {'psnr': psnr,
                      'ssim': ssim,
                      'mae': mae}
        print('train:', eval_value)

        self.metrics_train.reset()
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        log_dict = {"train_loss": loss, 'train_psnr': psnr, 'train_ssim': ssim, 'train_mae': mae}
        self.log_dict(log_dict, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        img, mask = batch['img'], batch['gt_semantic_seg']
        prediction = self.forward(img)
        for i in range(mask.shape[0]):
            self.metrics_val.add_batch(mask[i].cpu().numpy(), prediction[i].cpu().numpy())

        # vis.img_many({"Val Reference": mask[-1].cpu().detach(), "Val Prediction": prediction[-1].cpu().detach()})
        loss_val = self.loss(prediction, mask)
        return {"loss_val": loss_val}

    def validation_epoch_end(self, outputs):
        psnr = self.metrics_val.get_psnr()
        ssim = self.metrics_val.get_ssim()
        mae = self.metrics_val.get_mae()

        eval_value = {'psnr': psnr,
                      'ssim': ssim,
                      'mae': mae}
        print('train:', eval_value)

        self.metrics_val.reset()
        loss = torch.stack([x["loss_val"] for x in outputs]).mean()
        log_dict = {"val_loss": loss, 'val_psnr': psnr, 'val_ssim': ssim, 'val_mae': mae}
        self.log_dict(log_dict, prog_bar=True)

    def configure_optimizers(self):
        optimizer = self.config.optimizer
        lr_scheduler = self.config.lr_scheduler

        return [optimizer], [lr_scheduler]

    def train_dataloader(self):
        return self.config.train_loader

    def val_dataloader(self):
        return self.config.val_loader


# training
def main():
    args = get_args()
    config = py2cfg(args.config_path)
    seed_everything(42)

    checkpoint_callback = ModelCheckpoint(save_top_k=config.save_top_k, monitor=config.monitor,
                                          save_last=config.save_last, mode=config.monitor_mode,
                                          dirpath=config.weights_path,
                                          filename=config.weights_name)
    logger = CSVLogger('lightning_logs', name=config.log_name)

    model = Supervision_Train(config)
    if config.pretrained_ckpt_path:
        model = Supervision_Train.load_from_checkpoint(config.pretrained_ckpt_path, config=config)

    trainer = pl.Trainer(devices=config.gpus, max_epochs=config.max_epoch, accelerator='gpu',
                         check_val_every_n_epoch=config.check_val_every_n_epoch,
                         callbacks=[checkpoint_callback], strategy=config.strategy,
                         resume_from_checkpoint=config.resume_ckpt_path, logger=logger)
    trainer.fit(model=model)


if __name__ == "__main__":
    # vis = visualize.Visualizer("Super Resolution")
    main()