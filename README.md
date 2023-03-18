# GeoSR: A Benchmark for Super Resolution and Semantic Segmentation

The whole code will be released after the paper is accepted. 

Currently, as computer vision is not my major research topic, I don't have enough time and energy to further enhance the GeoSR repo. So, the pull requests are welcome such as the support for more classification, detection and segmentation datasets.

[Welcome to my homepage!](https://lironui.github.io)


## Introduction

**GeoSR** is an open-source super resolution and semantic segmentation toolbox based on PyTorch, [pytorch lightning](https://www.pytorchlightning.ai/) and [timm](https://github.com/rwightman/pytorch-image-models), which mainly focuses on developing advanced Vision Transformers for UAV super resolution and semantic segmentation.

## Folder Structure
Download [UAVid](https://uavid.nl/) and prepare the following folders to organize this repo:
```none
airs
├── GeoSR (code)
├── pretrain_weights (save the pretrained weights like vit, swin, etc)
├── model_weights (save the model weights)
├── lightning_logs (CSV format training logs)
├── data
│   ├── uavid
│   │   ├── uavid_train (original)
│   │   ├── uavid_val (original)
│   │   ├── uavid_test (original)
│   │   ├── uavid_train_val (Merge uavid_train and uavid_val)
│   ├── uavid_2x
...
```

## Install

Open the folder **airs** using **Linux Terminal** and create python environment:
```
conda create -n airs python=3.8
conda activate airs

conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -r GeoSR/requirements.txt
```

## Data Preprocessing

Download the datasets from the official website and split them yourself.

Split the training and validation sets for super resolution and semantic segmentation.
```
python GeoSR/tools/seg_sr_split.py \
--basic-path "data/uavid/uavid_train_val" \
--train-seg-path "data/uavid/train_seg" \
--train-sr-path "data/uavid/train_sr" \
--val-seg-path "data/uavid/val_seg" \
--val-sr-path "data/uavid/val_sr" 
```

Prepare the training and validation sets for super resolution and semantic segmentation (×2).
```
python GeoSR/tools/uavid_patch_split.py \
--input-dir "data/uavid/train_sr" \
--output-img-dir "data/UAVid_2x/train_sr/images" \
--output-ref-dir "data/UAVid_2x/train_sr/references" \
--output-mask-dir "data/UAVid_2x/train_sr/masks" \
--mode 'train' --split-size-h 256 --split-size-w 256 \
--stride-h 256 --stride-w 256 --scale 2
```
```
python GeoSR/tools/uavid_patch_split.py \
--input-dir "data/uavid/val_sr" \
--output-img-dir "data/UAVid_2x/val_sr/images" \
--output-ref-dir "data/UAVid_2x/val_sr/references" \
--output-mask-dir "data/UAVid_2x/val_sr/masks" \
--mode 'val' --split-size-h 256 --split-size-w 256 \
--stride-h 256 --stride-w 256 --scale 2
```
```
python GeoSR/tools/uavid_patch_split.py \
--input-dir "data/uavid/train_seg" \
--output-img-dir "data/UAVid_2x/train_seg/images" \
--output-ref-dir "data/UAVid_2x/train_seg/references" \
--output-mask-dir "data/UAVid_2x/train_seg/masks" \
--mode 'train' --split-size-h 256 --split-size-w 256 \
--stride-h 256 --stride-w 256 --scale 2
```
```
python GeoSR/tools/uavid_patch_split.py \
--input-dir "data/uavid/val_seg" \
--output-img-dir "data/UAVid_2x/val_seg/images" \
--output-ref-dir "data/UAVid_2x/val_seg/references" \
--output-mask-dir "data/UAVid_2x/val_seg/masks" \
--mode 'val' --split-size-h 256 --split-size-w 256 \
--stride-h 256 --stride-w 256 --scale 2
```

Prepare the test set (×2).
```
python GeoSR/tools/uavid_test_downsample.py \
--input-dir "data/uavid/uavid_test" \
--output-dir "data/UAVid_2x/test" \
--scale 2
```

Prepare the training and validation sets for super resolution and semantic segmentation (×4).
```
python GeoSR/tools/uavid_patch_split.py \
--input-dir "data/uavid/train_sr" \
--output-img-dir "data/UAVid_4x/train_sr/images" \
--output-ref-dir "data/UAVid_4x/train_sr/references" \
--output-mask-dir "data/UAVid_4x/train_sr/masks" \
--mode 'train' --split-size-h 512 --split-size-w 512 \
--stride-h 512 --stride-w 512 --scale 4
```
```
python GeoSR/tools/uavid_patch_split.py \
--input-dir "data/uavid/val_sr" \
--output-img-dir "data/UAVid_4x/val_sr/images" \
--output-ref-dir "data/UAVid_4x/val_sr/references" \
--output-mask-dir "data/UAVid_4x/val_sr/masks" \
--mode 'val' --split-size-h 512 --split-size-w 512 \
--stride-h 512 --stride-w 512 --scale 4
```
```
python GeoSR/tools/uavid_patch_split.py \
--input-dir "data/uavid/train_seg" \
--output-img-dir "data/UAVid_4x/train_seg/images" \
--output-ref-dir "data/UAVid_4x/train_seg/references" \
--output-mask-dir "data/UAVid_4x/train_seg/masks" \
--mode 'train' --split-size-h 512 --split-size-w 512 \
--stride-h 512 --stride-w 512 --scale 4
```
```
python GeoSR/tools/uavid_patch_split.py \
--input-dir "data/uavid/val_seg" \
--output-img-dir "data/UAVid_4x/val_seg/images" \
--output-ref-dir "data/UAVid_4x/val_seg/references" \
--output-mask-dir "data/UAVid_4x/val_seg/masks" \
--mode 'val' --split-size-h 512 --split-size-w 512 \
--stride-h 512 --stride-w 512 --scale 4
```

Prepare the test set (×4).
```
python GeoSR/tools/uavid_test_downsample.py \
--input-dir "data/uavid/uavid_test" \
--output-dir "data/UAVid_4x/test" \
--scale 4
```

Prepare the training and validation sets for super resolution and semantic segmentation (×8).
```
python GeoSR/tools/uavid_patch_split.py \
--input-dir "data/uavid/train_sr" \
--output-img-dir "data/UAVid_8x/train_sr/images" \
--output-ref-dir "data/UAVid_8x/train_sr/references" \
--output-mask-dir "data/UAVid_8x/train_sr/masks" \
--mode 'train' --split-size-h 1024 --split-size-w 1024 \
--stride-h 1024 --stride-w 1024 --scale 8
```
```
python GeoSR/tools/uavid_patch_split.py \
--input-dir "data/uavid/val_sr" \
--output-img-dir "data/UAVid_8x/val_sr/images" \
--output-ref-dir "data/UAVid_8x/val_sr/references" \
--output-mask-dir "data/UAVid_8x/val_sr/masks" \
--mode 'val' --split-size-h 1024 --split-size-w 1024 \
--stride-h 1024 --stride-w 1024 --scale 8
```
```
python GeoSR/tools/uavid_patch_split.py \
--input-dir "data/uavid/train_seg" \
--output-img-dir "data/UAVid_8x/train_seg/images" \
--output-ref-dir "data/UAVid_8x/train_seg/references" \
--output-mask-dir "data/UAVid_8x/train_seg/masks" \
--mode 'train' --split-size-h 1024 --split-size-w 1024 \
--stride-h 1024 --stride-w 1024 --scale 8
```
```
python GeoSR/tools/uavid_patch_split.py \
--input-dir "data/uavid/val_seg" \
--output-img-dir "data/UAVid_8x/val_seg/images" \
--output-ref-dir "data/UAVid_8x/val_seg/references" \
--output-mask-dir "data/UAVid_8x/val_seg/masks" \
--mode 'val' --split-size-h 1024 --split-size-w 1024 \
--stride-h 1024 --stride-w 1024 --scale 8
```

Prepare the test set (×8).
```
python GeoSR/tools/uavid_test_downsample.py \
--input-dir "data/uavid/uavid_test" \
--output-dir "data/UAVid_8x/test" \
--scale 8
```

## Training for Super Resolution

"-c" means the path of the config, use different **config** to train different models.

```
python GeoSR/train_supervision_SR.py -c GeoSR/config/uavid_SR/lswinsr.py
```

## Inference for the training, validation and test sets for segmentation

"-c" means the path of the config, use different settings of **config** to predict different scenarios.
```
python GeoSR/sr_for_seg.py --config-path GeoSR/config/uavid_SR/lswin2sr.py --prediction_mode 1
python GeoSR/sr_for_seg.py --config-path GeoSR/config/uavid_SR/lswin2sr.py --prediction_mode 2
python GeoSR/super_resolution_for_seg.py --config-path GeoSR/config/uavid_SR/lswin2sr.py
```

![Super Resolution](https://github.com/lironui/GeoSR/blob/main/figs/figure5.PNG) 
![Semantic Segmentation](https://github.com/lironui/GeoSR/blob/main/figs/figure9.PNG) 
![Demo](https://github.com/lironui/GeoSR/blob/main/figs/merge.mp4) 
<video src="https://github.com/lironui/GeoSR/blob/main/figs/merge.mp4" controls="controls" width="500" height="300">您的浏览器不支持播放该视频！</video>

## Citation

If you find this project useful in your research, please consider citing：

- [LSwinSR: UAV Imagery Super-Resolution based on Linear Swin Transformer](https://arxiv.org/)
- [UNetFormer: A UNet-like transformer for efficient semantic segmentation of remote sensing urban scene imagery](https://authors.elsevier.com/a/1fIji3I9x1j9Fs)

Other papers you might be interested in:

- [ABCNet: Attentive Bilateral Contextual Network for Efficient Semantic Segmentation of Fine-Resolution Remote Sensing Images](https://www.sciencedirect.com/science/article/pii/S0924271621002379)
- [Multiattention network for semantic segmentation of fine-resolution remote sensing images](https://ieeexplore.ieee.org/abstract/document/9487010)
- [A2-FPN for semantic segmentation of fine-resolution remotely sensed images](https://www.tandfonline.com/doi/full/10.1080/01431161.2022.2030071)
- [A Novel Transformer Based Semantic Segmentation Scheme for Fine-Resolution Remote Sensing Images](https://ieeexplore.ieee.org/abstract/document/9681903) 
- [Transformer Meets Convolution: A Bilateral Awareness Network for Semantic Segmentation of Very Fine Resolution Urban Scene Images](https://www.mdpi.com/2072-4292/13/16/3065)

Acknowlegement:
------- 
The GeoSR is constructed highly based on the repository **[GeoSeg](https://github.com/WangLibo1995/GeoSeg)**. We wish GeoSR could serve the growing research of UAV by providing a unified benchmark and inspiring researchers to develop their own super-resolution networks. Many thanks the following projects's contributions to GeoSR.
- [pytorch lightning](https://www.pytorchlightning.ai/)
- [timm](https://github.com/rwightman/pytorch-image-models)
- [pytorch-toolbelt](https://github.com/BloodAxe/pytorch-toolbelt)
- [ttach](https://github.com/qubvel/ttach)
- [catalyst](https://github.com/catalyst-team/catalyst)
- [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)
