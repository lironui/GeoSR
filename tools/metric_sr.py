import numpy as np
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio


def compare_mae(img_true, img_test):
    img_true = img_true.astype(np.float32)
    img_test = img_test.astype(np.float32)
    return np.sum(np.abs(img_true - img_test)) / np.sum(img_true + img_test)


class Evaluator(object):
    def __init__(self, normalization=True):
        self.normalization = normalization
        self.psnr = []
        self.ssim = []
        self.mae = []

    def add_batch(self, img_gt, img_pred):
        if self.normalization:
            self.psnr.append(peak_signal_noise_ratio(img_gt, img_pred, data_range=1))
            self.ssim.append(structural_similarity(img_gt, img_pred, data_range=1, channel_axis=0))
        else:
            img_gt = (img_gt * 255.0).round().astype(np.uint8)
            img_pred = (img_pred * 255.0).round().astype(np.uint8)
            self.psnr.append(peak_signal_noise_ratio(img_gt, img_pred, data_range=255))
            self.ssim.append(structural_similarity(img_gt, img_pred, data_range=255, channel_axis=0))
        self.mae.append(compare_mae(img_gt, img_pred))

    def get_psnr(self):
        return np.round(np.mean(self.psnr), 5)

    def get_ssim(self):
        return np.round(np.mean(self.ssim), 5)

    def get_mae(self):
        return np.round(np.mean(self.mae), 5)

    def reset(self):
        self.psnr = []
        self.ssim = []
        self.mae = []


if __name__ == '__main__':

    gt = np.array([[0, 2, 1],
                   [1, 2, 1],
                   [1, 0, 1]])

    pre = np.array([[0, 1, 1],
                   [2, 0, 1],
                   [1, 1, 1]])

    eval = Evaluator()
    eval.add_batch(gt, pre)
    print(eval.psnr)

