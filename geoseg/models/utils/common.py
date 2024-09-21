import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable, Function


def batched_index_select(values, indices):
    last_dim = values.shape[-1]
    return values.gather(1, indices[:, :, None].expand(-1, -1, last_dim))


def default_conv(in_channels, out_channels, kernel_size, stride=1, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), stride=stride, bias=bias)


class Conv2d_CG(nn.Conv2d):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, groups=1,
                 bias=True):
        super(Conv2d_CG, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        """
        # just uncomment this region if you want to use CDRR
        # weights for gate convolution network (context descriptor relationshop reasoning)
        self.weight_g = Parameter(torch.zeros(in_channels, in_channels), requires_grad=True)
        self.weight_r = Parameter(torch.zeros(in_channels, in_channels), requires_grad=True)
        nn.init.kaiming_normal_(self.weight_g)
        nn.init.kaiming_normal_(self.weight_r)
        # weight for affinity matrix
        self.weight_affinity_1 = Parameter(torch.zeros(in_channels, in_channels), requires_grad=True)
        self.weight_affinity_2 = Parameter(torch.zeros(in_channels, in_channels), requires_grad=True)
        nn.init.kaiming_normal_(self.weight_affinity_1)
        nn.init.kaiming_normal_(self.weight_affinity_2)
        """

        # weight & bias for content-gated-convolution
        self.weight_conv = Parameter(torch.zeros(out_channels, in_channels, kernel_size, kernel_size),
                                     requires_grad=True).cuda()
        self.bias_conv = Parameter(torch.zeros(out_channels), requires_grad=True)
        nn.init.kaiming_normal_(self.weight_conv)

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # for convolutional layers with a kernel size  of 1, just use traditional convolution
        if kernel_size == 1:
            self.ind = True
        else:
            self.ind = False
            self.oc = out_channels
            self.ks = kernel_size

            # target spatial size of the pooling layer
            ws = kernel_size
            self.avg_pool = nn.AdaptiveAvgPool2d((ws, ws))

            # the dimension of latent representation
            self.num_lat = int((kernel_size * kernel_size) / 2 + 1)

            # the context encoding module
            self.ce = nn.Linear(ws * ws, self.num_lat, False)
            self.ce_bn = nn.BatchNorm1d(in_channels)
            self.ci_bn2 = nn.BatchNorm1d(in_channels)

            self.act = nn.ReLU(inplace=True)

            # the number of groups in the channel interaction module
            if in_channels // 16:
                self.g = 16
            else:
                self.g = in_channels

            # the channel interacting module
            self.ci = nn.Linear(self.g, out_channels // (in_channels // self.g), bias=False)
            self.ci_bn = nn.BatchNorm1d(out_channels)

            # the gate decoding module (spatial interaction)
            self.gd = nn.Linear(self.num_lat, kernel_size * kernel_size, False)
            self.gd2 = nn.Linear(self.num_lat, kernel_size * kernel_size, False)

            # used to prepare the input feature map to patches
            self.unfold = nn.Unfold(kernel_size, dilation, padding, stride)

            # sigmoid function
            self.sig = nn.Sigmoid()

    def forward(self, x):
        # for convolutional layers with a kernel size of 1, just use the traditional convolution
        if self.ind:
            return F.conv2d(x, self.weight_conv, self.bias_conv, self.stride, self.padding, self.dilation, self.groups)
        else:
            b, c, h, w = x.size()  # x: batch x n_feat(=64) x h_patch x w_patch
            weight = self.weight_conv

            # allocate global information
            gl = self.avg_pool(x).view(b, c, -1)  # gl: batch x n_feat x 3 x 3 -> batch x n_feat x 9

            # context-encoding module
            out = self.ce(gl)  # out: batch x n_feat x 5

            """
            # just uncomment this region if you want to use CDRR
            # Conext Descriptor Relationship Reasoning
            weighted_out_1 = torch.matmul(self.weight_affinity_1, out)                      # weighted_out: batch x n_feat x 5
            weighted_out_2 = torch.matmul(self.weight_affinity_2, out)
            affinity = torch.bmm(weighted_out_1.permute(0, 2, 1), weighted_out_2)           # affinity: batch x 5 x 5
            out_1 = torch.matmul(affinity, out.permute(0, 2, 1))                        # out_1: batch x 5 x n_feat
            out_2 = torch.matmul(out_1, self.weight_g)                                  # out_2: batch x 5 x n_feat
            out_3 = torch.matmul(out_2, self.weight_r)                                  # out_3: batch x 5 x n_feat
            out_4 = out_3.permute(0, 2, 1)                                              # out_4: batch x n_feat x 5
            out_5 = torch.mul(out_4, out)                                               # out_5: batch x n_feat x 5
            out = out + out_5                                                                # out: batch x n_feat x 5
            """

            # use different bn for following two branches
            ce2 = out  # ce2: batch x n_feat x 5
            out = self.ce_bn(out)
            out = self.act(out)  # out: batch x n_feat x 5 (just batch normalization)

            # gate decoding branch 1 (spatial interaction)
            out = self.gd(out)  # out: batch x n_feat x 9 (5 --> 9 = 3x3)

            # channel interacting module
            if self.g > 3:
                oc = self.ci(self.act(self.ci_bn2(ce2).view(b, c // self.g, self.g, -1).transpose(2, 3))).transpose(2,
                                                                                                                    3).contiguous()
            else:
                oc = self.ci(self.act(self.ci_bn2(ce2).transpose(2, 1))).transpose(2, 1).contiguous()
            oc = oc.view(b, self.oc, -1)
            oc = self.ci_bn(oc)
            oc = self.act(oc)  # oc: batch x n_feat x 5 (after grouped linear layer)

            # gate decoding branch 2 (spatial interaction)
            oc = self.gd2(oc)  # oc: batch x n_feat x 9 (5 --> 9 = 3x3)

            # produce gate (equation (4) in the CRAN paper)
            out = self.sig(out.view(b, 1, c, self.ks, self.ks) + oc.view(b, self.oc, 1, self.ks,
                                                                         self.ks))  # out: batch x out_channel x in_channel x kernel_size x kernel_size (same dimension as conv2d weight)

            # unfolding input feature map to patches
            x_un = self.unfold(x)
            b, _, l = x_un.size()

            # gating
            out = (out * weight.unsqueeze(0)).view(b, self.oc, -1)

            # currently only handle square input and output


class MaskBranchDownUp(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(MaskBranchDownUp, self).__init__()

        MB_RB1 = []
        MB_RB1.append(ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))

        MB_Down = []
        MB_Down.append(nn.Conv2d(n_feat, n_feat, 3, stride=2, padding=1))

        MB_RB2 = []
        for i in range(2):
            MB_RB2.append(ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))

        MB_Up = []
        MB_Up.append(nn.ConvTranspose2d(n_feat, n_feat, 6, stride=2, padding=2))

        MB_RB3 = []
        MB_RB3.append(ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))

        MB_1x1conv = []
        MB_1x1conv.append(nn.Conv2d(n_feat, n_feat, 1, padding=0, bias=True))

        MB_sigmoid = []
        MB_sigmoid.append(nn.Sigmoid())

        self.MB_RB1 = nn.Sequential(*MB_RB1)
        self.MB_Down = nn.Sequential(*MB_Down)
        self.MB_RB2 = nn.Sequential(*MB_RB2)
        self.MB_Up = nn.Sequential(*MB_Up)
        self.MB_RB3 = nn.Sequential(*MB_RB3)
        self.MB_1x1conv = nn.Sequential(*MB_1x1conv)
        self.MB_sigmoid = nn.Sequential(*MB_sigmoid)

    def forward(self, x):
        x_RB1 = self.MB_RB1(x)
        x_Down = self.MB_Down(x_RB1)
        x_RB2 = self.MB_RB2(x_Down)
        x_Up = self.MB_Up(x_RB2)
        x_preRB3 = x_RB1 + x_Up
        x_RB3 = self.MB_RB3(x_preRB3)
        x_1x1 = self.MB_1x1conv(x_RB3)
        mx = self.MB_sigmoid(x_1x1)

        return mx


class ResAttModuleDownUpPlus(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResAttModuleDownUpPlus, self).__init__()
        RA_RB1 = []
        RA_RB1.append(ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        RA_TB = []
        RA_TB.append(TrunkBranch(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        RA_MB = []
        RA_MB.append(MaskBranchDownUp(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        RA_tail = []
        for i in range(2):
            RA_tail.append(ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))

        self.RA_RB1 = nn.Sequential(*RA_RB1)
        self.RA_TB = nn.Sequential(*RA_TB)
        self.RA_MB = nn.Sequential(*RA_MB)
        self.RA_tail = nn.Sequential(*RA_tail)

    def forward(self, input):
        RA_RB1_x = self.RA_RB1(input)
        tx = self.RA_TB(RA_RB1_x)
        mx = self.RA_MB(RA_RB1_x)
        txmx = tx * mx
        hx = txmx + RA_RB1_x
        hx = self.RA_tail(hx)

        return hx


class TrunkBranch(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(TrunkBranch, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(
                ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        tx = self.body(x)

        return tx


class NonLocalBlock2D(nn.Module):
    def __init__(self, in_channels, inter_channels):
        super(NonLocalBlock2D, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                           padding=0)

        self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1,
                           padding=0)
        nn.init.constant(self.W.weight, 0)
        nn.init.constant(self.W.bias, 0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                               padding=0)

        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                             padding=0)

    def forward(self, x):
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)

        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)

        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        f = torch.matmul(theta_x, phi_x)

        f_div_C = F.softmax(f, dim=1)

        y = torch.matmul(f_div_C, g_x)

        y = y.permute(0, 2, 1).contiguous()

        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


class NLMaskBranchDownUp(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(NLMaskBranchDownUp, self).__init__()

        MB_RB1 = []
        MB_RB1.append(NonLocalBlock2D(n_feat, n_feat // 2))
        MB_RB1.append(ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))

        MB_Down = []
        MB_Down.append(nn.Conv2d(n_feat, n_feat, 3, stride=2, padding=1))

        MB_RB2 = []
        for i in range(2):
            MB_RB2.append(ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))

        MB_Up = []
        MB_Up.append(nn.ConvTranspose2d(n_feat, n_feat, 6, stride=2, padding=2))

        MB_RB3 = []
        MB_RB3.append(ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))

        MB_1x1conv = []
        MB_1x1conv.append(nn.Conv2d(n_feat, n_feat, 1, padding=0, bias=True))

        MB_sigmoid = []
        MB_sigmoid.append(nn.Sigmoid())

        self.MB_RB1 = nn.Sequential(*MB_RB1)
        self.MB_Down = nn.Sequential(*MB_Down)
        self.MB_RB2 = nn.Sequential(*MB_RB2)
        self.MB_Up = nn.Sequential(*MB_Up)
        self.MB_RB3 = nn.Sequential(*MB_RB3)
        self.MB_1x1conv = nn.Sequential(*MB_1x1conv)
        self.MB_sigmoid = nn.Sequential(*MB_sigmoid)

    def forward(self, x):
        x_RB1 = self.MB_RB1(x)
        x_Down = self.MB_Down(x_RB1)
        x_RB2 = self.MB_RB2(x_Down)
        x_Up = self.MB_Up(x_RB2)
        x_preRB3 = x_RB1 + x_Up
        x_RB3 = self.MB_RB3(x_preRB3)
        x_1x1 = self.MB_1x1conv(x_RB3)
        mx = self.MB_sigmoid(x_1x1)

        return mx


## define nonlocal residual attention module
class NLResAttModuleDownUpPlus(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(NLResAttModuleDownUpPlus, self).__init__()
        RA_RB1 = []
        RA_RB1.append(ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        RA_TB = []
        RA_TB.append(TrunkBranch(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        RA_MB = []
        RA_MB.append(NLMaskBranchDownUp(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        RA_tail = []
        for i in range(2):
            RA_tail.append(ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))

        self.RA_RB1 = nn.Sequential(*RA_RB1)
        self.RA_TB = nn.Sequential(*RA_TB)
        self.RA_MB = nn.Sequential(*RA_MB)
        self.RA_tail = nn.Sequential(*RA_tail)

    def forward(self, input):
        RA_RB1_x = self.RA_RB1(input)
        tx = self.RA_TB(RA_RB1_x)
        mx = self.RA_MB(RA_RB1_x)
        txmx = tx * mx
        hx = txmx + RA_RB1_x
        hx = self.RA_tail(hx)

        return hx


class MeanShift(nn.Conv2d):
    def __init__(
            self, rgb_range,
            rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class BasicBlock(nn.Sequential):
    def __init__(
            self, conv, in_channels, out_channels, kernel_size, stride=1, bias=True,
            bn=False, act=nn.PReLU()):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)


class ResBlock(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size,
            bias=True, bn=False, act=nn.PReLU(), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias=bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias=bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class Shuffle_d(nn.Module):
    def __init__(self, scale=2):
        super(Shuffle_d, self).__init__()
        self.scale = scale

    def forward(self, x):
        def _space_to_channel(x, scale):
            b, C, h, w = x.size()
            Cout = C * scale ** 2
            hout = h // scale
            wout = w // scale
            x = x.contiguous().view(b, C, hout, scale, wout, scale)
            x = x.contiguous().permute(0, 1, 3, 5, 2, 4)
            x = x.contiguous().view(b, Cout, hout, wout)
            return x
        return _space_to_channel(x, self.scale)


class GaussianBlur(nn.Module):
    def __init__(self):
        super(GaussianBlur, self).__init__()
        kernel = np.array([[1. / 256., 4. / 256., 6. / 256., 4. / 256., 1. / 256.],
                           [4. / 256., 16. / 256., 24. / 256., 16. / 256., 4. / 256.],
                           [6. / 256., 24. / 256., 36. / 256., 24. / 256., 6. / 256.],
                           [4. / 256., 16. / 256., 24. / 256., 16. / 256., 4. / 256.],
                           [1. / 256., 4. / 256., 6. / 256., 4. / 256., 1. / 256.]])

        kernel = torch.FloatTensor(kernel)
        kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)
        self.gaussian = nn.Conv2d(3, 3, kernel_size=5, stride=1, padding=2, groups=3, bias=False)
        self.gaussian.weight = nn.Parameter(kernel, requires_grad=False)

    def forward(self, x):
        x = self.gaussian(x)
        return x


class pixelConv(nn.Module):
    # Generate pixel kernel  (3*k*k)xHxW
    def __init__(self, in_feats, out_feats=3, rate=4, ksize=3):
        super(pixelConv, self).__init__()
        self.padding = (ksize - 1) // 2
        self.ksize = ksize
        self.zero_padding = nn.ZeroPad2d(self.padding)
        mid_feats = in_feats * rate ** 2
        self.kernel_conv = nn.Sequential(*[
            nn.Conv2d(in_feats, mid_feats, kernel_size=3, padding=1),
            nn.Conv2d(mid_feats, mid_feats, kernel_size=3, padding=1),
            nn.Conv2d(mid_feats, 3 * ksize ** 2, kernel_size=3, padding=1)
        ])

    def forward(self, x_feature, x):
        kernel_set = self.kernel_conv(x_feature)

        dtype = kernel_set.data.type()
        ks = self.ksize
        N = self.ksize ** 2  # patch size
        # padding the input image with zero values
        if self.padding:
            x = self.zero_padding(x)

        p = self._get_index(kernel_set, dtype)
        p = p.contiguous().permute(0, 2, 3, 1).long()
        x_pixel_set = self._get_x_q(x, p, N)
        b, c, h, w = kernel_set.size()
        kernel_set_reshape = kernel_set.reshape(-1, self.ksize ** 2, 3, h, w).permute(0, 2, 3, 4, 1)
        x_ = x_pixel_set

        out = x_ * kernel_set_reshape
        out = out.sum(dim=-1, keepdim=True).squeeze(dim=-1)
        out = out
        return out

    def _get_index(self, kernel_set, dtype):
        '''
        get absolute index of each pixel in image
        '''
        N, b, h, w = self.ksize ** 2, kernel_set.size(0), kernel_set.size(2), kernel_set.size(3)
        # get absolute index of center index
        p_0_x, p_0_y = np.meshgrid(range(self.padding, h + self.padding), range(self.padding, w + self.padding),
                                   indexing='ij')
        p_0_x = p_0_x.flatten().reshape(1, 1, h, w).repeat(N, axis=1)
        p_0_y = p_0_y.flatten().reshape(1, 1, h, w).repeat(N, axis=1)
        p_0 = np.concatenate((p_0_x, p_0_y), axis=1)
        p_0 = Variable(torch.from_numpy(p_0).type(dtype), requires_grad=False)

        # get relative index around center pixel
        p_n_x, p_n_y = np.meshgrid(range(-(self.ksize - 1) // 2, (self.ksize - 1) // 2 + 1),
                                   range(-(self.ksize - 1) // 2, (self.ksize - 1) // 2 + 1), indexing='ij')
        # (2N, 1)
        p_n = np.concatenate((p_n_x.flatten(), p_n_y.flatten()))
        p_n = np.reshape(p_n, (1, 2 * N, 1, 1))
        p_n = Variable(torch.from_numpy(p_n).type(dtype), requires_grad=False)
        p = p_0 + p_n
        p = p.repeat(b, 1, 1, 1)
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()  # dimension of q: (b,h,w,2N)
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*padded_w)
        x = x.contiguous().view(b, c, -1)
        # (b, h, w, N)
        # index_x*w + index_y
        index = q[..., :N] * padded_w + q[..., N:]

        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)
        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)
        return x_offset


class GaussianBlur_Up(nn.Module):
    def __init__(self):
        super(GaussianBlur_Up, self).__init__()
        kernel = np.array([[1. / 256., 4. / 256., 6. / 256., 4. / 256., 1. / 256.],
                           [4. / 256., 16. / 256., 24. / 256., 16. / 256., 4. / 256.],
                           [6. / 256., 24. / 256., 36. / 256., 24. / 256., 6. / 256.],
                           [4. / 256., 16. / 256., 24. / 256., 16. / 256., 4. / 256.],
                           [1. / 256., 4. / 256., 6. / 256., 4. / 256., 1. / 256.]])
        kernel = kernel * 4
        kernel = torch.FloatTensor(kernel)
        kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)
        self.gaussian = nn.Conv2d(3, 3, kernel_size=5, stride=1, padding=2, groups=3, bias=False)
        self.gaussian.weight = nn.Parameter(kernel, requires_grad=False)

    def forward(self, x):
        x = self.gaussian(x)
        return x


class Laplacian_reconstruction(nn.Module):
    def __init__(self):
        super(Laplacian_reconstruction, self).__init__()
        self.Gau = GaussianBlur_Up()
    def forward(self, x_lap,x_gau):
        b,c,h,w = x_gau.size()
        up_x = torch.zeros((b,c,h*2,w*2),device='cuda')
        up_x[:,:,::2,::2]= x_gau
        up_x = self.Gau(up_x) + x_lap
        return up_x


class Laplacian_pyramid(nn.Module):
    def __init__(self, step=3):
        super(Laplacian_pyramid, self).__init__()
        self.Gau = GaussianBlur()
        self.Gau_up = GaussianBlur_Up()
        self.step = step

    def forward(self, x):
        Gaussian_lists = [x]
        Laplacian_lists = []
        size_lists = [x.size()[2:]]
        for _ in range(self.step - 1):
            gaussian_down = self.Prdown(Gaussian_lists[-1])
            Gaussian_lists.append(gaussian_down)
            size_lists.append(gaussian_down.size()[2:])
            Lap = Gaussian_lists[-2] - self.PrUp(Gaussian_lists[-1], size_lists[-2])
            Laplacian_lists.append(Lap)
        return Gaussian_lists, Laplacian_lists

    def Prdown(self, x):
        x_ = self.Gau(x)
        x_ = x_[:, :, ::2, ::2]
        return x_

    def PrUp(self, x, sizes):
        b, c, _, _ = x.size()
        h, w = sizes
        up_x = torch.zeros((b, c, h, w), device='cuda')
        up_x[:, :, ::2, ::2] = x
        up_x = self.Gau_up(up_x)
        return up_x