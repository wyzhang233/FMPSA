import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import torch
import math
from torch.nn import Module, Sequential, Conv2d, ReLU, AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
from torch.nn import functional as F
from torch.autograd import Variable


class Conv3x3(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, dilation=1):
        super(Conv3x3, self).__init__()
        reflect_padding = int(dilation * (kernel_size - 1) / 2)
        self.reflection_pad = nn.ReflectionPad2d(reflect_padding)
        self.conv2d = nn.Conv2d(in_dim, out_dim, kernel_size, stride, dilation=dilation, bias=False)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

class FMFE_left(nn.Module):

    def __init__(self, inplans, planes, pyconv_kernels=[3, 5, 7, 9], stride=1, pyconv_groups=[1, 4, 8, 16]):
        super(FMFE_left, self).__init__()

        self.conv2_1 = Conv2d(inplans, planes//4, kernel_size=pyconv_kernels[0], padding=pyconv_kernels[0]//2,
                            stride=stride, groups=pyconv_groups[0])
        self.conv2_2 = Conv2d(inplans, planes//4, kernel_size=pyconv_kernels[1], padding=pyconv_kernels[1]//2,
                            stride=stride, groups=pyconv_groups[1])
        self.conv2_3 = Conv2d(inplans, planes//4, kernel_size=pyconv_kernels[2], padding=pyconv_kernels[2]//2,
                            stride=stride, groups=pyconv_groups[2])
        self.conv2_4 = Conv2d(inplans, planes//4, kernel_size=pyconv_kernels[3], padding=pyconv_kernels[3]//2,
                            stride=stride, groups=pyconv_groups[3])

    def forward(self, x):
        return torch.cat((self.conv2_1(x), self.conv2_2(x), self.conv2_3(x), self.conv2_4(x)), dim=1)

class FMFE_right(nn.Module):

    def __init__(self, inplans, planes, pyconv_kernels=[9, 7, 5, 3], stride=1, pyconv_groups=[16, 8, 4, 1]):
        super(FMFE_right, self).__init__()
        self.conv2_1 = Conv2d(inplans, planes//4, kernel_size=pyconv_kernels[0], padding=pyconv_kernels[0]//2,
                            stride=stride, groups=pyconv_groups[0])
        self.conv2_2 = Conv2d(inplans, planes//4, kernel_size=pyconv_kernels[1], padding=pyconv_kernels[1]//2,
                            stride=stride, groups=pyconv_groups[1])
        self.conv2_3 = Conv2d(inplans, planes//4, kernel_size=pyconv_kernels[2], padding=pyconv_kernels[2]//2,
                            stride=stride, groups=pyconv_groups[2])
        self.conv2_4 = Conv2d(inplans, planes//4, kernel_size=pyconv_kernels[3], padding=pyconv_kernels[3]//2,
                            stride=stride, groups=pyconv_groups[3])

    def forward(self, x):
        return torch.cat((self.conv2_1(x), self.conv2_2(x), self.conv2_3(x), self.conv2_4(x)), dim=1)

class PSAB(nn.Module):
    def __init__(self, in_channels, reduction=8, dimension=2, sub_sample=False, bn_layer=False):
        super(PSAB, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = self.in_channels // reduction

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0, bias=False)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0, bias=False),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0, bias=False)
            nn.init.constant_(self.W.weight, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0, bias=False)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        f = self.count_cov_second(theta_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

    def count_cov_second(self, input):
        x = input
        batchSize, dim, M = x.data.shape
        x_mean_band = x.mean(2).view(batchSize, dim, 1).expand(batchSize, dim, M)
        y = (x - x_mean_band).bmm(x.transpose(1, 2)) / M
        return y

class PSNL(nn.Module):
    def __init__(self, channels):
        super(PSNL, self).__init__()
        self.non_local_p = PSAB(channels)

    def forward(self,x):
        batch_size, C, H, W = x.shape
        H1 = int(H / 2)
        W1 = int(W / 2)
        nonlocal_feat = torch.zeros_like(x)

        feat_sub_lu = x[:, :, :H1, :W1]
        feat_sub_ld = x[:, :, H1:, :W1]
        feat_sub_ru = x[:, :, :H1, W1:]
        feat_sub_rd = x[:, :, H1:, W1:]

        nonlocal_lu = self.non_local_p(feat_sub_lu)
        nonlocal_ld = self.non_local_p(feat_sub_ld)
        nonlocal_ru = self.non_local_p(feat_sub_ru)
        nonlocal_rd = self.non_local_p(feat_sub_rd)
        nonlocal_feat[:, :, :H1, :W1] = nonlocal_lu
        nonlocal_feat[:, :, H1:, :W1] = nonlocal_ld
        nonlocal_feat[:, :, :H1, W1:] = nonlocal_ru
        nonlocal_feat[:, :, H1:, W1:] = nonlocal_rd

        return nonlocal_feat

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Conv3x3(in_features, hidden_features, kernel_size=1, stride=1)
        self.act = act_layer()
        self.fc2 = Conv3x3(hidden_features, out_features, kernel_size=1, stride=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MARAB(nn.Module):
    def __init__(self, in_dim, out_dim, res_dim, k1_size=3, k2_size=1,norm_layer=nn.BatchNorm2d,dilation=1,mlp_ratio=4,act_layer=nn.GELU,drop=0):
        super(MARAB, self).__init__()
        self.conv1 = FMFE_left(in_dim, in_dim)
        self.relu1 = nn.PReLU()
        self.up_conv = FMFE_right(in_dim, in_dim)
        self.se = PSNL(res_dim)
        self.norm2 =norm_layer(res_dim)
        mlp_hidden_dim = int(res_dim * mlp_ratio)
        self.mlp = Mlp(in_features=res_dim, hidden_features=mlp_hidden_dim, out_features=out_dim,act_layer=act_layer, drop=drop)


    def forward(self, x, res):
        x_r = x
        x = self.relu1(self.conv1(x))
        x_s = x
        x = self.relu1(self.up_conv(x))
        x = self.se(x)
        x += x_s
        x_ss = x
        x = self.norm2(x)
        x = self.mlp(x)
        x += x_r
        x += x_ss

        return x, res

#主模块
class FMPSA(nn.Module):
    def __init__(self, inplanes=3, planes=31, channels=128, n_DRBs=8):
        super(FMPSA, self).__init__()
        self.input_conv2D = Conv3x3(inplanes, channels, 3, 1) 
        self.input_prelu2D = nn.PReLU() 
        self.head_conv2D = Conv3x3(channels, channels, 3, 1) 
        self.backbone = nn.ModuleList(
            [MARAB(in_dim=channels, out_dim=channels, res_dim=channels, k1_size=5, k2_size=3, dilation=1) for _ in
             range(n_DRBs)])

        self.tail_conv2D = Conv3x3(channels, channels, 3, 1)
        self.output_prelu2D = nn.PReLU()
        self.output_conv2D = Conv3x3(channels, planes, 3, 1)
        self.FMFE_left = FMFE_left(channels,channels)
        self.FMFE_right = FMFE_right(channels, channels)

    def forward(self, x):
        out = self.DRN2D(x)
        return out

    def DRN2D(self, x):
        out = self.input_prelu2D(x)
        print(out.shape)
        out = self.input_prelu2D(self.input_conv2D(x))
        out = self.FMFE_left(out)
        out = self.FMFE_right(out)
        residual = out
        res = out

        for i, block in enumerate(self.backbone):
            out, res = block(out, res)

        out = self.tail_conv2D(out)
        out = torch.add(out, residual)
        out = self.output_conv2D(self.output_prelu2D(out))

        return out

if __name__ == "__main__":

    input_tensor = torch.rand(1, 3, 64, 64)
    model = FMPSA(3, 31, 128, 8)
    with torch.no_grad():
        output_tensor = model(input_tensor)

    print(output_tensor.size())
    print('Parameters number is ', sum(param.numel() for param in model.parameters()))
    print(torch.__version__)





