# Code for "GENet: Excite Features from Different Perspectives for Action Recognition"
# Xiaoyang Li, Wenzhu Yang*, Kanglin Wang, Guodong Zhang, Ceren Zhang
# lixiaoyang@stumail.hbu.edu.cn, wenzhuyang@hbu.edu.cn

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelModule(nn.Module):
    def __init__(self, inplanes, planes, num_segments=8):
        super(ChannelModule, self).__init__()
        self.num_segments = num_segments
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.linear1 = nn.Linear(inplanes, planes, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.sigmoid = nn.Sigmoid()

        nn.init.normal_(self.linear1.weight, 0, 0.001)
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)

    def forward(self, x):
        bn, c, h, w = x.size()  # nt, c, h, w
        x = x.view(-1, self.num_segments, c, h, w)  # n, t, c, h, w
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # n, c, t, h, w

        t_fea, __ = x.split([self.num_segments - 1, 1], dim=2)  # n, c, t-1, h, w
        _, tplusone_fea = x.split([1, self.num_segments - 1], dim=2)  # n, c, t-1, h, w
        diff_fea = tplusone_fea - t_fea  # n, c, t-1, h, w
        diff_fea = diff_fea ** 2  # n, c, t-1, h, w
        avg_out = self.avg_pool(diff_fea)  # n, c, 1, 1, 1
        y = avg_out.contiguous().view(-1, c)  # n, c
        y = self.linear1(y)  # n, c
        y = self.bn1(y)
        y = self.sigmoid(y)
        y = y.contiguous().view(-1, c, 1, 1, 1)  # n, c, 1, 1, 1
        x = x * y.expand_as(x)  # n, c, t, h, w
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # n, t, c, h, w

        return x.view(-1, c, h, w)  # nt, c, h, w


class TemporalModule(nn.Module):
    def __init__(self, inplanes, planes, num_segments=8):
        super(TemporalModule, self).__init__()
        self.num_segments = num_segments
        self.pad = (0, 0, 0, 0, 0, 1, 0, 0)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv1d(inplanes, planes // 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv1d(planes // 4, planes, kernel_size=3, stride=1, padding=(3 * 2 - 1) // 2, bias=False, dilation=2)
        self.bn1 = nn.BatchNorm1d(planes // 4)
        self.bn2 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.xavier_normal_(self.conv2.weight)
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)
        nn.init.constant_(self.bn2.weight, 1)
        nn.init.constant_(self.bn2.bias, 0)

    def forward(self, x):
        bn, c, h, w = x.size()  # nt, c, h, w
        x = x.view(-1, self.num_segments, c, h, w)  # n, t, c, h, w
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # n, c, t, h, w

        t_fea, __ = x.split([self.num_segments - 1, 1], dim=2)  # n, c, t-1, h, w
        _, tplusone_fea = x.split([1, self.num_segments - 1], dim=2)  # n, c, t-1, h, w
        diff_fea = tplusone_fea - t_fea  # n, c, t-1, h, w
        diff_fea = diff_fea ** 2
        # diff_fea = torch.cat([diff_fea, diff_fea[:, :, 0:1, :, :]], dim=2)  # n, c, t, h, w
        diff_fea = F.pad(diff_fea, self.pad, mode="constant", value=0)  # n, c, t, h, w
        y = self.avg_pool(diff_fea).contiguous().view(-1, c, self.num_segments)  # n, c, t
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.sigmoid(y)  # n, c, t
        y = y.contiguous().view(-1, c, self.num_segments, 1, 1)  # n, c, t, 1, 1

        x = x * y.expand_as(x)  # n, c, t, h, w
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # n, t, c, h, w

        return x.view(-1, c, h, w)  # nt, c, h, w


class LocalTemporalModule(nn.Module):
    def __init__(self, inplanes, planes, num_segments=8):
        super(LocalTemporalModule, self).__init__()
        self.num_segments = num_segments
        self.conv1 = nn.Conv3d(inplanes, planes // 4, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0), bias=False)
        self.conv2 = nn.Conv3d(planes // 4, planes, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(2, 0, 0), dilation=(2, 1, 1), bias=False)
        self.bn1 = nn.BatchNorm3d(planes // 4)
        self.bn2 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.xavier_normal_(self.conv2.weight)
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)
        nn.init.constant_(self.bn2.weight, 1)
        nn.init.constant_(self.bn2.bias, 0)

    #
    def forward(self, x):
        bn, c, h, w = x.size()
        x = x.view(-1, self.num_segments, c, h, w)  # n, t, c, h, w
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # n, c, t, h, w
        y = self.conv1(x)  # n, c, t, h, w
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)  # n, c, t, h, w
        y = self.bn2(y)
        y = self.sigmoid(y)
        x = x*y  # n, c, t, h, w
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(-1, c, h, w)

        return x


class SpatialModule(nn.Module):
    def __init__(self, inplanes, planes, num_segments=8):
        super(SpatialModule, self).__init__()

        self.num_segments = num_segments
        self.conv1 = nn.Conv2d(inplanes, planes//4, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        self.conv2 = nn.Conv2d(planes//4, planes, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        self.bn1 = nn.BatchNorm2d(planes//4)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.xavier_normal_(self.conv2.weight)
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)
        nn.init.constant_(self.bn2.weight, 1)
        nn.init.constant_(self.bn2.bias, 0)

    def forward(self, x):
        bn, c, h, w = x.size()
        x = x.view(-1, self.num_segments, c, h, w)  # n, t, c, h, w
        y = x.mean(dim=1).squeeze(1)  # n, c, h, w
        y = self.conv1(y)  # n, c, h, w
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)  # n, c, h, w
        y = self.bn2(y)
        y = self.sigmoid(y).view(-1, 1, c, h, w)  # n, 1, c, h, w
        x = x*y.expand_as(x)  # n, t, c, h, w

        return x.view(-1, c, h, w)
