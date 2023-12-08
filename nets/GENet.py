import torch
import torch.nn as nn
# from .utils import load_state_dict_from_url
import torch.utils.model_zoo as model_zoo
import torchvision.models as models

from nets.ExcitationPath import ChannelModule, TemporalModule, SpatialModule, LocalTemporalModule

__all__ = ['ResNet', 'resnet50', 'resnet101', 'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    # 'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    # 'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    # 'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    # 'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

# def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
#     """3x3 convolution with padding"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
#                      padding=dilation, groups=groups, bias=False, dilation=dilation)


# def conv1x1(in_planes, out_planes, stride=1):
#     """1x1 convolution"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):
    expansion = 4
 
    def __init__(self, inplanes, planes, stride=1, downsample=None, use_ef=False, cdiv=8, num_segments=8, loop_id=0):
        super(Bottleneck, self).__init__()
        self.use_ef = use_ef
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        if self.use_ef:
            print('=> Using Partial Channel Excitation path with cdiv: {}'.format(cdiv))
            self.loop_id = loop_id
            self.eft_c = planes // cdiv
            self.eft1 = LocalTemporalModule(self.eft_c, self.eft_c, num_segments)
            self.eft2 = TemporalModule(self.eft_c, self.eft_c, num_segments)
            self.eft3 = SpatialModule(self.eft_c, self.eft_c, num_segments)
            self.eft4 = ChannelModule(self.eft_c, self.eft_c, num_segments)
            self.eft = (self.eft_c, self.eft_c, num_segments)
            self.start_c1 = loop_id*self.eft_c
            self.end_c1 = self.start_c1 + self.eft_c
            loop_id2 = (loop_id+1)%cdiv
            self.start_c2 = loop_id2*self.eft_c
            self.end_c2 = self.start_c2 + self.eft_c
            loop_id3 = (loop_id+2)%cdiv
            self.start_c3 = loop_id3*self.eft_c
            self.end_c3 = self.start_c3 + self.eft_c
            loop_id4 = (loop_id+3)%cdiv
            self.start_c4 = loop_id4*self.eft_c
            self.end_c4 = self.start_c4 + self.eft_c
            print('loop_ids: [{}:({}-{}), {}:({}-{}), {}:({}-{}), {}:({}-{})]'.format(loop_id, self.start_c1, self.end_c1, \
                loop_id2, self.start_c2, self.end_c2, loop_id3, self.start_c3, self.end_c3, loop_id4, self.start_c4, self.end_c4))
        self.downsample = downsample
        self.stride = stride
 
    def forward(self, x):
        # x = [bcz*n_seg, c, h, w]
        residual = x
 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
 
        out = self.conv2(out)
        out = self.bn2(out)
        #
        if self.use_ef:
            new_out = torch.zeros_like(out)
            BN, C_size, H_size, W_size = new_out.size()
            new_out[:, :, :, :] = out[:, :, :, :]

            new_out[:, self.start_c1:self.end_c1, :, :] = self.eft1(out[:, self.start_c1:self.end_c1, :, :])  # LT 1
            new_out[:, self.start_c2:self.end_c2, :, :] = self.eft2(out[:, self.start_c2:self.end_c2, :, :])  # temporal 1
            new_out[:, self.start_c3:self.end_c3, :, :] = self.eft3(out[:, self.start_c3:self.end_c3, :, :])  # spacial 1
            new_out[:, self.start_c4:self.end_c4, :, :] = self.eft4(out[:, self.start_c4:self.end_c4, :, :])  # channel 1
            # new_out = torch.zeros_like(out)
            # new_out[:, :self.eft_c, :, :] = self.eft(out[:, :self.eft_c, :, :])
            if self.end_c4 > self.start_c1:
                if self.start_c1 > 0:
                    new_out[:, :self.start_c1:, :, :] = out[:, :self.start_c1:, :, :]
                if self.end_c4 < C_size:
                    new_out[:, self.end_c4:, :, :] = out[:, self.end_c4:, :, :]
            elif self.end_c4 < self.start_c1:
                new_out[:, self.end_c4:self.start_c1:, :, :] = out[:, self.end_c4:self.start_c1:, :, :]

            out = new_out
            # out[:, self.eft_c:, :, :] = out[:, self.eft_c:, :, :]
        #
        out = self.relu(out)
 
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
 
        out += residual
        out = self.relu(out)
 
        return out


class ResNet(nn.Module):
 
    def __init__(self, block, layers, num_classes=1000, cdiv=2, num_segments=8, loop=False):
        self.inplanes = 64
        self.loop_id = 0
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.loop = loop
        self.layer1 = self._make_layer(block, 64, layers[0], use_ef=True, cdiv=cdiv, n_seg=num_segments)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, use_ef=True, cdiv=cdiv, n_seg=num_segments)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, use_ef=True, cdiv=cdiv, n_seg=num_segments)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, use_ef=True, cdiv=cdiv, n_seg=num_segments)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for name, m in self.named_modules():
            if 'eft' not in name:
                # if 'deconv' in name:
                #     nn.init.xavier_normal_(m.weight)
                # else:
                #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, use_ef=True, cdiv=2, n_seg=8):
        print('=> Processing stage with {} blocks'.format(blocks))
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
 
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_ef=use_ef, cdiv=cdiv, num_segments=n_seg, loop_id=self.loop_id))
        self.inplanes = planes * block.expansion
        if self.loop:
            self.loop_id = (self.loop_id+1)%cdiv

        #
        n_round = 1
        if blocks >= 23:
            n_round = 2
            print('=> Using n_round {} to insert Element Filter -T'.format(n_round))
        #
        for i in range(1, blocks):
            if i % n_round == 0:
                # use_ef = True
                layers.append(block(self.inplanes, planes, use_ef=use_ef, cdiv=cdiv, num_segments=n_seg, loop_id=self.loop_id))
                if self.loop:
                    self.loop_id = (self.loop_id+1)%cdiv
            else:
                # use_ef = False
                layers.append(block(self.inplanes, planes, use_ef=use_ef, cdiv=cdiv, num_segments=n_seg))
 
        return nn.Sequential(*layers)
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
 
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
 
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
 
        return x


# def _resnet(arch, block, layers, pretrained, progress, **kwargs):
#     model = ResNet(block, layers, **kwargs)
#     if pretrained:
#         state_dict = load_state_dict_from_url(model_urls[arch],
#                                               progress=progress)
#         model.load_state_dict(state_dict)
#     return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # EF_name = getattr(EF_zoo, EF)
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        checkpoint = model_zoo.load_url(model_urls['resnet50'])
        # checkpoint_keys = list(checkpoint.keys())
        model_dict = model.state_dict()
        model_dict.update(checkpoint)
        model.load_state_dict(model_dict)

    return model
 
 
def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # EF_name = getattr(EF_zoo, EF)
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        checkpoint = model_zoo.load_url(model_urls['resnet101'])
        # checkpoint_keys = list(checkpoint.keys())
        model_dict = model.state_dict()
        model_dict.update(checkpoint)
        model.load_state_dict(model_dict)
    return model
 
 
def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        checkpoint = model_zoo.load_url(model_urls['resnet152'])
        # checkpoint_keys = list(checkpoint.keys())
        model_dict = model.state_dict()
        model_dict.update(checkpoint)
        model.load_state_dict(model_dict)
    return model
