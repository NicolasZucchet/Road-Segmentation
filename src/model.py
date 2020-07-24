"""
This file is structured as:
 - Auxiliary modules
 - UNet class
 - Model interface
U-Net model adapted from https://github.com/milesial/Pytorch-UNet.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import os, sys
import urllib



"""
Auxiliary modules
"""

class ConvBNReLU(nn.Module):
    """(convolution => [BN] => ReLU)"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            ConvBNReLU(in_channels, mid_channels),
            ConvBNReLU(mid_channels, out_channels)
        )

    def forward(self, x):
        return self.double_conv(x)
        

class TripleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 3"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.triple_conv = nn.Sequential(
            ConvBNReLU(in_channels, mid_channels),
            ConvBNReLU(mid_channels, mid_channels),
            ConvBNReLU(mid_channels, out_channels)
        )

    def forward(self, x):
        return self.triple_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, n_convs=2, return_indices=False):
        super().__init__()
        self.return_indices = return_indices
        if n_convs == 2:
            self.conv = DoubleConv(in_channels, out_channels)
        elif n_convs == 3:
            self.conv = TripleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2, return_indices=return_indices)

    def forward(self, x):
        if self.return_indices:
            x_conv =  self.conv(x)
            output, indices = self.pool(x_conv)
            return x_conv.size(), indices, output
        else:
            x_pool = self.pool(x)
            return self.conv(x_pool)


class Up(nn.Module):
    """Upscaling then several ConvBNReLU"""

    def __init__(self, in_channels, out_channels, n_convs=2, mode='bilinear'):
        super().__init__()
        self.mode = mode

        # use the number of Conv->BN->ReLU wanted
        if n_convs == 2:
            conv = DoubleConv
        elif n_convs == 3:
            conv = TripleConv
        else:
            raise ValueError('n_convs should be 2 or 3')

        # if bilinear, use the normal convolutions to reduce the number of channels
        if mode=='bilinear':
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = conv(in_channels, out_channels // 2, in_channels // 2)
        elif mode=='max_unpool':
            self.up =  nn.MaxUnpool2d(2)
            self.conv = conv(in_channels, out_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = conv(in_channels, out_channels)

    def forward(self, x1, x2=None, mask=None, output_size=None):
        if self.mode == 'max_unpool':
            if mask is None or output_size is None:
                raise ValueError('mask and output_size should be given')
            x = self.up(x1, mask, output_size=output_size)
            return self.conv(x)

        else:
            if x2 is None:
                raise ValueError('x2 should be given')
            x1 = self.up(x1)
            # input is CHW
            diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
            diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
            # if you have padding issues, see
            # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
            # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
            x = torch.cat([x2, x1], dim=1)
            return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)



"""
UNet
"""

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        mode = 'bilinear' if bilinear else None
        factor = 2 if bilinear else 1

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512, mode=mode)
        self.up2 = Up(512, 256, mode=mode)
        self.up3 = Up(256, 128, mode=mode)
        self.up4 = Up(128, 64 * factor, mode=mode)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x2=x4)
        x = self.up2(x, x2=x3)
        x = self.up3(x, x2=x2)
        x = self.up4(x, x2=x1)
        logits = self.outc(x)
        return logits



"""
Model interface
"""

class Model(nn.Module):
    """
    Interface for choosing a specific model and share some methods
    """

    models = ['UNet']

    def __init__(self, model_name, in_channels, out_channels, device=torch.device('cpu')):
        super().__init__()
        if model_name not in self.models:
            raise ValueError('Model name should be in ' + str(models))
        self.model = getattr(sys.modules[__name__], model_name)(in_channels, out_channels)
        self.device = device
        self.model.to(self.device)

    def forward(self, x):
        return self.model(x)

    def load_weights(self, path=None):
        if path is not None:
            try:
                self.model.load_state_dict(torch.load(path, map_location=self.device))
            except:
                # handle the case when the names are not excatly the same (model. before each key)
                # it happens when the weights are saved on GPU and loaded on CPU
                saved_state_dict = torch.load(path, map_location=self.device)
                modified_state_dict = {}
                for key in saved_state_dict.keys():
                    if 'model.' in key:
                        modified_state_dict[key[6:]] = saved_state_dict[key]
                    else:
                        modified_state_dict[key] = saved_state_dict[key]
                self.model.load_state_dict(modified_state_dict)
        else:
            self.load_vgg_weights()
            print("Loaded VGG weights")

    def load_vgg_weights(self):
        vgg_weights = model_zoo.load_url("https://download.pytorch.org/models/vgg16_bn-6c64b313.pth")

        vgg_keys = list(vgg_weights.keys())
        model_keys  = list(self.state_dict().keys())

        # while corresponding size, match model weights to vgg weights
        mapped_weights = self.state_dict()
        i_model, i_vgg = 0, 0
        while i_model < len(model_keys) and i_vgg < len(vgg_keys):
            model_key, vgg_key = model_keys[i_model], vgg_keys[i_vgg]
            model_weight, vgg_weight = mapped_weights[model_key], vgg_weights[vgg_key]
            if 'num_batches_tracked' in model_key:
                i_model += 1
                continue
            if model_weight.size() == vgg_weight.size():
                mapped_weights[model_key] = vgg_weight
                i_vgg += 1
                i_model += 1
            else:
                break

        try:
            self.load_state_dict(mapped_weights)
            print("VGG-16 weights loaded")
        except:
            print("Error VGG-16 weights")
            raise
