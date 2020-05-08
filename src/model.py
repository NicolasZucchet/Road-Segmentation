"""
U-Net model adapted from https://github.com/milesial/Pytorch-UNet
SegNet model adapted from https://github.com/delta-onera/delta_tb
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import os, sys
import urllib


"""
This file consist in:
1. Auxiliary modules
2. UNet
3. Segnet
4. Model interface
"""


"""
1. Auxiliary modules
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
2. UNet
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
3. SegNet
"""

class SegNet(nn.Module):
    # Unet network
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal(m.weight.data)
    
    def __init__(self, in_channels, out_channels):
        super(SegNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.down1 = Down(in_channels, 64, n_convs=2, return_indices=True)
        self.down2 = Down(64, 128, n_convs=2, return_indices=True)
        self.down3 = Down(128, 256, n_convs=3, return_indices=True)
        self.down4 = Down(256, 512, n_convs=3, return_indices=True)
        self.down5 = Down(512, 512, n_convs=3, return_indices=True)

        self.up1 = Up(512, 512, n_convs=3, mode='max_unpool')
        self.up2 = Up(512, 256, n_convs=3, mode='max_unpool')
        self.up3 = Up(256, 128, n_convs=3, mode='max_unpool')
        self.up4 = Up(128, 64, n_convs=2, mode='max_unpool')
        self.unpool5 = nn.MaxUnpool2d(2)
        self.conv5 = nn.Sequential(
            ConvBNReLU(64, 64),
            nn.Conv2d(64, out_channels, 3, padding=1)
        )
        
        self.apply(self.weight_init)
        
    def forward(self, x):
        size1, mask1, x = self.down1(x)
        size2, mask2, x = self.down2(x)
        size3, mask3, x = self.down3(x)
        size4, mask4, x = self.down4(x)
        size5, mask5, x = self.down5(x)

        x = self.up1(x, mask=mask5, output_size = size5)
        x = self.up2(x, mask=mask4, output_size = size4)
        x = self.up3(x, mask=mask3, output_size = size3)
        x = self.up4(x, mask=mask2, output_size = size2)
        x = self.unpool5(x, mask1, output_size = size1)
        x = self.conv5(x)

        return x

    def load_pretrained_weights(self):

        vgg16_weights = model_zoo.load_url("https://download.pytorch.org/models/vgg16_bn-6c64b313.pth")

        count_vgg = 0
        count_this = 0

        vggkeys = list(vgg16_weights.keys())
        thiskeys  = list(self.state_dict().keys())

        corresp_map = []

        while(True):
            vggkey = vggkeys[count_vgg]
            thiskey = thiskeys[count_this]

            if "classifier" in vggkey:
                break
            
            while vggkey.split(".")[-1] not in thiskey:
                count_this += 1
                thiskey = thiskeys[count_this]


            corresp_map.append([vggkey, thiskey])
            count_vgg+=1
            count_this += 1

        mapped_weights = self.state_dict()
        for k_vgg, k_segnet in corresp_map:
            if (self.in_channels != 3) and "features" in k_vgg and "conv1_1." not in k_segnet:
                mapped_weights[k_segnet] = vgg16_weights[k_vgg]
            elif (self.in_channels == 3) and "features" in k_vgg:
                mapped_weights[k_segnet] = vgg16_weights[k_vgg]

        try:
            self.load_state_dict(mapped_weights)
            print("Loaded VGG-16 weights in Segnet !")
        except:
            print("Error VGG-16 weights in Segnet !")
            raise
    
    def load_from_filename(self, model_path):
        """Load weights from filename."""
        th = torch.load(model_path)  # load the weigths
        self.load_state_dict(th)


def segnet_bn_relu(in_channels, out_channels, pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SegNet(in_channels, out_channels)
    if pretrained:
        model.load_pretrained_weights()
    return model



"""
4. Model interface
"""

class Model(nn.Module):
    """
    Interface for choosing a specific model and share some methods
    """

    models = ['UNet', 'SegNet']

    def __init__(self, model_name, in_channels, out_channels, device=torch.device('cpu')):
        super().__init__()
        if model_name not in self.models:
            raise ValueError('Model name should be in ' + str(models))
        self.model = getattr(sys.modules[__name__], model_name)(in_channels, out_channels)
        print(self.model.forward)
        self.device = device
        print(device)
        self.model.to(self.device)

    def forward(self, x):
        return self.model(x)

    def load_weights(self, path=None):
        if path is not None:
            print(path)
            self.model.load_state_dict(torch.load(path, map_location=self.device))
