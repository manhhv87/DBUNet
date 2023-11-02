from __future__ import absolute_import, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


# Define the network structure of DeepLabV1
class DeepLabV1(nn.Sequential):
    """
    DeepLab v1: Dilated ResNet + 1x1 Conv
    Note that this is just a container for loading the pretrained COCO model 
    and not mentioned as "v1" in papers.
    """

    def __init__(self, n_classes, n_blocks,):
        super(DeepLabV1, self).__init__()
        ch = [64 * 2 ** p for p in range(6)]
        self.convs = nn.ModuleList([_Stem(ch[0]),
                                    _ResLayer(n_blocks[0], ch[0], ch[2], 1, 1),
                                    _ResLayer(n_blocks[1], ch[2], ch[3], 2, 1),
                                    _ResLayer(n_blocks[2], ch[3], ch[4], 1, 2),
                                    _ResLayer(n_blocks[3], ch[4], ch[5], 1, 4),
                                    nn.Conv2d(2048, n_classes, 1)])

    def forward(self, x):
        h = x.size()[2]
        w = x.size()[3]

        for lay in self.convs:
            x = lay(x)

        # (shape: (batch_size, num_classes, h, w))
        x = F.interpolate(x, size=(h, w), mode="bilinear")
        return x


# Here is a look at whether to use BatchNorm in the nn module of torch
# or BatchNorm defined in the encoding file.
_BATCH_NORM = nn.BatchNorm2d
_BOTTLENECK_EXPANSION = 4


# Define the components of Conv+BN+ReLU
class _ConvBnReLU(nn.Sequential):
    """
    Cascade of 2D convolution, batch norm, and ReLU.
    """

    BATCH_NORM = _BATCH_NORM

    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, dilation, relu=True):
        super(_ConvBnReLU, self).__init__()
        self.add_module("conv",
                        nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation, bias=False))
        self.add_module("bn",
                        _BATCH_NORM(out_ch, eps=1e-5, momentum=0.999))

        if relu:
            self.add_module("relu", nn.ReLU())


# Define Bottleneck, first use 1*1 convolution to reduce the dimension, then use 3*3 convolution,
# and finally use 1*1 convolution to increase the dimension, and then shortcut connection.
# The dimensionality reduction is determined by the _BOTTLENECK_EXPANSION parameter,
# which is the Bottleneck of ResNet.
class _Bottleneck(nn.Module):
    """
    Bottleneck block of MSRA ResNet.
    """

    def __init__(self, in_ch, out_ch, stride, dilation, downsample):
        super(_Bottleneck, self).__init__()
        mid_ch = out_ch // _BOTTLENECK_EXPANSION
        self.reduce = _ConvBnReLU(in_ch, mid_ch, 1, stride, 0, 1, True)
        self.conv3x3 = _ConvBnReLU(mid_ch, mid_ch, 3, 1,
                                   dilation, dilation, True)
        self.increase = _ConvBnReLU(mid_ch, out_ch, 1, 1, 0, 1, False)
        self.shortcut = (_ConvBnReLU(in_ch, out_ch, 1, stride, 0, 1, False)
                         if downsample
                         else lambda x: x  # identity
                         )

    def forward(self, x):
        h = self.reduce(x)
        h = self.conv3x3(h)
        h = self.increase(h)
        h += self.shortcut(x)
        return F.relu(h)


# Define ResLayer. The entire DeepLabv1 is stacked with ResLayer.
# Downsampling occurs in the first Bottleneck of each ResLayer.
class _ResLayer(nn.Sequential):
    """
    Residual layer with multi grids
    """

    def __init__(self, n_layers, in_ch, out_ch, stride, dilation, multi_grids=None):
        super(_ResLayer, self).__init__()

        if multi_grids is None:
            multi_grids = [1 for _ in range(n_layers)]
        else:
            assert n_layers == len(multi_grids)

        # Downsampling is only in the first block
        for i in range(n_layers):
            self.add_module(
                "block{}".format(i + 1),
                _Bottleneck(in_ch=(in_ch if i == 0 else out_ch),
                            out_ch=out_ch,
                            stride=(stride if i == 0 else 1),
                            dilation=dilation * multi_grids[i],
                            downsample=(True if i == 0 else False))
            )


# Before entering ResLayer, first use a 7*7 convolution kernel to slide
# on the original image to increase the receptive field. The padding mode
# is set to same and the size remains unchanged. The kernel size of
# the Pool layer is 3 and the stride is 2, which causes the resolution
# of the feature map to change.
class _Stem(nn.Sequential):
    """
    The 1st conv layer.
    Note that the max pooling is different from both MSRA and FAIR ResNet.
    """

    def __init__(self, out_ch):
        super(_Stem, self).__init__()
        self.add_module("conv1", _ConvBnReLU(3, out_ch, 7, 2, 3, 1))
        self.add_module("pool", nn.MaxPool2d(3, 2, 1, ceil_mode=True))


# Equivalent to Reshape, the network is not used
class _Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


# Main function, outputs the structure of the constructed DeepLab V1 model
# as well as the resolution of the original image and the resolution of the result image
if __name__ == "__main__":
    # model.eval()
    x = torch.randn(1, 3, 513, 513)
    h = x.size()[2]
    w = x.size()[3]
    model = DeepLabV1(n_classes=1, n_blocks=[3, 4, 23, 3])

    print(model)
    print("input:", x.shape)

    # Input image size does not match output image size
    print("output:", model(x).shape)
