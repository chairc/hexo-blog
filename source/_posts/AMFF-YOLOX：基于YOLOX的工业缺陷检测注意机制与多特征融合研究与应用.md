---
title: AMFF-YOLOX：基于YOLOX的工业缺陷检测注意机制与多特征融合研究与应用
date: 2024-12-15 17:48:50
hide: true
description: 在工业制造领域，保证工业产品的质量是该领域的一项重要任务。对于工业产品，一个小的缺陷有时会危及整体影响。例如，印刷电路板上的断点会影响设备信号的稳定传导，而金属裂纹会影响产品的美观性和强度。通常，一般的工业产品质量检查是由人进行的，这也有一些缺点，包括需要进行大量的初始检查人员培训，这增加了人员培训的成本。随着值班人员检查时间的增加，因其原因导致的误检率上升。随着计算机视觉的发展，自动化质量检测的使用已成为该行业的解决方案。工业缺陷的视觉检测可以降低成本，提高效率。
categories: 
- [人工智能, 计算机视觉, 检测模型]
- [Python项目应用]
tags: [检测模型, YOLO, Python]
math: true
---

# 0 介绍
在工业制造领域，保证工业产品的质量是该领域的一项重要任务。对于工业产品，一个小的缺陷有时会危及整体影响。例如，印刷电路板上的断点会影响设备信号的稳定传导，而金属裂纹会影响产品的美观性和强度。通常，一般的工业产品质量检查是由人进行的，这也有一些缺点，包括需要进行大量的初始检查人员培训，这增加了人员培训的成本。随着值班人员检查时间的增加，因其原因导致的误检率上升。随着计算机视觉的发展，自动化质量检测的使用已成为该行业的解决方案。工业缺陷的视觉检测可以降低成本，提高效率。

本文将根据现有技术与存在问题，进行工业缺陷检测的应用研究。包括[原文](https://www.mdpi.com/2079-9292/12/7/1662)、实现过程与代码，详细的代码请访问我的[GitHub](https://github.com/chairc/NRSD-MN-relabel)，如有问题可在评论区提出问题或在[此链接](https://github.com/chairc/NRSD-MN-relabel/issues)提出Issue。

# 1 网络整体结构
## 1.1 网络总览
YOLOX有6种不同的型号： YOLOX-nano、YOLOX-tinty、YOLOX-s、YOLOX-m、YOLOX-l和YOLOX-x。它使用CSP-Darknet和空间金字塔池（SPP）作为主干网络结构，路径聚合网络（PANet）作为网络的颈部部分，头部使用与上一代YOLO系列不同的解耦头部。

本文将选取YOLOX-s模型作为基准模型，如下图所示，本文改进了YOLOX的整体结构。改进后的网络结构分为主干网络、特征提取网络和检测网络。改进后的特征提取网络（蓝色部分）由主干网络注意提取层（红色部分）、带有注意模块的多尺度特征层和自适应空间特征融合层（紫色部分）组成。该编码器由底层主干网络和特征提取网络组成，检测解码器由三个解耦的检测头（绿色部分）组成。

![网络总览](/52087f507d3d1fedc485d676c977a4ed.png)

## 1.2 网络改进
### 1.2.1 特征提取网络PANet的改进
为了更好地关注工业缺陷，本文在主干网络的后三层和PANet的CSP层的输出位置添加了一个ECA（Efficient channel attention）模块，如下图所示。使用ECA模块并没有给模型添加太多的参数。同时，对不同特征映射的相关度分配加权系数，从而发挥强化重要特征的作用。本文在PANet后加入了自适应空间特征融合技术。它对特征提取网络后三层的三尺度特征信息输出进行加权和求和，以增强特征尺度的不变性。

![特征提取网络PANet的改进](/f1b7a0b820ebde76f84e51790586823f.png)

**该部分实现代码如下**：

```python
import torch
import torch.nn as nn

from .darknet import CSPDarknet
from .network_blocks import BaseConv, CSPLayer, DWConv
from .yolo_pafpn_attention import ECA
from .yolo_pafpn_asff import ASFF


class YOLOPAFPN(nn.Module):
    def __init__(
            self,
            depth=1.0,
            width=1.0,
            in_features=("dark3", "dark4", "dark5"),
            in_channels=[256, 512, 1024],
            depthwise=False,
            act="silu",
    ):
        super().__init__()
        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.lateral_conv0 = BaseConv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
        )
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.reduce_conv1 = BaseConv(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv2 = Conv(
            int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv1 = Conv(
            int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # 注意力通达大小
        self.neck_channels = [512, 256, 512, 1024]

        # ECA
        # dark5 1024
        self.eca_1 = ECA(int(in_channels[2] * width))
        # dark4 512
        self.eca_2 = ECA(int(in_channels[1] * width))
        # dark3 256
        self.eca_3 = ECA(int(in_channels[0] * width))
        # FPN CSPLayer 512
        self.eca_fp1 = ECA(int(self.neck_channels[0] * width))
        # PAN CSPLayer pan_out2 下采样 256
        self.eca_pa1 = ECA(int(self.neck_channels[1] * width))
        # PAN CSPLayer pan_out1 下采样 512
        self.eca_pa2 = ECA(int(self.neck_channels[2] * width))
        # PAN CSPLayer pan_out0 1024
        self.eca_pa3 = ECA(int(self.neck_channels[3] * width))

        # ASFF
        self.asff_1 = ASFF(level=0, multiplier=width)
        self.asff_2 = ASFF(level=1, multiplier=width)
        self.asff_3 = ASFF(level=2, multiplier=width)

    def forward(self, input):
        """
        Args:
            inputs: 输入图像

        Returns:
            Tuple[Tensor]: 特征信息.
        """

        # 主干网络
        out_features = self.backbone(input)
        features = [out_features[f] for f in self.in_features]
        [x2, x1, x0] = features

        # ECA
        x0 = self.eca_1(x0)
        x1 = self.eca_2(x1)
        x2 = self.eca_3(x2)

        # FPN

        # dark5
        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        # 上采样
        f_out0 = self.upsample(fpn_out0)  # 512/16
        # dark4 + 上采样
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16
        # ECA
        f_out0 = self.eca_fp1(f_out0)

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        # 上采样
        f_out1 = self.upsample(fpn_out1)  # 256/8
        # dark3 + 上采样
        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        #YOLO HEAD
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8
        # ECA
        pan_out2 = self.eca_pa1(pan_out2)

        # PAN

        # pan_out2 下采样
        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        # pan_out1 下采样 + CSPLayer fpn_out1
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        # YOLO HEAD
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16
        # ECA
        pan_out1 = self.eca_pa2(pan_out1)

        # 下采样
        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        # p_out0 下采样 + fpn_out0
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        # YOLO HEAD
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32
        # ECA
        pan_out0 = self.eca_pa3(pan_out0)

        outputs = (pan_out2, pan_out1, pan_out0)

        # ASFF
        pan_out0 = self.asff_1(outputs)
        pan_out1 = self.asff_2(outputs)
        pan_out2 = self.asff_3(outputs)
        outputs = (pan_out2, pan_out1, pan_out0)

        return outputs
```

### 1.2.2 注意力模块
ECA（Efficient channel attention）是一种轻量级的注意机制，它简单、有效，易于集成到现有的网络中，而不需要降维。使用一维卷积可以有效地捕获局部跨通道交互来提取通道间的依赖关系，允许在不向网络添加更多参数的情况下增强聚焦特征。为了使网络每次都能学习到所需的特征，本文在改进后的模型中添加了一个ECA模块，如下图所示。每个注意力组由一个CSP（Cross-Stage Partial ）层、一个ECA模块和一个基本卷积块组成。CSP层增强了整个网络学习特征的能力，并将特征提取的结果传递到ECA模块中。ECA模块的第一步是对传入的特征映射执行平均池化操作。第二步使用核为3的一维卷积来计算结果。第三步，应用上述结果，利用Sigmoid激活函数获得每个通道的权值。第四步，将权值与原始输入特征图的对应元素相乘，得到最终的输出特征图。最后，将结果输出到后续的基卷积块或单独输出。

![ECA注意力模块](/50bf64ea6e7d119de8bc7a3789b58224.png)


**该部分实现代码如下**：

```python
# ECANet
class ECA(nn.Module):
    def __init__(self, channel, b=1, gamma=2, kernel_size=3):
        super(ECA, self).__init__()
        # https://github.com/BangguWu/ECANet/issues/24#issuecomment-664926242
        # 自适应内核不容易实现，所以它是固定的“kernel_size=3”
        # 由于输入通道的原因，无法正确确认内核大小，可能会导致错误
        # kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        # kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 全局空间信息特征
        y = self.avg_pool(x)
        # ECA模块的两个不同分支
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # 多尺度信息融合
        y = self.sigmoid(y)
        return x * y.expand_as(x)
```

### 1.2.3 基于注意力模块的自适应特征融合
Liu等人（ 论文Learning spatial fusion for single-shot object detection）为了解决多尺度特征之间的不一致性问题，提出了自适应空间特征融合的方法。它使网络能够直接学习如何在其他层次上对特征进行空间过滤，以便只保留有用的信息用于组合。如下图所示，本文通过保留三种不同尺度的特征图的ECA模块的最终输出来进行特征提取层。自适应空间特征融合机制对这三个特征图尺度在20×20、40×40和80×80的不同尺度下的特征图信息进行权值和求和，并计算出相应的权值。

![基于注意力模块的自适应特征融合](/cd77e94e4bb8dc1ce7b18b82e552f69d.png)


公式（1）中$X^{eca1\rightarrow{level}}_{ij}$、$X^{eca2\rightarrow{level}}_{ij}$和$X^{eca3\rightarrow{level}}_{ij}$ 分别代表了PANet的三个注意力机制（上图中的ECA-1、ECA-2和ECA-3）的特征信息。本文将上述特征信息与权重参数$\alpha^{level}_{ij}$、$\beta^{level}_{ij}$ 和 $\gamma^{level}_{ij}$ （例如：$\alpha$、$\beta$和$\gamma$在位置 $(i, j)$ 的通道中共享的权重参数）调整到相同大小的特征图，然后将它们进行叠加操作形成一个新的融合层。

$$(1)\    y^{level}_{ij} = \alpha^{level}_{ij} \cdot X^{eca1\rightarrow{level}}_{ij} + \beta^{level}_{ij} \cdot X^{eca2\rightarrow{level}}_{ij} + \gamma^{level}_{ij} \cdot X^{eca3\rightarrow{level}}_{ij} $$

在公式（2）中， $\alpha^{level}_{ij}$、 $\beta^{level}_{ij}$ 和 $\gamma^{level}_{ij}$ 由softmax函数定义为一个和为1的参数并且 公式（3）中它们的范围属于 $[0, 1]$ 。 公式（4）中是权重$\alpha^{level}_{ij}$、 $\beta^{level}_{ij}$ 和 $\gamma^{level}_{ij}$的计算方法，其中 $\lambda^{level}_{\alpha}$、 $\lambda^{level}_{\beta}$ 和 $\lambda^{level}_{\gamma}$ 是通过 $X^{eca1\rightarrow{level}}$、$X^{eca2\rightarrow{level}}$ 和  $X^{eca3\rightarrow{level}}$ 计算所得 ，$\theta$ 是权重参数 $\alpha$, $\beta$ and $\gamma$ 的集合， $\theta^{level}_{ij}$是计算的权重参数名$\alpha^{level}_{ij}$、$\beta^{level}_{ij}$和$\gamma^{level}_{ij}$的统称。

$$(2)\    \alpha^{level}_{ij} + \beta^{level}_{ij} + \gamma^{level}_{ij} = 1 $$

$$(3)\    \alpha^{level}_{ij}, \beta^{level}_{ij}, \gamma^{level}_{ij} \in [0, 1] $$

$$(4)\    \theta^{level}_{ij} = \frac{e^{\lambda^{level}_{\theta_{ij}}}}{e^{\lambda^{level}_{\alpha_{ij}}} + e^{\lambda^{level}_{\beta_{ij}}} + e^{\lambda^{level}_{\gamma_{ij}}}}, \theta \in [\alpha, \beta, \gamma] $$


**该部分实现代码如下**：

```python
# ASFF实现方式
import torch
import torch.nn as nn
import torch.nn.functional as F


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution
    # ch_in, ch_out, kernel, stride, padding, groups
    def __init__(self, in_channels, out_channels, kernel=1, stride=1, padding=None, groups=1, act=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride, autopad(kernel, padding), groups=groups,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class ASFF(nn.Module):
    def __init__(self, level, multiplier=1, rfb=False, vis=False, act_cfg=True):
        """
        multiplier should be 1, 0.5
        which means, the channel of ASFF can be
        512, 256, 128 -> multiplier=0.5
        1024, 512, 256 -> multiplier=1
        For even smaller, you need change code manually.
        """
        super(ASFF, self).__init__()
        self.level = level
        self.dim = [int(1024 * multiplier), int(512 * multiplier),
                    int(256 * multiplier)]
        # print(self.dim)

        self.inter_dim = self.dim[self.level]
        if level == 0:
            self.stride_level_1 = Conv(int(512 * multiplier), self.inter_dim, 3, 2)

            self.stride_level_2 = Conv(int(256 * multiplier), self.inter_dim, 3, 2)

            self.expand = Conv(self.inter_dim, int(1024 * multiplier), 3, 1)
        elif level == 1:
            self.compress_level_0 = Conv(int(1024 * multiplier), self.inter_dim, 1, 1)
            self.stride_level_2 = Conv(int(256 * multiplier), self.inter_dim, 3, 2)
            self.expand = Conv(self.inter_dim, int(512 * multiplier), 3, 1)
        elif level == 2:
            self.compress_level_0 = Conv(int(1024 * multiplier), self.inter_dim, 1, 1)
            self.compress_level_1 = Conv(int(512 * multiplier), self.inter_dim, 1, 1)
            self.expand = Conv(self.inter_dim, int(256 * multiplier), 3, 1)

        # when adding rfb, we use half number of channels to save memory
        compress_c = 8 if rfb else 16
        self.weight_level_0 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = Conv(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = Conv(compress_c * 3, 3, 1, 1)
        self.vis = vis

    def forward(self, x):  # l,m,s
        """
        #
        256, 512, 1024
        from small -> large
        """
        # max feature
        global level_0_resized, level_1_resized, level_2_resized
        x_level_0 = x[2]
        # mid feature
        x_level_1 = x[1]
        # min feature
        x_level_2 = x[0]

        if self.level == 0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_downsampled_inter = F.max_pool2d(x_level_2, 3, stride=2, padding=1)
            level_2_resized = self.stride_level_2(level_2_downsampled_inter)
        elif self.level == 1:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(level_0_compressed, scale_factor=2, mode='nearest')
            level_1_resized = x_level_1
            level_2_resized = self.stride_level_2(x_level_2)
        elif self.level == 2:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(level_0_compressed, scale_factor=4, mode='nearest')
            x_level_1_compressed = self.compress_level_1(x_level_1)
            level_1_resized = F.interpolate(x_level_1_compressed, scale_factor=2, mode='nearest')
            level_2_resized = x_level_2

        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)

        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = level_0_resized * levels_weight[:, 0:1, :, :] + \
                            level_1_resized * levels_weight[:, 1:2, :, :] + \
                            level_2_resized * levels_weight[:, 2:, :, :]

        out = self.expand(fused_out_reduced)

        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out
```

```python
		# 三层从ECA中输出的特征信息
		outputs = (pan_out2, pan_out1, pan_out0)
		# 在特征提取网络中的ASFF使用
        pan_out0 = self.asff_1(outputs)
        pan_out1 = self.asff_2(outputs)
        pan_out2 = self.asff_3(outputs)
		# 三层从ASFF中输出的特征信息
        outputs = (pan_out2, pan_out1, pan_out0)
```

### 1.2.4 Bottleneck优化设计
CouvNeXt网络是由Liu等人（论文A convnet for the 2020s）的研究提出的。对大型卷积核采用逆瓶颈结构，并采用较少的归一化和激活函数来提高模型性能。基于这一思想，本文中的模型在每个CSP层中都进行了微观的瓶颈设计，并尝试采用逆瓶颈结构。然而，结果与原始结果相似，且效果没有明显改善，因此本文没有采用逆瓶颈结构。最后，本文基于CSP-Darknet模型，提出了ConvNeXt的瓶颈设计模式。在模型1×1卷积后去掉一个SiLU激活函数，在3×3卷积后删除一个归一化函数，如下图所示。通过对简化的归一化操作和激活函数操作分别进行测试，发现最终结果优于原始结构。

![改进的Bottleneck](/493cec1dfb927e7bf4059cce13f5ba66.png)

# 2 实验效果
## 2.1 数据集
以下是我们实验中使用的公共数据集： [NRSD-MN](https://github.com/zdfcvsn/MCnet)（重新标记的数据集链接：[NRSD-MN-relabel](https://drive.google.com/drive/folders/13r-l_OEUt63A8K-ol6jQiaKNuGdseZ7j?usp=sharing)）、[PCB](https://robotics.pkusz.edu.cn/resources/dataset/)和[NEU-DET](http://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/)。NRSD-MN数据集共有1个类别中的4101张图像，图像大小从400到800像素不等。实验共分为2971个训练集和1130个验证集。PCB数据集共有6个类别中的693张图像，它们的大小都大于1000像素。这6个类别分别是 missing hole、mouse bite、open circuit、short、spur和spurious copper。实验中分为554个训练集和139个验证集。NEU-DET数据集由1800张图像组成，其中6种缺陷被标记为crazing、inclusion、patches、 pitted surface、 rolled in scale和scratches。此外，图像大小为200×200像素。实验中分为1620个训练集和180个验证集。所有的验证集都来自于数据集自己的划分。
## 2.2 实验环境
本文的实验环境如下： Ubuntu 18.04、Python 3.8、Pytorch 1.8.1、CUDA 11.1，所有模型都使用相同的NVIDIA RTX 3060GPU 12GB。具体的实验参数可以参考文章或源代码。

## 2.3 实验结果

### 2.3.1 对比实验
在本研究中，我们设计了一组比较实验和四组消融实验，以mAP@0.5：0.95、mAP@0.5和FPS作为评价指标并以YOLOX为基线。在工业缺陷检测对比实验中，将YOLOX-tiny和YOLOX-s作为基本模型，并与YOLOv3-tiny、YOLOv5-s和YOLOv8-s进行了比较。在实验结果的对比图中，(a)表示图像的真实值，(b)表示基线的预测结果，(c)表示本文模型的预测结果。**详细分析可参考论文实验部分**。

**工业缺陷数据集检测的比较实验结果（mAP@0.5：0.95）**
|   Network    | NRSD(mAP@0.5:0.95) | PCB(mAP@0.5:0.95) | NEU(mAP@0.5:0.95) |
| :----------: | :----------------: | :---------------: | :---------------: |
| YOLOv3-tiny  |       46.29        |       42.48       |       21.32       |
|   YOLOv5-s   |       52.10        |       45.19       |       37.47       |
|   YOLOv8-s   |       56.29        |       51.16       |       43.31       |
|  YOLOX-tiny  |       56.50        |       45.91       |       41.04       |
|   YOLOX-s    |       57.74        |       49.72       |       47.61       |
| AMFF-YOLOX-s |     **61.06**      |     **51.58**     |     **49.08** 

**工业缺陷数据集检测的比较实验结果（mAP@0.5）**
|   Network    | NRSD(mAP@0.5) | PCB(mAP@0.5) | NEU(mAP@0.5) |
| :----------: | :-----------: | :----------: | :----------: |
| YOLOv3-tiny  |     78.26     |    90.69     |    55.02     |
|   YOLOv5-s   |     80.85     |    90.40     |    72.60     |
|   YOLOv8-s   |     80.48     |  **93.42**   |    75.64     |
|  YOLOX-tiny  |     81.68     |    88.07     |    77.98     |
|   YOLOX-s    |     80.50     |    89.51     |    78.49     |
| AMFF-YOLOX-s |   **85.00**   |    91.09     |  **80.48**   |

**NRSD对比结果**
![NRSD对比结果](/b5447f80f66164bddfea103ca7750a42.png)

**PCB对比结果**
![PCB对比结果](/e5ca9f7240813aa17af8383218904b64.png)

**NEU-DET对比结果**
![NEU-DET对比结果](/d727ece336c30bd11d3df5fa0624271e.png)

### 2.3.2 消融实验
为了验证模型各模块的有效性，本文首先在经典的VOC数据集中进行了烧蚀实验。该实验以YOLOX-tiny为基线，将输入大小设置为416×416，其他训练设置与基本设置相同，如下表所示。我们分别将其添加到YOLOX-tiny的FPN和PAN的不同位置的注意机制。在消融实验表中，本实验使用A作为ECA模块，B作为ASFF模块，C作为修改后的Bottleneck模块。**详细分析可参考论文实验部分**。

**在VOC2007数据集上的消融实验结果**
|       Network       | VOC(mAP@0.5:0.95) | VOC(mAP@0.5) |   FPS   |
| :-----------------: | :---------------: | :----------: | :-----: |
|      Baseline       |       35.85       |    59.49     | **340** |
|      + A (FPN)      |       36.62       |    60.45     |   331   |
|   + A (FPN + PAN)   |       36.73       |    60.10     |   328   |
|         + B         |       36.66       |    60.84     |   288   |
|    + A (FPN) + B    |       37.10       |    60.93     |   299   |
| + A (FPN + PAN) + B |     **37.41**     |  **61.06**   |   298   |

**在NRSD数据集上的消融实验结果**
|   Network   | NRSD(mAP@0.5:0.95) | NRSD(mAP@0.5) |   FPS   |
| :---------: | :----------------: | :-----------: | :-----: |
|  Baseline   |       58.27        |     81.89     |   144   |
|     + A     |       59.25        |     83.92     |   142   |
|     + B     |       59.67        |     83.55     |   124   |
|     + C     |       58.79        |     83.29     | **151** |
| + A + B + C |     **61.06**      |   **85.00**   |   129   |

**在PCB数据集上的消融实验结果**
|   Network   | PCB(mAP@0.5:0.95) | PCB(mAP@0.5) |   FPS   |
| :---------: | :---------------: | :----------: | :-----: |
|  Baseline   |       49.72       |    89.51     |   144   |
|     + A     |       50.44       |    90.02     |   142   |
|     + B     |       51.27       |    90.37     |   119   |
|     + C     |       51.02       |    90.86     | **150** |
| + A + B + C |     **51.58**     |  **91.09**   |   125   |

**在NEU-DET数据集上的消融实验结果**
|   Network   | NEU(mAP@0.5:0.95) | NEU(mAP@0.5) |   FPS   |
| :---------: | :---------------: | :----------: | :-----: |
|  Baseline   |       47.61       |    78.49     | **153** |
|     + A     |       48.48       |    80.20     |   151   |
|     + B     |       48.18       |    79.53     |   123   |
|     + C     |       48.53       |    79.59     |   151   |
| + A + B + C |     **49.08**     |  **80.48**   |   131   |

# 3 最后
在本文中我们提出了一种改进的工业缺陷检测网络AMFF-YOLOX，该网络结合了注意机制、自适应空间特征融合和改进的瓶颈模块，以在不牺牲太多速度的情况下提高缺陷检测的准确性。通过大量的消融实验和与现有的最先进的方法的比较，验证了该模型的整体有效性和竞争力。

如果有任何关于结构、代码调参、项目部署、数据集等相关问题欢迎访问我的[GitHub](https://github.com/chairc)或给我发邮件（chenyu1998424@gmail.com）。

如果你的我的项目感兴趣，可以访问项目[源码网址](https://github.com/chairc/NRSD-MN-relabel)。

如果你对AIGC方向感兴趣，可以来看看我的其它项目：[基于分布式的工业缺陷检测扩散模型](https://github.com/chairc/Industrial-Defect-Diffusion-Model)，文章[讲解链接](https://blog.csdn.net/qq_43226466/article/details/132288853)。