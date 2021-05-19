import torch.nn as nn
import torch.nn.functional as F
import torch

from einops.layers.torch import Rearrange
from einops import rearrange

import numpy as np

from typing import Any, List
import math
import warnings
from collections import OrderedDict

__all__ = ['ConTBlock', 'ConTNet']


r""" The following trunc_normal method is pasted from timm https://github.com/rwightman/pytorch-image-models/tree/master/timm      
"""
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

def fixed_padding(inputs, kernel_size, dilation):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs

class ConvBN(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1, bn=True):
        padding = (kernel_size - 1) // 2
        if bn:
            super(ConvBN, self).__init__(OrderedDict([
                ('conv', nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                                padding=padding, groups=groups, bias=False)),
                ('bn', nn.BatchNorm2d(out_planes))
            ]))
        else:
            super(ConvBN, self).__init__(OrderedDict([
                ('conv', nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                                padding=padding, groups=groups, bias=False)),
            ]))

class MHSA(nn.Module):
    r"""
    Build a Multi-Head Self-Attention:
        - https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
        - https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    """
    def __init__(self,
                 planes,
                 head_num,
                 dropout,
                 patch_size,
                 qkv_bias,
                 relative):
        super(MHSA, self).__init__()
        self.head_num = head_num 
        head_dim = planes // head_num
        self.qkv = nn.Linear(planes, 3*planes, bias=qkv_bias)
        self.relative = relative
        self.patch_size = patch_size
        self.scale = head_dim ** -0.5
        
        if self.relative:
            # print('### relative position embedding ###')
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * patch_size - 1) * (2 * patch_size - 1), head_num))  
            coords_w = coords_h = torch.arange(patch_size)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  
            coords_flatten = torch.flatten(coords, 1)  
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  
            relative_coords[:, :, 0] += patch_size - 1 
            relative_coords[:, :, 1] += patch_size - 1
            relative_coords[:, :, 0] *= 2 * patch_size - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)
            trunc_normal_(self.relative_position_bias_table, std=.02)

        self.attn_drop = nn.Dropout(p=dropout)
        self.proj = nn.Linear(planes, planes)
        self.proj_drop = nn.Dropout(p=dropout)

    def forward(self, x):
        B, N, C, H = *x.shape, self.head_num
        # print(x.shape)
        qkv = self.qkv(x).reshape(B, N, 3, H, C // H).permute(2, 0, 3, 1, 4) # x: (3, B, H, N, C//H)
        q, k, v = qkv[0], qkv[1], qkv[2]  # x: (B, H, N, C//N) 

        q = q * self.scale 
        attn = (q @ k.transpose(-2, -1)) # attn: (B, H, N, N)

        if self.relative:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.patch_size ** 2, self.patch_size ** 2, -1)  
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous() 
            attn = attn + relative_position_bias.unsqueeze(0)

        attn = attn.softmax(dim=-1) 
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C) 
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class MLP(nn.Module):
    r"""
    Build a Multi-Layer Perceptron
        - https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    """
    def __init__(self,
                 planes,
                 mlp_dim,
                 dropout):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(planes, mlp_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, planes)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x
        

class STE(nn.Module):
    r"""
    Build a Standard Transformer Encoder(STE)
    input: Tensor (b, c, h, w)
    output: Tensor (b, c, h, w)
    """
    def __init__(self,
                 planes: int,
                 mlp_dim: int,
                 head_num: int,
                 dropout: float,
                 patch_size: int,
                 relative: bool,
                 qkv_bias: bool,
                 pre_norm: bool,
                 **kwargs):
        super(STE, self).__init__()
        self.patch_size = patch_size
        self.pre_norm = pre_norm
        self.relative = relative

        self.flatten = nn.Sequential(
                Rearrange('b c pnh pnw psh psw -> (b pnh pnw) psh psw c'),
            )
        if not relative:
            self.pe = nn.ParameterList(
                    [nn.Parameter(torch.zeros(1, patch_size, 1, planes//2)), nn.Parameter(torch.zeros(1, 1, patch_size, planes//2))]
                )
        self.attn = MHSA(planes, head_num, dropout, patch_size, qkv_bias=qkv_bias, relative=relative)
        self.mlp = MLP(planes, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(planes)
        self.norm2 = nn.LayerNorm(planes)

    def forward(self, x):
        bs, c, h, w = x.shape 
        patch_size = self.patch_size
        patch_num_h, patch_num_w = h // patch_size, w // patch_size
    
        x = (
            x.unfold(2, self.patch_size, self.patch_size)
                .unfold(3, self.patch_size, self.patch_size)
        ) # x: (b, c, patch_num, patch_num, patch_size, patch_size)
        x = self.flatten(x) # x: (b, patch_size, patch_size, c)
        ### add 2d position embedding ###
        if not self.relative:
            x_h, x_w = x.split(c // 2, dim=3)
            x = torch.cat((x_h + self.pe[0], x_w + self.pe[1]), dim=3) # x: (b, patch_size, patch_size, c)
        
        x = rearrange(x, 'b psh psw c -> b (psh psw) c')

        if self.pre_norm:
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
        else:
            x = self.norm1(x + self.attn(x))
            x = self.norm2(x + self.mlp(x))

        x = rearrange(x, '(b pnh pnw) (psh psw) c -> b c (pnh psh) (pnw psw)', pnh=patch_num_h, pnw=patch_num_w, psh=patch_size, psw=patch_size) 

        return x

class ConTBlock(nn.Module):
    r"""
    Build a ConTBlock
    """
    def __init__(self,
                 planes: int,
                 out_planes: int,
                 mlp_dim: int,
                 head_num: int,
                 dropout: float,
                 patch_size: List[int],
                 downsample: nn.Module = None, 
                 stride: int=1,
                 last_dropout: float=0.3,
                 **kwargs):
        super(ConTBlock, self).__init__()
        self.downsample = downsample
        self.identity = nn.Identity()
        self.dropout = nn.Identity()
        
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.ste1 = STE(planes=planes, mlp_dim=mlp_dim, head_num=head_num, dropout=dropout, patch_size=patch_size[0], **kwargs)
        self.ste2 = STE(planes=planes, mlp_dim=mlp_dim, head_num=head_num, dropout=dropout, patch_size=patch_size[1], **kwargs)
        
        if stride == 1 and downsample is not None: 
            self.dropout = nn.Dropout(p=last_dropout)
            kernel_size = 1
        else:
            kernel_size = 3

        self.out_conv = ConvBN(planes, out_planes, kernel_size, stride, bn=False)

    def forward(self, x):
        x_preact = self.relu(self.bn(x))
        identity = self.identity(x)

        if self.downsample is not None:
            identity = self.downsample(x_preact)

        residual = self.ste1(x_preact)
        residual = self.ste2(residual)
        residual = self.out_conv(residual)
        out = self.dropout(residual+identity)

        return out

class ConTNet(nn.Module):
    r"""
    Build a ConTNet backbone
    """
    def __init__(self, 
                 block: nn.Module,
                 layers: List[int],
                 mlp_dim: List[int],
                 head_num: List[int],
                 dropout: List[int],
                 in_channels: int=3,
                 inplanes: int=64,
                 num_classes: int=1000,
                 init_weights: bool=True,
                 first_embedding: bool=False,
                 tweak_C: bool=False,
                 **kwargs):
        r"""
        Args:
            block: ConT Block
            layers: number of blocks at each layer
            mlp_dim: dimension of mlp in each stage
            head_num: number of head in each stage
            dropout: dropout in the last two stage
            relative: if True, relative Position Embedding is used
            groups: nunmber of group at each conv layer in the Network
            depthwise: if True, depthwise convolution is adopted
            in_channels: number of channels of input image
            inplanes: channel of the first convolution layer
            num_classes: number of classes for classification task
                         only useful when `with_classifier` is True
            with_avgpool: if True, an average pooling is added at the end of resnet stage5
            with_classifier: if True, FC layer is registered for classification task
            first_embedding: if True, a conv layer with both stride and kernel of 7 is placed at the top
            tweakC: if true, the first layer of ResNet-C replace the ori layer
        """
    
        super(ConTNet, self).__init__()
        self.inplanes = inplanes
        self.block = block

        # build the top layer
        if tweak_C:
            self.layer0 = nn.Sequential(OrderedDict([
                ('conv_bn1', ConvBN(in_channels, inplanes//2, kernel_size=3, stride=2)),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv_bn2', ConvBN(inplanes//2, inplanes//2, kernel_size=3, stride=1)),
                ('relu2', nn.ReLU(inplace=True)),
                ('conv_bn3', ConvBN(inplanes//2, inplanes, kernel_size=3, stride=1)),
                ('relu3', nn.ReLU(inplace=True)),
                ('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            ]))
        elif first_embedding:
            self.layer0 = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(in_channels, inplanes, kernel_size=4, stride=4)),
                ('norm', nn.LayerNorm())
            ]))
        else:
            self.layer0 = nn.Sequential(OrderedDict([
                ('conv', ConvBN(in_channels, inplanes, kernel_size=7, stride=2, bn=False)),
                # ('relu', nn.ReLU(inplace=True)),
                ('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            ]))

        # build cont layers
        self.cont_layers = []
        self.out_channels = OrderedDict()

        for i in range(len(layers)):
            stride = 2,
            patch_size = [7,14]
            if i == len(layers)-1:
                stride, patch_size[1] = 1, 7 # the last stage does not conduct downsampling
            cont_layer = self._make_layer(inplanes * 2**i, layers[i], stride=stride, mlp_dim=mlp_dim[i], head_num=head_num[i], dropout=dropout[i], patch_size=patch_size, **kwargs)
            layer_name = 'layer{}'.format(i + 1)
            self.add_module(layer_name, cont_layer)
            self.cont_layers.append(layer_name)
            self.out_channels[layer_name] = 2 * inplanes * 2**i

        self.last_out_channels = next(reversed(self.out_channels.values()))
        self.fc = nn.Linear(self.last_out_channels, num_classes) 

        if init_weights:
            self._initialize_weights()

    def _make_layer(self,
                    planes: int,
                    blocks: int,
                    stride: int,
                    mlp_dim: int,
                    head_num: int,
                    dropout: float,
                    patch_size: List[int],
                    use_avgdown: bool=False,
                    **kwargs):
        
        layers = OrderedDict()
        for i in range(0, blocks-1):
            layers[f'{self.block.__name__}{i}'] = self.block(
                planes, planes, mlp_dim, head_num, dropout, patch_size, **kwargs)

        downsample = None
        if stride != 1:
            if use_avgdown:
                downsample = nn.Sequential(OrderedDict([
                    ('avgpool', nn.AvgPool2d(kernel_size=2, stride=2)),
                    ('conv', ConvBN(planes, planes * 2, kernel_size=1, stride=1, bn=False))]))
            else:
                downsample = ConvBN(planes, planes * 2, kernel_size=1, 
                                        stride=2, bn=False)
        else:
            downsample = ConvBN(planes, planes * 2, kernel_size=1, stride=1, bn=False)

        layers[f'{self.block.__name__}{blocks-1}'] = self.block(
            planes, planes*2, mlp_dim, head_num, dropout, patch_size, downsample, stride, **kwargs) 

        return nn.Sequential(layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    
    def forward(self, x):   
        x = self.layer0(x)  

        for _, layer_name in enumerate(self.cont_layers):
            cont_layer = getattr(self, layer_name)
            x = cont_layer(x)
        
        x = x.mean([2, 3])  
        x = self.fc(x)

        return x

def create_ConTNet_Ti(kwargs):
    return ConTNet(block=ConTBlock, 
                   mlp_dim=[196, 392, 768, 768], 
                   dropout=[0,0,0,0], 
                   inplanes=48, 
                   layers=[1,1,1,1], 
                   last_dropout=0, 
                   **kwargs)

def create_ConTNet_S(kwargs):
    return ConTNet(block=ConTBlock, 
                   mlp_dim=[256, 512, 1024, 1024], 
                   dropout=[0,0,0,0], 
                   inplanes=64, 
                   layers=[1,1,1,1], 
                   last_dropout=0, 
                   **kwargs)

def create_ConTNet_M(kwargs):
    return ConTNet(block=ConTBlock, 
                   mlp_dim=[256, 512, 1024, 1024], 
                   dropout=[0,0,0,0], 
                   inplanes=64, 
                   layers=[2,2,2,2], 
                   last_dropout=0, 
                   **kwargs)

def create_ConTNet_B(kwargs):
    return ConTNet(block=ConTBlock, 
                   mlp_dim=[256, 512, 1024, 1024], 
                   dropout=[0,0,0.1,0.1], 
                   inplanes=64, 
                   layers=[3,4,6,3], 
                   last_dropout=0.2, 
                   **kwargs)

def build_model(arch, use_avgdown, relative, qkv_bias, pre_norm):
    type = arch.split('-')[-1]
    func = f'create_ConTNet_{type}'
    kwargs = dict(use_avgdown=use_avgdown, relative=relative, qkv_bias=qkv_bias, pre_norm=pre_norm)
    return func(kwargs)