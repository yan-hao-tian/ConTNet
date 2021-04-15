# ConTNet

## Introduction

**ConTNet** (**Con**vlution-**T**ranformer Network) is proposed mainly in response to the following two issues: (1) ConvNets lack a large receptive field, limiting the performance of ConvNets on downstream tasks. (2) Transformer-based model is not robust enough and requires special training settings or hundreds of millions of images as the pretrain dataset, thereby limiting their adoption. **ConTNet** combines convolution and transformer alternately, which is very robust and can be optimized like ResNet unlike the recently-proposed transformer-based models (e.g., ViT, DeiT) that are sensitive to hyper-parameters and need many tricks when trained from scratch on a midsize dataset (e.g., ImageNet).



## Main Results on ImageNet

|  name   |   resolution  |   acc@1   |   #params(M) |   FLOPs(G)   |   model   |
| ----  |   ----    |   ----    |   ----    |   ----    |   ----    |
|   Res-18  |   224x224 |  71.5     |   11.7    |   1.8 |       |
|   ConT-S  |   224x224 |  **74.9** |   10.1    |   1.5 |       |
|   Res-50  |   224x224 |  77.1     |   25.6    |   4.0 |       |
|   ConT-M  |   224x224 |  **77.6** |   19.2    |   3.1 |       |
|   Res-101 |   224x224 |  **78.2** |   44.5    |   7.6 |       |
|   ConT-B  |   224x224 |   77.9    |   39.6    |   6.4 |       |
|   DeiT-Ti<sup>*</sup>  |   224x224 |  72.2    |   5.7    |   1.3 |       |
|   ConT-Ti<sup>*</sup>  |   224x224 |  **74.9**|   5.8    |   0.8 |       |
|   Res-18<sup>*</sup>  |   224x224 |  73.2     |   11.7    |   1.8 |       |
|   ConT-S<sup>*</sup>  |   224x224 |  **76.5** |   10.1    |   1.5 |       |
|   Res-50<sup>*</sup>  |   224x224 |  78.6     |   25.6    |   4.0 |       |
|   DeiT-S<sup>*</sup>  |   224x224 |  79.8     |   22.1    |   4.6 |       |
|   ConT-M<sup>*</sup>  |   224x224 |  **80.2** |   19.2    |   3.1 |       |
|   Res-101<sup>*</sup> |   224x224 |  80.0     |   44.5    |   7.6 |       |
|   DeiT-B<sup>*</sup>  |   224x224 |  **81.8** |   86.6    |   17.6|       |
|   ConT-B<sup>*</sup>  |   224x224 |  **81.8** |   39.6    |   6.4 |       |

Note: <sup>*</sup> indicates training with strong augmentations.

## Main Results on Downstream Tasks

