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

Object detection results on COCO.

| method  | backbone  | #params(M)  | FLOPs(G)  | AP    | AP</sup>s<sup>  | AP</sup>m<sup>  | AP</sup>l<sup>  |
| ----    | ----      | ----        | ----      | ----  | --------        | -----           | -----           |
|RetinaNet| Res-50 <br> ConTNet-M|  32.0 <br> 27.0  | 235.6 <br> 217.2  | 36.5 <br> **37.9**  | 20.4 <br> **23.0** | 40.3 <br> **40.6** | 48.1 <br> **50.4** |
| FCOS    | Res-50 <br> ConTNet-M|  32.2 <br> 27.2  | 242.9 <br> 228.4  | 36.6 <br> **39.3**  | 21.0 <br> **23.1** | 40.6 <br> **43.1** | 47.0 <br> **51.9** |
| faster rcnn | Res-50 <br> ConTNet-M|  41.5 <br> 36.6  | 241.0 <br> 225.6  | 37.4 <br> **40.0**  | 21.2 <br> **25.4** | 41.0 <br> **43.0** | 48.1 <br> **52.0** |
  
Instance segmentation results on Cityscapes based on Mask-RCNN.
| backbone  | AP<sup>bb</sup> | AP<sub>s</sub><sup>bb</sup> | AP<sub>m</sub><sup>bb</sup> | AP<sub>l</sub><sup>bb</sup> | AP<sup>mk</sup> | AP<sub>s</sub><sup>mk</sup> | AP<sub>m</sub><sup>mk</sup> | AP<sub>l</sub><sup>mk</sup> |
| ----      | ----    | ----  | ----  | ----  | ----  | ----  | ----  | ----  |
| Res-50 <br> ConT-M  | 38.2 <br> **40.5**  | 21.9 <br> **25.1**  | 40.9 <br> **44.4** | 49.5 <br> **52.7** | 34.7 <br> **38.1** | 18.3 <br> **20.9** | 37.4 <br> **41.0** | 47.2 <br> **50.3** |

Semantic segmentation results on cityscapes.
| model | mIOU  |
| ----- | ----  |
|PSP-Res50| 77.12 |
|PSP-ConTM| **78.28** |
