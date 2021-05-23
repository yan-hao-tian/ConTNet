import torch.nn as nn
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.data import Mixup


def build_criterion(mixup, label_smoothing):
    mixup_fn = None
    if mixup > 0.:
        criterion = SoftTargetCrossEntropy()

        mixup_fn = Mixup(
            mixup_alpha=mixup, cutmix_alpha=1, cutmix_minmax=None,
            prob=1, switch_prob=0.5, mode='batch',
            label_smoothing=label_smoothing, num_classes=1000)

    elif label_smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss()

    return criterion, mixup_fn
