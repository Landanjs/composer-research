# Copyright 2021 MosaicML. All Rights Reserved.
"""DeepLabV3 model extending :class:`.ComposerClassifier`."""

import textwrap
import warnings
from typing import Any, Callable, List, Optional, Union

import monai
import torch
import torch.distributed as dist
import torch.nn.functional as F
from composer.core.types import BatchPair
from composer.loss import soft_cross_entropy
from composer.metrics import CrossEntropy, MIoU
from composer.models.base import ComposerModel
from composer.models.initializers import Initializer
from monai.networks.utils import one_hot
from monai.utils import LossReduction
from torch.nn.modules.loss import _Loss
from torchmetrics import MetricCollection
from torchvision.models import _utils, resnet

__all__ = ["deeplabv3_builder", "ComposerDeepLabV3"]


class SimpleSegmentationModel(torch.nn.Module):

    def __init__(self, backbone, classifier):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        logits = self.classifier(tuple(features.values()))
        logits = F.interpolate(logits,
                               size=input_shape,
                               mode="bilinear",
                               align_corners=False)
        return logits


def deeplabv3_builder(num_classes: int,
                      backbone_arch: str = 'resnet101',
                      is_backbone_pretrained: bool = True,
                      backbone_url: str = '',
                      sync_bn: bool = True,
                      use_plus: bool = True,
                      initializers: List[Initializer] = [],
                      loss: str = 'ce'):
    """Helper function to build a torchvision DeepLabV3 model with a 3x3 convolution layer and dropout removed.

    Args:
        num_classes (int): Number of classes in the segmentation task.
        backbone_arch (str, optional): The architecture to use for the backbone. Must be either [``'resnet50'``, ``'resnet101'``].
            Default: ``'resnet101'``.
        is_backbone_pretrained (bool, optional): If ``True``, use pretrained weights for the backbone. Default: ``True``.
        backbone_url (str, optional): Url used to download model weights. If empty, the PyTorch url will be used.
            Default: ``''``.
        sync_bn (bool, optional): If ``True``, replace all BatchNorm layers with SyncBatchNorm layers. Default: ``True``.
        use_plus (bool, optional): If ``True``, use DeepLabv3+ head instead of DeepLabv3. Default: ``True``.
        initializers (List[Initializer], optional): Initializers for the model. ``[]`` for no initialization. Default: ``[]``.

    Returns:
        deeplabv3: A DeepLabV3 :class:`torch.nn.Module`.

    Example:

    .. code-block:: python

        from composer.models.deeplabv3.deeplabv3 import deeplabv3_builder

        pytorch_model = deeplabv3_builder(num_classes=150, backbone_arch='resnet101', is_backbone_pretrained=False)
    """

    # check that the specified architecture is in the resnet module
    if not hasattr(resnet, backbone_arch):
        raise ValueError(
            f"backbone_arch must be part of the torchvision resnet module, got value: {backbone_arch}"
        )

    # change the model weight url if specified
    if backbone_url:
        resnet.model_urls[backbone_arch] = backbone_url
    backbone = getattr(resnet, backbone_arch)(
        pretrained=is_backbone_pretrained,
        replace_stride_with_dilation=[False, True, True])

    # specify which layers to extract activations from
    return_layers = {
        'layer1': 'layer1',
        'layer4': 'layer4'
    } if use_plus else {
        'layer4': 'layer4'
    }
    backbone = _utils.IntermediateLayerGetter(backbone,
                                              return_layers=return_layers)

    try:
        from mmseg.models import ASPPHead  # type: ignore
        from mmseg.models import DepthwiseSeparableASPPHead
    except ImportError as e:
        raise ImportError(
            textwrap.dedent("""\
            Either mmcv or mmsegmentation is not installed. To install mmcv, please run pip install mmcv-full==1.4.4 -f
             https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html where {cu_version} and
             {torch_version} refer to your CUDA and PyTorch versions, respectively. To install mmsegmentation, please
             run pip install mmsegmentation==0.22.0 on command-line.""")
        ) from e
    norm_type = 'SyncBN' if sync_bn else 'BN'
    norm_cfg = dict(type=norm_type, requires_grad=True)
    if use_plus:
        # mmseg config:
        # https://github.com/open-mmlab/mmsegmentation/blob/master/configs/_base_/models/deeplabv3plus_r50-d8.py
        head = DepthwiseSeparableASPPHead(in_channels=2048,
                                          in_index=-1,
                                          channels=512,
                                          dilations=(1, 12, 24, 36),
                                          c1_in_channels=256,
                                          c1_channels=48,
                                          dropout_ratio=0.1,
                                          num_classes=num_classes,
                                          norm_cfg=norm_cfg,
                                          align_corners=False)
    else:
        # mmseg config:
        # https://github.com/open-mmlab/mmsegmentation/blob/master/configs/_base_/models/deeplabv3_r50-d8.py
        head = ASPPHead(in_channels=2048,
                        in_index=-1,
                        channels=512,
                        dilations=(1, 12, 24, 36),
                        dropout_ratio=0.1,
                        num_classes=num_classes,
                        norm_cfg=norm_cfg,
                        align_corners=False)

    model = SimpleSegmentationModel(backbone, head)

    if initializers:
        for initializer in initializers:
            initializer_fn = Initializer(initializer).get_initializer()

            # Only apply initialization to classifier head if pre-trained weights are used
            if is_backbone_pretrained:
                model.classifier.apply(initializer_fn)
            else:
                model.apply(initializer_fn)
    if loss == 'bce':
        torch.nn.init.constant_(model.classifier.conv_seg.bias, -2.17609125906)
    #model.classifier.conv_seg.bias = torch.ones_like(model.classifier.conv_seg.bias) * (-2.17609125906)
    if sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    return model


class ComposerDeepLabV3(ComposerModel):
    """DeepLabV3 model extending :class:`.ComposerClassifier`. Logs Mean Intersection over Union (MIoU) and Cross
    Entropy during training and validation.

    From `Rethinking Atrous Convolution for Semantic Image Segmentation <https://arxiv.org/abs/1706.05587>`_ (Chen et al, 2017).

    Args:
        num_classes (int): Number of classes in the segmentation task.
        backbone_arch (str, optional): The architecture to use for the backbone. Must be either [``'resnet50'``, ``'resnet101'``].
            Default: ``'resnet101'``.
        is_backbone_pretrained (bool, optional): If ``True``, use pretrained weights for the backbone. Default: ``True``.
        backbone_url (str, optional): Url used to download model weights. If empty, the PyTorch url will be used.
            Default: ``''``.
        sync_bn (bool, optional): If ``True``, replace all BatchNorm layers with SyncBatchNorm layers. Default: ``True``.
        use_plus (bool, optional): If ``True``, use DeepLabv3+ head instead of DeepLabv3. Default: ``True``.
        initializers (List[Initializer], optional): Initializers for the model. ``[]`` for no initialization. Default: ``[]``.


    Example:

    .. code-block:: python

        from composer.models import ComposerDeepLabV3

        model = ComposerDeepLabV3(num_classes=150, backbone_arch='resnet101', is_backbone_pretrained=False)
    """

    def __init__(self,
                 num_classes: int,
                 backbone_arch: str = 'resnet101',
                 is_backbone_pretrained: bool = True,
                 backbone_url: str = '',
                 sync_bn: bool = True,
                 use_plus: bool = True,
                 initializers: List[Initializer] = [],
                 pixelwise_loss: str = 'ce',
                 sigmoid=False,
                 softmax=False,
                 jaccard=False,
                 batch=False,
                 squared_pred=False,
                 no_class_weight=1.0,
                 gamma=0.0,
                 focal_weight=None,
                 lambda_dice=0.0,
                 lambda_focal=1.0):

        super().__init__()
        self.num_classes = num_classes
        self.pixelwise_loss = pixelwise_loss
        self.no_class_weight = no_class_weight
        self.model = deeplabv3_builder(
            backbone_arch=backbone_arch,
            is_backbone_pretrained=is_backbone_pretrained,
            backbone_url=backbone_url,
            use_plus=use_plus,
            num_classes=num_classes,
            sync_bn=sync_bn,
            initializers=initializers,
            loss=self.pixelwise_loss)

        # Metrics
        self.train_miou = MIoU(self.num_classes, ignore_index=-1)
        self.train_ce = CrossEntropy(ignore_index=-1)
        self.val_miou = MIoU(self.num_classes, ignore_index=-1)
        self.val_ce = CrossEntropy(ignore_index=-1)
        self.lambda_dice = lambda_dice
        self.lambda_focal = lambda_focal
        self.gamma = gamma
        self.dice_loss = monai.losses.DiceLoss(include_background=True,
                                               to_onehot_y=False,
                                               sigmoid=sigmoid,
                                               softmax=softmax,
                                               jaccard=jaccard,
                                               batch=False,
                                               squared_pred=squared_pred,
                                               reduction='none')
        self.focal_loss = monai.losses.FocalLoss(include_background=True,
                                                 to_onehot_y=True,
                                                 gamma=gamma,
                                                 weight=focal_weight,
                                                 reduction="none")

    def forward(self, batch: BatchPair):
        x = batch[0]
        logits = self.model(x)
        return logits

    def loss(self, outputs: Any, batch: BatchPair):
        target = batch[1]
        loss = 0
        if self.lambda_dice:
            one_hot_targets = monai.networks.utils.one_hot(
                (target + 1).unsqueeze(1), num_classes=(outputs.shape[1] + 1))
            dice_loss = self.dice_loss(outputs, one_hot_targets[:, 1:]).view(
                outputs.shape[0], -1)
            dice_loss = dice_loss.pow(1 / self.gamma)
            class_counts = one_hot_targets[:, 1:].sum(dim=[2, 3]) # B x C
            present_class_mask = (class_counts != 0) # B x C

            # Start present classes with a weight of 1
            weights = torch.zeros_like(dice_loss) # B x C
            weights[present_class_mask] = 1

            epsilon = 1e-5

            # Get the total number of pixels for each class across devices
            total_class_counts = class_counts.sum(dim=0, keepdim=True) # 1 x C
            dist.all_reduce(total_class_counts)

            # Scale weight of each object by the its proportion of total class area
            weights *= class_counts / (total_class_counts + epsilon)

            # Weights by the number of classes in each sample
            num_classes_in_batch = present_class_mask.float().sum(dim=1, keepdim=True) # B x 1
            weights /= (num_classes_in_batch + epsilon)

            #print(weights, num_classes_in_batch)

            loss += (dice_loss * weights).sum(dim=1).mean() * self.lambda_dice
        if self.lambda_focal:
            if self.pixelwise_loss == 'ce':
                ce_loss = soft_cross_entropy(outputs, target, ignore_index=-1)
                if False:
                    confidences = F.softmax(outputs, dim=1).gather(
                        dim=1, index=target.unsqueeze(1)).squeeze(1)

                    loss += ((1 - confidences).pow(self.gamma) *
                             ce_loss)[target != 0].mean() * self.lambda_focal
                else:
                    loss += ce_loss * self.lambda_focal
            elif self.pixelwise_loss == 'bce':
                focal_loss = self.focal_loss(outputs, target.unsqueeze(1))
                focal_loss = focal_loss.sum(1)
                loss += focal_loss[target != 0].mean() * self.lambda_focal

        return loss

    def metrics(self, train: bool = False):
        metric_list = [self.train_miou, self.train_ce
                       ] if train else [self.val_miou, self.val_ce]
        return MetricCollection(metric_list)

    def validate(self, batch: BatchPair):
        assert self.training is False, "For validation, model must be in eval mode"
        target = batch[1]
        logits = self.forward(batch)
        return logits, target


class DiceLoss(_Loss):
    """
    Compute average Dice loss between two tensors. It can support both multi-classes and multi-labels tasks.
    The data `input` (BNHW[D] where N is number of classes) is compared with ground truth `target` (BNHW[D]).
    Note that axis N of `input` is expected to be logits or probabilities for each class, if passing logits as input,
    must set `sigmoid=True` or `softmax=True`, or specifying `other_act`. And the same axis of `target`
    can be 1 or N (one-hot format).
    The `smooth_nr` and `smooth_dr` parameters are values added to the intersection and union components of
    the inter-over-union calculation to smooth results respectively, these values should be small.
    The original paper: Milletari, F. et. al. (2016) V-Net: Fully Convolutional Neural Networks forVolumetric
    Medical Image Segmentation, 3DV, 2016.
    """

    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Optional[Callable] = None,
        squared_pred: bool = False,
        jaccard: bool = False,
        reduction: Union[LossReduction, str] = LossReduction.MEAN,
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
    ) -> None:
        """
        Args:
            include_background: if False, channel index 0 (background category) is excluded from the calculation.
                if the non-background segmentations are small compared to the total image size they can get overwhelmed
                by the signal from the background so excluding it in such cases helps convergence.
            to_onehot_y: whether to convert `y` into the one-hot format. Defaults to False.
            sigmoid: if True, apply a sigmoid function to the prediction.
            softmax: if True, apply a softmax function to the prediction.
            other_act: if don't want to use `sigmoid` or `softmax`, use other callable function to execute
                other activation layers, Defaults to ``None``. for example:
                `other_act = torch.tanh`.
            squared_pred: use squared versions of targets and predictions in the denominator or not.
            jaccard: compute Jaccard Index (soft IoU) instead of dice or not.
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.
                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.
            smooth_nr: a small constant added to the numerator to avoid zero.
            smooth_dr: a small constant added to the denominator to avoid nan.
            batch: whether to sum the intersection and union areas over the batch dimension before the dividing.
                Defaults to False, a Dice loss value is computed independently from each item in the batch
                before any `reduction`.
        Raises:
            TypeError: When ``other_act`` is not an ``Optional[Callable]``.
            ValueError: When more than 1 of [``sigmoid=True``, ``softmax=True``, ``other_act is not None``].
                Incompatible values.
        """
        super().__init__(reduction=LossReduction(reduction).value)
        if other_act is not None and not callable(other_act):
            raise TypeError(
                f"other_act must be None or callable but is {type(other_act).__name__}."
            )
        if int(sigmoid) + int(softmax) + int(other_act is not None) > 1:
            raise ValueError(
                "Incompatible values: more than 1 of [sigmoid=True, softmax=True, other_act is not None]."
            )
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.other_act = other_act
        self.squared_pred = squared_pred
        self.jaccard = jaccard
        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)
        self.batch = batch

    def forward(self, input: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD], where N is the number of classes.
            target: the shape should be BNH[WD] or B1H[WD], where N is the number of classes.
        Raises:
            AssertionError: When input and target (after one hot transform if set)
                have different shapes.
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].
        Example:
            >>> from monai.losses.dice import *  # NOQA
            >>> import torch
            >>> from monai.losses.dice import DiceLoss
            >>> B, C, H, W = 7, 5, 3, 2
            >>> input = torch.rand(B, C, H, W)
            >>> target_idx = torch.randint(low=0, high=C - 1, size=(B, H, W)).long()
            >>> target = one_hot(target_idx[:, None, ...], num_classes=C)
            >>> self = DiceLoss(reduction='none')
            >>> loss = self(input, target)
            >>> assert np.broadcast_shapes(loss.shape, input.shape) == input.shape
        """
        if self.sigmoid:
            input = torch.sigmoid(input)

        n_pred_ch = input.shape[1]
        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn(
                    "single channel prediction, `softmax=True` ignored.")
            else:
                input = torch.softmax(input, 1)

        if self.other_act is not None:
            input = self.other_act(input)

        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn(
                    "single channel prediction, `to_onehot_y=True` ignored.")
            else:
                target = one_hot(target, num_classes=n_pred_ch)

        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn(
                    "single channel prediction, `include_background=False` ignored."
                )
            else:
                # if skipping background, removing first channel
                target = target[:, 1:]
                input = input[:, 1:]

        if target.shape != input.shape:
            raise AssertionError(
                f"ground truth has different shape ({target.shape}) from input ({input.shape})"
            )

        # reducing only spatial dimensions (not batch nor channels)
        reduce_axis: List[int] = torch.arange(2, len(input.shape)).tolist()
        if self.batch:
            # reducing spatial dimensions and batch
            reduce_axis = [0] + reduce_axis

        intersection = torch.sum(target * input, dim=reduce_axis)

        if self.squared_pred:
            target = torch.pow(target, 2)
            input = torch.pow(input, 2)

        ground_o = torch.sum(target, dim=reduce_axis)
        pred_o = torch.sum(input, dim=reduce_axis)
        dist.all_reduce(ground_o)
        dist.all_reduce(pred_o)

        denominator = ground_o + pred_o

        if self.jaccard:
            denominator = 2.0 * (denominator - intersection)

        if self.batch:
            dist.all_reduce(intersection)
            #dist.all_reduce(denominator)

        f: torch.Tensor = 1.0 - (2.0 * intersection + self.smooth_nr) / (
            denominator + self.smooth_dr)

        if self.reduction == LossReduction.MEAN.value:
            f = torch.mean(f)  # the batch and channel average
        elif self.reduction == LossReduction.SUM.value:
            f = torch.sum(f)  # sum over the batch and channel dims
        elif self.reduction == LossReduction.NONE.value:
            # If we are not computing voxelwise loss components at least
            # make sure a none reduction maintains a broadcastable shape
            broadcast_shape = list(f.shape[0:2]) + [1] * (len(input.shape) - 2)
            f = f.view(broadcast_shape)
        else:
            raise ValueError(
                f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].'
            )

        return f, ground_o
