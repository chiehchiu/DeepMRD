import torch
import torch.nn as nn
import numpy as np

from ..builder import CLASSIFIERS, build_backbone, build_head, build_neck
from .base import BaseClassifier
from ..backbones.acsconv.converters import ConvertModel

from mmcls.core import auto_fp16


@CLASSIFIERS.register_module()
class FImageClassifier(BaseClassifier):

    def __init__(self, backbone, converter=None, neck=None, fuse_neck=None, head=None, pretrained=None):
        super(FImageClassifier, self).__init__()
        self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        if fuse_neck is not None:
            self.fuse_neck = build_neck(fuse_neck)

        if head is not None:
            self.head = build_head(head)

        self.init_weights(pretrained=pretrained)

        if converter is not None:
            # convert 2d model to 3d after init_weights()
            # optional: [ACS, I3D, 2.5D]
            print('Convert model with \'{}\' function'.format(converter))
            self.backbone = ConvertModel(self.backbone, converter).model
            print(self.backbone)

    @property
    def with_fneck(self):
        return hasattr(self, 'fuse_neck') and self.fuse_neck is not None

    def init_weights(self, pretrained=None):
        super(FImageClassifier, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_fneck:
            if isinstance(self.fuse_neck, nn.Sequential):
                for m in self.fuse_neck:
                    m.init_weights()
            else:
                self.fuse_neck.init_weights()
        if self.with_head:
            self.head.init_weights()

    def extract_feat(self, img):
        """Directly extract features from the backbone + neck
        """
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    @auto_fp16(apply_to=('img', ))
    def forward_train(self, img, gt_label, aux_feat, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            gt_label (Tensor): of shape (N, 1) encoding the ground-truth label
                of input images.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(img)
        x = self.fuse_neck(x, aux_feat)
        losses = dict()
        loss = self.head.forward_train(x, gt_label)
        losses.update(loss)

        return losses

    @auto_fp16(apply_to=('img', ))
    def simple_test(self, img, aux_feat):
        """Test without augmentation."""
        x = self.extract_feat(img)
        x = self.fuse_neck(x, aux_feat)
        return self.head.simple_test(x)

