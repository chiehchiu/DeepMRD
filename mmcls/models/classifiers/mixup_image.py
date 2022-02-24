import torch
import torch.nn as nn
import numpy as np

from ..builder import CLASSIFIERS, build_backbone, build_head, build_neck
from .base import BaseClassifier
from ..backbones.acsconv.converters import ConvertModel

from mmcls.core import auto_fp16


@CLASSIFIERS.register_module()
class MixupImageClassifier(BaseClassifier):

    def __init__(self, backbone, converter=None, neck=None, head=None, pretrained=None, save_feat=False, alpha=1.0):
        super(MixupImageClassifier, self).__init__()
        self.save_feat = save_feat
        self.backbone = build_backbone(backbone)
        self.alpha = alpha

        if neck is not None:
            self.neck = build_neck(neck)

        if head is not None:
            self.head = build_head(head)

        self.init_weights(pretrained=pretrained)

        if converter is not None:
            # convert 2d model to 3d after init_weights()
            # optional: [ACS, I3D, 2.5D]
            print('Convert model with \'{}\' function'.format(converter))
            self.backbone = ConvertModel(self.backbone, converter).model
            print(self.backbone)

    def init_weights(self, pretrained=None):
        super(MixupImageClassifier, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_head:
            self.head.init_weights()

    def mixup_data(self, x, y, alpha=1.0, use_cuda=True):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        if use_cuda:
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def mixup_extract_feat(self, img, gt_label):
        """Directly extract features from the backbone + neck
        """
        mixed_img, gt_a, gt_b, lam = self.mixup_data(img, gt_label, self.alpha)

        x = self.backbone(mixed_img)
        if self.with_neck:
            x = self.neck(x)
        return x, gt_a, gt_b, lam

    def extract_feat(self, img):
        """Directly extract features from the backbone + neck
        """
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    @auto_fp16(apply_to=('img', ))
    def forward_train(self, img, gt_label, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            gt_label (Tensor): of shape (N, 1) encoding the ground-truth label
                of input images.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x, gt_a, gt_b, lam = self.mixup_extract_feat(img, gt_label)

        losses = dict()
        loss = self.head.mixup_forward_train(x, gt_a, gt_b, lam)
        losses.update(loss)

        return losses

    @auto_fp16(apply_to=('img', ))
    def simple_test(self, img):
        """Test without augmentation."""
        x = self.extract_feat(img)
        if not self.save_feat:
            return self.head.simple_test(x)
        else:
            return list(x.detach().cpu().numpy())

