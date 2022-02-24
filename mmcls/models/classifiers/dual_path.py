import torch
import torch.nn as nn
import numpy as np

from ..builder import CLASSIFIERS, build_backbone, build_head, build_neck
from .base import BaseClassifier
from ..backbones.acsconv.converters import ConvertModel

from mmcls.core import auto_fp16


@CLASSIFIERS.register_module()
class DPImageClassifier(BaseClassifier):

    def __init__(self, backbone, aux_backbone=None, converter=None, neck=None, head=None, pretrained=None, aux_pretrained=None, aux_neck=None):
        super(DPImageClassifier, self).__init__()
        self.backbone = build_backbone(backbone)
        self.aux_backbone = build_backbone(aux_backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        if aux_neck is not None:
            self.aux_neck = build_neck(aux_neck)

        if head is not None:
            self.head = build_head(head)

        self.init_weights(pretrained=pretrained)
        self.aux_init_weights(pretrained=aux_pretrained)

        if converter is not None:
            # convert 2d model to 3d after init_weights()
            # optional: [ACS, I3D, 2.5D]
            print('Convert model with \'{}\' function'.format(converter))
            self.backbone = ConvertModel(self.backbone, converter).model
            print(self.backbone)

    @property
    def with_aneck(self):
        return hasattr(self, 'aux_neck') and self.aux_neck is not None

    def init_weights(self, pretrained=None):
        super(DPImageClassifier, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_aneck:
            if isinstance(self.aux_neck, nn.Sequential):
                for m in self.aux_neck:
                    m.init_weights()
            else:
                self.aux_neck.init_weights()

        if self.with_head:
            self.head.init_weights()

    def aux_init_weights(self, pretrained=None):
        super(DPImageClassifier, self).init_weights(pretrained)
        self.aux_backbone.init_weights(pretrained=pretrained)
        # aux_neck and aux_head not supported yet

    def extract_feat(self, img):
        """Directly extract features from the backbone + neck
        """
        x = self.backbone(img)
        aux_x = self.aux_backbone(img)
        if self.with_neck:
            x = self.neck(x)
            aux_x = self.neck(aux_x)
        x = self.aux_neck(x, aux_x)
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
        x = self.extract_feat(img)
        losses = dict()
        loss = self.head.forward_train(x, gt_label)
        losses.update(loss)

        return losses

    @auto_fp16(apply_to=('img', ))
    def simple_test(self, img):
        """Test without augmentation."""
        x = self.extract_feat(img)
        return self.head.simple_test(x)

    # LJ add multicrop_test
    @auto_fp16(apply_to=('img', ))
    def multicrop_test(self, img):
        """Test with multicrop."""
        b, crop_num = img.shape[:2]
        # avg
        pred_ = np.zeros((b, self.head.num_classes))
        # max
        # pred_ = []
        for i in range(crop_num):
            img_ = img[:, i, ...].unsqueeze(1)
            x = self.extract_feat(img_)
            pred = self.head.simple_test(x)
            # avg
            pred_ += np.array(pred)
            # max
            # pred_.append(pred)

        # max
        # pred_ = np.stack(pred_)
        # pred_ = np.max(pred_, axis=0)

        # avg
        pred_ = pred_ / crop_num
        pred_ = pred_.tolist()

        pred_ = [np.array(p, dtype=np.float16) for p in pred_]

        return pred_


    # LJ add multipatch_test
    @auto_fp16(apply_to=('img', ))
    def multipatch_test(self, img):
        """Test with multipatch."""
        b, patch_num = img.shape[:2]
        # Noisy-or method
        # pred_ = np.ones((b, self.head.num_classes))
        # max method
        pred_ = []
        for i in range(patch_num):
            img_ = img[:, i, ...].unsqueeze(1)
            x = self.extract_feat(img_)
            pred = self.head.simple_test(x)
            # Noisy-or method
            # pred_ *= (1.0 - np.array(pred))
            # max method
            pred_.append(pred)

        # Noisy-or method
        # pred_[:, 0] = pred_[:, 1] + 1e-6
        # pred_[:, 1] = 1 - pred_[:, 1] + 1e-6
        # pred_ = pred_.tolist()
        # pred_ = [np.array(p, dtype=np.float16) for p in pred_]

        # max method
        pred_ = np.stack(pred_)
        pred_ = np.max(pred_, axis=0)
        pred_[:, 0] = 1 - pred_[:, 1] + 1e-6
        pred_ = [np.array(p, dtype=np.float16) for p in pred_]
        # pdb.set_trace()

        return pred_
