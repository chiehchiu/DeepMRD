import torch
import torch.nn as nn

from ..builder import NECKS
from mmcls.models.plugins import AsyNonLocal2D

@NECKS.register_module()
class SumFusion(nn.Module):
    """neck layer for feature fusion with element-wise sum
    """

    def __init__(self):
        super(SumFusion, self).__init__()

    def init_weights(self):
        pass

    def forward(self, feat1, feat2):
        if isinstance(feat1, tuple):
            outs = tuple([fea1+fea2 for fea1, fea2 in zip(feat1, feat2)])
            outs = tuple(
                [out.view(x.size(0), -1) for out, x in zip(outs, feat1)])
        elif isinstance(feat1, torch.Tensor):
            outs = feat1 + feat2
            outs = outs.view(feat1.size(0), -1)
        else:
            raise TypeError('neck inputs should be tuple or torch.tensor')
        return outs


@NECKS.register_module()
class NonLocalFusion(nn.Module):
    """neck layer for feature fusion with non-local operation
    """

    def __init__(self,                  
                 in_channels=1,
                 refer_in_channels=1,
                 reduction=1,
                 use_scale=True,
                 conv_cfg=None,
                 norm_cfg=None,
                 mode='embedded_gaussian'):
        super(NonLocalFusion, self).__init__()
        self.asy_att = AsyNonLocal2D(in_channels,
                 refer_in_channels,
                 reduction,
                 use_scale,
                 conv_cfg,
                 norm_cfg,
                 mode)

    def init_weights(self):
        self.asy_att.init_weights()

    def forward(self, feat1, aux_feat):
        feat1 = feat1.view(feat1.size(0), 1, 1, -1)
        aux_feat = aux_feat.view(aux_feat.size(0), 1, 1, -1)
        outs = self.asy_att(feat1, aux_feat)
        outs = outs.view(feat1.size(0), -1)
        return outs

