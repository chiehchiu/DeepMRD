import torch
import torch.nn as nn
from mmcv.cnn import constant_init, normal_init, ConvModule

class AsyNonLocal2D(nn.Module):
    """Non-local module for fusing two features
    Args:
        in_channels (int): Channels of the input feature map.
        reduction (int): Channel reduction ratio.
        use_scale (bool): Whether to scale pairwise_weight by 1/inter_channels.
        conv_cfg (dict): The config dict for convolution layers.
            (only applicable to conv_out)
        norm_cfg (dict): The config dict for normalization layers.
            (only applicable to conv_out)
        mode (str): Options are `embedded_gaussian` and `dot_product`.

        self.g is used for the reference input
        self.theta is used for the querry input
        self.phi is also used for the reference input
        cross_attention is conducted by: dot_product(self.theta, self.phi) * self.g
    """

    def __init__(self,
                 in_channels,
                 refer_in_channels,
                 reduction=2,
                 use_scale=True,
                 conv_cfg=None,
                 norm_cfg=None,
                 mode='embedded_gaussian'):
        super(AsyNonLocal2D, self).__init__()
        self.in_channels = in_channels
        self.refer_in_channels = refer_in_channels
        self.reduction = reduction
        self.use_scale = use_scale
        self.inter_channels = in_channels // reduction
        self.mode = mode
        assert mode in ['embedded_gaussian', 'dot_product']

        # g, theta, phi are actually `nn.Conv2d`. Here we use ConvModule for
        # potential usage.
        self.g = ConvModule(
            self.refer_in_channels,
            self.inter_channels,
            kernel_size=1,
            act_cfg=None)
        self.theta = ConvModule(
            self.in_channels,
            self.inter_channels,
            kernel_size=1,
            act_cfg=None)
        self.phi = ConvModule(
            self.refer_in_channels,
            self.inter_channels,
            kernel_size=1,
            act_cfg=None)
        self.conv_out = ConvModule(
            self.inter_channels,
            self.in_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

        self.init_weights()

    def init_weights(self, std=0.01, zeros_init=True):
        for m in [self.g, self.theta, self.phi]:
            normal_init(m.conv, std=std)
        if zeros_init:
            constant_init(self.conv_out.conv, 0)
        else:
            normal_init(self.conv_out.conv, std=std)

    def embedded_gaussian(self, theta_x, phi_x):
        # pairwise_weight: [N, HxW, HxW]
        pairwise_weight = torch.matmul(theta_x, phi_x)
        if self.use_scale:
            # theta_x.shape[-1] is `self.inter_channels`
            pairwise_weight /= theta_x.shape[-1]**0.5
        pairwise_weight = pairwise_weight.softmax(dim=-1)
        return pairwise_weight

    def dot_product(self, theta_x, phi_x):
        # pairwise_weight: [N, HxW, HxW]
        pairwise_weight = torch.matmul(theta_x, phi_x)
        pairwise_weight /= pairwise_weight.shape[-1]
        return pairwise_weight

    def forward(self, querry, reference):
        ##forward by: dot_product(self.theta(q), self.phi(ref)) * self.g(ref)
        rn, _, rh, rw = querry.shape
        qn, _, qh, qw = reference.shape

        # g_x: [N, DxH'xW', C] for reference 
        # reference in N C DH' W'; g(reference) in N C' DH' W';
        g_x = self.g(reference).view(rn, self.inter_channels, -1) # gx in N C' DH'W'
        g_x = g_x.permute(0, 2, 1) #gx in N DH'W' C'

        # theta_x: [N, HxW, C] for querry
        theta_x = self.theta(querry).view(qn, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        # phi_x: [N, C, DH'xW'] for reference
        phi_x = self.phi(reference).view(rn, self.inter_channels, -1) # phi_x in (N C' DH'W')

        pairwise_func = getattr(self, self.mode)
        # pairwise_weight: [N, HxW, DH'xW']
        pairwise_weight = pairwise_func(theta_x, phi_x)

        # y: [N, HxW, C]
        y = torch.matmul(pairwise_weight, g_x)
        # y: [N, C, H, W]
        y = y.permute(0, 2, 1).reshape(rn, self.inter_channels, rh, rw)

        output = querry + self.conv_out(y)

        return output
