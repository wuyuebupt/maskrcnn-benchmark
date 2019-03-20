# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.backbone import resnet
from maskrcnn_benchmark.modeling.poolers import Pooler
from maskrcnn_benchmark.modeling.poolers import PoolerNeighbor
from maskrcnn_benchmark.modeling.poolers import PoolerNeighborMap
from maskrcnn_benchmark.modeling.make_layers import group_norm
from maskrcnn_benchmark.modeling.make_layers import make_fc

from .attention import NONLocalBlock2D
from .attention import NONLocalBlock2D_Group
from .attention import ListModule
from maskrcnn_benchmark.layers import Conv2d



@registry.ROI_BOX_FEATURE_EXTRACTORS.register("ResNet50Conv5ROIFeatureExtractorNeighbor")
class ResNet50Conv5ROIFeatureExtractorNeighbor(nn.Module):
    def __init__(self, config):
        super(ResNet50Conv5ROIFeatureExtractorNeighbor, self).__init__()

        neighbor_expand = config.MODEL.ROI_BOX_HEAD.NEIGHBOR_EXPAND
        roi_expand = config.MODEL.ROI_BOX_HEAD.ROI_EXPAND
        resolution = config.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = config.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = config.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        # pooler = Pooler(
        pooler = PoolerNeighbor(
            neighbor_expand=neighbor_expand,
            roi_expand=roi_expand,
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )

        stage = resnet.StageSpec(index=4, block_count=3, return_features=False)
        head = resnet.ResNetHead(
            block_module=config.MODEL.RESNETS.TRANS_FUNC,
            stages=(stage,),
            num_groups=config.MODEL.RESNETS.NUM_GROUPS,
            width_per_group=config.MODEL.RESNETS.WIDTH_PER_GROUP,
            stride_in_1x1=config.MODEL.RESNETS.STRIDE_IN_1X1,
            stride_init=None,
            res2_out_channels=config.MODEL.RESNETS.RES2_OUT_CHANNELS,
            dilation=config.MODEL.RESNETS.RES5_DILATION
        )

        self.pooler = pooler
        self.head = head

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = self.head(x)
        return x


@registry.ROI_BOX_FEATURE_EXTRACTORS.register("ResNet50Conv5ROIFeatureExtractor")
class ResNet50Conv5ROIFeatureExtractor(nn.Module):
    def __init__(self, config):
        super(ResNet50Conv5ROIFeatureExtractor, self).__init__()

        resolution = config.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = config.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = config.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )

        stage = resnet.StageSpec(index=4, block_count=3, return_features=False)
        head = resnet.ResNetHead(
            block_module=config.MODEL.RESNETS.TRANS_FUNC,
            stages=(stage,),
            num_groups=config.MODEL.RESNETS.NUM_GROUPS,
            width_per_group=config.MODEL.RESNETS.WIDTH_PER_GROUP,
            stride_in_1x1=config.MODEL.RESNETS.STRIDE_IN_1X1,
            stride_init=None,
            res2_out_channels=config.MODEL.RESNETS.RES2_OUT_CHANNELS,
            dilation=config.MODEL.RESNETS.RES5_DILATION
        )

        self.pooler = pooler
        self.head = head

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = self.head(x)
        return x

class FPNUpChannels(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FPNUpChannels, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        ## top
        self.top = nn.Sequential(
                     nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                     nn.BatchNorm2d(out_channels),
                   )
        ## bottom
        self.bottom = nn.Sequential(
                        nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(in_channels),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                        nn.BatchNorm2d(out_channels)
                      )

        # Conv2d(num_inputs, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
        # nn.BatchNorm2d(out_channels)

    def forward(self, x):
        ## top
        out = self.top(x)
        out0 = self.bottom(x)
        ## bottom
        ## residual
        out1 = out + out0
        out1 = self.relu(out1)

        return out1


# class FPNFFConv(nn.Module):
#     def __init__(self, in_channels):
#         super(FPNFFConv, self).__init()
# 
#         inter_channels = in_channels // 4
#         out_channels = in_channels
# 
#         self.relu = nn.ReLU(inplace=True)
#         ## top
#         self.conv1 = Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=False)
#         self.bn1 = nn.BatchNorm2d(inter_channels)
#         ## bottom
#         self.conv2 = Conv2d(inter_channels, inter_channels, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(inter_channels)
# 
#         self.conv3 = Conv2d(inter_channels, out_channels, kernel_size=1, stride=1, padding=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(out_channels)
# 
#         # Conv2d(num_inputs, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
#         # nn.BatchNorm2d(out_channels)
# 
#     def forward(self, x):
#         identity = x
#         ## bottom
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#         out = self.conv3(out)
#         out = self.bn3(out)
#         ## residual
#         out = out + identity
#         out = self.relu(out)
# 
#         return out
# 









@registry.ROI_BOX_FEATURE_EXTRACTORS.register("FPN2MLPFeatureExtractorNeighbor")
class FPN2MLPFeatureExtractorNeighbor(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg):
        super(FPN2MLPFeatureExtractorNeighbor, self).__init__()

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO

        conv_neighbor_expand = cfg.MODEL.ROI_BOX_HEAD.NEIGHBOR_CONV_EXPAND
        fc_neighbor_expand = cfg.MODEL.ROI_BOX_HEAD.NEIGHBOR_FC_EXPAND
        roi_expand = cfg.MODEL.ROI_BOX_HEAD.ROI_EXPAND

        maplevel_conv = cfg.MODEL.ROI_BOX_HEAD.POOLER_MAP_LEVEL_CONV 
        maplevel_fc = cfg.MODEL.ROI_BOX_HEAD.POOLER_MAP_LEVEL_FC
        masklevel_conv = cfg.MODEL.ROI_BOX_HEAD.POOLER_MASK_CONV 
        masklevel_fc = cfg.MODEL.ROI_BOX_HEAD.POOLER_MASK_FC

        self.pooler_conv = PoolerNeighborMap(
            neighbor_expand=conv_neighbor_expand,
            roi_expand=roi_expand,
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
            maplevel = maplevel_conv,
            masklevel  = masklevel_conv
        )
        self.pooler_fc = PoolerNeighborMap(
            neighbor_expand=fc_neighbor_expand,
            roi_expand=roi_expand,
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
            maplevel = maplevel_fc,
            masklevel = masklevel_fc
        )
        # pooler = Pooler(
        #     output_size=(resolution, resolution),
        #     scales=scales,
        #     sampling_ratio=sampling_ratio,
        # )

        num_inputs = cfg.MODEL.BACKBONE.OUT_CHANNELS
        input_size = cfg.MODEL.BACKBONE.OUT_CHANNELS * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN
        # self.pooler = pooler

        nonlocal_use_bn = cfg.MODEL.ROI_BOX_HEAD.NONLOCAL_USE_BN
        nonlocal_use_relu = cfg.MODEL.ROI_BOX_HEAD.NONLOCAL_USE_RELU
        nonlocal_use_softmax = cfg.MODEL.ROI_BOX_HEAD.NONLOCAL_USE_SOFTMAX
        nonlocal_use_ffconv = cfg.MODEL.ROI_BOX_HEAD.NONLOCAL_USE_FFCONV
        nonlocal_inter_channels = cfg.MODEL.ROI_BOX_HEAD.NONLOCAL_INTER_CHANNELS

        ## add conv and pool like faster rcnn
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=7)
        # out_channels = num_inputs
        out_channels = cfg.MODEL.ROI_BOX_HEAD.NONLOCAL_OUT_CHANNELS

        ## depreciate
        # self.nonlocal_conv = nn.Sequential(
        #      Conv2d(num_inputs, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
        #      nn.BatchNorm2d(out_channels),
        #      nn.ReLU()
        # )
        self.nonlocal_conv = FPNUpChannels(num_inputs, out_channels)

        ## shared non-local
        # if self.nonlocal_use_shared == True:
        shared_num_group = cfg.MODEL.ROI_BOX_HEAD.NONLOCAL_SHARED_NUM_GROUP
        self.shared_num_stack = cfg.MODEL.ROI_BOX_HEAD.NONLOCAL_SHARED_NUM_STACK
        shared_nonlocal = []
        for i in range(self.shared_num_stack):
            shared_nonlocal.append(NONLocalBlock2D_Group(out_channels, num_group=shared_num_group, inter_channels=nonlocal_inter_channels, sub_sample=False, bn_layer=nonlocal_use_bn, relu_layer=nonlocal_use_relu, use_softmax=nonlocal_use_softmax, use_ffconv=nonlocal_use_ffconv))
        self.shared_nonlocal = ListModule(*shared_nonlocal)


        ## seperate group non-local, before fc6 and fc7
        cls_num_group = cfg.MODEL.ROI_BOX_HEAD.NONLOCAL_CLS_NUM_GROUP
        self.cls_num_stack = cfg.MODEL.ROI_BOX_HEAD.NONLOCAL_CLS_NUM_STACK

        reg_num_group = cfg.MODEL.ROI_BOX_HEAD.NONLOCAL_REG_NUM_GROUP
        self.reg_num_stack = cfg.MODEL.ROI_BOX_HEAD.NONLOCAL_REG_NUM_STACK

        nonlocal_use_bn = cfg.MODEL.ROI_BOX_HEAD.NONLOCAL_USE_BN
        nonlocal_use_relu = cfg.MODEL.ROI_BOX_HEAD.NONLOCAL_USE_RELU

        cls_nonlocal = []
        for i in range(self.cls_num_stack):
            cls_nonlocal.append(NONLocalBlock2D_Group(out_channels, num_group=cls_num_group, inter_channels=nonlocal_inter_channels, sub_sample=False, bn_layer=nonlocal_use_bn, relu_layer=nonlocal_use_relu, use_softmax=nonlocal_use_softmax, use_ffconv=nonlocal_use_ffconv))
        self.cls_nonlocal = ListModule(*cls_nonlocal)

        reg_nonlocal = []
        for i in range(self.reg_num_stack):
            reg_nonlocal.append(NONLocalBlock2D_Group(out_channels, num_group=reg_num_group, inter_channels=nonlocal_inter_channels, sub_sample=False, bn_layer=nonlocal_use_bn, relu_layer=nonlocal_use_relu, use_softmax=nonlocal_use_softmax, use_ffconv=nonlocal_use_ffconv))
        self.reg_nonlocal = ListModule(*reg_nonlocal)


        ## mlp 
        self.fc6 = make_fc(input_size, representation_size, use_gn)
        self.fc7 = make_fc(representation_size, representation_size, use_gn)

    def forward(self, x, proposals):
        x_conv = x
        x_fc = x
        x_conv, mask = self.pooler_conv(x_conv, proposals)
        x_fc, mask_fc = self.pooler_fc(x_fc, proposals)
        # x_conv, levels_conv = self.pooler_conv(x_conv, proposals)
        # x_fc, levels_fc = self.pooler_fc(x_fc, proposals)


        identity = x_fc
        ### model Y
        x_conv =  self.nonlocal_conv(x_conv)
        ## shared
        for i in range(self.shared_num_stack):
            x_conv = self.shared_nonlocal[i](x_conv)

        ## seperate
        x_cls = x_conv
        x_reg = x_conv
        for i in range(self.cls_num_stack):
            x_cls = self.cls_nonlocal[i](x_cls)
        for i in range(self.reg_num_stack):
            x_reg = self.reg_nonlocal[i](x_reg)
         
        x_cls = self.avgpool(x_cls)
        x_cls = x_cls.view(x_cls.size(0), -1)
        x_reg = self.avgpool(x_reg)
        x_reg = x_reg.view(x_reg.size(0), -1)

        ### MLP
        identity = identity.view(identity.size(0), -1)

        identity = F.relu(self.fc6(identity))
        identity = F.relu(self.fc7(identity))

        return tuple((x_cls, x_reg, identity, mask, mask_fc))




@registry.ROI_BOX_FEATURE_EXTRACTORS.register("FPN2MLPFeatureExtractor")
class FPN2MLPFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg):
        super(FPN2MLPFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        input_size = cfg.MODEL.BACKBONE.OUT_CHANNELS * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN
        self.pooler = pooler
        self.fc6 = make_fc(input_size, representation_size, use_gn)
        self.fc7 = make_fc(representation_size, representation_size, use_gn)

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x


@registry.ROI_BOX_FEATURE_EXTRACTORS.register("FPNXconv1fcFeatureExtractor")
class FPNXconv1fcFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg):
        super(FPNXconv1fcFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        self.pooler = pooler
        
        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN
        in_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        conv_head_dim = cfg.MODEL.ROI_BOX_HEAD.CONV_HEAD_DIM
        num_stacked_convs = cfg.MODEL.ROI_BOX_HEAD.NUM_STACKED_CONVS
        dilation = cfg.MODEL.ROI_BOX_HEAD.DILATION

        xconvs = []
        for ix in range(num_stacked_convs):
            xconvs.append(
                nn.Conv2d(
                    in_channels,
                    conv_head_dim,
                    kernel_size=3,
                    stride=1,
                    padding=dilation,
                    dilation=dilation,
                    bias=False if use_gn else True
                )
            )
            in_channels = conv_head_dim
            if use_gn:
                xconvs.append(group_norm(in_channels))
            xconvs.append(nn.ReLU(inplace=True))

        self.add_module("xconvs", nn.Sequential(*xconvs))
        for modules in [self.xconvs,]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    if not use_gn:
                        torch.nn.init.constant_(l.bias, 0)

        input_size = conv_head_dim * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        self.fc6 = make_fc(input_size, representation_size, use_gn=False)

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = self.xconvs(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        return x


def make_roi_box_feature_extractor(cfg):
    func = registry.ROI_BOX_FEATURE_EXTRACTORS[
        cfg.MODEL.ROI_BOX_HEAD.FEATURE_EXTRACTOR
    ]
    return func(cfg)
