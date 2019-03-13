# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from maskrcnn_benchmark.modeling import registry
from torch import nn

from maskrcnn_benchmark.layers import Conv2d
import torch.nn.functional as F
from maskrcnn_benchmark.modeling.make_layers import group_norm
from .attention import NONLocalBlock2D
from .attention import NONLocalBlock2D_Group
from .attention import ListModule



@registry.ROI_BOX_PREDICTOR.register("FastRCNNPredictor")
class FastRCNNPredictor(nn.Module):
    def __init__(self, config, pretrained=None):
        super(FastRCNNPredictor, self).__init__()

        stage_index = 4
        stage2_relative_factor = 2 ** (stage_index - 1)
        res2_out_channels = config.MODEL.RESNETS.RES2_OUT_CHANNELS
        num_inputs = res2_out_channels * stage2_relative_factor

        num_classes = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=7)
        self.cls_score = nn.Linear(num_inputs, num_classes)
        num_bbox_reg_classes = 2 if config.MODEL.CLS_AGNOSTIC_BBOX_REG else num_classes
        ## old bbox_pred
        # self.bbox_pred = nn.Linear(num_inputs, num_bbox_reg_classes * 4)

        nn.init.normal_(self.cls_score.weight, mean=0, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)

        self.bbox_pred = nn.Linear(num_inputs, num_bbox_reg_classes * 4)
        nn.init.normal_(self.bbox_pred.weight, mean=0, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)

        nonlocal_use_bn = config.MODEL.ROI_BOX_HEAD.NONLOCAL_USE_BN
        nonlocal_use_relu = config.MODEL.ROI_BOX_HEAD.NONLOCAL_USE_RELU
        nonlocal_inter_channels = config.MODEL.ROI_BOX_HEAD.NONLOCAL_INTER_CHANNELS
        nonlocal_use_softmax = config.MODEL.ROI_BOX_HEAD.NONLOCAL_USE_SOFTMAX 

        self.nonlocal_use_shared = config.MODEL.ROI_BOX_HEAD.NONLOCAL_USE_SHARED


        ## shared non-local
        if self.nonlocal_use_shared == True:
            shared_num_group = config.MODEL.ROI_BOX_HEAD.NONLOCAL_SHARED_NUM_GROUP
            self.shared_num_stack = config.MODEL.ROI_BOX_HEAD.NONLOCAL_SHARED_NUM_STACK

            shared_nonlocal = []
            for i in range(self.shared_num_stack):
                shared_nonlocal.append(NONLocalBlock2D_Group(num_inputs, num_group=shared_num_group, inter_channels=nonlocal_inter_channels, sub_sample=False, bn_layer=nonlocal_use_bn, relu_layer=nonlocal_use_relu, use_softmax=nonlocal_use_softmax))
            self.shared_nonlocal = ListModule(*shared_nonlocal)

        else:
            ## seperate group non-local, before average pooling
            ### non-local 
            ## original non-local
            #self.reg_nonlocal = NONLocalBlock2D(num_inputs, sub_sample=False, bn_layer=False) 
            ## group non-local
            cls_num_group = config.MODEL.ROI_BOX_HEAD.NONLOCAL_CLS_NUM_GROUP
            self.cls_num_stack = config.MODEL.ROI_BOX_HEAD.NONLOCAL_CLS_NUM_STACK

            reg_num_group = config.MODEL.ROI_BOX_HEAD.NONLOCAL_REG_NUM_GROUP
            self.reg_num_stack = config.MODEL.ROI_BOX_HEAD.NONLOCAL_REG_NUM_STACK


            cls_nonlocal = []
            for i in range(self.cls_num_stack):
                cls_nonlocal.append(NONLocalBlock2D_Group(num_inputs, num_group=cls_num_group, inter_channels=nonlocal_inter_channels, sub_sample=False, bn_layer=nonlocal_use_bn, relu_layer=nonlocal_use_relu, use_softmax=nonlocal_use_softmax))
            self.cls_nonlocal = ListModule(*cls_nonlocal)
            
            reg_nonlocal = []
            for i in range(self.reg_num_stack):
                reg_nonlocal.append(NONLocalBlock2D_Group(num_inputs, num_group=reg_num_group, inter_channels=nonlocal_inter_channels,  sub_sample=False, bn_layer=nonlocal_use_bn, relu_layer=nonlocal_use_relu, use_softmax=nonlocal_use_softmax))
            self.reg_nonlocal = ListModule(*reg_nonlocal)
        

        ### convolution weights
        ## an extra conv with 2048 to 256
        # out_channels = 256
        # self.conv_reg = Conv2d(
        #     num_inputs, out_channels, kernel_size=1, stride=1, padding=0, bias=False
        # )        
        # self.bn_reg = group_norm(out_channels)
        ## v1
        # self.conv_grid = Conv2d(out_channels, out_channels, kernel_size=7, stride=1, padding=0, bias=False)
        # self.bbox_pred = nn.Linear(out_channels, num_bbox_reg_classes * 4)

        ## v2
        # self.conv_grid = Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        # self.bbox_pred = nn.Linear(out_channels * 4 * 4, num_bbox_reg_classes * 4)

        ## v3
        # self.conv_grid = Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        # self.bbox_pred = nn.Linear(out_channels * 7 * 7, num_bbox_reg_classes * 4)

        # print(self.bbox_pred)
        # exit()

    def forward(self, x):


        if self.nonlocal_use_shared == True:
            for i in range(self.shared_num_stack):
                x = self.shared_nonlocal[i](x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            cls_logit = self.cls_score(x)
            bbox_pred = self.bbox_pred(x)
        else: 
            x_cls = x
            x_reg = x
            # print (x.shape)
            # exit()
            ### non-local for cls
            for i in range(self.cls_num_stack):
                x_cls = self.cls_nonlocal[i](x_cls)
            x_cls = self.avgpool(x_cls)
            x_cls = x_cls.view(x_cls.size(0), -1)
            cls_logit = self.cls_score(x_cls)

            ### non-local for reg
            for i in range(self.reg_num_stack):
                x_reg = self.reg_nonlocal[i](x_reg)
            # print (out.shape)
            # exit()
            x_reg = self.avgpool(x_reg)
            x_reg = x_reg.view(x_reg.size(0), -1)
            bbox_pred = self.bbox_pred(x_reg)

        return cls_logit, bbox_pred


@registry.ROI_BOX_PREDICTOR.register("FPNPredictorNeighbor")
class FPNPredictorNeighbor(nn.Module):
    def __init__(self, cfg):
        super(FPNPredictorNeighbor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        # representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        # representation_size = cfg.MODEL.ROI_BOX_HEAD.NONLOCAL_OUT_CHANNELS
        representation_size = cfg.MODEL.BACKBONE.OUT_CHANNELS

        self.cls_score = nn.Linear(representation_size, num_classes)
        num_bbox_reg_classes = 2 if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG else num_classes
        self.bbox_pred = nn.Linear(representation_size, num_bbox_reg_classes * 4)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

        self.nonlocal_use_shared = cfg.MODEL.ROI_BOX_HEAD.NONLOCAL_USE_SHARED

    def forward(self, x):

        # if self.nonlocal_use_shared == True:
        #     scores = self.cls_score(x)
        #     bbox_deltas = self.bbox_pred(x)
        # else:
        #     scores = self.cls_score(x[0])
        #     bbox_deltas = self.bbox_pred(x[1])

        scores = self.cls_score(x[0])
        bbox_deltas = self.bbox_pred(x[1])

        return scores, bbox_deltas

@registry.ROI_BOX_PREDICTOR.register("FPNPredictor")
class FPNPredictor(nn.Module):
    def __init__(self, cfg):
        super(FPNPredictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM

        self.cls_score = nn.Linear(representation_size, num_classes)
        num_bbox_reg_classes = 2 if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG else num_classes
        self.bbox_pred = nn.Linear(representation_size, num_bbox_reg_classes * 4)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

    def forward(self, x):
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas



def make_roi_box_predictor(cfg):
    func = registry.ROI_BOX_PREDICTOR[cfg.MODEL.ROI_BOX_HEAD.PREDICTOR]
    return func(cfg)
