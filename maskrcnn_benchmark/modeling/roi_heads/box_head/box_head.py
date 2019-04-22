# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

from .roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_box_predictors import make_roi_box_predictor
from .inference import make_roi_box_post_processor
from .loss import make_roi_box_loss_evaluator
from .attention import NONLocalBlock2D

from ...utils import cat

class ProposalMapper(object):
    """
    map proposal with a level   
    """
    def __init__(self, threshold=224):
        self.threshold = threshold

    def __call__(self, boxlists):
        s = torch.sqrt(cat([boxlist.area() for boxlist in boxlists]))
        idx_fc = torch.nonzero(s < self.threshold).squeeze(1)
        idx_conv = torch.nonzero(s >= self.threshold).squeeze(1)
        return idx_conv, idx_fc

class ROIBoxHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg):
        super(ROIBoxHead, self).__init__()
        self.feature_extractor = make_roi_box_feature_extractor(cfg)
        self.predictor = make_roi_box_predictor(cfg)
        self.post_processor = make_roi_box_post_processor(cfg)

        self.num_evaluation = len(self.post_processor)

        self.loss_evaluator = make_roi_box_loss_evaluator(cfg)

        conv_fc_threshold = cfg.MODEL.ROI_BOX_HEAD.CONV_FC_THRESHOLD
        self.map_proposal_threshold=ProposalMapper(conv_fc_threshold)

        mask_loss = cfg.MODEL.ROI_BOX_HEAD.MASK_LOSS

        self.conv_cls_weight = mask_loss[0]
        self.conv_reg_weight = mask_loss[1]
        self.fc_cls_weight = mask_loss[2]
        self.fc_reg_weight = mask_loss[3]
        self.fusion_cls_weight = mask_loss[4]

        # self.evaluation_flags = cfg.TEST.EVALUATION_FLAGS
        self.head_fusion = cfg.MODEL.ROI_BOX_HEAD.HEAD_FUSION



    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        if self.training:
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            with torch.no_grad():
                proposals = self.loss_evaluator.subsample(proposals, targets)

        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        x = self.feature_extractor(features, proposals)
        # final classifier that converts the features into predictions
        # class_logits, box_regression = self.predictor(x)
        if self.head_fusion == True:
            class_logits, box_regression, class_logits_fc, box_regression_fc, class_logits_fusion, mask, mask_fc = self.predictor(x)
        else:
            class_logits, box_regression, class_logits_fc, box_regression_fc, mask, mask_fc = self.predictor(x)

        if not self.training:
            # print (class_logits.shape)
            # print (box_regression.shape)
            # print (class_logits_fc.shape)
            # print (box_regression_fc.shape)
            # print (len(proposals))
            # print ((proposals[0]))
            # print (levels)
            # print (levels.shape)

            ## combine two results based on the level
            dtype, device = class_logits.dtype, class_logits.device
            # class_logits_combine = torch.zeros(
            #     class_logits.shape,
            #     dtype=dtype,
            #     device=device,
            # )
            # box_regression_combine = torch.zeros(
            #     box_regression.shape,
            #     dtype=dtype,
            #     device=device,
            # )

            ## 0 1 from fc
            ## 2 3 from conv
            #### find index for some thresholds, depreciated 
            ### idx_conv, idx_fc = self.map_proposal_threshold(proposals)
            # print (idx_conv.shape)
            # print (idx_fc.shape)
            # exit()

            # idx_in_level_01 = torch.cat((torch.nonzero(levels == 0).squeeze(1), torch.nonzero(levels == 1).squeeze(1)))
            # idx_in_level_23 = torch.cat((torch.nonzero(levels == 2).squeeze(1), torch.nonzero(levels == 3).squeeze(1)))
            # print (idx_in_level_01.shape)
            # print (idx_in_level_23.shape)


            ## cls from fc, reg from conv
            # class_logits_combine   = class_logits_fc
            # box_regression_combine = box_regression

            ## get 
            #$ class_logits_combine[idx_conv]   = class_logits[idx_conv]
            #$ box_regression_combine[idx_conv] = box_regression[idx_conv]

            #$ class_logits_combine[idx_fc]   = class_logits_fc[idx_fc]
            #$ box_regression_combine[idx_fc] = box_regression_fc[idx_fc]

            # class_logits_combine[idx_in_level_01]   = class_logits[idx_in_level_01]
            # box_regression_combine[idx_in_level_01] = box_regression[idx_in_level_01]
            # class_logits_combine[idx_in_level_23]   = class_logits_fc[idx_in_level_23]
            # box_regression_combine[idx_in_level_23] = box_regression_fc[idx_in_level_23]

            # result = self.post_processor((class_logits, box_regression), proposals)
            # result_fc = self.post_processor((class_logits_fc, box_regression_fc), proposals)
            # print (result)
            # print (result_fc)
            # exit()

            ## results from conv
            # result = self.post_processor((class_logits, box_regression), proposals)
            ## results from fc
            # result = self.post_processor((class_logits_fc, box_regression_fc), proposals)
            ## results from conv + fc
            # result = self.post_processor((class_logits_combine, box_regression_combine), proposals)
            result = []
            for i in range(self.num_evaluation):
                if self.head_fusion == True:
                    result_ = self.post_processor[i]((class_logits, box_regression, class_logits_fc, box_regression_fc, class_logits_fusion), proposals)
                else:
                    result_ = self.post_processor[i]((class_logits, box_regression, class_logits_fc, box_regression_fc, class_logits), proposals)
                result.append(result_)
            # print (len(result))
            # exit()
            return x, result, {}

        loss_classifier, loss_box_reg = self.loss_evaluator(
            [class_logits], [box_regression], [mask]
        )

        loss_classifier_fc, loss_box_reg_fc = self.loss_evaluator(
            [class_logits_fc], [box_regression_fc], [mask_fc]
        )

        if self.head_fusion == True:
            loss_classifier_fusion, loss_box_reg_fusion = self.loss_evaluator(
                [class_logits_fusion], [box_regression], [mask]
            )

        ## loss weights
        # print (loss_classifier)
        # print (loss_box_reg)
        # print (loss_classifier_fc)
        # print (loss_box_reg_fc)
        loss_classifier = loss_classifier * self.conv_cls_weight
        loss_box_reg = loss_box_reg * self.conv_reg_weight
        loss_classifier_fc = loss_classifier_fc * self.fc_cls_weight
        loss_box_reg_fc = loss_box_reg_fc * self.fc_reg_weight

        if self.head_fusion == True:
            loss_classifier_fusion =  loss_classifier_fusion * self.fusion_cls_weight
            return (
                x,
                proposals,
                dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg, loss_classifier_fc=loss_classifier_fc, loss_box_reg_fc=loss_box_reg_fc, loss_classifier_fusion=loss_classifier_fusion)
            )


        else:
            # print (loss_classifier)
            # print (loss_box_reg)
            # print (loss_classifier_fc)
            # print (loss_box_reg_fc)
            # exit()


            return (
                x,
                proposals,
                dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg, loss_classifier_fc=loss_classifier_fc, loss_box_reg_fc=loss_box_reg_fc)
            )


def build_roi_box_head(cfg):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIBoxHead(cfg)
