# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

from .roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_box_predictors import make_roi_box_predictor
from .inference import make_roi_box_post_processor
from .loss import make_roi_box_loss_evaluator
from .attention import NONLocalBlock2D


class ROIBoxHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg):
        super(ROIBoxHead, self).__init__()
        self.feature_extractor = make_roi_box_feature_extractor(cfg)
        self.predictor = make_roi_box_predictor(cfg)
        self.post_processor = make_roi_box_post_processor(cfg)
        self.post_processor_fc = make_roi_box_post_processor(cfg)
        self.loss_evaluator = make_roi_box_loss_evaluator(cfg)
        # self.loss_evaluator_fc = make_roi_box_loss_evaluator(cfg)

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
        class_logits, box_regression, class_logits_fc, box_regression_fc, levels = self.predictor(x)

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
            class_logits_combine = torch.zeros(
                class_logits.shape,
                dtype=dtype,
                device=device,
            )
            box_regression_combine = torch.zeros(
                box_regression.shape,
                dtype=dtype,
                device=device,
            )

            ## 0 1 from fc
            ## 2 3 from conv
            idx_in_level_01 = torch.cat((torch.nonzero(levels == 0).squeeze(1), torch.nonzero(levels == 1).squeeze(1)))
            idx_in_level_23 = torch.cat((torch.nonzero(levels == 2).squeeze(1), torch.nonzero(levels == 3).squeeze(1)))
            # print (idx_in_level_01.shape)
            # print (idx_in_level_23.shape)

            ## get 
            class_logits_combine[idx_in_level_01]   = class_logits_fc[idx_in_level_01]
            box_regression_combine[idx_in_level_01] = box_regression_fc[idx_in_level_01]
            class_logits_combine[idx_in_level_23]   = class_logits_fc[idx_in_level_23]
            box_regression_combine[idx_in_level_23] = box_regression_fc[idx_in_level_23]

            # result = self.post_processor((class_logits, box_regression), proposals)
            # result_fc = self.post_processor((class_logits_fc, box_regression_fc), proposals)
            # print (result)
            # print (result_fc)
            # exit()
            result = self.post_processor((class_logits_combine, box_regression_combine), proposals)
            # print (result)
            # exit()
            return x, result, {}

        loss_classifier, loss_box_reg = self.loss_evaluator(
            [class_logits], [box_regression]
        )

        loss_classifier_fc, loss_box_reg_fc = self.loss_evaluator(
            [class_logits_fc], [box_regression_fc]
        )
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
