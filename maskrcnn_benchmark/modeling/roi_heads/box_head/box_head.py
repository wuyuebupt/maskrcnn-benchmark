# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

from .roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_box_predictors import make_roi_box_predictor
from .inference import make_roi_box_post_processor
from .loss import make_roi_box_loss_evaluator
from .attention import NONLocalBlock2D

from ...utils import cat

from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.structures.bounding_box import BoxList
import copy

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

        cfg_stage2 = copy.deepcopy(cfg)
        cfg_stage2.MODEL.ROI_HEADS.FG_IOU_THRESHOLD=0.6
        cfg_stage2.MODEL.ROI_HEADS.BG_IOU_THRESHOLD=0.6
        print (cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD)
        cfg.freeze()
        cfg_stage2.freeze()
        
        self.loss_evaluator_stage2 = make_roi_box_loss_evaluator(cfg_stage2)

        conv_fc_threshold = cfg.MODEL.ROI_BOX_HEAD.CONV_FC_THRESHOLD
        self.map_proposal_threshold=ProposalMapper(conv_fc_threshold)

        mask_loss = cfg.MODEL.ROI_BOX_HEAD.MASK_LOSS

        self.conv_cls_weight = mask_loss[0]
        self.conv_reg_weight = mask_loss[1]
        self.fc_cls_weight = mask_loss[2]
        self.fc_reg_weight = mask_loss[3]

        # self.evaluation_flags = cfg.TEST.EVALUATION_FLAGS
        bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
        self.box_coder = BoxCoder(weights=bbox_reg_weights)




    def regress_by_class(self, rois, proposals, label):
        """Regress the bbox for the predicted class. Used in Cascade R-CNN.
        Args:
            rois (Tensor): shape (n, 4) or (n, 5)
            label (Tensor): shape (n, )
            bbox_pred (Tensor): shape (n, 4*(#class+1)) or (n, 4)
        Returns:
            Tensor: Regressed bboxes, the same shape as input rois.
        """
        label = label * 4
        inds = torch.stack((label, label + 1, label + 2, label + 3), 1)
        bbox_pred = torch.gather(rois, 1, inds)
        # print (bbox_pred.shape)

        # label.shape

        ## pos
        # inds_pos = torch.nonzero(label > 0).squeeze(1)
        # proposals[inds_pos, :] = bbox_pred[inds_pos, :]
        # return proposals
        ## neg
        # inds_neg = torch.nonzero(label == 0).squeeze(1)
        # bbox_pred[inds_neg, :] = proposals[inds_neg, :]
        return bbox_pred

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

        # proposals_all = proposals
        # print (proposals_all)
        if self.training:
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            with torch.no_grad():
                proposals = self.loss_evaluator.subsample(proposals, targets)

        # # proposals_all = proposals
        # print (proposals_all)


        
        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        x = self.feature_extractor(features, proposals)
        # final classifier that converts the features into predictions
        # class_logits, box_regression = self.predictor(x)
        # class_logits, box_regression, class_logits_fc, box_regression_fc, mask, mask_fc = self.predictor(x)
        class_logits, box_regression, class_logits_fc, box_regression_fc, class_logits_fc_stage2,  box_regression_fc_stage2, mask, mask_fc = self.predictor(x)

        ## 
        # print (class_logits_fc_stage2_.shape)
        ### stage 2
        ## update proposal regress by classes
        # print (proposals[0].shape)
        # print (proposals[0])
        # rois = proposals[0].bbox
        boxes = proposals
      
        # print (rois)
        # print (class_logits_fc.shape)

        ## fc cls results 
        bbox_label = class_logits_fc.argmax(1)

        ## conv cls results 
        # bbox_label = class_logits.argmax(1)
        # print (bbox_label.shape)
        # print (bbox_label)
         
        boxes_per_image = [len(box) for box in boxes]
        # print (boxes_per_image)
        # exit()
        concat_boxes = torch.cat([a.bbox for a in boxes], dim=0)
        # print (concat_boxes.shape)

        rois_new = self.box_coder.decode(box_regression.view(sum(boxes_per_image), -1), concat_boxes)
        # rois_new = self.box_coder.decode(box_regression_fc.view(sum(boxes_per_image), -1), concat_boxes)

        #### decode from conv box results, but class from fc? 


        # print (rois_new.shape)
        proposals_new_box = self.regress_by_class(rois_new, concat_boxes, bbox_label)

        ## keep negative rois maybe???

        # rint (proposals_new_box)

        image_shape = proposals[0].size
        proposals_new_ = BoxList(proposals_new_box, image_shape, mode="xyxy")
        # print (proposals[0].extra_fields)
        for extra_field in proposals[0].extra_fields:
            proposals_new_.add_field(extra_field, proposals[0].get_field(extra_field))

        proposals_new = [proposals_new_]

        if self.training:
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            with torch.no_grad():
                proposals_new = self.loss_evaluator_stage2.subsample(proposals_new, targets)
        
        ## update prediction
        x_stage2 = self.feature_extractor(features, proposals_new)
        class_logits_, box_regression_, class_logits_fc_, box_regression_fc_, class_logits_fc_stage2_,  box_regression_fc_stage2_, mask_, mask_fc_ = self.predictor(x_stage2)



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
                result_ = self.post_processor[i]((class_logits, box_regression, class_logits_fc, box_regression_fc, class_logits_fc_stage2), proposals)
                # result_ = self.post_processor[i]((class_logits_fc, box_regression, class_logits_fc_stage2, box_regression, class_logits_fc_stage2), proposals)
                # result_ = self.post_processor[i]((class_logits_, box_regression_, class_logits_fc_stage2_, box_regression_fc_, class_logits_fc_stage2_), proposals_new)
                result.append(result_)

            for i in range(self.num_evaluation):
                result_ = self.post_processor[i]((class_logits_, box_regression_, class_logits_fc_stage2_, box_regression_fc_, class_logits_fc_stage2_), proposals_new)
                # result_ = self.post_processor[i]((class_logits_fc, box_regression_, class_logits_fc_stage2_, box_regression_, class_logits_fc_stage2_), proposals_new)
                # result_ = self.post_processor[i]((class_logits_, box_regression_, class_logits_fc_, box_regression_fc_, class_logits_fc_stage2_), proposals_new)
                # result_ = self.post_processor[i]((class_logits_, box_regression_, class_logits_fc_, box_regression_fc_), proposals)
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

        loss_classifier_fc_stage2, loss_box_reg_fc_stage2 = self.loss_evaluator_stage2(
            [class_logits_fc_stage2_], [box_regression_fc_stage2_], [mask_fc]
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

        loss_classifier_fc_stage2 = loss_classifier_fc_stage2 * self.fc_cls_weight
        loss_box_reg_fc_stage2 = loss_box_reg_fc_stage2 * self.fc_reg_weight
        # print (loss_classifier)
        # print (loss_box_reg)
        # print (loss_classifier_fc)
        # print (loss_box_reg_fc)
        # exit()


        return (
            x,
            proposals,
            # dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg, loss_classifier_fc=loss_classifier_fc, loss_box_reg_fc=loss_box_reg_fc)
            dict(loss_classifier_fc_stage2=loss_classifier_fc_stage2, loss_box_reg_fc_stage2=loss_box_reg_fc_stage2)
        )


def build_roi_box_head(cfg):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIBoxHead(cfg)
