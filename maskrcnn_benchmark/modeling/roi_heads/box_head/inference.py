# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
import scipy.io as sio

from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
import numpy as np


class PostProcessor(nn.Module):
    """
    From a set of classification scores, box regression and proposals,
    computes the post-processed boxes, and applies NMS to obtain the
    final results
    """

    def __init__(
        self,
        score_thresh=0.05,
        nms=0.5,
        detections_per_img=100,
        box_coder=None,
        cls_agnostic_bbox_reg=False,
        mode = 2
    ):
        """
        Arguments:
            score_thresh (float)
            nms (float)
            detections_per_img (int)
            box_coder (BoxCoder)
        """
        super(PostProcessor, self).__init__()
        self.score_thresh = score_thresh
        self.nms = nms
        self.detections_per_img = detections_per_img
        if box_coder is None:
            box_coder = BoxCoder(weights=(10., 10., 5., 5.))
        self.box_coder = box_coder
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg

        self.mode = mode

    def forward(self, x, boxes, path=None):
        """
        Arguments:
            x (tuple[tensor, tensor]): x contains the class logits
                and the box_regression from the model.
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for ech image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra fields labels and scores
        """
        # class_logits, box_regression = x
        # class_prob = F.softmax(class_logits, -1)
        class_logits_conv, box_regression_conv,  class_logits_fc, box_regression_fc = x
        class_prob_conv = F.softmax(class_logits_conv, -1)
        class_prob_fc = F.softmax(class_logits_fc, -1)

        # 0 : conv cls + conv reg
        # 1 : fc cls + fc cls
        # 2 : fc cls + conv reg
        # 3 : fc cls + conv reg (in posterior bayesian manner)
        class_prob = class_prob_fc
        box_regression = box_regression_conv

        if self.mode == 0:
            class_prob = class_prob_conv
            box_regression = box_regression_conv

        if self.mode == 1:
            class_prob = class_prob_fc
            box_regression = box_regression_fc

        if self.mode == 2:
            class_prob = class_prob_fc
            box_regression = box_regression_conv

        if self.mode == 3:
            class_prob = 1 - (1 - class_prob_conv) * (1 - class_prob_fc)
            box_regression = box_regression_conv

        # class_prob_conv = F.softmax(class_logits_conv, -1)
        # class_prob_fc = F.softmax(class_logits_fc, -1)
        # class_prob = (class_prob_conv + class_prob_fc) / 2
        # class_prob = 1 - (1 - class_prob_conv) * (1 - class_prob_fc)
        # class_prob = class_prob_fc


        # TODO think about a representation of batch of boxes
        image_shapes = [box.size for box in boxes]
        boxes_per_image = [len(box) for box in boxes]
        concat_boxes = torch.cat([a.bbox for a in boxes], dim=0)
        ## input concat_boxes
        concat_boxes_np = concat_boxes.cpu().detach().numpy()
        print (concat_boxes.shape)
        # print (concat_boxes_np)
        # print (concat_boxes_np.shape)

        # print (boxes_per_image)
        # gt_labels = boxes[0].extra_fields['labels']
        gt_labels = boxes[0].extra_fields['labels']
        gt_box = boxes[0].extra_fields['gt_box']
        print (gt_labels)
        print (gt_box)
        # exit()

        if self.cls_agnostic_bbox_reg:
            box_regression = box_regression[:, -4:]

        
        proposals = self.box_coder.decode(
            box_regression.view(sum(boxes_per_image), -1), concat_boxes
        )
        print (proposals.shape)
        ## get the gt bbox
        proposals = proposals[:, 4*gt_labels:4*gt_labels+4]
        # print (proposals.shape)
        # exit()



        if self.cls_agnostic_bbox_reg:
            proposals = proposals.repeat(1, class_prob.shape[1])

        num_classes = class_prob.shape[1]

        # print (proposals)
        # print (class_prob.shape)

        ## selected gt
        class_prob = class_prob[:, gt_labels]

        ## save all prob
        # class_prob = class_prob

        # print (class_prob.shape)
        concat_boxes = concat_boxes.split(boxes_per_image, dim=0)
        proposals = proposals.split(boxes_per_image, dim=0)
        class_prob = class_prob.split(boxes_per_image, dim=0)
        # print (proposals)
        # print (class_prob.shape)
        # exit()

        results = []
        for prob, boxes_per_img, concat_box, image_shape in zip(
            class_prob, proposals, concat_boxes, image_shapes
        ):
            print (boxes_per_img.shape)
            print (prob.shape)
            boxlist = self.prepare_boxlist(boxes_per_img, prob, image_shape)
            boxlist_proposal = self.prepare_boxlist(concat_box, prob, image_shape)
            boxlist.add_field('labels', gt_labels)
            boxlist.add_field('gt_box', gt_box)
            boxlist_proposal.add_field('labels', gt_labels)
            boxlist_proposal.add_field('gt_box', gt_box)

            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist_proposal = boxlist_proposal.clip_to_image(remove_empty=False)
            print (boxlist)
            # print (boxlist.bbox)
            # print (boxlist.extra_fields['scores'])
            boxlist, iou_prob = self.filter_results_nms_analysis(boxlist, num_classes, boxlist_proposal)
            if len(iou_prob) == 0:
                iou_before  = -1 
                prob_before = -1 
                iou_after   = -1 
                prob_after  = -1 
                iou_before_proposal = -1 
                continue
            # print (boxlist)
            print (iou_prob)
            iou_before  = iou_prob[0].cpu().detach().numpy()
            prob_before = iou_prob[1].cpu().detach().numpy()
            iou_after   = iou_prob[2].cpu().detach().numpy()
            prob_after  = iou_prob[3].cpu().detach().numpy()
            iou_before_proposal  = iou_prob[4].cpu().detach().numpy()


            out_boxes_np  = boxlist.bbox.cpu().detach().numpy()
            cls_np  = boxlist.extra_fields['scores'].cpu().detach().numpy()
            # boxlist = self.filter_results(boxlist, num_classes)
            # print (boxlist)
            # print (boxlist.bbox)
            # print (boxlist.extra_fields['scores'])
            # print (boxlist.extra_fields['labels'])
         
            results.append(boxlist)
        ## 
        
        # save_name = 'bbox_weight' + str(self.mode) + '.mat'
        if self.mode == 0:
            save_name = 'weights/' + path + '_bbox_weight_conv.mat'
        elif self.mode == 1:
            save_name = 'weights/' + path + '_bbox_weight_fc.mat'
        else:
            save_name = 'weights/' + path + '_bbox_weight_shoudNotHappen.mat'
        print (save_name)

        # sio.savemat(save_name, {'bbox': concat_boxes_np, 'out_bbox':out_boxes_np, 'cls':cls_np})
        sio.savemat(save_name, {'iou_before': iou_before, 'iou_after': iou_after, 'prob_before': prob_before, 'prob_after': prob_after, 'iou_before_proposal': iou_before_proposal})

        # sio.savemat('bbox_weight.mat', {'bbox': concat_boxes_np, 'out_bbox':out_boxes_np, 'cls':cls_np})
        return results

    def prepare_boxlist(self, boxes, scores, image_shape):
        """
        Returns BoxList from `boxes` and adds probability scores information
        as an extra field
        `boxes` has shape (#detections, 4 * #classes), where each row represents
        a list of predicted bounding boxes for each of the object classes in the
        dataset (including the background class). The detections in each row
        originate from the same object proposal.
        `scores` has shape (#detection, #classes), where each row represents a list
        of object detection confidence scores for each of the object classes in the
        dataset (including the background class). `scores[i, j]`` corresponds to the
        box at `boxes[i, j * 4:(j + 1) * 4]`.
        """
        boxes = boxes.reshape(-1, 4)
        scores = scores.reshape(-1)
        boxlist = BoxList(boxes, image_shape, mode="xyxy")
        boxlist.add_field("scores", scores)
        return boxlist

    def gt_filter_results(self, boxlist, gt_box, boxlist_proposal):
        # print (boxlist)
        # print (gt_box)
        image_shape = boxlist.size
        # print (image_shape)
        gt_boxlist = self.prepare_boxlist(gt_box, np.ones(1,), image_shape)
        # print (gt_boxlist)
        match_quality_matrix = boxlist_iou(gt_boxlist, boxlist)
        # print (match_quality_matrix)
        # print (match_quality_matrix.shape)
        match_quality_matrix = match_quality_matrix.reshape(-1)
        # print (match_quality_matrix.shape)
        pos_ind = torch.nonzero(match_quality_matrix > 0).squeeze(1)

        # print (pos_ind)
        selected_boxes = boxlist.bbox[pos_ind]
        selected_boxes_proposal = boxlist_proposal.bbox[pos_ind]
        # print (selected_boxes)
        print (selected_boxes.shape)
        print (selected_boxes_proposal.shape)


        scores = boxlist.extra_fields['scores']
        # print (scores)
        # print (scores.shape)
        selected_scores = scores[pos_ind]
        # print (selected_scores)
        selected_boxlist = self.prepare_boxlist(selected_boxes, selected_scores, boxlist.size)
        selected_boxlist_proposal = self.prepare_boxlist(selected_boxes_proposal, selected_scores, boxlist.size)
        # print (selected_boxlist)
        # exit()
        return selected_boxlist, selected_boxlist_proposal




    def filter_results_nms_analysis(self, boxlist, num_classes, boxlist_proposal):
        """Returns bounding-box detection results by thresholding on scores and
        applying non-maximum suppression (NMS).
        """
        # unwrap the boxlist to avoid additional overhead.
        # if we had multi-class NMS, we could perform this directly on the boxlist
        # boxes = boxlist.bbox.reshape(-1, num_classes * 4)
        # scores = boxlist.get_field("scores").reshape(-1, num_classes)
        boxes = boxlist.bbox.reshape(-1, 4)
        boxes_proposal = boxlist_proposal.bbox.reshape(-1, 4)
        scores = boxlist.get_field("scores").reshape(-1, 1)
        gt_labels = boxlist.get_field("labels")
        gt_box = boxlist.get_field("gt_box")
        # print (boxlist.extra_fields)
        print (boxes.shape)
        print (boxes_proposal.shape)
        print (scores.shape)
        print (gt_box)

        device = scores.device
        result = []
        result_iou_prob = []
        # Apply threshold on detection probabilities and apply NMS
        # Skip j = 0, because it's the background class
        inds_all = scores > self.score_thresh
        print (inds_all)
        # for j in range(1, num_classes):
        for j in range(1):
            inds = inds_all[:, j].nonzero().squeeze(1)
            print (inds)
            # if  
            #     return [], []
            scores_j = scores[inds, j]
            boxes_j = boxes[inds, j * 4 : (j + 1) * 4]
            boxes_j_proposal = boxes_proposal[inds, j * 4 : (j + 1) * 4]
            boxlist_for_class = BoxList(boxes_j, boxlist.size, mode="xyxy")
            boxlist_for_class.add_field("scores", scores_j)

            boxlist_for_class_proposal = BoxList(boxes_j_proposal, boxlist.size, mode="xyxy")
            boxlist_for_class_proposal.add_field("scores", scores_j)

            ### gt filter and match make sure their is only one output at the end
            boxlist_for_class, boxlist_for_class_proposal = self.gt_filter_results(boxlist_for_class, gt_box, boxlist_for_class_proposal) 
            if boxlist_for_class.bbox.shape[0] == 0:
                return [], []


            
            print (boxlist_for_class.bbox.shape)

            ### nms
            ## before nms, calculate max 
            gt_boxlist = self.prepare_boxlist(gt_box, np.ones(1,), boxlist_for_class.size)
            match_quality_matrix_pre = boxlist_iou(gt_boxlist, boxlist_for_class)
            match_quality_matrix_pre = match_quality_matrix_pre.reshape(-1)

            ## proposal IoU
            match_quality_matrix_pre_proposal = boxlist_iou(gt_boxlist, boxlist_for_class_proposal)
            match_quality_matrix_pre_proposal = match_quality_matrix_pre_proposal.reshape(-1)
           
            max_ind = torch.argmax(match_quality_matrix_pre)
            max_iou_before = match_quality_matrix_pre[max_ind]
            iou_before = match_quality_matrix_pre
            iou_before_proposal = match_quality_matrix_pre_proposal

            # prob_before = boxlist_for_class.extra_fields['scores'][max_ind]
            prob_before = boxlist_for_class.extra_fields['scores']
            # print (boxlist_for_class.extra_fields['scores'])
            # print (boxlist_for_class.extra_fields['scores'].shape)
            # print (max_iou_before)
            #  print (prob_before)
            # print (boxlist_for_class)
           
            print (match_quality_matrix_pre)
            print (match_quality_matrix_pre.shape)
            
            # exit()
              
            boxlist_for_class = boxlist_nms(
                boxlist_for_class, self.nms
            )

            if boxlist_for_class.bbox.shape[0] > 1:
                return [], []

 
            match_quality_matrix = boxlist_iou(gt_boxlist, boxlist_for_class)
            match_quality_matrix = match_quality_matrix.reshape(-1)
            # print (match_quality_matrix)
            max_ind = torch.argmax(match_quality_matrix)
            max_iou_after = match_quality_matrix[max_ind]
            prob_after = boxlist_for_class.extra_fields['scores'][max_ind]

            # print (max_iou_after)
            # print (prob_after)
            # print (boxlist_for_class)
            # exit()
           
            # result_iou_prob = [max_iou_before, prob_before, max_iou_after, prob_after]
            result_iou_prob = [iou_before, prob_before, max_iou_after, prob_after, iou_before_proposal]
 
            num_labels = len(boxlist_for_class)
            boxlist_for_class.add_field(
                "labels", torch.full((num_labels,), j, dtype=torch.int64, device=device)
            )
            result.append(boxlist_for_class)

        result = cat_boxlist(result)
        number_of_detections = len(result)

        # Limit to max_per_image detections **over all classes**
        if number_of_detections > self.detections_per_img > 0:
            cls_scores = result.get_field("scores")
            image_thresh, _ = torch.kthvalue(
                cls_scores.cpu(), number_of_detections - self.detections_per_img + 1
            )
            keep = cls_scores >= image_thresh.item()
            keep = torch.nonzero(keep).squeeze(1)
            result = result[keep]
        return result, result_iou_prob

    def filter_results(self, boxlist, num_classes):
        """Returns bounding-box detection results by thresholding on scores and
        applying non-maximum suppression (NMS).
        """
        # unwrap the boxlist to avoid additional overhead.
        # if we had multi-class NMS, we could perform this directly on the boxlist
        boxes = boxlist.bbox.reshape(-1, num_classes * 4)
        scores = boxlist.get_field("scores").reshape(-1, num_classes)

        device = scores.device
        result = []
        # Apply threshold on detection probabilities and apply NMS
        # Skip j = 0, because it's the background class
        inds_all = scores > self.score_thresh
        for j in range(1, num_classes):
            inds = inds_all[:, j].nonzero().squeeze(1)
            scores_j = scores[inds, j]
            boxes_j = boxes[inds, j * 4 : (j + 1) * 4]
            boxlist_for_class = BoxList(boxes_j, boxlist.size, mode="xyxy")
            boxlist_for_class.add_field("scores", scores_j)
            boxlist_for_class = boxlist_nms(
                boxlist_for_class, self.nms
            )
            num_labels = len(boxlist_for_class)
            boxlist_for_class.add_field(
                "labels", torch.full((num_labels,), j, dtype=torch.int64, device=device)
            )
            result.append(boxlist_for_class)

        result = cat_boxlist(result)
        number_of_detections = len(result)

        # Limit to max_per_image detections **over all classes**
        if number_of_detections > self.detections_per_img > 0:
            cls_scores = result.get_field("scores")
            image_thresh, _ = torch.kthvalue(
                cls_scores.cpu(), number_of_detections - self.detections_per_img + 1
            )
            keep = cls_scores >= image_thresh.item()
            keep = torch.nonzero(keep).squeeze(1)
            result = result[keep]
        return result


def make_roi_box_post_processor(cfg):
    use_fpn = cfg.MODEL.ROI_HEADS.USE_FPN

    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights)

    score_thresh = cfg.MODEL.ROI_HEADS.SCORE_THRESH
    nms_thresh = cfg.MODEL.ROI_HEADS.NMS
    detections_per_img = cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG
    cls_agnostic_bbox_reg = cfg.MODEL.CLS_AGNOSTIC_BBOX_REG

    evaluation_flags = cfg.TEST.EVALUATION_FLAGS

    #### return multiple post processor with different mode
    # 0 : conv cls + conv reg
    # 1 : fc cls + fc cls
    # 2 : fc cls + conv reg
    # 3 : fc cls + conv reg (in posterior bayesian manner)
    #------
    # evaluation_flags: 1 1 1 1

    postprocessor = []
    for i, value in enumerate(evaluation_flags):
        print (i, value)
        if value == 1:
            postprocessor_ = PostProcessor(
                score_thresh,
                nms_thresh,
                detections_per_img,
                box_coder,
                cls_agnostic_bbox_reg,
                mode = i
            )      
            postprocessor.append(postprocessor_)

    # print (len(postprocessor))
    assert (len(postprocessor) > 0)        
    # exit()

    return postprocessor
