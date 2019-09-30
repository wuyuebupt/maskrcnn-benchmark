# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou



class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg)
        self.roi_heads = build_roi_heads(cfg)

    def expand_bbox(self, gt_box, split = 1, scale=1.0):
        # print (gt_box) 
        # print (gt_box.shape) 
        width  = gt_box[0, 2]  - gt_box[0, 0] + 1.0
        height = gt_box[0, 3]  - gt_box[0, 1] + 1.0
        height = height * scale
        width = width * scale
        # print (width, height)
        num = split * 2 + 1

        total_num = num * num
        new_bbox = torch.zeros_like(gt_box)
        # print (new_bbox.shape)
        new_bbox = new_bbox.expand(total_num, -1).clone()

        if split == 0:
            width_split = width / 2.0 
            height_split = height / 2.0 
        else:
            width_split = width / 2.0 / split
            height_split = height / 2.0 / split
        # print (new_bbox.shape)
        for i in range(num):
            ## x
            delta_x = (i - split ) * width_split
            for j in range(num):
                ## y
                delta_y = (j - split ) * height_split
                ## index
                index = i * num + j
                ## x
                #print (index, delta_x, delta_y)
                new_bbox[index, 0] = gt_box[0, 0] + delta_x
                new_bbox[index, 1] = gt_box[0, 1] + delta_y
                new_bbox[index, 2] = gt_box[0, 2] + delta_x
                new_bbox[index, 3] = gt_box[0, 3] + delta_y
                
        # print (new_bbox)
        # exit()
        # return gt_box
        return new_bbox


    def forward(self, images, targets=None, path=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        
        # print (images)
        # print (images.tensors.shape)
        # print (images.image_sizes)
        images = to_image_list(images)
        # print (images)
        # exit()
        features = self.backbone(images.tensors)
        proposals, proposal_losses = self.rpn(images, features, targets)
        # print (proposals)


        proposals = [targets[0]]
        # print (proposals)
        # proposals[0].bbox = proposals[0].bbox[0,:]
        # print (proposals[0].bbox)
        # print (proposals[0].extra_fields['labels'])
        # print (proposals[0].extra_fields['labels'])

        num_objs = proposals[0].extra_fields['labels'].shape[0] 
        print (num_objs)
        print (path)

        ## compute gt iou, max occlusion scores for each gt 
        gts = proposals[0]
        print (gts)
        match_quality_matrix = boxlist_iou(gts, gts)
        # print (match_quality_matrix)

        ## loop for all objects
        result = []
        for target_index in range(num_objs):

            # if target_index == 2:
            #     break
            ## reduce target and  generate bboxlist
            # target_index = 0
            print (target_index)
            print (proposals[0].bbox.shape)
            selected_gt = proposals[0].bbox[target_index,:]

            ### find the occlusion
            if num_objs > 1:
                ious = match_quality_matrix[target_index, :]
                print (ious)
                ious_ = torch.cat([ious[:target_index], ious[target_index+1:]])
                print (ious_)
                max_ind = torch.argmax(ious_)
                max_iou = ious_[max_ind]
                print (max_ind, max_iou)
            else:
                max_iou = torch.tensor(0.0)

            ### save to file 

            # print (selected_gt)
            selected_gt = selected_gt.reshape(-1, 4)
            image_shape=  proposals[0].size
            # print (image_shape)
            label = int(targets[0].extra_fields['labels'][target_index])
            print (label)
            
            
            # print (selected_gt)

            ## expend the gt bbox
            # neighbors = self.expand_bbox(selected_gt, 35, 0.75)
            # neighbors = self.expand_bbox(selected_gt, 35, 1.0)
            # neighbors = self.expand_bbox(selected_gt, 35, 1.25)
            neighbors = self.expand_bbox(selected_gt, 0, 1.0)
            # print (neighbors)
            # exit()
            
             
            # boxlist = BoxList(selected_gt, image_shape, mode="xyxy")
            boxlist = BoxList(neighbors, image_shape, mode="xyxy")
            boxlist.add_field('labels', label)
            boxlist.add_field('occlusion_iou', max_iou)
      
            # print (boxlist)
            ## bug was here
            proposals_ = [boxlist]

            ## new path
            newpath = path[0] + '_object_' + str(target_index) + '_class_'+ str(label) 
            if self.roi_heads:
                x, result, detector_losses = self.roi_heads(features, proposals_, targets, newpath)
            else:
                # RPN-only models don't have roi_heads
                x = features
                result = proposals
                detector_losses = {}


            # print (result)
            # exit()

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

        # print (len(result))
        # exit()
        return result
