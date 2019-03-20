# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn

from maskrcnn_benchmark.layers import ROIAlign

from .utils import cat


class LevelMapper(object):
    """Determine which FPN level each RoI in a set of RoIs should map to based
    on the heuristic in the FPN paper.
    """

    def __init__(self, k_min, k_max, canonical_scale=224, canonical_level=4, eps=1e-6):
        """
        Arguments:
            k_min (int)
            k_max (int)
            canonical_scale (int)
            canonical_level (int)
            eps (float)
        """
        self.k_min = k_min
        self.k_max = k_max
        self.s0 = canonical_scale
        self.lvl0 = canonical_level
        self.eps = eps

    def __call__(self, boxlists):
        """
        Arguments:
            boxlists (list[BoxList])
        """
        # Compute level ids
        s = torch.sqrt(cat([boxlist.area() for boxlist in boxlists]))

        # Eqn.(1) in FPN paper
        target_lvls = torch.floor(self.lvl0 + torch.log2(s / self.s0 + self.eps))
        target_lvls = torch.clamp(target_lvls, min=self.k_min, max=self.k_max)
        return target_lvls.to(torch.int64) - self.k_min

class LevelMapperInput(object):
    """Determine which FPN level each RoI in a set of RoIs should map to based
    on the heuristic in the FPN paper.
    """

    def __init__(self, scales, mask):
        """
        Arguments:
            k_min (int)
            k_max (int)
            canonical_scale (int)
            canonical_level (int)
            eps (float)
        """
        self.scales = scales
        self.scales_min = scales[:-1]
        self.scales_max = scales[1:]
        self.scales_mask = mask

        print (self.scales_min)
        print (self.scales_max)
        print (self.scales_mask)

        # exit()


        # self.k_max = k_max
        # self.s0 = canonical_scale
        # self.lvl0 = canonical_level
        # self.eps = eps

    def __call__(self, boxlists):
        """
        Arguments:
            boxlists (list[BoxList])
        """
        # Compute level ids
        s = torch.sqrt(cat([boxlist.area() for boxlist in boxlists]))
        # print (s)
        # print (s.shape)
        # print (s.dtype)

        device = s.device
        lvls = torch.zeros(
                s.shape,
                dtype=torch.int64,
                device=device,
            )


        masks = torch.zeros(
                s.shape,
                dtype=torch.float,
                device=device,
            )

        for level, (scale_min, scale_max, scale_mask) in enumerate(zip(self.scales_min, self.scales_max, self.scales_mask)):
            # print (level, scale_min, scale_max)
            ## find index
            scale_min = scale_min
            scale_max = scale_max

            s_min = (s>=scale_min)
            s_max = (s<scale_max)
            # print (s_min)
            # print (s_max)
            s_min_max = s_min * s_max
            idx_in_level = torch.nonzero(s_min_max).squeeze(1)
            # print (idx_in_level.shape)
            lvls[idx_in_level] = level
            masks[idx_in_level] = scale_mask 
        # print (lvls)
        # print (masks)
        # exit()


        ### keep for comparison, original 
        # self.lvl0 = 4
        # self.s0 = 224
        # self.k_min = 2
        # self.k_max = 5
        # self.eps = 1e-6

        # # Eqn.(1) in FPN paper
        # target_lvls = torch.floor(self.lvl0 + torch.log2(s / self.s0 + self.eps))
        # target_lvls = torch.clamp(target_lvls, min=self.k_min, max=self.k_max)
        # res  = target_lvls.to(torch.int64) - self.k_min
        # print (res)
        # exit()
        return lvls.to(torch.int64), masks.to(torch.float)


class Pooler(nn.Module):
    """
    Pooler for Detection with or without FPN.
    It currently hard-code ROIAlign in the implementation,
    but that can be made more generic later on.
    Also, the requirement of passing the scales is not strictly necessary, as they
    can be inferred from the size of the feature map / size of original image,
    which is available thanks to the BoxList.
    """

    def __init__(self, output_size, scales, sampling_ratio):
        """
        Arguments:
            output_size (list[tuple[int]] or list[int]): output size for the pooled region
            scales (list[float]): scales for each Pooler
            sampling_ratio (int): sampling ratio for ROIAlign
        """
        super(Pooler, self).__init__()
        poolers = []
        for scale in scales:
            poolers.append(
                ROIAlign(
                    output_size, spatial_scale=scale, sampling_ratio=sampling_ratio
                )
            )
        self.poolers = nn.ModuleList(poolers)
        self.output_size = output_size
        # get the levels in the feature map by leveraging the fact that the network always
        # downsamples by a factor of 2 at each level.
        lvl_min = -torch.log2(torch.tensor(scales[0], dtype=torch.float32)).item()
        lvl_max = -torch.log2(torch.tensor(scales[-1], dtype=torch.float32)).item()
        self.map_levels = LevelMapper(lvl_min, lvl_max)

    def convert_to_roi_format(self, boxes):
        concat_boxes = cat([b.bbox for b in boxes], dim=0)
        device, dtype = concat_boxes.device, concat_boxes.dtype
        ids = cat(
            [
                torch.full((len(b), 1), i, dtype=dtype, device=device)
                for i, b in enumerate(boxes)
            ],
            dim=0,
        )
        rois = torch.cat([ids, concat_boxes], dim=1)
        return rois

    def forward(self, x, boxes):
        """
        Arguments:
            x (list[Tensor]): feature maps for each level
            boxes (list[BoxList]): boxes to be used to perform the pooling operation.
        Returns:
            result (Tensor)
        """
        num_levels = len(self.poolers)
        rois = self.convert_to_roi_format(boxes)
        if num_levels == 1:
            return self.poolers[0](x[0], rois)

        levels = self.map_levels(boxes)

        num_rois = len(rois)
        num_channels = x[0].shape[1]
        output_size = self.output_size[0]

        dtype, device = x[0].dtype, x[0].device
        result = torch.zeros(
            (num_rois, num_channels, output_size, output_size),
            dtype=dtype,
            device=device,
        )
        for level, (per_level_feature, pooler) in enumerate(zip(x, self.poolers)):
            idx_in_level = torch.nonzero(levels == level).squeeze(1)
            rois_per_level = rois[idx_in_level]
            result[idx_in_level] = pooler(per_level_feature, rois_per_level)

        return result

class PoolerNeighbor(nn.Module):
    """
    Pooler for Detection with or without FPN.
    It currently hard-code ROIAlign in the implementation,
    but that can be made more generic later on.
    Also, the requirement of passing the scales is not strictly necessary, as they
    can be inferred from the size of the feature map / size of original image,
    which is available thanks to the BoxList.
    """

    def __init__(self, neighbor_expand, roi_expand, output_size, scales, sampling_ratio):
        """
        Arguments:
            neighbor_expand (float): scale for enlarged proposals
            roi_expand (bool): if the output size is expanded like the proposals
            output_size (list[tuple[int]] or list[int]): output size for the pooled region
            scales (list[float]): scales for each Pooler
            sampling_ratio (int): sampling ratio for ROIAlign
        """
        super(PoolerNeighbor, self).__init__()

        self.neighbor_expand = neighbor_expand
        ## expand the output as well
        print (output_size)
        # print (type(output_size))
        if roi_expand:
            output_size = tuple([ int(x * neighbor_expand) for x in output_size])
            print (output_size)

        poolers = []
        for scale in scales:
            poolers.append(
                ROIAlign(
                    output_size, spatial_scale=scale, sampling_ratio=sampling_ratio
                )
            )
        self.poolers = nn.ModuleList(poolers)
        self.output_size = output_size

        # get the levels in the feature map by leveraging the fact that the network always
        # downsamples by a factor of 2 at each level.
        lvl_min = -torch.log2(torch.tensor(scales[0], dtype=torch.float32)).item()
        lvl_max = -torch.log2(torch.tensor(scales[-1], dtype=torch.float32)).item()
        self.map_levels = LevelMapper(lvl_min, lvl_max)

    def boxes_expand(self, boxes):
        """
        Arguments:
            boxes (Tensor):
        Returns:
            result (Tensor)
        """
        neighbor_expand = self.neighbor_expand
        # print (boxes.shape)

        TO_REMOVE = 1  # TODO remove
        widths = boxes[:, 2] - boxes[:, 0] + TO_REMOVE
        heights = boxes[:, 3] - boxes[:, 1] + TO_REMOVE
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        ## 
        expand_widths = widths * neighbor_expand
        expand_heights = heights * neighbor_expand

        pred_boxes = torch.zeros_like(boxes)

        ## print (pred_boxes.shape)
        # x1
        pred_boxes[:, 0::4] = ctr_x[:, None] - 0.5 * expand_widths[:, None]
        # y1
        pred_boxes[:, 1::4] = ctr_y[:, None] - 0.5 * expand_heights[:, None]
        # x2 (note: "- 1" is correct; don't be fooled by the asymmetry)
        pred_boxes[:, 2::4] = ctr_x[:, None] + 0.5 * expand_widths[:, None] - 1
        # y2 (note: "- 1" is correct; don't be fooled by the asymmetry)
        pred_boxes[:, 3::4] = ctr_y[:, None] + 0.5 * expand_heights[:, None] - 1

        # print (pred_boxes)
        # print (pred_boxes.shape)

        # exit()
        return pred_boxes


    def convert_to_roi_format(self, boxes):
        concat_boxes = cat([b.bbox for b in boxes], dim=0)
        # print (concat_boxes)
        ## expand the boxes
        concat_boxes = self.boxes_expand(concat_boxes)
        # print (concat_boxes)

        device, dtype = concat_boxes.device, concat_boxes.dtype
        ids = cat(
            [
                torch.full((len(b), 1), i, dtype=dtype, device=device)
                for i, b in enumerate(boxes)
            ],
            dim=0,
        )
        rois = torch.cat([ids, concat_boxes], dim=1)
        return rois

    def forward(self, x, boxes):
        """
        Arguments:
            x (list[Tensor]): feature maps for each level
            boxes (list[BoxList]): boxes to be used to perform the pooling operation.
        Returns:
            result (Tensor)
        """
        num_levels = len(self.poolers)
        # print (num_levels)
        # print (self.poolers[0])

        # print (boxes)
        rois = self.convert_to_roi_format(boxes)
        # print (rois)
        if num_levels == 1:
            # print (self.poolers[0](x[0], rois))
            # print (self.poolers[0](x[0], rois).shape)
            # exit()
            return self.poolers[0](x[0], rois)

        levels = self.map_levels(boxes)
        # print (levels)

        num_rois = len(rois)
        num_channels = x[0].shape[1]
        output_size = self.output_size[0]

        dtype, device = x[0].dtype, x[0].device
        result = torch.zeros(
            (num_rois, num_channels, output_size, output_size),
            dtype=dtype,
            device=device,
        )
        for level, (per_level_feature, pooler) in enumerate(zip(x, self.poolers)):
            idx_in_level = torch.nonzero(levels == level).squeeze(1)
            rois_per_level = rois[idx_in_level]
            result[idx_in_level] = pooler(per_level_feature, rois_per_level)
            # print (idx_in_level)
            # print (result.shape)

        # print (result)
        # print (result.shape)
        # exit()
 
        return result, levels

class PoolerNeighborMap(nn.Module):
    """
    Pooler for Detection with or without FPN.
    It currently hard-code ROIAlign in the implementation,
    but that can be made more generic later on.
    Also, the requirement of passing the scales is not strictly necessary, as they
    can be inferred from the size of the feature map / size of original image,
    which is available thanks to the BoxList.
    """

    def __init__(self, neighbor_expand, roi_expand, output_size, scales, sampling_ratio, maplevel, masklevel):
        """
        Arguments:
            neighbor_expand (float): scale for enlarged proposals
            roi_expand (bool): if the output size is expanded like the proposals
            output_size (list[tuple[int]] or list[int]): output size for the pooled region
            scales (list[float]): scales for each Pooler
            sampling_ratio (int): sampling ratio for ROIAlign
        """
        super(PoolerNeighborMap, self).__init__()

        self.neighbor_expand = neighbor_expand
        ## expand the output as well
        print (output_size)
        # print (type(output_size))
        if roi_expand:
            output_size = tuple([ int(x * neighbor_expand) for x in output_size])
            print (output_size)

        poolers = []
        for scale in scales:
            poolers.append(
                ROIAlign(
                    output_size, spatial_scale=scale, sampling_ratio=sampling_ratio
                )
            )
        self.poolers = nn.ModuleList(poolers)
        self.output_size = output_size

        # get the levels in the feature map by leveraging the fact that the network always
        # downsamples by a factor of 2 at each level.
        # lvl_min = -torch.log2(torch.tensor(scales[0], dtype=torch.float32)).item()
        # lvl_max = -torch.log2(torch.tensor(scales[-1], dtype=torch.float32)).item()
        # self.map_levels = LevelMapper(lvl_min, lvl_max)
        self.map_levels = LevelMapperInput(maplevel, masklevel)
        # print (scales)
        # print (lvl_min)
        # print (lvl_max)
        # print (self.map_levels)
        # exit()

    def boxes_expand(self, boxes):
        """
        Arguments:
            boxes (Tensor):
        Returns:
            result (Tensor)
        """
        neighbor_expand = self.neighbor_expand
        # print (boxes.shape)

        TO_REMOVE = 1  # TODO remove
        widths = boxes[:, 2] - boxes[:, 0] + TO_REMOVE
        heights = boxes[:, 3] - boxes[:, 1] + TO_REMOVE
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        ## 
        expand_widths = widths * neighbor_expand
        expand_heights = heights * neighbor_expand

        pred_boxes = torch.zeros_like(boxes)

        ## print (pred_boxes.shape)
        # x1
        pred_boxes[:, 0::4] = ctr_x[:, None] - 0.5 * expand_widths[:, None]
        # y1
        pred_boxes[:, 1::4] = ctr_y[:, None] - 0.5 * expand_heights[:, None]
        # x2 (note: "- 1" is correct; don't be fooled by the asymmetry)
        pred_boxes[:, 2::4] = ctr_x[:, None] + 0.5 * expand_widths[:, None] - 1
        # y2 (note: "- 1" is correct; don't be fooled by the asymmetry)
        pred_boxes[:, 3::4] = ctr_y[:, None] + 0.5 * expand_heights[:, None] - 1

        # print (pred_boxes)
        # print (pred_boxes.shape)

        # exit()
        return pred_boxes


    def convert_to_roi_format(self, boxes):
        concat_boxes = cat([b.bbox for b in boxes], dim=0)
        # print (concat_boxes)
        ## expand the boxes
        concat_boxes = self.boxes_expand(concat_boxes)
        # print (concat_boxes)

        device, dtype = concat_boxes.device, concat_boxes.dtype
        ids = cat(
            [
                torch.full((len(b), 1), i, dtype=dtype, device=device)
                for i, b in enumerate(boxes)
            ],
            dim=0,
        )
        rois = torch.cat([ids, concat_boxes], dim=1)
        return rois

    def forward(self, x, boxes):
        """
        Arguments:
            x (list[Tensor]): feature maps for each level
            boxes (list[BoxList]): boxes to be used to perform the pooling operation.
        Returns:
            result (Tensor)
        """
        num_levels = len(self.poolers)
        # print (num_levels)
        # print (self.poolers[0])

        # print (boxes)
        rois = self.convert_to_roi_format(boxes)
        # print (rois)
        if num_levels == 1:
            # print (self.poolers[0](x[0], rois))
            # print (self.poolers[0](x[0], rois).shape)
            # exit()
            return self.poolers[0](x[0], rois)

        levels, masks = self.map_levels(boxes)
        # print (levels)

        num_rois = len(rois)
        num_channels = x[0].shape[1]
        output_size = self.output_size[0]

        dtype, device = x[0].dtype, x[0].device
        result = torch.zeros(
            (num_rois, num_channels, output_size, output_size),
            dtype=dtype,
            device=device,
        )
        for level, (per_level_feature, pooler) in enumerate(zip(x, self.poolers)):
            idx_in_level = torch.nonzero(levels == level).squeeze(1)
            rois_per_level = rois[idx_in_level]
            result[idx_in_level] = pooler(per_level_feature, rois_per_level)
            # print (idx_in_level)
            # print (result.shape)

        # print (result)
        # print (result.shape)
        # exit()
 
        return result, masks
