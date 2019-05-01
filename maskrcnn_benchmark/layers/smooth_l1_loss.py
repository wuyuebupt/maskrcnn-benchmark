# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch


# TODO maybe push this to nn?
def smooth_l1_loss(input, target, beta=1. / 9, size_average=True):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    n = torch.abs(input - target)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if size_average:
        return loss.mean()
    return loss.sum()

def smooth_l1_loss_debug(input, target, beta=1. / 9, size_average=True):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    # print (input.shape)
    # print (target.shape)
    
    n = torch.abs(input - target)
    # print (n.shape)
    # exit()
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    # print (loss)
    # print (loss.shape)
    # exit()
    if size_average:
        return loss.mean()
    # print (loss)
    # print (loss.shape)
    return loss.sum()




# TODO maybe push this to nn?
def smooth_l1_loss_mask(input, target, beta=1. / 9, size_average=True, mask=None):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    n = torch.abs(input - target)
    # print (n)
    cond = n < beta
    ## after cond
    # print (n)
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    # print (loss.shape)
    # print (mask.shape)
    # print (mask.size())

    mask = mask.view(-1, 1)
    # print (mask.shape)
    mask = mask.expand_as(loss)
    # mask = mask.expand(mask.size()[0], 4)

    # print (mask)
    if mask is not None:
        loss = loss * mask
    # print (loss)
    if size_average:
        return loss.mean()
    # exit()
    return loss.sum()
