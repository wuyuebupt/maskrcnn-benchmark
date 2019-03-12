import numpy as np
import os,sys

import cv2


# image is the original color image
# filename is the path for saving the result image with attention. 
# gcam here is the heat map of attention

def boxes_expand(boxes, neighbor_expand):
    """
    Arguments:
        boxes (Tensor):
    Returns:
        result (Tensor)
    """
    # print (boxes.shape)

    TO_REMOVE = 1  # TODO remove
    widths = boxes[:, 2] - boxes[:, 0] + TO_REMOVE
    heights = boxes[:, 3] - boxes[:, 1] + TO_REMOVE
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    ## 
    expand_widths = widths * neighbor_expand
    expand_heights = heights * neighbor_expand

    pred_boxes = np.zeros_like(boxes)

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

def getpatch(img, proposal):

    proposal = np.clip(proposal, 0, 1400)
    x1 = int(proposal[0,0])
    y1 = int(proposal[0,1])
    x2 = int(proposal[0,2])
    y2 = int(proposal[0,3])

    crop = img[y1:y2, x1:x2, :]
    return crop


def save_cam(image, gcam):
    gcam = gcam - np.min(gcam)
    gcam = gcam / np.max(gcam)
    #print (gcam)
    # b,g,r = cv2.split(image)       # get b,g,r
    # image = cv2.merge([r,g,b]) 
    h, w, d = image.shape
    gcam = cv2.resize(gcam, (w,h))
    gcam = cv2.applyColorMap(
        np.uint8(255 * gcam), cv2.COLORMAP_JET)
    gcam = np.asarray(gcam, dtype=np.float) + \
        np.asarray(image, dtype=np.float)
    gcam = 255 * gcam / np.max(gcam)
    gcam = np.uint8(gcam)

    return gcam

    # cv2.imwrite(filename, gcam)



if __name__ == '__main__':
    proposal_file = sys.argv[1]
    fid = open(proposal_file, 'rb')
    proposal = np.fromfile(fid, dtype='f4')
    proposal = np.reshape(proposal, (1,4))
    proposal = boxes_expand(proposal, 1.2)
    # proposal = boxes_expand(proposal, 1.2)
    # proposal = boxes_expand(proposal, 1.2)
    print (proposal)
    print (proposal.shape)

    ## prediction 

    pred_file = sys.argv[2]
    fid = open(pred_file, 'rb')
    pred = np.fromfile(fid, dtype='f4')
    pred = np.reshape(pred, (1,4))


    attention_file = sys.argv[3]
    fid = open(attention_file, 'rb')
    attention = np.fromfile(fid, dtype='f4')
    attention = np.reshape(attention, (4, 7, 7))

    print (attention.shape)
    # print (attention)

    img_file = sys.argv[4]

    img = cv2.imread(img_file)

    x1 = proposal[0, 0]
    y1 = proposal[0, 1]
    x2 = proposal[0, 2]
    y2 = proposal[0, 3]
    cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)

    x1 = pred[0, 0]
    y1 = pred[0, 1]
    x2 = pred[0, 2]
    y2 = pred[0, 3]
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255,0), 2)

    # gcam = save_cam(img, attention[0,:,:])
    # cv2.imshow('def', gcam)
    cv2.imshow('def', img)
    cv2.waitKey()


    ## crop proposal out
    patch = getpatch(img, proposal)
    print (patch.shape)
    cv2.imshow('abc', patch)
    cv2.waitKey()

    save_prefix = 'attention/att_'
    gcam = save_cam(patch, attention[0,:,:])
    cv2.imshow('abc', gcam)
    save_0 = save_prefix + '0.jpg'
    cv2.imwrite(save_0, gcam)
    cv2.waitKey()
    gcam = save_cam(patch, attention[1,:,:])
    cv2.imshow('abc', gcam)
    save_0 = save_prefix + '1.jpg'
    cv2.imwrite(save_0, gcam)
    cv2.waitKey()
    gcam = save_cam(patch, attention[2,:,:])
    cv2.imshow('abc', gcam)
    save_0 = save_prefix + '2.jpg'
    cv2.imwrite(save_0, gcam)
    cv2.waitKey()
    gcam = save_cam(patch, attention[3,:,:])
    cv2.imshow('abc', gcam)
    save_0 = save_prefix + '3.jpg'
    cv2.imwrite(save_0, gcam)
    cv2.waitKey()
    
   



    # print ()


# image is the original color image
# filename is the path for saving the result image with attention. 
# gcam here is the heat map of attention

# import cv2
# import numpy as np
# 
# def save_cam(image, filename, gcam):
#     gcam = gcam - np.min(gcam)
#     gcam = gcam / np.max(gcam)
#     #print (gcam)
#     # b,g,r = cv2.split(image)       # get b,g,r
#     # image = cv2.merge([r,g,b]) 
#     h, w, d = image.shape
#     gcam = cv2.resize(gcam, (w,h))
#     gcam = cv2.applyColorMap(
#         np.uint8(255 * gcam), cv2.COLORMAP_JET)
#     gcam = np.asarray(gcam, dtype=np.float) + \
#         np.asarray(image, dtype=np.float)
#     gcam = 255 * gcam / np.max(gcam)
#     gcam = np.uint8(gcam)
# 
#     cv2.imwrite(filename, gcam)
