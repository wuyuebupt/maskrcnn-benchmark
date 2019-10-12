import os,sys
import numpy as np
import scipy.io as sio
import cv2


def cal_iou(bbox1, bbox2):
  inter_x1 = max(bbox1[0], bbox2[0])
  inter_y1 = max(bbox1[1], bbox2[1])
  inter_x2 = min(bbox1[2], bbox2[2])
  inter_y2 = min(bbox1[3], bbox2[3])

  if inter_x1 > inter_x2 or inter_y1 > inter_y2:
      return 0.0

  inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
  bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
  bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
  area = inter_area / (bbox1_area + bbox2_area - inter_area)
  return area


def assign_occlusion_type(occ_file):
  data = sio.loadmat(occ_file)
  occlusion = data['occlusion']
  if occlusion <= 0.05:
    return -1
  elif occlusion <= 0.3:
    return 1
  elif occlusion <=0.6:
    return 2
  else:
    return 3

      
        
def assign_type(gt_box, scale):
  gt_box_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1]) * scale * scale
  if gt_box_area <= 32**2:
    return 1
  elif gt_box_area <=96**2:
    return 2
  else:
    return 3



if __name__ == '__main__':
  filename = sys.argv[1]
  with open(filename) as f:
    lines = f.readlines()
  lines = [line.strip() for line in lines]
  print (len(lines))

  s_ = 71

  num_bbox = s_ * s_

  ## all
  all_iou     = np.zeros((num_bbox, 1), dtype=float)
  all_cls     = np.zeros((num_bbox, 1), dtype=float)
  all_out_iou = np.zeros((num_bbox, 1), dtype=float)
  all_iou     = all_iou.reshape(s_, s_)
  all_cls     = all_cls.reshape(s_, s_)
  all_out_iou = all_out_iou.reshape(s_, s_)
  ## small
  small_iou     = np.zeros((num_bbox, 1), dtype=float)
  small_cls     = np.zeros((num_bbox, 1), dtype=float)
  small_out_iou = np.zeros((num_bbox, 1), dtype=float)
  small_iou     = small_iou.reshape(s_, s_)
  small_cls     = small_cls.reshape(s_, s_)
  small_out_iou = small_out_iou.reshape(s_, s_)
  ## medium
  medium_iou     = np.zeros((num_bbox, 1), dtype=float)
  medium_cls     = np.zeros((num_bbox, 1), dtype=float)
  medium_out_iou = np.zeros((num_bbox, 1), dtype=float)
  medium_iou     = medium_iou.reshape(s_, s_)
  medium_cls     = medium_cls.reshape(s_, s_)
  medium_out_iou = medium_out_iou.reshape(s_, s_)
  ## large
  large_iou     = np.zeros((num_bbox, 1), dtype=float)
  large_cls     = np.zeros((num_bbox, 1), dtype=float)
  large_out_iou = np.zeros((num_bbox, 1), dtype=float)
  large_iou     = large_iou.reshape(s_, s_)
  large_cls     = large_cls.reshape(s_, s_)
  large_out_iou = large_out_iou.reshape(s_, s_)
  
  count = 0
  small_count = 0
  medium_count = 0
  large_count = 0
  
  small  = [0**2, 32**2],     # small
  medium = [32**2, 96**2],    # medium
  large  = [96**2, 1e5**2],   # large

  for line in lines:
    # print (line)
    # datafile = 'weights/' + line
    datafile = 'weights_ori/' + line
    occlusionfile = 'weights_occlusion/' + line.replace('conv', 'occlusion').replace('fc','occlusion') 
    print (occlusionfile)

    imgname = 'val2014/' + line[:29]
    # print (imgname)
    img = cv2.imread(imgname)
    # print (img.shape)
    ## short to 800
    width = img.shape[1]
    height = img.shape[0]
    scale = -1
    if width < height:
      scale = width / 800.0 
    else:
      scale = height / 800.0
    print (scale)
    assert ((scale > 0))
    
    # datafile = 'weights-run1/' + line
    # data = sio.loadmat(line)
    data = sio.loadmat(datafile)
    # print (data)
    bbox = data['bbox']
    cls = data['cls']
    out_bbox = data['out_bbox']
    # print (bbox.shape)
    # print (cls.shape)
    # print (out_bbox.shape)
    num_bbox = bbox.shape[0]
    
    ## zero index
    gt_index = (num_bbox-1)/2
    gt_bbox = bbox[gt_index,:]

    iou = np.zeros((num_bbox, 1), dtype=float)
    out_iou = np.zeros((num_bbox, 1), dtype=float)
    for i in range(num_bbox):
        iou[i,0] = cal_iou(gt_bbox, bbox[i,:])
        out_iou[i,0] = cal_iou(gt_bbox, out_bbox[i,:])
    
    # print (iou)
    # print (out_iou)
    iou = iou.reshape(s_, s_)
    out_iou = out_iou.reshape(s_, s_)
    cls = cls.reshape(s_, s_)
    # print (out_iou)
    # print (cls)

    all_iou = all_iou + iou;
    all_out_iou= all_out_iou + out_iou;
    all_cls= all_cls + cls;

    # type_ = assign_type(gt_bbox, scale)
    type_ = assign_occlusion_type(occlusionfile)
    if type_ == 1:
      # small
      small_iou     = small_iou + iou;
      small_out_iou = small_out_iou + out_iou;
      small_cls     = small_cls + cls;
      small_count   = small_count + 1
    elif type_ == 2:
      # small
      medium_iou     = medium_iou + iou;
      medium_out_iou = medium_out_iou + out_iou;
      medium_cls     = medium_cls + cls;
      medium_count   = medium_count + 1
    elif type_ == 3:
      # small
      large_iou     = large_iou + iou;
      large_out_iou = large_out_iou + out_iou;
      large_cls     = large_cls + cls;
      large_count   = large_count + 1


    count = count + 1
    # print (iou)
    # print (np.max(iou))
    # print (cls)
    # print (np.max(cls))
    # print (out_iou)
    # print (np.max(out_iou))

    # after_reshape = line + '.iou_cls.mat'
    # after_reshape = after_reshape.replace('weights', 'weights_ious')
    # after_reshape = 'weights_ious/' + line + '.iou_cls.mat'
    # print (after_reshape)
    # sio.savemat(after_reshape, {'iou': iou, 'cls':cls, 'out_iou': out_iou})
  
    print (count, type_)
  # print (all_cls_)
  # print (count)
  all_iou     = all_iou     / count
  all_out_iou = all_out_iou / count
  all_cls     = all_cls     / count

  if small_count ==0:
      small_count = 1
  small_iou     = small_iou     / small_count
  small_out_iou = small_out_iou / small_count
  small_cls     = small_cls     / small_count

  if medium_count == 0:
      medium_count = 1
  medium_iou     = medium_iou     / medium_count
  medium_out_iou = medium_out_iou / medium_count
  medium_cls     = medium_cls     / medium_count

  if large_count == 0:
      large_count = 1
  large_iou     = large_iou     / large_count
  large_out_iou = large_out_iou / large_count
  large_cls     = large_cls     / large_count


  # sio.savemat(filename+'ave.mat', {'iou': iou, 'cls':cls, 'out_iou': out_iou})
  sio.savemat(filename+'ave.mat', {'iou': all_iou, 'cls':all_cls, 'out_iou': all_out_iou, \
                                   'small_iou': small_iou, 'small_cls':small_cls, 'small_out_iou': small_out_iou,  \
                                   'medium_iou': medium_iou, 'medium_cls':medium_cls, 'medium_out_iou':  medium_out_iou,  \
                                   'large_iou': large_iou, 'large_cls':large_cls, 'large_out_iou': large_out_iou})
    
    # exit()
  
