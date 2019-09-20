import os,sys
import numpy as np
import scipy.io as sio


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
       
        

if __name__ == '__main__':
  filename = sys.argv[1]
  with open(filename) as f:
    lines = f.readlines()
  lines = [line.strip() for line in lines]
  print (len(lines))

  s_ = 71

  num_bbox = s_ * s_
  all_iou = np.zeros((num_bbox, 1), dtype=float)
  all_iou = all_iou.reshape(s_, s_)
  all_cls = np.zeros((num_bbox, 1), dtype=float)
  all_cls = all_cls.reshape(s_, s_)
  all_out_iou = np.zeros((num_bbox, 1), dtype=float)
  all_out_iou = all_out_iou.reshape(s_, s_)
  
  count = 0
  for line in lines:
    # print (line)
    datafile = 'weights/' + line
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

    all_iou = all_iou + iou;
    all_out_iou= all_out_iou + out_iou;
    all_cls= all_cls + cls;
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
  
    print (count)
  all_iou = all_iou / count;
  all_out_iou= all_out_iou / count;
  all_cls= all_cls / count;
  # sio.savemat(filename+'ave.mat', {'iou': iou, 'cls':cls, 'out_iou': out_iou})
  sio.savemat(filename+'ave.mat', {'iou': all_iou, 'cls':all_cls, 'out_iou': all_out_iou})
    
    # exit()
  
