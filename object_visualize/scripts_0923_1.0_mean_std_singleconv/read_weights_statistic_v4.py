import os,sys
import numpy as np
import scipy.io as sio
import math


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
  num_samples = len(lines)
  # exit()

  s_ = 71

  num_bbox = s_ * s_
  all_iou = np.zeros((num_samples, num_bbox, 1), dtype=float)
  all_iou = all_iou.reshape(num_samples, s_, s_)
  all_cls = np.zeros((num_samples, num_bbox, 1), dtype=float)
  all_cls = all_cls.reshape(num_samples, s_, s_)
  all_out_iou = np.zeros((num_samples, num_bbox, 1), dtype=float)
  all_out_iou = all_out_iou.reshape(num_samples, s_, s_)
  print (all_out_iou.shape)
  # exit()


  step = 5 #  must be devided by 100
  start = 0
  end = 100
  
  
  ## generate the list of list
  iou_cls_bin = []
  iou_out_bin = []
  thres = []
  # for i in range(start, end+step, step):
  for i in range(start, end, step):
    iou_cls_bin.append([])
    iou_out_bin.append([])
    thres.append(i/100.0)
  print (len(iou_cls_bin))
  print (thres)
  # iou_index = np.zeros((s_, s_), dtype=int) 
  iou_index = np.zeros((num_bbox, 1), dtype=int) 
  print (iou_index.shape)
  ## init the bin index
  line  = lines[0]
  datafile = 'weights/' + line
  data = sio.loadmat(datafile)
  bbox = data['bbox']

  num_bbox = bbox.shape[0]
  gt_index = (num_bbox-1)/2
  gt_bbox = bbox[gt_index,:]
  iou = np.zeros((num_bbox, 1), dtype=float)

  for i in range(num_bbox):
      iou[i,0] = cal_iou(gt_bbox, bbox[i,:])
      # iou_index_ = int(math.ceil(iou[i,0] / 0.05))
      iou_index_ = int(math.floor(iou[i,0] / step * 100))
      # print (iou_index_)
      if iou_index_ == 100 / step:
          # print (iou_index_)
          iou_index_ = 100 / step - 1

      iou_index[i,0] = iou_index_
      print (i, iou_index_, iou[i,0])
  ## 
  iou_index = iou_index.reshape(s_, s_)
  print (iou_index.shape)
  # print (iou_index)
  # exit()

  count = 0
  for line in lines:
    print (line)
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
    # exit()
    num_bbox = bbox.shape[0]
    
    ## zero index
    gt_index = (num_bbox-1)/2
    gt_bbox = bbox[gt_index,:]

    iou = np.zeros((num_bbox, 1), dtype=float)
    out_iou = np.zeros((num_bbox, 1), dtype=float)

    # cls = cls.reshape(s_, s_)
    # cls = cls.reshape(num_bbox, 1)
    for i in range(num_bbox):
        iou[i,0] = cal_iou(gt_bbox, bbox[i,:])
        out_iou[i,0] = cal_iou(gt_bbox, out_bbox[i,:])

        ## append data
        # print (cls.shape)
        # exit()
        # iou_cls_bin[iou_index[i,0]].append(cls[0, i])
    iou = iou.reshape(s_, s_)
    out_iou = out_iou.reshape(s_, s_)
    cls = cls.reshape(s_, s_)
    all_iou[count,:, : ] = iou
    all_cls[count,:, : ] = cls
    all_out_iou[count, :, :] = out_iou

    for i in range(s_):
      for j in range(s_):
        iou_cls_bin[iou_index[i,j]].append(cls[i,j])
        iou_out_bin[iou_index[i,j]].append(out_iou[i,j])

    count = count + 1

    # print (iou)
    # print (np.max(iou))
    # print (cls)
    # print (np.max(cls))
    # print (out_iou)
    # print (np.max(out_iou))

    # after_reshape = line + '.iou_cls.mat'
    # after_reshape = after_reshape.replace('weights', 'weights_ious')
        
        
    
    # print (iou)
    # print (out_iou)

    # iou = iou.reshape(s_, s_)
    # out_iou = out_iou.reshape(s_, s_)
    # cls = cls.reshape(s_, s_)

    # all_iou[count, :, :] = iou;
    # all_out_iou[count, :, :] =  out_iou;
    # all_cls[count, :, :] = cls;


    # all_out_iou= all_out_iou + out_iou;
    # all_cls= all_cls + cls;

    # count = count + 1

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
  # print (all_iou)
  # print (iou_cls_bin)


  ## get statistic
  cls_mean = []
  cls_std = []

  # print (iou_cls_bin[-1])
  # print (iou_cls_bin[-2])
  # arr = np.array(iou_cls_bin[-2])
  # mean_ = np.mean(arr, axis=0)
  # std_ = np.std(arr, axis=0)
  # print (mean_, std_)
  # print (len(iou_cls_bin))
  # exit()
  for cls_i, cls_bin in enumerate(iou_cls_bin):
    print (cls_i, len(cls_bin))
    if len(cls_bin) == 0:
      print (len(cls_bin))
      cls_mean.append(0)
      cls_std.append(0)
    else:
      # print(len(cls_bin))
      sum__ = sum(cls_bin)
      arr = np.array(cls_bin)
      sum_ = np.sum(arr)
      print (sum_)
      print (arr)
      # mean_ = np.mean(arr, axis=0)
      # std_ = np.std(arr, axis=0)
      mean_ = np.mean(arr,dtype=np.float64)
      std_ = np.std(arr,dtype=np.float64)
      # print (mean_, std_)
      cls_mean.append(mean_)
      cls_std.append(std_)

  out_mean = []
  out_std = []
  for out_bin in iou_out_bin:
    if len(out_bin) == 0:
      out_mean.append(0)
      out_std.append(0)
    else:
      arr = np.array(out_bin)
      sum_ = np.sum(arr)
      print (sum_)
      
      # mean_ = np.mean(arr, axis=0)
      # std_ = np.std(arr, axis=0)
      mean_ = np.mean(arr,dtype=np.float64)
      std_ = np.std(arr,dtype=np.float64)
      # print (mean_, std_)
      out_mean.append(mean_)
      out_std.append(std_)

  thres = np.array(thres)
  cls_mean = np.array(cls_mean)
  cls_std = np.array(cls_std)
  out_mean = np.array(out_mean)
  out_std = np.array(out_std)
  print (thres)
  print (cls_mean)
  print (cls_std)
  print (out_mean)
  print (out_std)

  sio.savemat(filename+'_statistic.mat', {'thres': thres, 'cls_mean':cls_mean, 'cls_std': cls_std, 'out_mean': out_mean, 'out_std': out_std})
  sio.savemat(filename+'_debug.mat', {'all_iou': all_iou, 'all_cls':all_cls, 'all_out_iou': all_out_iou})

  exit()
  # sio.savemat(filename+'_statistic.mat', {'iou': iou, 'cls':cls, 'out_iou': out_iou})



