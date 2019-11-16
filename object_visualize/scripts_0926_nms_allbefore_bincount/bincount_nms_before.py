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

  results = sio.loadmat(filename)
  print (results.keys())
  iou_before_nms = results['iou_before_nms'].reshape(-1)
  iou_after_nms = results['iou_after_nms'].reshape(-1)
  prob_before_nms = results['prob_before_nms'].reshape(-1)
  prob_after_nms = results['prob_after_nms'].reshape(-1)
  print (iou_before_nms.shape)
  print (iou_after_nms.shape)
  print (prob_before_nms.shape)
  print (prob_after_nms.shape)


  step = 5 #  must be devided by 100
  start = 0
  end = 100
  
  
  ## generate the list of list
  iou_cls_bin = []
  thres = []
  # for i in range(start, end+step, step):
  for i in range(start, end, step):
    iou_cls_bin.append([])
    thres.append(i/100.0)
  print (len(iou_cls_bin))
  print (thres)


  #### go through after nms iou and score
  count = 0

  # for i in range(iou_after_nms.shape[0]):
  for i in range(iou_before_nms.shape[0]):
    # iou_index = int(math.floor(iou_after_nms[i]/ (1.0/ end * step) ))
    iou_index = int(math.floor(iou_before_nms[i]/ (1.0/ end * step) ))
    if iou_index == end / step:
        iou_index = end / step - 1
    print (iou_index)


    iou_cls_bin[iou_index].append(prob_before_nms[i])
    # iou_cls_bin[iou_index].append(prob_after_nms[i])

    count = count + 1

  print (count)
  # print (all_iou)
  # print (iou_cls_bin)


  ## get statistic
  cls_mean = []
  cls_std = []
  cls_min = []
  cls_max = []
  cls_median = []
  cls_bincount = []

  for cls_i, cls_bin in enumerate(iou_cls_bin):
    print (cls_i, len(cls_bin))
    if len(cls_bin) == 0:
      print (len(cls_bin))
      cls_mean.append(0)
      cls_std.append(0)
      cls_min.append(0)
      cls_max.append(0)
      cls_median.append(0)
      cls_bincount.append(np.zeros((20,), dtype=int))
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
      min_ = np.min(arr)
      max_ = np.max(arr)
      median_ = np.median(arr)

      # print (mean_, std_)
      cls_mean.append(mean_)
      cls_std.append(std_)
      cls_min.append(min_)
      cls_max.append(max_)
      cls_median.append(median_)

      arr = arr / 0.05
      arr = arr.astype(int)
      arr = np.sort(arr)
      # print (arr)
      bin_count = np.bincount(arr, minlength=20)
      # print (bin_count)
      print (bin_count.shape)
      # exit()
 
      cls_bincount.append(bin_count)


  thres = np.array(thres)
  cls_mean = np.array(cls_mean)
  cls_std = np.array(cls_std)
  cls_min = np.array(cls_min)
  cls_max = np.array(cls_max)
  cls_median = np.array(cls_median)
  cls_bincount = np.array(cls_bincount)
  
  print (thres)
  print (cls_mean)
  print (cls_std)

  sio.savemat(filename+'_statistic_beforenms.mat', {'thres': thres, 'cls_min':cls_min, 'cls_max': cls_max, 'cls_median': cls_median, 'cls_mean':cls_mean, 'cls_std': cls_std, 'cls_bincount':cls_bincount})
  # sio.savemat(filename+'_statistic.mat', {'thres': thres, 'cls_min':cls_min, 'cls_max': cls_max, 'cls_median': cls_median, 'out_min':out_min, 'out_max':out_max, 'out_median': out_median, 'cls_mean':cls_mean, 'cls_std': cls_std, 'out_mean': out_mean, 'out_std': out_std, 'out_bincount': out_bincount, 'cls_bincount':cls_bincount})
  # sio.savemat(filename+'_debug.mat', {'all_iou': all_iou, 'all_cls':all_cls, 'all_out_iou': all_out_iou})

  exit()
  # sio.savemat(filename+'_statistic.mat', {'iou': iou, 'cls':cls, 'out_iou': out_iou})



