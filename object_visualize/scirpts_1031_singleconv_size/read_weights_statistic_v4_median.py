import os,sys
import numpy as np
import scipy.io as sio
import math
import cv2


def assign_type(gt_box, scale):
  gt_box_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1]) * scale * scale
  if gt_box_area <= 32**2:
    return 1
  elif gt_box_area <=96**2:
    return 2
  else:
    return 3

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
  selected_type = int(sys.argv[2])

  with open(filename) as f:
    lines = f.readlines()
  lines = [line.strip() for line in lines]
  print (len(lines))
  num_samples = len(lines)
  # exit()

  s_ = 71

  num_bbox = s_ * s_
  # all_iou = np.zeros((num_samples, num_bbox, 1), dtype=float)
  # all_iou = all_iou.reshape(num_samples, s_, s_)
  # all_cls = np.zeros((num_samples, num_bbox, 1), dtype=float)
  # all_cls = all_cls.reshape(num_samples, s_, s_)
  # all_out_iou = np.zeros((num_samples, num_bbox, 1), dtype=float)
  # all_out_iou = all_out_iou.reshape(num_samples, s_, s_)
  # print (all_out_iou.shape)
  all_iou     = []
  all_cls     = [] 
  all_out_iou = []


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
  # iou_index = np.zeros((num_bbox, 1), dtype=int) 
  # print (iou_index.shape)
  ## init the bin index
  line  = lines[0]
  # datafile = 'weights/ + line


  # ### 100 as the gt boxes
  # datafile = 'weights_100/' + line
  # data = sio.loadmat(datafile)
  # bbox = data['bbox']

  # num_bbox = bbox.shape[0]
  # gt_index = (num_bbox-1)/2
  # gt_bbox = bbox[gt_index,:]
  # iou = np.zeros((num_bbox, 1), dtype=float)

  # for i in range(num_bbox):
  #     iou[i,0] = cal_iou(gt_bbox, bbox[i,:])
  #     # iou_index_ = int(math.ceil(iou[i,0] / 0.05))
  #     iou_index_ = int(math.floor(iou[i,0] / step * 100))
  #     # print (iou_index_)
  #     if iou_index_ == 100 / step:
  #         # print (iou_index_)
  #         iou_index_ = 100 / step - 1

  #     iou_index[i,0] = iou_index_
  #     print (i, iou_index_, iou[i,0])
  # ## 
  # iou_index = iou_index.reshape(s_, s_)
  # print (iou_index.shape)
  # # print (iou_index)
  # # exit()


  # weights_array = ['weights_075', 'weights_100', 'weights_125']

  ### 100 as the gt boxes
  iou_index = [np.zeros((num_bbox, 1), dtype=int), np.zeros((num_bbox, 1), dtype=int), np.zeros((num_bbox, 1), dtype=int)]
  datafile =  'weights_100/' + line
  # datafile =  'weights/' + line

  data = sio.loadmat(datafile)
  bbox = data['bbox']

  num_bbox = bbox.shape[0]
  gt_index = (num_bbox-1)/2
  gt_bbox = bbox[gt_index,:]


  weights_array = ['weights_075', 'weights_100', 'weights_125']
  # weights_array = ['weights']

  for counter, weight_name in enumerate(weights_array):
      # datafile = 'weights_100/' + line
      datafile =  weight_name + '/' + line

      data = sio.loadmat(datafile)
      bbox = data['bbox']

      num_bbox = bbox.shape[0]
      # gt_index = (num_bbox-1)/2
      # gt_bbox = bbox[gt_index,:]
      iou = np.zeros((num_bbox, 1), dtype=float)

      for i in range(num_bbox):
          iou[i,0] = cal_iou(gt_bbox, bbox[i,:])
          # iou_index_ = int(math.ceil(iou[i,0] / 0.05))
          iou_index_ = int(math.floor(iou[i,0] / step * 100))
          # print (iou_index_)
          if iou_index_ == 100 / step:
              # print (iou_index_)
              iou_index_ = 100 / step - 1

          iou_index[counter][i,0] = iou_index_
          print (i, iou_index_, iou[i,0])
      ## 
      iou_index[counter] = iou_index[counter].reshape(s_, s_)
      print (iou_index[counter].shape)



  ################################### 
  count = 0
  for line in lines:
    ## choose size
    # print (line)
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

    #### has to be 010
    datafile = 'weights_100/' + line
    # datafile = 'weights/' + line
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


    type_ = assign_type(gt_bbox, scale)
    if type_ != selected_type:
        continue    

    ### 


    # for weight_name in weights_array:
    for counter, weight_name in enumerate(weights_array):
        print (weight_name)
        datafile = weight_name + '/' + line
        # datafile = 'weights/' + line
        # datafile = 'weights-run1/' + line
        # data = sio.loadmat(line)
        data = sio.loadmat(datafile)
        # print (data)
        bbox = data['bbox']
        cls = data['cls']
        out_bbox = data['out_bbox']
        num_bbox = bbox.shape[0]
 
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
        # all_iou[count,:, : ] = iou
        # all_cls[count,:, : ] = cls
        # all_out_iou[count, :, :] = out_iou

        for i in range(s_):
          for j in range(s_):
            # if cls[i,j] > 0.05:
            # if cls[i,j] > 0.05:
              # iou_cls_bin[iou_index[i,j]].append(cls[i,j])
              # iou_out_bin[iou_index[i,j]].append(out_iou[i,j])
              iou_cls_bin[iou_index[counter][i,j]].append(cls[i,j])
              iou_out_bin[iou_index[counter][i,j]].append(out_iou[i,j])
              all_iou.append(iou[i, j])
              all_cls.append(cls[i, j]) 
              all_out_iou.append(out_iou[i, j])

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
      cls_min.append(0)
      cls_max.append(0)
      cls_median.append(0)
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

  out_mean = []
  out_std = []
  out_min = []
  out_max = []
  out_median = []
  for out_bin in iou_out_bin:
    if len(out_bin) == 0:
      out_mean.append(0)
      out_std.append(0)
      out_min.append(0)
      out_max.append(0)
      out_median.append(0)
    else:
      arr = np.array(out_bin)
      sum_ = np.sum(arr)
      print (sum_)
      
      # mean_ = np.mean(arr, axis=0)
      # std_ = np.std(arr, axis=0)
      mean_ = np.mean(arr,dtype=np.float64)
      std_ = np.std(arr,dtype=np.float64)
      min_ = np.min(arr)
      max_ = np.max(arr)
      median_ = np.median(arr)
      # print (mean_, std_)
      out_mean.append(mean_)
      out_std.append(std_)
      out_min.append(min_)
      out_max.append(max_)
      out_median.append(median_)

  thres = np.array(thres)
  cls_mean = np.array(cls_mean)
  cls_std = np.array(cls_std)
  cls_min = np.array(cls_min)
  cls_max = np.array(cls_max)
  cls_median = np.array(cls_median)
  out_mean = np.array(out_mean)
  out_std = np.array(out_std)
  out_min = np.array(out_min)
  out_max = np.array(out_max)
  out_median = np.array(out_median)
  print (thres)
  print (cls_mean)
  print (cls_std)
  print (out_mean)
  print (out_std)

  all_iou = np.array(all_iou)
  all_cls = np.array(all_cls)
  all_out_iou = np.array(all_out_iou)


  if selected_type == 1:
      type_string = 'small'
  if selected_type == 2:
      type_string = 'medium'
  if selected_type == 3:
      type_string = 'large'


  sio.savemat(filename+'_statistic_'+ type_string +'.mat', {'thres': thres, 'cls_min':cls_min, 'cls_max': cls_max, 'cls_median': cls_median, 'out_min':out_min, 'out_max':out_max, 'out_median': out_median, 'cls_mean':cls_mean, 'cls_std': cls_std, 'out_mean': out_mean, 'out_std': out_std})


  ## for p-corr value
  sio.savemat(filename+'_pairs_'+ type_string + '.mat', {'all_iou': all_iou, 'all_cls':all_cls, 'all_out_iou': all_out_iou})

  exit()
  # sio.savemat(filename+'_statistic.mat', {'iou': iou, 'cls':cls, 'out_iou': out_iou})



