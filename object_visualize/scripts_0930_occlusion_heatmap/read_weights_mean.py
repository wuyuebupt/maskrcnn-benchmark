import os,sys
import math
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

  count = 0
  correct = 0

  step = 5 #  must be devided by 100
  start = 0
  end = 100


  ## generate the list of list
  occlusion_bin = []
  thres = []
  # for i in range(start, end+step, step):
  for i in range(start, end, step):
    occlusion_bin.append([])
    thres.append(i/100.0)

  
  occlusions = []
  for line in lines:
    # print (line)
    class_gt = int(line.split('_')[6] )
    # print (class_gt)

    datafile = 'weights/' + line
    # datafile = 'weights-run1/' + line
    # data = sio.loadmat(line)
    data = sio.loadmat(datafile)
    # print (data)
    occlusion = data['occlusion']   
    # print (occlusion)

    occlusion_index_ = int(math.floor(occlusion/ step * 100))
    occlusion_bin[occlusion_index_].append(occlusion)


    occlusions.append(occlusion)
  occlusions = np.array(occlusions)
  arr = occlusions/0.05
  arr = arr.astype(int)
  arr = arr.reshape(-1)
  print (arr.shape)
  arr = np.sort(arr)

  occlusions_count = np.bincount(arr, minlength=20)
  print (occlusions_count)

  print (occlusions.shape)

  sio.savemat(filename+'ave.mat', {'occlusions': occlusions, 'thres': thres, 'occlusions_count': occlusions_count})
    
    # exit()
  
