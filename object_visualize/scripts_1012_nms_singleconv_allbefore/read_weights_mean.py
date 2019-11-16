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
  correct = 0

  iou_before_nms = []
  iou_before_proposal_nms = []
  iou_after_nms = []
  prob_before_nms = []
  prob_after_nms = []
  for line in lines:
    # print (line)
    class_gt = int(line.split('_')[6] )
    # print (class_gt)

    datafile = 'weights/' + line
    # datafile = 'weights-run1/' + line
    # data = sio.loadmat(line)
    data = sio.loadmat(datafile)
    # print (data)
    
    iou_before_max = data['iou_before']
    iou_after = data['iou_after']

    prob_before_iou_max = data['prob_before']
    prob_after_max = data['prob_after']

    iou_before_proposal = data['iou_before_proposal']

    if iou_after < -0.5:
        continue


    print (iou_before_max.shape)
    print (prob_before_iou_max.shape)

    # if prob_after_max < prob_before_iou_max:
    #     print(prob_before_iou_max, prob_after_max)
    #     print(iou_before_max, iou_after)

    print(prob_before_iou_max, prob_after_max)
    print(iou_before_max, iou_after)

    iou_before_max = iou_before_max.reshape(-1)
    iou_before_proposal = iou_before_proposal.reshape(-1)
    prob_before_iou_max = prob_before_iou_max.reshape(-1)
    # iou_after = iou_after.reshape(-1)
    # prob_after_max = prob_after_max.reshape(-1)
    print (iou_before_max.shape)
    # exit()

    
    # iou_before_nms.append(iou_before_max)
    iou_before_nms.extend(iou_before_max)
    iou_before_proposal_nms.extend(iou_before_proposal)
    iou_after_nms.append(iou_after)
    prob_before_nms.extend(prob_before_iou_max)
    prob_after_nms.append(prob_after_max)

  iou_before_nms = np.array(iou_before_nms)
  # iou_before_nms = iou_before_nms.reshape(-1,1)
  iou_after_nms = np.array(iou_after_nms)
  prob_before_nms = np.array(prob_before_nms)
  # prob_before_nms = prob_before_nms.reshape(-1,1)
  prob_after_nms = np.array(prob_after_nms)
  iou_before_proposal_nms = np.array(iou_before_proposal_nms)
  print (iou_before_nms.shape)
  print (prob_before_nms.shape)
  print (iou_after_nms.shape)
  print (prob_after_nms.shape)
  print (iou_before_proposal_nms.shape)

  sio.savemat(filename+'nms.mat', {'iou_before_nms': iou_before_nms,  \
                                   'iou_after_nms': iou_after_nms,  \
                                   'prob_before_nms': prob_before_nms,  \
                                   'prob_after_nms': prob_after_nms,  \
                                   'iou_before_proposal_nms': iou_before_proposal_nms,  \
                                    }) 
    

