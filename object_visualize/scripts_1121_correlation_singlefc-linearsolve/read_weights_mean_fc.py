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
       
        

def cal_corr(featmap):
    # print (featmap.shape)
    featmap = featmap.reshape(featmap.shape[0],-1)
    # print (featmap.shape)

    norms = np.zeros(featmap.shape[1],)
    for i in range(featmap.shape[1]):
        norms[i] = np.linalg.norm(featmap[:, i])
    # print (norms)

    corr = np.matmul(featmap.transpose(), featmap)
    # print (corr)
    # print (corr.shape)


    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            # print (corr[i, j])
            corr[i, j] = corr[i, j]/norms[i] / norms[j]

    # print (corr)

    corr_reshape = np.zeros(corr.shape)
    for i in range(7):
        for j in range(7):
            for m in range(7):
                for n in range(7):
                    line = (i) * 7 + j
                    elements =  (m )* 7 + n 

                    current_x = (i ) * 7 + m
                    current_y = (j ) * 7 + n

                    corr_reshape[current_x, current_y] = corr[line, elements]

    # print (corr_reshape)
    return corr_reshape

    




if __name__ == '__main__':
  filename = sys.argv[1]
  with open(filename) as f:
    lines = f.readlines()
  lines = [line.strip() for line in lines]
  print (len(lines))

  ##  
  weights = sio.loadmat(sys.argv[2])
  w = weights['w']
  b = weights['b']
  print (weights.keys())
  print (w.shape)
  print (b.shape)
  b = b.reshape(-1)
  print (b.shape)
  print (filename)
  w = w.transpose()
  w = w.reshape( 256, 7, 7, 1024)
  w = w.transpose(3, 2, 1, 0)
  # w = w.reshape(1024, 7, 7, 256)
  print (w.shape)



  corr_all = np.zeros((49, 49), dtype=float)
  count = 0
  for line in lines:
    # print (line)
    class_gt = int(line.split('_')[6] )
    # print (class_gt)

    datafile = 'weights/' + line
    # datafile = 'weights-run1/' + line
    # data = sio.loadmat(line)
    data = sio.loadmat(datafile)
    # print (data)
    print (data.keys())
    print (datafile)
    print (count)

    # conv_after_roi  = data['conv_after_roi']
    # conv_before_avg = data['conv_before_avg']
    # conv_after_avg  = data['conv_after_avg']

    fc_after_roi = data['fc_after_roi']
    fc_first_ = data['fc_first_']
    fc_last_out  = data['fc_last_out']

    # print (fc_after_roi.shape)
    # print (fc_last_out.shape)

    if len(fc_after_roi.shape)  == 2:
        continue

    ## 

    for i in range(fc_after_roi.shape[0]):
        featmap = fc_after_roi[i,:,:,:]
        # print (featmap.shape)
        featmap = np.transpose(featmap, (2, 1, 0))
        # print (featmap.shape)
        featmap = np.reshape(featmap, (1, 7, 7, 256))
        # featmap = np.repeat(featmap, 1024, axis=0)
           
        # print (featmap.shape)
        # print (w.shape)
        featmap_w = np.multiply(w, featmap)
        # featmap_w = np.matmul(w.transpose(), featmap.transpose())
        print (featmap_w.shape)


        featmap_w = np.sum(featmap_w, 3)
        print (featmap_w.shape)
        featmap_w = np.sum(featmap_w, 2)
        featmap_w = np.sum(featmap_w, 1)
        featmap_w = featmap_w + b
        print (featmap_w.shape)
        
        ###  
        fc_first_i = fc_first_[i,:]

        print (fc_first_i.shape)
        print (featmap_w)
        print (fc_first_i)
        exit()
        corr_map = cal_corr(featmap_w)
        count += 1
        corr_all += corr_map
        # print (corr_map)
        # exit()

    
    ###### 
    # conv_after_avg = data['conv_after_avg']
    # conv_after_avg = data['conv_after_avg']

  print (corr_all/count)
  corr_avg  = corr_all / count
  sio.savemat(filename+'_corr.mat', {'corr_avg': corr_avg} )
  exit()


