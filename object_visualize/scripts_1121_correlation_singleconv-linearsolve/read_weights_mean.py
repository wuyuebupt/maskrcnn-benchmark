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

  corr_all = np.zeros((49, 49), dtype=float)

  ## 
  input_feat = []
  output_feat = []

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
    # print (data.keys())
    print (datafile)
    print (count)

    conv_after_roi  = data['conv_after_roi']
    conv_before_avg = data['conv_before_avg']
    conv_after_avg  = data['conv_after_avg']
    # print (conv_after_roi.shape)
    # print (conv_before_avg.shape)
    # print (conv_after_avg.shape)
    # print (conv_after_roi.shape)
    # print (conv_before_avg.shape)
    # print (conv_before_avg.shape)
    # print (len(conv_before_avg.shape))
    if len(conv_before_avg.shape)  == 2:
        continue

    ## 
    # conv_before_avg = conv_before_avg.reshape(conv_before_avg.shape[0],)

    # for i in range(conv_before_avg.shape[0]):
    for i in range(conv_after_roi.shape[0]):
        featmap = conv_after_roi[i, :, :, :]
        print (featmap.shape)
        featmap = featmap.reshape(-1)
        print (featmap.shape)

        out = conv_after_avg[i]
        out = out.reshape(-1)
        print (out.shape)
      
        # corr_map = cal_corr(featmap)
        count += 1
        # corr_all += corr_map
        # print (corr_map)
        input_feat.append(featmap)
        output_feat.append(out)

        ## 

   
  ## inside the loop
  print (len(input_feat))
  input_feat = np.asarray(input_feat)
  output_feat = np.asarray(output_feat)
  print (input_feat.shape)
  print (output_feat.shape)
  matrix = np.dot(np.linalg.pinv(input_feat), output_feat)
  print (matrix.shape)

  matrix = matrix.reshape(256, 7, 7, 1024)
  print (matrix.shape)
  matrix = matrix.transpose(1,2,0,3)
  print (matrix.shape)
  matrix = matrix.reshape(7, 7, 256*1024)
  print (matrix.shape)
  matrix = matrix.transpose(2,1,0)
  print (matrix.shape)

  corr_map = cal_corr(matrix)
  print (corr_map)
    

 
  # print (corr_all/count)
  corr_avg  = corr_all / count
  # sio.savemat(filename+'_corr.mat', {'corr_avg': corr_avg} )
  sio.savemat(filename+'_corr.mat', {'corr_map': corr_map} )
  exit()


