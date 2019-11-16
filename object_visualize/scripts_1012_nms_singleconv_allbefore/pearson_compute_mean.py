import os,sys
import scipy.io as sio

import scipy.stats as stat 
import numpy as np

if __name__ == '__main__':
    file_name = sys.argv[1]

    data = sio.loadmat(file_name)
    print (data.keys())

    x = data['iou_before_proposal_nms']
    y = data['prob_before_nms']
    z = data['iou_before_nms']
    p = data['iou_after_nms']
    q = data['prob_after_nms']

    # pearson_res = stat.pearsonr(x, y)
    print (x.shape)
    print (y.shape)
    print (z.shape)
    print (p.shape)
    print (q.shape)
    
    x = x.reshape(-1)
    y = y.reshape(-1)
    z = z.reshape(-1)
    p = p.reshape(-1)
    q = q.reshape(-1)
    print (x.shape)
    print (y.shape)
    print (z.shape)
    print (p.shape)
    print (q.shape)
    # print (pearson_res)
    # print (pearson_res)

    print (stat.pearsonr(x, y))
    print (stat.pearsonr(z, y))
    print (stat.pearsonr(p, q))
    print (stat.pearsonr(x, z))
    print (np.mean(x), np.std(x) )
    print (np.mean(z), np.std(z) )
    print (np.mean(p), np.std(p) )

    print (np.mean(y), np.std(y) )
    print (np.mean(q), np.std(q) )

