import os,sys
import scipy.io as sio

import scipy.stats as stat 

if __name__ == '__main__':
    file_name = sys.argv[1]

    data = sio.loadmat(file_name)
    print (data.keys())

    x = data['all_iou']
    y = data['all_cls']
    z = data['all_out_iou']

    # pearson_res = stat.pearsonr(x, y)
    print (x.shape)
    print (y.shape)
    print (z.shape)
    
    x = x.reshape(-1)
    y = y.reshape(-1)
    z = z.reshape(-1)
    print (x.shape)
    print (y.shape)
    print (z.shape)
    # print (pearson_res)
    # print (pearson_res)

    print (stat.pearsonr(x, y))
    # print (stat.pearsonr(x, z))
    # print (stat.pearsonr(y, z))

