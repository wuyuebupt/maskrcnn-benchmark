import scipy.io as sio
import sys, os
import numpy as np

if __name__ == '__main__':
    file_name = sys.argv[1]
    aps = sio.loadmat(file_name)['aps'].reshape(-1)

    print (aps.shape)
 
    print (aps)
    sorted_index = [i[0] for i in sorted(np.ndenumerate(aps), key=lambda x:x[1], reverse=True)]
    sorted_aps = [i[1] for i in sorted(np.ndenumerate(aps), key=lambda x:x[1], reverse=True)]
    # sorted_list = sorted(np.ndenumerate(aps), key=lambda x:x[1], reverse=True)

    # for ele in sorted_list:
    #     print (ele)

    # sorted_aps = np.array(sorted_list)
    sorted_index = np.array(sorted_index)
    sorted_aps = np.array(sorted_aps)
    sorted_index = sorted_index.reshape(-1)
    sorted_aps = sorted_aps.reshape(-1)
    print (sorted_aps)
    print (sorted_index)


    # sio.savemat('sorted_class_index.mat', {'sorted_aps': sorted_aps, 'sorted_index':sorted_index})
    print (sorted_index[:27].shape)
    print (sorted_index[27:27+27].shape)
    print (sorted_index[27+27:].shape)
    print (sorted_index[:27])
    print (sorted_index[27:27+27])
    print (sorted_index[27+27:])
    

