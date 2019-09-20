


figure


load('conv.txt_statistic.mat')
errorbar(thres,cls_mean, cls_std)
hold on;
errorbar(thres,out_mean, out_std)

load('fc.txt_statistic.mat')
errorbar(thres,cls_mean, cls_std)
errorbar(thres,out_mean, out_std)