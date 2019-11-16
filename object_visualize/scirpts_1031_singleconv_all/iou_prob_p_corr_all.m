
% load('conv.txtnms.mat')
% load('fc.txt_pairs_small.mat')
% load('conv.txt-debugnms.mat')
% figure;
% scatter(iou_before_nms, iou_after_nms);



load('fc.txt_pairs_small.mat')

iou_before_nms = all_iou;
prob_before_nms = all_cls;
% % 
load('fc.txt_pairs_medium.mat')
% % 
iou_before_nms = [iou_before_nms all_iou];
prob_before_nms = [prob_before_nms all_cls];
% % 
% % 
load('fc.txt_pairs_large.mat')
% % 
iou_before_nms = [iou_before_nms all_iou];
prob_before_nms = [prob_before_nms all_cls];
% % 
% % 
% % % 
% % % iou_before_nms = all_iou;
% % % 
% % % prob_before_nms = all_cls;


% R = corrcoef(iou_before_nms,prob_before_nms);
% R
% 
a = [iou_before_nms; prob_before_nms];
% 
% 
R = corrcoef(a');
% R = xcorr(a');
% 
R
% 
% rho
% pval

load('conv.txt_pairs_small.mat')

iou_before_nms = all_iou;
prob_before_nms = all_cls;
% % 
load('conv.txt_pairs_medium.mat')
% % 
iou_before_nms = [iou_before_nms all_iou];
prob_before_nms = [prob_before_nms all_cls];
% % 
% % 
load('conv.txt_pairs_large.mat')
% % 
iou_before_nms = [iou_before_nms all_iou];
prob_before_nms = [prob_before_nms all_cls];
% % 
% % 
% % % 
% % % iou_before_nms = all_iou;
% % % 
% % % prob_before_nms = all_cls;


% R = corrcoef(iou_before_nms,prob_before_nms);
% R
% 
a = [iou_before_nms; prob_before_nms];
% 
% 
% [R,P,RL,RU] = corrcoef(a', 'alpha', 0.05);
R = corrcoef(a');
% R = xcorr(a');
% 
R
% 
% rho

% 
% iou_before_nms = all_iou;
% prob_before_nms = all_out_iou;
% 
% 
%  
% R = corrcoef(iou_before_nms,prob_before_nms);
% R
% 


