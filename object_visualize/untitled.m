%  load('fc6_weight.mat')
 
% load('bbox_weight.mat')


% load('bbox_weight_2_10.mat')
load('bbox_weight0.mat');
% load('bbox_weight1.mat');

num_bbox = size(bbox,1);
gt_index = (num_bbox+1)/2;
gt = bbox(gt_index,:);

iou = zeros(num_bbox,1);
out_iou = zeros(num_bbox,1);

gt = gt+1;
out_bbox = out_bbox + 1;
for i = 1:num_bbox
    % i
    % bb = out_bbox(i,:);
    bb = bbox(i,:);
    iou(i,1) = bboxOverlapRatio(bb,gt);
    
    out_iou(i,1) = bboxOverlapRatio(out_bbox(i,:),gt);
end

offset = sqrt(num_bbox);
cls = reshape(cls, [offset, offset]);
iou = reshape(iou, [offset,offset]);
out_iou = reshape(out_iou, [offset, offset]);
% cls = -log(cls);

% plot_block = 1000;
D1=iou;
% D1 = C/max(max(C));


% figure
% imshow(D, 'InitialMagnification', plot_block);
% colormap(jet)


D2=cls;
% D2 = C/max(max(C));
% figure
% imshow(D, 'InitialMagnification', plot_block);
% colormap(jet)

D3=out_iou;
% D3 = C/max(max(C));

% figure
% imshow(D, 'InitialMagnification', plot_block);
% colormap(jet)

D4 = D2 .* D3;

D2_conv = D2;

D_conv = [D1,D2,D3,D4];


% load('bbox_weight0.mat');
load('bbox_weight1.mat');

num_bbox = size(bbox,1);
gt_index = (num_bbox+1)/2;
gt = bbox(gt_index,:);

iou = zeros(num_bbox,1);
out_iou = zeros(num_bbox,1);

gt = gt+1;
out_bbox = out_bbox + 1;
for i = 1:num_bbox
    % i
    % bb = out_bbox(i,:);
    bb = bbox(i,:);
    iou(i,1) = bboxOverlapRatio(bb,gt);
    
    out_iou(i,1) = bboxOverlapRatio(out_bbox(i,:),gt);
end

offset = sqrt(num_bbox);
cls = reshape(cls, [offset, offset]);
iou = reshape(iou, [offset,offset]);
out_iou = reshape(out_iou, [offset, offset]);
% cls = -log(cls);

plot_block = 1000;
D1=iou;
% D1 = C/max(max(C));


% figure
% imshow(D, 'InitialMagnification', plot_block);
% colormap(jet)


D2=cls;
% D2 = C/max(max(C));
% figure
% imshow(D, 'InitialMagnification', plot_block);
% colormap(jet)

D3=out_iou;
% D3 = C/max(max(C));

% figure
% imshow(D, 'InitialMagnification', plot_block);
% colormap(jet)


D4 = D2 .* D3;


D2_fc = D2;


D_fc = [D1,D2,D3,D4];

D = [D_conv;D_fc];

figure
imshow(D, 'InitialMagnification', plot_block);
colormap(jet);


% cls = -log(cls);

% w = reshape(w, [256,7,7,1024]);

% w = reshape(w, [256,49,1024]);
% w = permute(w, [2 1 3]);

w = reshape(w, [49,256,1024]);

% w = reshape(w, [1024,256,49]);
% w = permute(w, [3 1 2]);
% w = reshape(w, [1024,256,49]);
% w = permute(w, [3 1 2]);

% w = permute(w, [2 1 3]);
w = reshape(w, [49, 256*1024]);

w = reshape(w, [7, 7, 256*1024]);
w = permute(w, [2 1 3]);
w = reshape(w, [49, 256*1024]);
w2 = w*w';

w3 = zeros(size(w2));
%% sort the weight
for i =1:7
    for j=1:7
        for m = 1:7
            for n=1:7
                line =  (i-1) * 7 + j;
                
                current_x =  (i-1) * 7 + m;
                current_y =  (j-1) * 7 + n;
                
                elements = (m-1) * 7 + n;
%                 previous_y = (n-1) * 7 + j;
                w3(current_x, current_y) = w2(line,elements);
            end
        end
    end   
end


C=w2;
D = C/max(max(C));
figure
imshow(D, 'InitialMagnification', 1000);
colormap(jet)




C=w3;
D = C/max(max(C));
figure
imshow(D, 'InitialMagnification', 1000);
colormap(jet)

% 12544x1024


r_w = rank(w);

% matlab svd

[u, s, v] = svd(w);
s_ = svd(w);

% cai deng svd

[U, S, V] = mySVD(w);

S = full(S);

idx = logical(eye(size(S)));
value = S(idx);


figure;
% plot(value);
bar(value);