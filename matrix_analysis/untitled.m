 load('fc6_weight.mat')
 

w = w';

% 12544x1024

r_w = rank(w);

% matlab svd

[u, s, v] = svd(w);
s_ = svd(w);

% cai deng svd

[U, S, V] = mySVD(w);

S = full(S);