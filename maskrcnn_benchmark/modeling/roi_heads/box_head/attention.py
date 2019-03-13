import torch
from torch import nn
from torch.nn import functional as F


class ListModule(nn.Module):
    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)


# class _Attention_Module(nn.Module):
#     def forward(self, x, y):
#         '''
#         : input:
#         :  x
#         :  y
#         : output:
#         :  new x
#         '''
###### v1 0312,  
### ### group non local
### class _NonLocalBlockND_Cross(nn.Module):
###     def __init__(self, in_channels, num_group, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True, relu_layer=True, use_softmax=True,  use_cross=False):
###         super(_NonLocalBlockND_Cross, self).__init__()
### 
###         assert dimension in [1, 2, 3]
###         assert dimension == 2
###         assert num_group in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
### 
###         self.dimension = dimension
###         self.sub_sample = sub_sample
### 
###         self.in_channels = in_channels
###         self.inter_channels = inter_channels
###         self.num_group = num_group
### 
###         if self.inter_channels is None:
###             self.inter_channels = in_channels // 2
###             if self.inter_channels == 0:
###                 self.inter_channels = 1
### 
###         ## inner channels are divided by num of groups
###         conv_nd = nn.Conv2d
###         max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
###         bn = nn.BatchNorm2d
###         relu = nn.ReLU
###         self.relu_layer = relu_layer
### 
###         self.use_softmax = use_softmax
### 
###         if self.use_softmax:
###             self.softmax = nn.Softmax(dim=2)
###      
###         ## for different input
###         self.use_cross = use_cross
### 
###         assert self.num_group <= self.inter_channels
### 
###         self.inter_channels_group = self.inter_channels // self.num_group
###         print (self.inter_channels_group)
### 
###         ## for cls 
###         g_cls     = []
###         theta_cls = []
###         phi_cls   = []
###         for i in range(self.num_group):
###             g_cls.append(conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels_group,
###                     kernel_size=1, stride=1, padding=0))
###             theta_cls.append(conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels_group,
###                     kernel_size=1, stride=1, padding=0))
###             phi_cls.append(conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels_group,
###                     kernel_size=1, stride=1, padding=0))
### 
###         self.g_cls     = ListModule(*g_cls)
###         self.theta_cls = ListModule(*theta_cls)
###         self.phi_cls   = ListModule(*phi_cls)
### 
###         ## for reg
###         g_reg     = []
###         theta_reg = []
###         phi_reg   = []
###         for i in range(self.num_group):
###             g_reg.append(conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels_group,
###                     kernel_size=1, stride=1, padding=0))
###             theta_reg.append(conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels_group,
###                     kernel_size=1, stride=1, padding=0))
###             phi_reg.append(conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels_group,
###                     kernel_size=1, stride=1, padding=0))
### 
###         self.g_reg     = ListModule(*g_reg)
###         self.theta_reg = ListModule(*theta_reg)
###         self.phi_reg   = ListModule(*phi_reg)
### 
### 
###         ## old
###         # g = []
###         # theta = []
###         # phi = []
###         # for i in range(self.num_group):
###         #     g.append(conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels_group,
###         #             kernel_size=1, stride=1, padding=0))
###         #     theta.append(conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels_group,
###         #             kernel_size=1, stride=1, padding=0))
###         #     phi.append(conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels_group,
###         #             kernel_size=1, stride=1, padding=0))
### 
###         # self.g = ListModule(*g)
###         # self.theta = ListModule(*theta)
###         # self.phi = ListModule(*phi)
### 
###         # print (self.g)
### 
###         ## oldold
###         # self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
###         #                  kernel_size=1, stride=1, padding=0)
### 
###         # self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
###         #                      kernel_size=1, stride=1, padding=0)
### 
###         # self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
###         #                    kernel_size=1, stride=1, padding=0)
### 
###         assert sub_sample==False
###         ## not used and not modified
###         if sub_sample:
###             self.g = nn.Sequential(self.g, max_pool_layer)
###             self.phi = nn.Sequential(self.phi, max_pool_layer)
### 
### 
### 
###         ## v2 
###         self.W_cls = nn.Sequential(conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
###                          kernel_size=1, stride=1, padding=0))
###         self.W_reg = nn.Sequential(conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
###                          kernel_size=1, stride=1, padding=0))
###         # print (self.W_cls)
###         # print (self.W_reg)
###         ## BN first then RELU
###         if bn_layer:
###             self.W_cls.add_module(
###                 'bn', bn(self.in_channels)
###             )
###             self.W_reg.add_module(
###                 'bn', bn(self.in_channels)
###             )
### 
###         print (self.W_cls)
###         print (self.W_reg)
###         # if relu_layer:
###         #     self.W.add_module( 
###         #         'relu', relu(inplace=True)
###         #     )
### 
###         ## init the weights
###         nn.init.constant_(self.W_cls[0].weight, 0)
###         nn.init.constant_(self.W_cls[0].bias, 0)
###         nn.init.constant_(self.W_reg[0].weight, 0)
###         nn.init.constant_(self.W_reg[0].bias, 0)
###         if bn_layer:
###             nn.init.constant_(self.W_cls[1].weight, 0)
###             nn.init.constant_(self.W_cls[1].bias, 0)
###             nn.init.constant_(self.W_reg[1].weight, 0)
###             nn.init.constant_(self.W_reg[1].bias, 0)
### 
###         #if relu_layer or bn_layer:
###         #    nn.init.constant_(self.W[1].weight, 0)
###         #    nn.init.constant_(self.W[1].bias, 0)
###         #else:
###         #    nn.init.constant_(self.W.weight, 0)
###         #    nn.init.constant_(self.W.bias, 0)
###         
###         ## v1
###         # if bn_layer & relu_layer:
###         #     self.W = nn.Sequential(
###         #         conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
###         #                 kernel_size=1, stride=1, padding=0),
###         #         relu(inplace=True), 
###         #         bn(self.in_channels)
###         #     )
###         #     nn.init.constant_(self.W[1].weight, 0)
###         #     nn.init.constant_(self.W[1].bias, 0)
###         # elif bn_layer & not relu_layer:
###         #     self.W = nn.Sequential(
###         #         conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
###         #                 kernel_size=1, stride=1, padding=0),
###         #         bn(self.in_channels)
###         #     )
###         #     nn.init.constant_(self.W[1].weight, 0)
###         #     nn.init.constant_(self.W[1].bias, 0)
###         #  elif not bn_layer & relu_layer:
###         #     self.W = nn.Sequential(
###         #         conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
###         #                 kernel_size=1, stride=1, padding=0),
###         #         relu(inplace=True)
###         #     )
###         #     nn.init.constant_(self.W[1].weight, 0)
###         #     nn.init.constant_(self.W[1].bias, 0)
###         # else:
###         #     self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
###         #                      kernel_size=1, stride=1, padding=0)
###         #     nn.init.constant_(self.W.weight, 0)
###         #     nn.init.constant_(self.W.bias, 0)
### 
### 
###     # def forward(self, x):
###     def forward(self, x_cls, x_reg):
###         '''
###         :param x_cls: (b, c, t, h, w)
###         :param x_reg: (b, c, t, h, w)
###         :return:
###         '''
###         # print (x.shape)
###         batch_size = x_cls.size(0)
###         # print (batch_size) 
###         # print (x_cls.shape)
###         # print (x_reg.shape)
### 
### 
###         if self.use_cross == False:
###             ## self-attention
###             #### cls 
###             y_cls_group = []
### 
###             for i in range(self.num_group):
###                 
###                 g_x_cls = self.g_cls[i](x_cls).view(batch_size, self.inter_channels_group, -1)
###                 g_x_cls = g_x_cls.permute(0, 2, 1)
### 
###                 theta_x_cls = self.theta_cls[i](x_cls).view(batch_size, self.inter_channels_group, -1)
###                 theta_x_cls = theta_x_cls.permute(0, 2, 1)
###                 phi_x_cls = self.phi_cls[i](x_cls).view(batch_size, self.inter_channels_group, -1)
### 
###                 f = torch.matmul(theta_x_cls, phi_x_cls)
### 
###                 if self.use_softmax == True:
###                     f_div_C = self.softmax(f)
###                 else:
###                     N = f.size(-1)
###                     f_div_C = f / N
### 
###                 y = torch.matmul(f_div_C, g_x_cls)
###                 y = y.permute(0, 2, 1).contiguous()
###                 y_cls_group.append(y)
### 
###             y_cls_out = torch.cat(y_cls_group, dim=1)
### 
###             y_cls_out = y_cls_out.view(batch_size, self.inter_channels, *x_cls.size()[2:])
###             W_y_cls = self.W_cls(y_cls_out)
###             z_cls = W_y_cls + x_cls
### 
###             ## relu after residual
###             if self.relu_layer:
###                 z_cls = F.relu_(z_cls)
### 
###             #### reg
###             y_reg_group = []
### 
###             for i in range(self.num_group):
###                 
###                 g_x_reg = self.g_reg[i](x_reg).view(batch_size, self.inter_channels_group, -1)
###                 g_x_reg = g_x_reg.permute(0, 2, 1)
### 
###                 theta_x_reg = self.theta_reg[i](x_reg).view(batch_size, self.inter_channels_group, -1)
###                 theta_x_reg = theta_x_reg.permute(0, 2, 1)
###                 phi_x_reg = self.phi_reg[i](x_reg).view(batch_size, self.inter_channels_group, -1)
### 
###                 f = torch.matmul(theta_x_reg, phi_x_reg)
### 
###                 if self.use_softmax == True:
###                     f_div_C = self.softmax(f)
###                 else:
###                     N = f.size(-1)
###                     f_div_C = f / N
### 
###                 y = torch.matmul(f_div_C, g_x_reg)
###                 y = y.permute(0, 2, 1).contiguous()
###                 y_reg_group.append(y)
### 
###             y_reg_out = torch.cat(y_reg_group, dim=1)
### 
###             y_reg_out = y_reg_out.view(batch_size, self.inter_channels, *x_reg.size()[2:])
###             W_y_reg = self.W_reg(y_reg_out)
###             z_reg = W_y_reg + x_reg
### 
###             ## relu after residual
###             if self.relu_layer:
###                 z_reg = F.relu_(z_reg)
### 
###             return z_cls, z_reg
###         elif self.use_cross == True:
###             ## cross-attention
###             #### cls 
###             y_cls_group = []
### 
###             for i in range(self.num_group):
###                 
###                 ## g and phi are from encoder, that is reg as input
###                 g_x_cls = self.g_cls[i](x_reg).view(batch_size, self.inter_channels_group, -1)
###                 g_x_cls = g_x_cls.permute(0, 2, 1)
### 
###                 theta_x_cls = self.theta_cls[i](x_cls).view(batch_size, self.inter_channels_group, -1)
###                 theta_x_cls = theta_x_cls.permute(0, 2, 1)
###                 phi_x_cls = self.phi_cls[i](x_reg).view(batch_size, self.inter_channels_group, -1)
### 
###                 f = torch.matmul(theta_x_cls, phi_x_cls)
### 
###                 if self.use_softmax == True:
###                     f_div_C = self.softmax(f)
###                 else:
###                     N = f.size(-1)
###                     f_div_C = f / N
### 
###                 y = torch.matmul(f_div_C, g_x_cls)
###                 y = y.permute(0, 2, 1).contiguous()
###                 y_cls_group.append(y)
### 
###             y_cls_out = torch.cat(y_cls_group, dim=1)
### 
###             y_cls_out = y_cls_out.view(batch_size, self.inter_channels, *x_cls.size()[2:])
###             W_y_cls = self.W_cls(y_cls_out)
###             z_cls = W_y_cls + x_cls
### 
###             ## relu after residual
###             if self.relu_layer:
###                 z_cls = F.relu_(z_cls)
### 
###             #### reg
###             y_reg_group = []
### 
###             for i in range(self.num_group):
###                 
###                 ## g and phi are from encoder, that is cls as input
###                 g_x_reg = self.g_reg[i](x_cls).view(batch_size, self.inter_channels_group, -1)
###                 g_x_reg = g_x_reg.permute(0, 2, 1)
### 
###                 theta_x_reg = self.theta_reg[i](x_reg).view(batch_size, self.inter_channels_group, -1)
###                 theta_x_reg = theta_x_reg.permute(0, 2, 1)
###                 phi_x_reg = self.phi_reg[i](x_cls).view(batch_size, self.inter_channels_group, -1)
### 
###                 f = torch.matmul(theta_x_reg, phi_x_reg)
### 
###                 if self.use_softmax == True:
###                     f_div_C = self.softmax(f)
###                 else:
###                     N = f.size(-1)
###                     f_div_C = f / N
### 
###                 y = torch.matmul(f_div_C, g_x_reg)
###                 y = y.permute(0, 2, 1).contiguous()
###                 y_reg_group.append(y)
### 
###             y_reg_out = torch.cat(y_reg_group, dim=1)
### 
###             y_reg_out = y_reg_out.view(batch_size, self.inter_channels, *x_reg.size()[2:])
###             W_y_reg = self.W_reg(y_reg_out)
###             z_reg = W_y_reg + x_reg
### 
###             ## relu after residual
###             if self.relu_layer:
###                 z_reg = F.relu_(z_reg)
### 
###             return z_cls, z_reg
###         else:
###             assert (False)
### 
### class NONLocalBlock2D_Cross(_NonLocalBlockND_Cross):
###     def __init__(self, in_channels, num_group=1, inter_channels=None, sub_sample=True, bn_layer=True, relu_layer=True, use_softmax=True,  use_cross=True):
###         super(NONLocalBlock2D_Cross, self).__init__(in_channels,
###                                               num_group=num_group,
###                                               inter_channels=inter_channels,
###                                               dimension=2, sub_sample=sub_sample,
###                                               bn_layer=bn_layer, relu_layer=relu_layer, use_softmax=use_softmax, 
###                                               use_cross=use_cross)
### 



### group non local
### find out that it can use _NonLocalBlockND_Group by modifying the forward function 
class _NonLocalBlockND_Cross(nn.Module):
    def __init__(self, in_channels, num_group, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True, relu_layer=True, use_softmax=True,  mode_code=0):
        super(_NonLocalBlockND_Cross, self).__init__()


        assert mode_code in [0, 1, 2, 3]
        self.mode_code = mode_code
        # assert dimension in [1, 2, 3]
        # assert dimension == 2
        # assert num_group in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

        # self.dimension = dimension
        # self.sub_sample = sub_sample

        # self.in_channels = in_channels
        # self.inter_channels = inter_channels
        # self.num_group = num_group

        # if self.inter_channels is None:
        #     self.inter_channels = in_channels // 2
        #     if self.inter_channels == 0:
        #         self.inter_channels = 1

        # ## inner channels are divided by num of groups
        # conv_nd = nn.Conv2d
        # max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
        # bn = nn.BatchNorm2d
        # relu = nn.ReLU
        # self.relu_layer = relu_layer

        # self.use_softmax = use_softmax

        # if self.use_softmax:
        #     self.softmax = nn.Softmax(dim=2)
     
        ## for different input
        # assert self.num_group <= self.inter_channels

        # self.inter_channels_group = self.inter_channels // self.num_group
        # print (self.inter_channels_group)

        self.cls_nonlocal = _NonLocalBlockND_Group(in_channels,
                                              num_group=num_group,
                                              inter_channels=inter_channels,
                                              dimension=2, sub_sample=sub_sample,
                                              bn_layer=bn_layer, relu_layer=relu_layer, use_softmax=use_softmax)

        self.reg_nonlocal = _NonLocalBlockND_Group(in_channels,
                                              num_group=num_group,
                                              inter_channels=inter_channels,
                                              dimension=2, sub_sample=sub_sample,
                                              bn_layer=bn_layer, relu_layer=relu_layer, use_softmax=use_softmax)


        ## for cls 
        # g_cls     = []
        # theta_cls = []
        # phi_cls   = []
        # for i in range(self.num_group):
        #     g_cls.append(conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels_group,
        #             kernel_size=1, stride=1, padding=0))
        #     theta_cls.append(conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels_group,
        #             kernel_size=1, stride=1, padding=0))
        #     phi_cls.append(conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels_group,
        #             kernel_size=1, stride=1, padding=0))

        # self.g_cls     = ListModule(*g_cls)
        # self.theta_cls = ListModule(*theta_cls)
        # self.phi_cls   = ListModule(*phi_cls)

        # ## for reg
        # g_reg     = []
        # theta_reg = []
        # phi_reg   = []
        # for i in range(self.num_group):
        #     g_reg.append(conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels_group,
        #             kernel_size=1, stride=1, padding=0))
        #     theta_reg.append(conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels_group,
        #             kernel_size=1, stride=1, padding=0))
        #     phi_reg.append(conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels_group,
        #             kernel_size=1, stride=1, padding=0))

        # self.g_reg     = ListModule(*g_reg)
        # self.theta_reg = ListModule(*theta_reg)
        # self.phi_reg   = ListModule(*phi_reg)


        ## old
        # g = []
        # theta = []
        # phi = []
        # for i in range(self.num_group):
        #     g.append(conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels_group,
        #             kernel_size=1, stride=1, padding=0))
        #     theta.append(conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels_group,
        #             kernel_size=1, stride=1, padding=0))
        #     phi.append(conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels_group,
        #             kernel_size=1, stride=1, padding=0))

        # self.g = ListModule(*g)
        # self.theta = ListModule(*theta)
        # self.phi = ListModule(*phi)

        # print (self.g)

        ## oldold
        # self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
        #                  kernel_size=1, stride=1, padding=0)

        # self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
        #                      kernel_size=1, stride=1, padding=0)

        # self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
        #                    kernel_size=1, stride=1, padding=0)

        # assert sub_sample==False
        # ## not used and not modified
        # if sub_sample:
        #     self.g = nn.Sequential(self.g, max_pool_layer)
        #     self.phi = nn.Sequential(self.phi, max_pool_layer)



        # ## v2 
        # self.W_cls = nn.Sequential(conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
        #                  kernel_size=1, stride=1, padding=0))
        # self.W_reg = nn.Sequential(conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
        #                  kernel_size=1, stride=1, padding=0))
        # # print (self.W_cls)
        # # print (self.W_reg)
        # ## BN first then RELU
        # if bn_layer:
        #     self.W_cls.add_module(
        #         'bn', bn(self.in_channels)
        #     )
        #     self.W_reg.add_module(
        #         'bn', bn(self.in_channels)
        #     )

        # print (self.W_cls)
        # print (self.W_reg)
        # # if relu_layer:
        # #     self.W.add_module( 
        # #         'relu', relu(inplace=True)
        # #     )

        # ## init the weights
        # nn.init.constant_(self.W_cls[0].weight, 0)
        # nn.init.constant_(self.W_cls[0].bias, 0)
        # nn.init.constant_(self.W_reg[0].weight, 0)
        # nn.init.constant_(self.W_reg[0].bias, 0)
        # if bn_layer:
        #     nn.init.constant_(self.W_cls[1].weight, 0)
        #     nn.init.constant_(self.W_cls[1].bias, 0)
        #     nn.init.constant_(self.W_reg[1].weight, 0)
        #     nn.init.constant_(self.W_reg[1].bias, 0)

        #if relu_layer or bn_layer:
        #    nn.init.constant_(self.W[1].weight, 0)
        #    nn.init.constant_(self.W[1].bias, 0)
        #else:
        #    nn.init.constant_(self.W.weight, 0)
        #    nn.init.constant_(self.W.bias, 0)
        
        ## v1
        # if bn_layer & relu_layer:
        #     self.W = nn.Sequential(
        #         conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
        #                 kernel_size=1, stride=1, padding=0),
        #         relu(inplace=True), 
        #         bn(self.in_channels)
        #     )
        #     nn.init.constant_(self.W[1].weight, 0)
        #     nn.init.constant_(self.W[1].bias, 0)
        # elif bn_layer & not relu_layer:
        #     self.W = nn.Sequential(
        #         conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
        #                 kernel_size=1, stride=1, padding=0),
        #         bn(self.in_channels)
        #     )
        #     nn.init.constant_(self.W[1].weight, 0)
        #     nn.init.constant_(self.W[1].bias, 0)
        #  elif not bn_layer & relu_layer:
        #     self.W = nn.Sequential(
        #         conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
        #                 kernel_size=1, stride=1, padding=0),
        #         relu(inplace=True)
        #     )
        #     nn.init.constant_(self.W[1].weight, 0)
        #     nn.init.constant_(self.W[1].bias, 0)
        # else:
        #     self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
        #                      kernel_size=1, stride=1, padding=0)
        #     nn.init.constant_(self.W.weight, 0)
        #     nn.init.constant_(self.W.bias, 0)


    # def forward(self, x):
    def forward(self, x_cls, x_reg):
        '''
        :param x_cls: (b, c, t, h, w)
        :param x_reg: (b, c, t, h, w)
        :return:
        '''
        # print (x.shape)
        # batch_size = x_cls.size(0)
        # print (batch_size) 
        # print (x_cls.shape)
        # print (x_reg.shape)
        if self.mode_code == 0:
            # x <- x, x
            # y <- y, y
            x_cls = self.cls_nonlocal(x_cls, x_cls) 
            x_reg = self.reg_nonlocal(x_reg, x_reg) 
        elif self.mode_code == 1:
            # x <- x, y 
            # y <- y, y
            x_cls = self.cls_nonlocal(x_cls, x_reg) 
            x_reg = self.reg_nonlocal(x_reg, x_reg) 
        elif self.mode_code == 2:
            # x <- x, x
            # y <- y, x
            x_cls = self.cls_nonlocal(x_cls, x_cls) 
            x_reg = self.reg_nonlocal(x_reg, x_cls) 
        elif self.mode_code == 3:
            # x <- x, y
            # y <- y, x
            x_cls = self.cls_nonlocal(x_cls, x_reg) 
            x_reg = self.reg_nonlocal(x_reg, x_cls) 
        else:
            assert False

        return x_cls, x_reg

        # if self.use_cross == False:
        #     ## self-attention
        #     #### cls 
        #     y_cls_group = []

        #     for i in range(self.num_group):
        #         
        #         g_x_cls = self.g_cls[i](x_cls).view(batch_size, self.inter_channels_group, -1)
        #         g_x_cls = g_x_cls.permute(0, 2, 1)

        #         theta_x_cls = self.theta_cls[i](x_cls).view(batch_size, self.inter_channels_group, -1)
        #         theta_x_cls = theta_x_cls.permute(0, 2, 1)
        #         phi_x_cls = self.phi_cls[i](x_cls).view(batch_size, self.inter_channels_group, -1)

        #         f = torch.matmul(theta_x_cls, phi_x_cls)

        #         if self.use_softmax == True:
        #             f_div_C = self.softmax(f)
        #         else:
        #             N = f.size(-1)
        #             f_div_C = f / N

        #         y = torch.matmul(f_div_C, g_x_cls)
        #         y = y.permute(0, 2, 1).contiguous()
        #         y_cls_group.append(y)

        #     y_cls_out = torch.cat(y_cls_group, dim=1)

        #     y_cls_out = y_cls_out.view(batch_size, self.inter_channels, *x_cls.size()[2:])
        #     W_y_cls = self.W_cls(y_cls_out)
        #     z_cls = W_y_cls + x_cls

        #     ## relu after residual
        #     if self.relu_layer:
        #         z_cls = F.relu_(z_cls)

        #     #### reg
        #     y_reg_group = []

        #     for i in range(self.num_group):
        #         
        #         g_x_reg = self.g_reg[i](x_reg).view(batch_size, self.inter_channels_group, -1)
        #         g_x_reg = g_x_reg.permute(0, 2, 1)

        #         theta_x_reg = self.theta_reg[i](x_reg).view(batch_size, self.inter_channels_group, -1)
        #         theta_x_reg = theta_x_reg.permute(0, 2, 1)
        #         phi_x_reg = self.phi_reg[i](x_reg).view(batch_size, self.inter_channels_group, -1)

        #         f = torch.matmul(theta_x_reg, phi_x_reg)

        #         if self.use_softmax == True:
        #             f_div_C = self.softmax(f)
        #         else:
        #             N = f.size(-1)
        #             f_div_C = f / N

        #         y = torch.matmul(f_div_C, g_x_reg)
        #         y = y.permute(0, 2, 1).contiguous()
        #         y_reg_group.append(y)

        #     y_reg_out = torch.cat(y_reg_group, dim=1)

        #     y_reg_out = y_reg_out.view(batch_size, self.inter_channels, *x_reg.size()[2:])
        #     W_y_reg = self.W_reg(y_reg_out)
        #     z_reg = W_y_reg + x_reg

        #     ## relu after residual
        #     if self.relu_layer:
        #         z_reg = F.relu_(z_reg)

        #     return z_cls, z_reg
        # elif self.use_cross == True:
        #     ## cross-attention
        #     #### cls 
        #     y_cls_group = []

        #     for i in range(self.num_group):
        #         
        #         ## g and phi are from encoder, that is reg as input
        #         g_x_cls = self.g_cls[i](x_reg).view(batch_size, self.inter_channels_group, -1)
        #         g_x_cls = g_x_cls.permute(0, 2, 1)

        #         theta_x_cls = self.theta_cls[i](x_cls).view(batch_size, self.inter_channels_group, -1)
        #         theta_x_cls = theta_x_cls.permute(0, 2, 1)
        #         phi_x_cls = self.phi_cls[i](x_reg).view(batch_size, self.inter_channels_group, -1)

        #         f = torch.matmul(theta_x_cls, phi_x_cls)

        #         if self.use_softmax == True:
        #             f_div_C = self.softmax(f)
        #         else:
        #             N = f.size(-1)
        #             f_div_C = f / N

        #         y = torch.matmul(f_div_C, g_x_cls)
        #         y = y.permute(0, 2, 1).contiguous()
        #         y_cls_group.append(y)

        #     y_cls_out = torch.cat(y_cls_group, dim=1)

        #     y_cls_out = y_cls_out.view(batch_size, self.inter_channels, *x_cls.size()[2:])
        #     W_y_cls = self.W_cls(y_cls_out)
        #     z_cls = W_y_cls + x_cls

        #     ## relu after residual
        #     if self.relu_layer:
        #         z_cls = F.relu_(z_cls)

        #     #### reg
        #     y_reg_group = []

        #     for i in range(self.num_group):
        #         
        #         ## g and phi are from encoder, that is cls as input
        #         g_x_reg = self.g_reg[i](x_cls).view(batch_size, self.inter_channels_group, -1)
        #         g_x_reg = g_x_reg.permute(0, 2, 1)

        #         theta_x_reg = self.theta_reg[i](x_reg).view(batch_size, self.inter_channels_group, -1)
        #         theta_x_reg = theta_x_reg.permute(0, 2, 1)
        #         phi_x_reg = self.phi_reg[i](x_cls).view(batch_size, self.inter_channels_group, -1)

        #         f = torch.matmul(theta_x_reg, phi_x_reg)

        #         if self.use_softmax == True:
        #             f_div_C = self.softmax(f)
        #         else:
        #             N = f.size(-1)
        #             f_div_C = f / N

        #         y = torch.matmul(f_div_C, g_x_reg)
        #         y = y.permute(0, 2, 1).contiguous()
        #         y_reg_group.append(y)

        #     y_reg_out = torch.cat(y_reg_group, dim=1)

        #     y_reg_out = y_reg_out.view(batch_size, self.inter_channels, *x_reg.size()[2:])
        #     W_y_reg = self.W_reg(y_reg_out)
        #     z_reg = W_y_reg + x_reg

        #     ## relu after residual
        #     if self.relu_layer:
        #         z_reg = F.relu_(z_reg)

        #     return z_cls, z_reg
        # else:
        #     assert (False)

class NONLocalBlock2D_Cross(_NonLocalBlockND_Cross):
    def __init__(self, in_channels, num_group=1, inter_channels=None, sub_sample=True, bn_layer=True, relu_layer=True, use_softmax=True, mode_code=0):
        super(NONLocalBlock2D_Cross, self).__init__(in_channels,
                                              num_group=num_group,
                                              inter_channels=inter_channels,
                                              dimension=2, sub_sample=sub_sample,
                                              bn_layer=bn_layer, relu_layer=relu_layer, use_softmax=use_softmax, 
                                              mode_code=0)







### group non local
class _NonLocalBlockND_Group(nn.Module):
    def __init__(self, in_channels, num_group, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True, relu_layer=True, use_softmax=True):
        super(_NonLocalBlockND_Group, self).__init__()

        assert dimension in [1, 2, 3]
        assert dimension == 2
        assert num_group in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.num_group = num_group

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        ## inner channels are divided by num of groups
        conv_nd = nn.Conv2d
        max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
        bn = nn.BatchNorm2d
        relu = nn.ReLU
        self.relu_layer = relu_layer

        self.use_softmax = use_softmax

        if self.use_softmax:
            self.softmax = nn.Softmax(dim=2)
     

        assert self.num_group <= self.inter_channels

        self.inter_channels_group = self.inter_channels // self.num_group
        print (self.inter_channels_group)

        g = []
        theta = []
        phi = []
        for i in range(self.num_group):
            g.append(conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels_group,
                    kernel_size=1, stride=1, padding=0))
            theta.append(conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels_group,
                    kernel_size=1, stride=1, padding=0))
            phi.append(conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels_group,
                    kernel_size=1, stride=1, padding=0))

        self.g = ListModule(*g)
        self.theta = ListModule(*theta)
        self.phi = ListModule(*phi)

        print (self.g)
        # self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
        #                  kernel_size=1, stride=1, padding=0)

        # self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
        #                      kernel_size=1, stride=1, padding=0)

        # self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
        #                    kernel_size=1, stride=1, padding=0)

        assert sub_sample==False
        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)


        ## v2 
        self.W = nn.Sequential(conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                         kernel_size=1, stride=1, padding=0))
        print (self.W)
        ## BN first then RELU
        if bn_layer:
            self.W.add_module(
                'bn', bn(self.in_channels)
            )

        # if relu_layer:
        #     self.W.add_module( 
        #         'relu', relu(inplace=True)
        #     )
        print (self.W)

        ## init the weights
        nn.init.constant_(self.W[0].weight, 0)
        nn.init.constant_(self.W[0].bias, 0)

        #if relu_layer or bn_layer:
        #    nn.init.constant_(self.W[1].weight, 0)
        #    nn.init.constant_(self.W[1].bias, 0)
        #else:
        #    nn.init.constant_(self.W.weight, 0)
        #    nn.init.constant_(self.W.bias, 0)
        
        ## v1
        # if bn_layer & relu_layer:
        #     self.W = nn.Sequential(
        #         conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
        #                 kernel_size=1, stride=1, padding=0),
        #         relu(inplace=True), 
        #         bn(self.in_channels)
        #     )
        #     nn.init.constant_(self.W[1].weight, 0)
        #     nn.init.constant_(self.W[1].bias, 0)
        # elif bn_layer & not relu_layer:
        #     self.W = nn.Sequential(
        #         conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
        #                 kernel_size=1, stride=1, padding=0),
        #         bn(self.in_channels)
        #     )
        #     nn.init.constant_(self.W[1].weight, 0)
        #     nn.init.constant_(self.W[1].bias, 0)
        #  elif not bn_layer & relu_layer:
        #     self.W = nn.Sequential(
        #         conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
        #                 kernel_size=1, stride=1, padding=0),
        #         relu(inplace=True)
        #     )
        #     nn.init.constant_(self.W[1].weight, 0)
        #     nn.init.constant_(self.W[1].bias, 0)
        # else:
        #     self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
        #                      kernel_size=1, stride=1, padding=0)
        #     nn.init.constant_(self.W.weight, 0)
        #     nn.init.constant_(self.W.bias, 0)


    def forward(self, x, y):
        '''
        :param x: (b, c, t, h, w)
        :param y: (b, c, t, h, w)
        :return:
        '''
        # print (x.shape)
        batch_size = x.size(0)
        # print (batch_size)


        # theta  : x
        # phy, g : y
        # 
        y_group = []
        for i in range(self.num_group):
            
            g_x = self.g[i](y).view(batch_size, self.inter_channels_group, -1)
            ## relu
            # g_x = F.relu_(g_x) 
            g_x = g_x.permute(0, 2, 1)

            theta_x = self.theta[i](x).view(batch_size, self.inter_channels_group, -1)
            theta_x = theta_x.permute(0, 2, 1)
            phi_x = self.phi[i](y).view(batch_size, self.inter_channels_group, -1)
            f = torch.matmul(theta_x, phi_x)

            if self.use_softmax == True:
                # f_div_C = F.softmax(f, 1, _stacklevel=4)
                f_div_C = self.softmax(f)
                # print (f_div_C.shape)
                # print (f[0,0,:])
                # print (f_div_C[0,0,:])
                # print (torch.sum(f_div_C[0,0,:]))

                # exit()
            else:
                N = f.size(-1)
                # print (f.shape)
                f_div_C = f / N
                # print (N)
                # print (f_div_C.shape)
                # exit()

                # print (f_div_C[0,:,:])
                # print (f[0,:,:])
                # print (f_softmax[0,:,:])
                # print (f_softmax.shape)

            y = torch.matmul(f_div_C, g_x)
            y = y.permute(0, 2, 1).contiguous()
            # print (y.shape)
            y_group.append(y)

        # y_out = torch.stack(y_group, dim=1)
        y_out = torch.cat(y_group, dim=1)
        # print (y_out.shape)
        # all_features = torch.stack(features)


        # y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        # W_y = self.W(y)
        y_out = y_out.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y_out)
        z = W_y + x

        ## relu after residual
        if self.relu_layer:
            z = F.relu_(z)

        return z

class NONLocalBlock2D_Group(_NonLocalBlockND_Group):
    def __init__(self, in_channels, num_group=1, inter_channels=None, sub_sample=True, bn_layer=True, relu_layer=True, use_softmax=True):
        super(NONLocalBlock2D_Group, self).__init__(in_channels,
                                              num_group=num_group,
                                              inter_channels=inter_channels,
                                              dimension=2, sub_sample=sub_sample,
                                              bn_layer=bn_layer, relu_layer=relu_layer, use_softmax=use_softmax)


## original non local
class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NONLocalBlock3D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock3D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=3, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


if __name__ == '__main__':
    import torch

    for (sub_sample, bn_layer) in [(True, True), (False, False), (True, False), (False, True)]:
        img = torch.zeros(2, 3, 20)
        net = NONLocalBlock1D(3, sub_sample=sub_sample, bn_layer=bn_layer)
        out = net(img)
        print(out.size())

        img = torch.zeros(2, 3, 20, 20)
        net = NONLocalBlock2D(3, sub_sample=sub_sample, bn_layer=bn_layer)
        out = net(img)
        print(out.size())

        img = torch.randn(2, 3, 8, 20, 20)
        net = NONLocalBlock3D(3, sub_sample=sub_sample, bn_layer=bn_layer)
        out = net(img)
        print(out.size())


