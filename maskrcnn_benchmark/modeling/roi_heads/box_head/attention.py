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

### group non local
class _NonLocalBlockND_Group(nn.Module):
    def __init__(self, in_channels, num_group, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True, relu_layer=True):
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


    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''
        # print (x.shape)
        batch_size = x.size(0)
        # print (batch_size)

        y_group = []
        for i in range(self.num_group):
            
            g_x = self.g[i](x).view(batch_size, self.inter_channels_group, -1)
            ## relu
            # g_x = F.relu_(g_x) 
            g_x = g_x.permute(0, 2, 1)

            theta_x = self.theta[i](x).view(batch_size, self.inter_channels_group, -1)
            theta_x = theta_x.permute(0, 2, 1)
            phi_x = self.phi[i](x).view(batch_size, self.inter_channels_group, -1)
            f = torch.matmul(theta_x, phi_x)
            N = f.size(-1)
            f_div_C = f / N
            # print (N)
            # print (f_div_C.shape)

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
    def __init__(self, in_channels, num_group=1, inter_channels=None, sub_sample=True, bn_layer=True, relu_layer=True):
        super(NONLocalBlock2D_Group, self).__init__(in_channels,
                                              num_group=num_group,
                                              inter_channels=inter_channels,
                                              dimension=2, sub_sample=sub_sample,
                                              bn_layer=bn_layer, relu_layer=relu_layer)


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


