import torch
from torch import nn
from torch.nn import functional as F



class CirclePool(nn.Module):
    def __init__(self, flag, input_dimension):
        super(CirclePool, self).__init__()

        self.flag = flag # NOT USED 

        ## TODO 
        ## generalize with arbitrary input and stride
        ## self.num_pool = 

        self.output_dimension = -1

        if self.flag == 'avg':
            self.output_dimension = input_dimension
            self.avgpool_7 = nn.AvgPool2d(kernel_size=7, stride=7)
        elif self.flag == 'noPool':
            self.output_dimension = input_dimension * 7 * 7
        elif self.flag == 'circle':
            self.output_dimension = input_dimension * 3        ## 7x7
            self.avgpool_7 = nn.AvgPool2d(kernel_size=7, stride=7)
            ## 5x5
            self.avgpool_5 = nn.AvgPool2d(kernel_size=5, stride=5)
            ## 3x3
            self.avgpool_3 = nn.AvgPool2d(kernel_size=3, stride=3)
        else:
            assert (False, "not supported pool method")

    def forward(self, features):
        """
        : input : 256x7x7
        : avg output: 256
        : noPool output: 256x7x7
        : circle output: 256x3
        : view change inside the pool (not like  )
        :-----------------------------
        :
        """
        ## avg pool
        if self.flag == 'avg':
            pool_7_ =  self.avgpool_7(features)
            output = pool_7_.view(pool_7_.size(0), -1) 
            return output
        elif self.flag == 'noPool':
            output = features.view(features.size(0), -1) 
            return output
        elif self.flag == 'circle':
            ## utilize average pool by cascade
            pool_7_ =  self.avgpool_7(features)

            features_5 = features[:,:,1:1+5, 1:1+5]
            pool_5_ = self.avgpool_5(features_5)

            features_3= features[:,:,2:2+3, 2:2+3]
            pool_3_ = self.avgpool_3(features_3)

            ## 
            pool_3 = pool_3_
            pool_5 = (pool_5_ * 25 - pool_3_ * 9) / (25 - 9)
            pool_7 = (pool_7_ * 49 - pool_3_ * 25) / (49 - 25)

            # print (pool_3.shape)

            output = torch.cat([pool_3, pool_5, pool_7], dim=2)
            ## utilize average pool by cascade
            pool_7_ =  self.avgpool_7(features)

            features_5 = features[:,:,1:1+5, 1:1+5]
            pool_5_ = self.avgpool_5(features_5)

            features_3= features[:,:,2:2+3, 2:2+3]
            pool_3_ = self.avgpool_3(features_3)

            ## 
            pool_3 = pool_3_
            pool_5 = (pool_5_ * 25 - pool_3_ * 9) / (25 - 9)
            pool_7 = (pool_7_ * 49 - pool_3_ * 25) / (49 - 25)

            # print (pool_3.shape)

            output = torch.cat([pool_3, pool_5, pool_7], dim=2)
            output = output.view(output.size(0), -1)
            return output
        else:
            assert (False, "not supported pool method")


if __name__ == '__main__':
    # import torch
    for flag in ['avg', 'noPool', 'circle']:
        img = torch.zeros(2, 256, 7, 7)
        net = CirclePool(flag, input_dimension=256)
        out = net(img)
        print (net.output_dimension)
        print (out.size())
        img = torch.zeros(2, 1024, 7, 7)
        net = CirclePool(flag, input_dimension=1024)
        out = net(img)
        print (net.output_dimension)
        print (out.size())


