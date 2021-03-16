
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import sys



class M_Metric_test(nn.Module):
    def __init__(self,dim):
        super(M_Metric_test, self).__init__()

        self.dim=dim
        #self.M=nn.Parameter(torch.randn(self.dim, self.dim),requires_grad =True)
        #self.M=nn.Parameter(torch.eye(self.dim),requires_grad =True)
        self.M=nn.Parameter(manifold=nn.PositiveDefinite(self.dim))


    def forward(self, input):

        #print('input feature',input[0])

        x0=input[0].repeat(input.shape[0]-1,1)
        x1=input[1:input.shape[0],:]

        d_m=((x0-x1).mm(self.M)).mm( (x0-x1).t())
        #print('d_m',d_m)
        p=torch.diag(d_m)
        #print('p',p)

        return p