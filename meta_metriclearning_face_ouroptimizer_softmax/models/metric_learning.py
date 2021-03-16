
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import sys



class M_Metric(nn.Module):
    def __init__(self,dim,batchsize_para):
        super(M_Metric, self).__init__()

        self.dim=dim
        M=torch.randn(batchsize_para,self.dim, self.dim)

        #M=(M+M.permute(0,2,1))/2

        for i in range(batchsize_para):
            U = torch.empty(self.dim, self.dim)
            nn.init.orthogonal_(U)
            D=torch.diag(torch.rand(self.dim))
            M[i]=(U.mm(D)).mm(U.t())

        #print('M rank',)
        self.M=nn.Parameter(M,requires_grad =True)
        #self.M=nn.Parameter(torch.eye(self.dim),requires_grad =True)
        #self.M=nn.Parameter(manifold=nn.PositiveDefinite(self.dim))
        #nn.init.eye_(self.M)
        #print(self.M)
        self.M.requires_grad = True


    def forward(self, input):

        #print('input feature',input[0])

        x0=input.repeat(input.shape[0],1)
        x1=(input.repeat(1,input.shape[0])).view(input.shape[0]*input.shape[0],-1)

        #print('x0-x1 shape',(x0-x1).shape)
        #print('self.M shape',self.M.shape)
        d_m=torch.matmul(torch.matmul(x0-x1,self.M),(x0-x1).t())

        #d_m=((x0-x1).mm(self.M)).mm( (x0-x1).t())
        #print('d_m',d_m)
        #p=torch.diag(d_m).view(input.shape[0],input.shape[0])
        #print('d_m shape',d_m.shape)
        p=torch.diagonal(d_m,offset=0,dim1=1,dim2=2).view(-1,input.shape[0],input.shape[0])
        #print('p shape',p.shape)
        #print('p',p)

        return p