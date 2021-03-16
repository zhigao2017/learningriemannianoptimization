import torch
import torch.nn as nn
from torch.autograd import Variable as V
from torch.autograd import Function

from models.EigLayer import EigLayer
from models.m_sqrt import M_Sqrt


class Retraction(nn.Module):
    def __init__(self,lr):
        super(Retraction, self).__init__()

        self.beta=lr
        self.msqrt2=M_Sqrt(-1)
        self.eiglayer1=EigLayer()


    def forward(self, inputs, grad,lr):


        new_point=torch.zeros(inputs.shape).cuda()
        n=inputs.shape[0]

        P=-lr*grad
        PV=inputs+P

        PV_p=torch.matmul(PV.permute(0,2,1),PV)

        PV_S,PV_U=self.eiglayer1(PV_p)
        PV_S2=self.msqrt2(PV_S)
        PV_p=torch.matmul( torch.matmul( PV_U, PV_S2 ), PV_U.permute(0,2,1) )

        new_point=torch.matmul(PV,PV_p)


        return new_point