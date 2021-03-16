import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import Function

'''
not correct
'''

class SVDLayerF(Function):
    @staticmethod
    def forward(self,input):

        n=(input.shape)[0]
        n1=(input.shape)[1]
        n2=(input.shape)[2]
        n_min=min(n1,n2)

        U=torch.zeros(n,n1,n_min).cuda()
        S=torch.zeros(n,n_min,n_min).cuda()
        V=torch.zeros(n,n2,n2).cuda()

        for i in range(n):
            leftvector, value, rightvector=torch.svd( input[i])
            U[i]=leftvector
            S[i]=torch.diag(value)
            V[i]=rightvector

            
        self.save_for_backward(input, U,S,V)
        #print('EigLayer finish')
        return U,S,V


    @staticmethod
    def backward(self, grad_U,grad_S,grad_V ):

        input, U,S,V = self.saved_tensors

        n=input.shape[0]
        n1=input.shape[1]
        n2=input.shape[2]
        n_min=min(n1,n2)

        grad_input=Variable(torch.zeros(n,n1,n2 )) .cuda()

        e=torch.eye(n_min).cuda()

        P_i=torch.matmul(S,torch.ones(n_min,n_min).cuda())
        P_i=P_i*P_i
        
        P=(P_i-P_i.permute(0,2,1))+e
        epo=(torch.ones(P.shape).cuda())*0.000001

        P=torch.where(P!=0,P,epo)
        P=(1/P)-e
        

        g1= torch.matmul(V.permute(0,2,1),grad_V)
        g1=torch.mul(P.permute(0,2,1),g1)
        g1=(g1+g1.permute(0,2,1))/2
        g1=2*torch.matmul(S,g1)
        g2=grad_S


        grad_input=torch.matmul(torch.matmul(U,g1+g2),V.permute(0,2,1))

        #print('grad_input',torch.sum(grad_input))

        return grad_input




class SVDLayer(nn.Module):
    def __init__(self):
        super(SVDLayer, self).__init__()
    

    def forward(self, input1):
        return SVDLayerF().apply(input1)

