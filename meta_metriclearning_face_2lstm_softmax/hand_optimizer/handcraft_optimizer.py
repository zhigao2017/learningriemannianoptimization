import torch
import torch.nn as nn
from torch.autograd import Variable as V
from MatrixLSTM.MatrixLSTM import MatrixLSTM
from hand_optimizer.retraction import Retraction

class Hand_Optimizee_Model(nn.Module): 
    def __init__(self,lr):
        super(Hand_Optimizee_Model,self).__init__()
        self.lr=lr
        self.retraction=Retraction(self.lr)

    def forward(self,grad,M,state):

        PU=torch.matmul(M.permute(0,2,1),grad)
        PU=(PU+PU.permute(0,2,1))/2
        PPU=torch.matmul(M,PU)
        grad_R=grad-PPU
        
        M = self.retraction(M,grad_R)
        #grad_R=grad
        #M=M-self.lr*grad_R

        return M,state

