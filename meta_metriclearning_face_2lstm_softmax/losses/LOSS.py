from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np

"""
Baseline loss function in BIER

Deep Metric Learning with BIER: Boosting Independent Embeddings Robustly
"""



class ContrastiveLoss(nn.Module):
    def __init__(self, beta=0.01):
        super(ContrastiveLoss, self).__init__()
        self.beta = beta
        self.lamba=1
        self.min_margin=50
        self.max_margin=120
        print('loss beta',self.beta)
        # self.alpha = alpha

    def forward(self, sim_mat, targets, M):

        #print('sim_mat',sim_mat)

        n = sim_mat.size(1)
        bp = sim_mat.size(0)
        dim = M.shape[1]
        #print('n bp',n,bp)
        targets = targets.cuda()
        pos_mask = (targets.repeat(1, n).view(bp,n,n)).eq(targets.repeat(1, n).view(bp,n,n).permute(0,2,1))
        #pos_mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        #print('pos_mask shape',pos_mask.shape)


        label_v=torch.Tensor(bp,int(n*(n-1)/2)).cuda()
        sim_v=torch.Tensor(bp,int(n*(n-1)/2)).cuda()
        #sim_v=torch.Tensor(int(n*(n-1)/2)).cuda()
        #label_v=torch.Tensor(int(n*(n-1)/2)).cuda()
        
        count=0
        for i in range(n-1):
            label_v[:,count:count+n-i-1]=pos_mask[:,i,i+1:n]
            #label_v[count:count+n-i-1]=pos_mask[i,i+1:n]
            sim_v[:,count:count+n-i-1]=sim_mat[:,i,i+1:n]
            count=count+n-i-1

        #print('label_v',label_v.shape)
        #print('sim_v',sim_v.shape)
        #print('sim_v',sim_v)
        zero = torch.zeros(sim_v.shape).cuda()
        #loss1=torch.sum(   torch.sum(label_v.mul(torch.pow((torch.max(sim_v-self.min_margin,zero)),2)),1)  ) /(torch.sum(label_v))
        #loss2=torch.sum(   torch.sum((1-label_v).mul(torch.pow((torch.max(self.max_margin-sim_v,zero)),2)),1)   ) /(torch.sum(1-label_v))
        #print('----', label_v.mul(torch.pow((torch.max(sim_v-self.min_margin,zero)),2)).shape  )
        #print('torch.sum(label_v.mul(torch.pow((torch.max(sim_v-self.min_margin,zero)),2)),1)',torch.sum(label_v.mul(torch.pow((torch.max(sim_v-self.min_margin,zero)),2)),1).shape)
        #print('torch.sum(label_v,1)',torch.sum(label_v,1).shape)
        #print('torch.pow(torch.max(sim_v-self.min_margin,zero),2)',label_v.mul(torch.pow(torch.max(sim_v-self.min_margin,zero),2)) )
        #print('torch.pow(torch.max(sim_v-self.min_margin,zero),2)',torch.pow(torch.max(sim_v-self.min_margin,zero),2).shape)
        loss1=torch.sum(    label_v.mul(torch.pow(torch.max(sim_v-self.min_margin,zero),2))    ,1) /(torch.sum(label_v,1))
        loss2=torch.sum(    (1-label_v).mul(torch.pow(torch.max(self.max_margin-sim_v,zero),2)),1) /(torch.sum(1-label_v,1))
        #loss=loss1/(torch.sum(label_v)*bp)+loss2/(torch.sum((1-label_v))*bp)
        loss1=torch.sum(loss1)/bp
        loss2=torch.sum(loss2)/bp
        
        '''
        p=torch.exp(self.beta*sim_v)
        #loss=torch.sum(label_v*( torch.log(1+p) )+(1-label_v)*( torch.log(1+(1/p)) ))
        loss1=torch.sum(label_v*( torch.log(1+p) ))
        loss2=torch.sum((1-label_v)*( torch.log(1+(1/p)) ))
        #loss1=torch.sum(torch.sum(label_v* torch.log(1+p),1 )/torch.sum(label_v))  
        #loss2=torch.sum(torch.sum((1-label_v)* torch.log(1+(1/p)),1)/torch.sum(1-label_v)) 
        #loss=loss1+loss2
        '''

        '''
        loss_r=loss1
        loss_r=0
        for i in range(bp):
            #print('logdet',torch.logdet(M[i]))
            loss_r=loss_r+torch.trace(M[i])-torch.logdet(M[i])-dim
        loss_r=loss_r/bp
        '''


        pos_d=torch.sum(label_v*sim_v)/(torch.sum(label_v))
        neg_d=torch.sum((1-label_v)*sim_v)/(torch.sum(1-label_v))

        #print('torch.sum((1-label_v)*sim_v)',torch.sum((1-label_v)*sim_v))

        #print('DDDDDDDDDDDDDDistance---------------pos_d',pos_d)
        #print('DDDDDDDDDDDDDDistance---------------neg_d',neg_d)
        #print('LLLLLLLLLLLLLLoss-----training_step-----','loss1',loss1)
        #print('LLLLLLLLLLLLLLoss-----training_step-----','loss2',loss2)
        #print('LLLLLLLLLLLLLLoss-----training_step-----','loss_r',loss_r)
        

        #return loss, pos_d, neg_d
        #return loss1+loss2+loss_r
        return loss1+loss2





