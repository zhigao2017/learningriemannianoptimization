import torch
import torch.nn as nn
from torch.autograd import Variable 
from losses.LOSS import ContrastiveLoss
import time
import math

from models.EigLayer import EigLayer
from models.m_exp import M_Exp
from retraction import Retraction

retraction=Retraction(1)
criterion = nn.CrossEntropyLoss()


def f(inputs,M):

    X=torch.matmul(M.permute(0,2,1),inputs)

    return X




class Learner( object ):
    def __init__(self, opt,DIM, outputDIM,batchsize_para,  optimizee, train_steps ,  
                                            retain_graph_flag=False,
                                            reset_theta = False ,
                                            reset_function_from_IID_distirbution = True):
        self.criterion = ContrastiveLoss().cuda()
        self.retraction=Retraction(1)
        self.optimizee = optimizee

        self.opt=opt

        self.beta=1
        self.eiglayer1=EigLayer()
        self.mexp=M_Exp()

        self.train_steps = train_steps
        #self.num_roll=num_roll
        self.retain_graph_flag = retain_graph_flag
        self.reset_theta = reset_theta
        self.reset_function_from_IID_distirbution = reset_function_from_IID_distirbution  
        self.state = None


        self.DIM=DIM
        self.outputDIM=outputDIM
        self.batchsize_para=batchsize_para

        self.global_loss_graph = 0 # global loss for optimizing LSTM
        self.losses = []   # KEEP each loss of all epoches

        for parameters in optimizee.parameters():
            print(torch.sum(parameters))
            
        self.M=torch.randn(self.batchsize_para,self.DIM, self.outputDIM).cuda()
        for i in range(self.batchsize_para):
            '''
            U = torch.empty(self.DIM, self.DIM)
            nn.init.orthogonal_(U)
            D=torch.abs(torch.diag(torch.rand(self.DIM)+0.6))
            self.M[i]=(U.mm(D)).mm(U.t())
            #self.M[i]=self.M[i]+0.001*torch.trace(self.M[i])*torch.eye(self.DIM)
            '''
            nn.init.orthogonal_(self.M[i])

        self.M=self.M.cuda()
        self.M.requires_grad=True




        
            
    def Reset_Or_Reuse(self ,num_roll, M, state):
        ''' re-initialize the `W, Y, x , state`  at the begining of each global training
            IF `num_roll` == 0    '''

        reset_theta =self.reset_theta
        reset_function_from_IID_distirbution = self.reset_function_from_IID_distirbution

       
        if num_roll == 0 and reset_theta == True:

            M=torch.randn(self.batchsize_para,self.DIM, self.outputDIM).cuda()
            for i in range(self.batchsize_para):

                '''
                U = torch.empty(self.DIM, self.DIM)
                nn.init.orthogonal_(U)
                D=torch.abs(torch.diag(torch.rand(self.DIM)+0.5))
                M[i]=(U.mm(D)).mm(U.t())
                #self.M[i]=self.M[i]+1*torch.trace(self.M[i])*(torch.eye(self.DIM).cuda())
                '''

                nn.init.orthogonal_(M[i])
                #print('logdet',torch.logdet(M[i]))
            
            
        if num_roll == 0:
            state = None
            print('reset metric model state ')
            
        M = M.cuda()
        M.requires_grad=True

        return M,state
          
            
    def __call__(self, train_loader,train_test_loader,num_roll=0) : 
        '''
        Total Training steps = Unroll_Train_Steps * the times of  `Learner` been called
        
        SGD,RMS,LSTM FROM defination above
         but Adam is adopted by pytorch~ This can be improved later'''
        M,state =  self.Reset_Or_Reuse( num_roll, self.M, self.state )
        self.global_loss_graph = 0
        optimizee = self.optimizee

        #print('state is None = {}'.format(state == None))

        #for i in range(self.train_steps):

        flag=False

        count=0
        while(1):


            for j, data in enumerate(train_loader, 0):

                
                total_loss=0
                for l in range(8):
                    for k, subdata in enumerate(train_test_loader, 0):

                        inputs, labels = subdata
                        inputs = Variable(inputs.cuda())
                        labels = Variable(labels).cuda()

                        inputs=inputs.view(self.batchsize_para,inputs.shape[0]//self.batchsize_para,-1)
                        #labels=labels.view(batchsize_para,labels.shape[0]//batchsize_para,-1)
                        inputs=inputs.permute(0,2,1)

                        output = f(inputs,M)
                        output=output.permute(0,2,1)
                        output=torch.reshape(output, (output.shape[0]*output.shape[1],output.shape[2]) )
                        #labels=torch.reshape(labels, (labels.shape[0]*labels.shape[1]) )
                        labels=labels-1
                        loss = criterion(output, labels)

                        total_loss += (loss.detach())*8
                        #print('single loss',loss)
                print('count',count,'total loss',total_loss/(2414*8))
                

                count=count+1

                #print('---------------------------------------------------------------------------')
                #print('M',M)
                inputs, labels = data
                inputs = Variable(inputs.cuda())
                labels = Variable(labels).cuda()


                inputs=inputs.view(self.batchsize_para,inputs.shape[0]//self.batchsize_para,-1)
                inputs=inputs.permute(0,2,1)
                
                output = f(inputs,M)
                output=output.permute(0,2,1)
                output=torch.reshape(output, (output.shape[0]*output.shape[1],output.shape[2]) )
                #labels=torch.reshape(labels, (labels.shape[0]*labels.shape[1],labels.shape[2]) )
                labels=labels-1
                loss = criterion(output, labels)

                
                ct=time.time()
                localtime = time.localtime(ct) 
                data_head=time.strftime("%Y-%m-%d %H:%M:%S",localtime)
                data_secs=(ct-int(ct))*1000
                time_stamp="%s.%03d"%(data_head,data_secs)
                print('count',count,'loss',loss,'localtime',time_stamp)
                

                loss.backward() # default as False,set to True for LSTMS
                lr, update, state = optimizee(M.grad, state)
                #lr=lr/100000
                lr=lr/(1/self.opt.hand_optimizer_lr)


                update=update+M.grad

                #M.grad.data.zero_()

                PU=torch.matmul(M.permute(0,2,1),update)
                PU=(PU+PU.permute(0,2,1))/2
                PPU=torch.matmul(M,PU)
                update_R=update-PPU

                M = retraction(M,update_R,lr)

                state = (state[0].detach(),state[1].detach(),state[2].detach(),state[3].detach())
                M = M.detach()
                M.requires_grad = True
                M.retain_grad()
                



        return self.losses ,self.global_loss_graph,flag



