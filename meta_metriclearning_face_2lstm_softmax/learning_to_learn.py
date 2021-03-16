import torch
import torch.nn as nn
from torch.autograd import Variable
from timeit import default_timer as timer
import time
import math, random

from losses.LOSS import ContrastiveLoss
from ReplyBuffer import ReplayBuffer
from retraction import Retraction

retraction=Retraction(1)


criterion = nn.CrossEntropyLoss()


def f(inputs,M):


    X=torch.matmul(M.permute(0,2,1),inputs)

    return X



def adjust_learning_rate(optimizer, decay_rate):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate


def Learning_to_learn_global_training(opt,hand_optimizee,optimizee,train_loader):

    DIM=opt.DIM
    outputDIM=opt.outputDIM
    batchsize_para=opt.batchsize_para
    Observe=opt.Observe
    Epochs=opt.Epochs
    Optimizee_Train_Steps=opt.Optimizee_Train_Steps
    optimizer_lr=opt.optimizer_lr
    Decay=opt.Decay
    Decay_rate=opt.Decay_rate
    Imcrement=opt.Imcrement

    adam_global_optimizer = torch.optim.Adam(optimizee.parameters(),lr = optimizer_lr)
    #adam_global_optimizer = torch.optim.Adamax(optimizee.parameters(),lr = optimizer_lr)

    RB=ReplayBuffer(600*batchsize_para)

    Square=torch.eye(DIM)

    for i in range(Observe):
        RB.shuffle()
        if i ==0:
            M=torch.randn(batchsize_para,DIM, outputDIM).cuda()
            for k in range(batchsize_para):
                nn.init.orthogonal_(M[k])


            state = (torch.zeros(batchsize_para,DIM,outputDIM).cuda(),
                                     torch.zeros(batchsize_para,DIM,outputDIM).cuda(),
                                     torch.zeros(batchsize_para,DIM,outputDIM).cuda(),
                                     torch.zeros(batchsize_para,DIM,outputDIM).cuda(),
                                     ) 
            iteration=torch.zeros(batchsize_para)
            #M.retain_grad()
            M.requires_grad=True

            RB.push(state,M,iteration)  
            count=1
            print ('observe finish',count)

        break_flag=False
        for j, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = Variable(inputs.cuda())
            labels = Variable(labels).cuda()
            inputs=inputs.view(batchsize_para,inputs.shape[0]//batchsize_para,-1)
            #labels=labels.view(batchsize_para,labels.shape[0]//batchsize_para,-1)
            inputs=inputs.permute(0,2,1)

            output = f(inputs,M)
            output=output.permute(0,2,1)
            output=torch.reshape(output, (output.shape[0]*output.shape[1],output.shape[2]) )
            #labels=torch.reshape(labels, (labels.shape[0]*labels.shape[1]) )
            labels=labels-1
            loss = criterion(output, labels)


            loss.backward()
            M, state = hand_optimizee(M.grad, M, state)

            print('-------------------------')
            #print('MtM', torch.mm(M[k].t(),M[k]))


            iteration=iteration+1
            for k in range(batchsize_para):
                if iteration[k]>=Optimizee_Train_Steps-opt.train_steps:
                    nn.init.orthogonal_(M[k])
                    state[0][k]=torch.zeros(DIM,outputDIM).cuda()
                    state[1][k]=torch.zeros(DIM,outputDIM).cuda()
                    state[2][k]=torch.zeros(DIM,outputDIM).cuda()
                    state[3][k]=torch.zeros(DIM,outputDIM).cuda()   
                    iteration[k]=0


            state = (state[0].detach(),state[1].detach(),state[2].detach(),state[3].detach())
            M=M.detach()
            
            M.requires_grad=True
            M.retain_grad()


            RB.push(state,M,iteration)
            count=count+1
            print ('loss',loss)
            print ('observe finish',count)
            localtime = time.asctime( time.localtime(time.time()) )

            if count==Observe:
                break_flag=True
                break
        if break_flag==True:
            break                         
    
    RB.shuffle()

    check_point=optimizee.state_dict()
    check_point2=optimizee.state_dict()
    check_point3=optimizee.state_dict()
    for i in range(Epochs): 
        print('\n=======> global training steps: {}'.format(i))
        if (i+1) % Decay==0 and (i+1) != 0:
            count=count+1
            adjust_learning_rate(adam_global_optimizer, Decay_rate)

        if opt.Imcrementflag==True:
            if (i+1) % Imcrement==0 and (i+1) != 0:
                Optimizee_Train_Steps=Optimizee_Train_Steps+50

        if (i+1) % opt.modelsave==0 and (i+1) != 0:
            if opt.Pretrain==True:
                torch.save(optimizee.state_dict(), 'snapshot/traindata'+str(i)+'_'+str(opt.optimizer_lr*1000)+'_Decay'+str(opt.Decay)+'_Observe'+str(opt.Observe)+'_Epochs'+str(opt.Epochs)+'_Optimizee_Train_Steps'+str(opt.Optimizee_Train_Steps)+'_train_steps'+str(opt.train_steps)+'_hand_optimizer_lr'+str(opt.hand_optimizer_lr)+'_lrdim1024.pth')
            else:
                torch.save(optimizee.state_dict(), 'snapshot/traindata'+str(i)+'_'+str(opt.optimizer_lr*1000)+'_Decay'+str(opt.Decay)+'_Observe'+str(opt.Observe)+'_Epochs'+str(opt.Epochs)+'_Optimizee_Train_Steps'+str(opt.Optimizee_Train_Steps)+'_train_steps'+str(opt.train_steps)+'_hand_optimizer_lr'+str(opt.hand_optimizer_lr)+'nopretrain_newlr_meanvar_devide2'+'.pth')



        if i==0:
            global_loss_graph=0
        else:
            global_loss_graph=global_loss_graph.detach()
            global_loss_graph=0

        state_read, M_read,iteration_read= RB.sample(batchsize_para) 
        state=(state_read[0].detach(),state_read[1].detach(),state_read[2].detach(),state_read[3].detach())
        M=M_read.detach()
        iteration=iteration_read.detach()
        M.requires_grad=True
        M.retain_grad()
        
        flag=False
        break_flag=False
        count=0
        adam_global_optimizer.zero_grad()
        while(1):
            for j, data in enumerate(train_loader, 0):
                print('---------------------------------------------------------------------------')

                ct=time.time()
                localtime = time.localtime(ct) 
                data_head=time.strftime("%Y-%m-%d %H:%M:%S",localtime)
                data_secs=(ct-int(ct))*1000
                time_stamp="%s.%03d"%(data_head,data_secs)
                print('count',count,'localtime',time_stamp)

                #print('M',M)
                inputs, labels = data
                inputs = Variable(inputs.cuda())
                labels = Variable(labels).cuda()
                inputs=inputs.view(batchsize_para,inputs.shape[0]//batchsize_para,-1)
                inputs=inputs.permute(0,2,1)
                
                output = f(inputs,M)
                output=output.permute(0,2,1)
                output=torch.reshape(output, (output.shape[0]*output.shape[1],output.shape[2]) )
                labels=labels-1
                loss = criterion(output, labels)

                if count>0:
                    global_loss_graph=global_loss_graph+(loss)

                print('count',count,'loss',loss)

                if count==opt.train_steps:
                    break_flag=True
                    break

                M=M.detach()
                M.requires_grad=True
                M.retain_grad()

                output2 = f(inputs,M)
                output2=output2.permute(0,2,1)
                output2=torch.reshape(output2, (output2.shape[0]*output2.shape[1],output2.shape[2]) )
                loss2 = criterion(output2, labels)
                print('count',count,'loss2',loss2)
                loss2.backward()
                
                

                g=M.grad.detach()
                lr, update, state = optimizee(g, state)
                lr=lr/(1/opt.hand_optimizer_lr)
                update=update+g

                
                s=torch.sum(state[0])+torch.sum(state[1])+torch.sum(state[2])+torch.sum(state[3])
                if s > 100000:
                    break_flag=True
                    flag=True
                    break
                

                M.grad.data.zero_()

                PU=torch.matmul(M.permute(0,2,1),update)
                PU=(PU+PU.permute(0,2,1))/2
                PPU=torch.matmul(M,PU)
                update_R=update-PPU

                M = retraction(M,update_R,lr)
                M.retain_grad()

                localtime = time.asctime( time.localtime(time.time()) )
                iteration=iteration+1
                update.retain_grad()
                count=count+1

 
            if break_flag==True:
                break 

        
        global_loss_graph.backward() 
        if flag==False:
            adam_global_optimizer.step()

            for k in range(batchsize_para):
                if iteration[k] >= Optimizee_Train_Steps-opt.train_steps:
                    nn.init.orthogonal_(M[k])
                    state[0][k]=torch.zeros(DIM,outputDIM).cuda()
                    state[1][k]=torch.zeros(DIM,outputDIM).cuda()
                    state[2][k]=torch.zeros(DIM,outputDIM).cuda()
                    state[3][k]=torch.zeros(DIM,outputDIM).cuda()     
                    iteration[k]=0

            RB.push((state[0].detach(),state[1].detach(),state[2].detach(),state[3].detach()),M.detach(),iteration.detach())            

            check_point=check_point2
            check_point2=check_point3
            check_point3=optimizee.state_dict()         
        else:
            print('=====>eigenvalue break, reloading check_point')
            optimizee.load_state_dict(check_point)

        print('=======>global_loss_graph',global_loss_graph)
