import torch
import torch.nn as nn
from torch.autograd import Variable as V
from timeit import default_timer as timer

from learner import Learner

def adjust_learning_rate(optimizer, decay_rate):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate


def Learning_to_learn_global_training(opt,optimizee,DIM,outputDIM,batchsize_para,Global_Train_Steps,Optimizee_Train_Steps,UnRoll_STEPS,optimizer_lr,Decay,Decay_rate,train_loader,train_test_loader):

    count=0
    global_loss_list = []
    Total_Num_Unroll = Optimizee_Train_Steps // UnRoll_STEPS
    #adam_global_optimizer = torch.optim.Adam(optimizee.parameters(),lr = optimizer_lr)
    #adam_global_optimizer = torch.optim.Adamax(optimizee.parameters(),lr = optimizer_lr)


    LSTM_Learner = Learner(opt,DIM,outputDIM, batchsize_para, optimizee, UnRoll_STEPS, retain_graph_flag=True, reset_theta=True,reset_function_from_IID_distirbution = True)

    #LSTM_Learner=LSTM_Learner.cuda()
    best_sum_loss = 999999
    best_final_loss = 999999
    best_flag = False
    for i in range(Global_Train_Steps): 

        print('\n=======> global training steps: {}'.format(i))
        if i % Decay==0 and i != 0:
            count=count+1
            #Total_Num_Unroll = Optimizee_Train_Steps[count] // UnRoll_STEPS[count]
            #LSTM_Learner = Learner(DIM, batchsize_para, optimizee, UnRoll_STEPS[count], retain_graph_flag=True, reset_theta=True,reset_function_from_IID_distirbution = True)
            #adjust_learning_rate(adam_global_optimizer, Decay_rate)

        if i % opt.modelsave==0 and i != 0:
            torch.save(optimizee.state_dict(), 'snapshot/constraintmodel_meta_metriclearning_Riemannian_tangentspace_oneoptimizer_itera'+str(i)+'_'+str(opt.optimizer_lr*1000)+'_Optimizee_Train_Steps'+str(opt.Optimizee_Train_Steps)+'_UnRoll_STEPS'+str(opt.UnRoll_STEPS)+'_batchsize_para'+str(opt.batchsize_para)+'d128_twoLSTM_Optimizee_lr0.001_iteration1000_nomoment_30_3_c1'+'.pth')
            
        for num in range(Total_Num_Unroll):
            
            start = timer()
            _,global_loss,flag = LSTM_Learner(train_loader,train_test_loader,num)   

            #adam_global_optimizer.zero_grad()
            #global_loss.backward() 
       
            #if global_loss<20000 and flag==False:
                #adam_global_optimizer.step()
            #else:
                #break
                
            global_loss_list.append(global_loss.cpu().detach_().numpy())
            time = timer() - start

            print('-> time consuming [{:.1f}s] optimizee train steps :  [{}] | Global_Loss = [{:.1f}]  '\
                  .format(time,(num +1)* UnRoll_STEPS,global_loss,))


    return global_loss_list,best_flag