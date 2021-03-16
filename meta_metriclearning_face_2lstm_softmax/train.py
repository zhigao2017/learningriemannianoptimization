import torch
import torch.nn as nn

from learning_to_learn import Learning_to_learn_global_training
from LSTM_Optimizee_Model import LSTM_Optimizee_Model
from hand_optimizer.handcraft_optimizer import Hand_Optimizee_Model
from DataSet.YaleB import YaleB
from utils import FastRandomIdentitySampler

import config

opt = config.parse_opt()
print(opt)

LSTM_Optimizee = LSTM_Optimizee_Model(opt,opt.DIM, opt.outputDIM, batchsize_data=opt.batchsize_data, batchsize_para=opt.batchsize_para).cuda()

if opt.Pretrain==True:
	checkpoint2 = torch.load(opt.prepath2)
	LSTM_Optimizee.load_state_dict(checkpoint2,strict=False)
	checkpoint = torch.load(opt.prepath)
	LSTM_Optimizee.load_state_dict(checkpoint,strict=False)



Hand_Optimizee = Hand_Optimizee_Model(opt.hand_optimizer_lr).cuda()

print('pretrain finsist')

train_mnist = YaleB(opt.datapath, train=True)
'''
train_loader = torch.utils.data.DataLoader(
    train_mnist, batch_size=opt.batchsize_data,
    sampler=FastRandomIdentitySampler(train_mnist, num_instances=opt.num_instances),
    drop_last=True, pin_memory=True, num_workers=opt.nThreads)
'''

train_loader = torch.utils.data.DataLoader(
        train_mnist, batch_size=opt.batchsize_data,shuffle=True, drop_last=True, num_workers=0)


global_loss_list ,flag = Learning_to_learn_global_training(opt,Hand_Optimizee,LSTM_Optimizee,train_loader)