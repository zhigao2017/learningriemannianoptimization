import torch
import torch.nn as nn

from learning_to_learn import Learning_to_learn_global_training
from LSTM_Optimizee_Model import LSTM_Optimizee_Model
from DataSet.YaleB import YaleB
from utils import FastRandomIdentitySampler

import config

opt = config.parse_opt()

print(opt)

LSTM_Optimizee = LSTM_Optimizee_Model(opt,opt.DIM, opt.outputDIM, batchsize_data=opt.batchsize_data, batchsize_para=opt.batchsize_para).cuda()

#checkpoint = torch.load('/home/mcislab/gaozhi/meta_metriclearning/meta_metriclearning_Riemannian_tangentspace_oneoptimizer_retraction_dir+lr/snapshot/constraintmodel_meta_metriclearning_Riemannian_tangentspace_oneoptimizer_itera150_0.1_Optimizee_Train_Steps300_UnRoll_STEPS5_batchsize_para12d128_twoLSTM_Optimizee_lr0.001_iteration1000_nomoment_30_3.pth')
#checkpoint = torch.load('/home/mcislab/gaozhi/meta_metriclearning/meta_metriclearning_Riemannian_tangentspace_oneoptimizer_retraction_dir+lr/snapshot/constraintmodel_meta_metriclearning_Riemannian_tangentspace_oneoptimizer_itera150_0.1_Optimizee_Train_Steps300_UnRoll_STEPS5_batchsize_para12d128_twoLSTM_Optimizee_lr0.001_iteration1000_nomoment_30_3.pth')
#checkpoint = torch.load('/home/mcislab/gaozhi/meta_metriclearning/meta_metriclearning_Riemannian_tangentspace_oneoptimizer_pool_multiplestep_handopti_differentdata_break_retraction_dir+lr/snapshot/c1_centroid_itera99_0.01_Decay1500_Imcrement30_Observe200_Epochs50000_Optimizee_Train_Steps300_train_steps5d128_twoLSTM_Optimizee_lr0.0005_iteration1000_nomoment_30_3.pth')
#checkpoint = torch.load('/home/mcislab/gaozhi/meta_metriclearning/meta_metriclearning_Riemannian_tangentspace_oneoptimizer_pool_multiplestep_handopti_differentdata_break_retraction_dir+lr/snapshot/oldlstm_centroid_itera1399_0.1_Decay1000_Imcrement30_Observe200_Epochs50000_Optimizee_Train_Steps300_train_steps5d128_twoLSTM_Optimizee_lr0.0005_iteration1000_nomoment_30_3.pth')
#checkpoint = torch.load('/home/mcislab/gaozhi/meta_metriclearning/meta_metriclearning_Riemannian_tangentspace_oneoptimizer_retraction_dir+lr/snapshot/constraintmodel_meta_metriclearning_Riemannian_tangentspace_oneoptimizer_itera150_0.1_Optimizee_Train_Steps300_UnRoll_STEPS5_batchsize_para12d128_twoLSTM_Optimizee_lr0.001_iteration1000_nomoment_30_3.pth')
#checkpoint = torch.load('/home/mcislab/gaozhi/meta_metriclearning/meta_metriclearning_Riemannian_tangentspace_oneoptimizer_retraction_dir+lr/snapshot/constraintmodel_meta_metriclearning_Riemannian_tangentspace_oneoptimizer_itera50_0.1_Optimizee_Train_Steps300_UnRoll_STEPS5_batchsize_para12d128_twoLSTM_Optimizee_lr0.001_iteration1000_nomoment_30_3.pth')
#checkpoint = torch.load('/home/mcislab/gaozhi/meta_metriclearning/meta_metriclearning_Riemannian_tangentspace_oneoptimizer_retraction_dir+lr/snapshot/newlstmlr_constraintmodel_meta_metriclearning_Riemannian_tangentspace_oneoptimizer_itera2000_0.1_Optimizee_Train_Steps300_UnRoll_STEPS5_batchsize_para12d128_twoLSTM_Optimizee_lr0.001_iteration1000_nomoment_30_3_c1.pth')
checkpoint = torch.load(opt.prepath1)


LSTM_Optimizee.load_state_dict(checkpoint)

print('pretrain finsist')

train_mnist = YaleB(opt.datapath, train=False)
'''
train_loader = torch.utils.data.DataLoader(
    train_mnist, batch_size=opt.batchsize_data,
    sampler=FastRandomIdentitySampler(train_mnist, num_instances=opt.num_instances),
    drop_last=True, pin_memory=True, num_workers=opt.nThreads)
'''

train_loader = torch.utils.data.DataLoader(
        train_mnist, batch_size=opt.batchsize_data,shuffle=True, drop_last=True, num_workers=0)

train_loader_all = torch.utils.data.DataLoader(
        train_mnist, batch_size=1920,shuffle=True, drop_last=True, num_workers=0)

train_test_loader = torch.utils.data.DataLoader(
        train_mnist, batch_size=8,shuffle=True, drop_last=True, num_workers=0)



global_loss_list ,flag = Learning_to_learn_global_training(opt,LSTM_Optimizee,
														opt.DIM,
                                                        opt.outputDIM,
														opt.batchsize_para,
                                                        opt.Global_Train_Steps,
                                                        opt.Optimizee_Train_Steps,
                                                        opt.UnRoll_STEPS,
                                                        opt.optimizer_lr,
                                                        opt.Decay,
                                                        opt.Decay_rate,
                                                        train_loader,train_test_loader)