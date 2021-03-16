import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    # Data input settings
    parser.add_argument('--DIM', type=int, default=1024)
    parser.add_argument('--outputDIM', type=int, default=38)
    parser.add_argument('--batchsize_para', type=int, default=8)
    parser.add_argument('--batchsize_data', type=int, default=512)
    parser.add_argument('--datapath', type=str, default='data')
    parser.add_argument('--prepath1', type=str, default='../meta_metriclearning_face_2lstm_softmax_continue/snapshot/traindata19999_10.0_Decay100000_Observe2000_Epochs1000000_Optimizee_Train_Steps2000_train_steps10_hand_optimizer_lr0.0001_lrdim1024.pth')
    parser.add_argument('--num_instances', type=int, default=12)
    parser.add_argument('--nThreads', type=int, default=0)
    parser.add_argument('--Global_Train_Steps', type=int, default=1000000)
    parser.add_argument('--Decay', type=int,default=400)
    parser.add_argument('--modelsave', type=int,default=50)
    parser.add_argument('--Decay_rate', type=float,default=0.5)
    parser.add_argument('--Optimizee_Train_Steps', type=int, default=300)
    parser.add_argument('--UnRoll_STEPS', type=int, default=5)
    parser.add_argument('--optimizer_lr', type=float, default=0.00001)
    parser.add_argument('--hand_optimizer_lr', type=float, default=0.0001)
    args = parser.parse_args()
    return args