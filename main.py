import argparse
import random

import numpy as np
import torch

from exp.exp_sleep_acl_cross import Exp_Sleep_ACL_Cross
from utils.print_args import print_args

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='sleepstage')

    # basic config
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model', type=str, default='SleepAC', help='model name, options: [SleepAC]')

    # data loader
    parser.add_argument('--dataset', type=str, default='ISRUC-S3',
                        help='dataset type, options:[ISRUC-S1, ISRUC-S3, Sleep-EDF-20, SHHS1]')
    parser.add_argument('--data_path', type=str, default='./datasets/EEG/',
                        help='root path of the data file')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/SleepAC/', help='location of model checkpoints')
    parser.add_argument('--result_path', type=str, default='./results', help='results save dir')
    parser.add_argument('--test_fold', type=int, default=1, help='test fold')
    parser.add_argument('--exp_mode', type=str, default='normal-random', help='')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--device', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', help='use multiple gpus', default=True)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')


    parser.add_argument('--epochs', type=int, default=80, help='train epochs')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=20, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='data loader num workers')
    parser.add_argument('--seed', type=int, default=1024, help='random seed')
    parser.add_argument('--weight', type=float, nargs='+', default=[1, 1, 1], help='loss weight')
    parser.add_argument('--ratio', type=float, default=0.05, help='')
    parser.add_argument('--total', type=int, default=3, help='')


    parser.add_argument('--num_nodes', type=int, default=10, help='')
    parser.add_argument('--seq_len', type=int, default=3000, help='')
    parser.add_argument('--freq_len', type=int, default=3000, help='')
    parser.add_argument('--num_class', type=int, default=5)
    parser.add_argument('--out_channels', type=int, default=512, help='')



    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.device = args.device_ids[0]

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print('Args in experiment:')
    print_args(args)

    Exp = Exp_Sleep_ACL_Cross

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            exp = Exp(args)
            setting =  '{}_{}_bs{}_{}'.format(
                args.model,
                args.dataset,
                args.batch_size,
                ii)

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            # print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            # exp.test(setting)
            torch.cuda.empty_cache()
    else:
        setting = '{}_{}_bs{}_'.format(
            args.model,
            args.dataset,
            args.batch_size)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)
        torch.cuda.empty_cache()