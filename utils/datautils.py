import argparse
import os
import random
import numpy as np
import torch
from scipy import signal

def get_dataset_cross(configs):
    path = configs.data_path + configs.dataset + '.npz'
    ReadList = np.load(path, allow_pickle=True)
    num = ReadList['Fold_len']
    data = ReadList['Fold_data']
    stft = ReadList['Fold_stft']
    label = ReadList['Fold_label']

    data = torch.tensor(data).float()
    stft = torch.tensor(stft).float()
    label = torch.tensor(label).squeeze()

    unique_labels, counts = torch.unique(label, return_counts=True)
    print("labels counts: ", counts)

    Data = []
    Stft = []
    Label = []
    start = 0
    for i in range(num.shape[0]):
        start_idx = start
        end_idx = start_idx + num[i] - 1
        start = end_idx + 1
        Data.append(data[start_idx:end_idx])
        Stft.append(stft[start_idx:end_idx])
        Label.append(label[start_idx:end_idx])

    train_datas = []
    train_stfts = []
    train_labels = []
    test_datas = []
    test_stfts = []
    test_labels = []
    for k in range(10):
        train_data = []
        train_stft = []
        train_label = []
        test_data = []
        test_stft = []
        test_label = []
        for i in range(num.shape[0]):
            data = Data[i]
            stft = Stft[i]
            label = Label[i]
            if i%10 != k:
                train_data.append(data)
                train_stft.append(stft)
                train_label.append(label)
            else:
                test_data.append(data)
                test_stft.append(stft)
                test_label.append(label)

        Train_Data = torch.cat(train_data)
        Train_Stft = torch.cat(train_stft)
        Train_Label = torch.cat(train_label)
        Test_Data = torch.cat(test_data)
        Test_Stft = torch.cat(test_stft)
        Test_Label = torch.cat(test_label)

        print("Fold:", k+1)
        _, counts = torch.unique(Train_Label, return_counts=True)
        print("train labels counts: ", counts)
        _, counts = torch.unique(Test_Label, return_counts=True)
        print("test labels counts: ", counts)
        train_datas.append(Train_Data)
        train_stfts.append(Train_Stft)
        train_labels.append(Train_Label)
        test_datas.append(Test_Data)
        test_stfts.append(Test_Stft)
        test_labels.append(Test_Label)

    return train_datas, train_stfts, train_labels, test_datas, test_stfts, test_labels

def get_dataset_random(configs):
    path = configs.data_path + configs.dataset
    assert configs.dataset == 'SHHS1'

    all_files = [f for f in os.listdir(path) if f.endswith('.npz')]

    random.shuffle(all_files)
    print(len(all_files))
    num_selected_files = int(len(all_files))
    all_files = all_files[:num_selected_files]
    num_train_files = int(0.7 * len(all_files))
    train_files = all_files[:num_train_files]
    test_files = all_files[num_train_files:]
    print(len(all_files), len(train_files), len(test_files))

    # 初始化存储列表
    train_datas, train_stfts, train_labels = [], [], []
    test_datas, test_stfts, test_labels = [], [], []

    def load_data(file_list, data_list, stft_list, label_list):
        for file_name in file_list:
            file_path = os.path.join(path, file_name)
            ReadList = np.load(file_path, allow_pickle=True)

            data_list.append(torch.tensor(ReadList['Fold_data'], dtype=torch.float32))
            stft_list.append(torch.tensor(ReadList['Fold_stft'], dtype=torch.float32))
            label_list.append(torch.tensor(ReadList['Fold_label'], dtype=torch.long))

    # 加载训练数据
    print('加载train data...')
    load_data(train_files, train_datas, train_stfts, train_labels)
    print('加载train data完成')
    # 加载测试数据
    print('加载test data...')
    load_data(test_files, test_datas, test_stfts, test_labels)
    print('加载test data完成')

    # 合并不同文件的数据为单个 Tensor
    train_datas = torch.cat(train_datas, dim=0)
    train_stfts = torch.cat(train_stfts, dim=0)
    train_labels = torch.cat(train_labels, dim=0)

    test_datas = torch.cat(test_datas, dim=0)
    test_stfts = torch.cat(test_stfts, dim=0)
    test_labels = torch.cat(test_labels, dim=0)
    print(train_datas.shape, test_datas.shape)
    _, counts = torch.unique(train_labels, return_counts=True)
    print("train labels counts: ", counts)
    _, counts = torch.unique(test_labels, return_counts=True)
    print("test labels counts: ", counts)
    return train_datas, train_stfts, train_labels, test_datas, test_stfts, test_labels

def get_dataset_cross_transformer(configs):
    path = configs.data_path + configs.dataset + '.npz'
    ReadList = np.load(path, allow_pickle=True)
    num = ReadList['Fold_len']
    data = ReadList['Fold_data']
    label = ReadList['Fold_label']

    data = torch.tensor(data).float()
    label = torch.tensor(label).squeeze()

    unique_labels, counts = torch.unique(label, return_counts=True)
    print("labels counts: ", counts)

    Data = []
    Label = []
    start = 0
    for i in range(num.shape[0]):
        start_idx = start
        end_idx = start_idx + num[i] - 1
        start = end_idx + 1
        cur_data = data[start_idx:end_idx]
        cur_label = label[start_idx:end_idx]
        nums = cur_data.shape[0]
        new_nums = nums // 5 * 5
        # 丢弃多余的数据
        data_trimmed = cur_data[:new_nums]  # (new_nums, lens, channels)
        label_trimmed = cur_label[:new_nums]  # (new_nums,)

        # 重新 reshape Data 和 Label
        data_reshaped = data_trimmed.reshape(-1, 5, cur_data.shape[1], cur_data.shape[2])  # (num, 20, lens, channels)
        label_reshaped = label_trimmed.reshape(-1, 5)  # (num, 20)
        Data.append(data_reshaped)
        Label.append(label_reshaped)

    train_datas = []
    train_labels = []
    test_datas = []
    test_labels = []
    for k in range(10):
        train_data = []
        train_label = []
        test_data = []
        test_label = []
        for i in range(num.shape[0]):
            data = Data[i]
            label = Label[i]
            if i % 10 != k:
                train_data.append(data)
                train_label.append(label)
            else:
                test_data.append(data)
                test_label.append(label)

        Train_Data = torch.cat(train_data)
        Train_Label = torch.cat(train_label)
        Test_Data = torch.cat(test_data)
        Test_Label = torch.cat(test_label)

        print("Fold:", k+1)
        _, counts = torch.unique(Train_Label, return_counts=True)
        print("train labels counts: ", counts)
        _, counts = torch.unique(Test_Label, return_counts=True)
        print("test labels counts: ", counts)
        train_datas.append(Train_Data)
        train_labels.append(Train_Label)
        test_datas.append(Test_Data)
        test_labels.append(Test_Label)

    return train_datas, train_labels, test_datas, test_labels

# parser = argparse.ArgumentParser(description='sleepstage')
# parser.add_argument('--dataset', type=str, default='Sleep-EDF-20',
#                         help='dataset type, options:[ISRUC-S1, ISRUC-S3, Sleep-EDF-20, SHHS1]')
# parser.add_argument('--data_path', type=str, default='/lab/2023/lss/datasets/EEG/idea1/',
#                         help='root path of the data file')
# args = parser.parse_args()
# # # get_dataset_cross(args)
# # get_dataset_random(args)
# get_dataset_cross_transformer(args)
