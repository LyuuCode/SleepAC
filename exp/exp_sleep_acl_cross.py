import os
import time
from collections import Counter

import torch
from torch import nn, optim
from torch.utils.data import Subset, DataLoader, TensorDataset

from exp.exp_basic import Exp_Basic
from layers.losses import FocalLoss, ContrastiveLoss
from utils.ACL_strategy import select_most_valuable_samples, select_simple_samples
from utils.datautils import *
from utils.tools import EarlyStopping, generate_mask, adjust_learning_rate, save_results, \
    get_accuracy


class Exp_Sleep_ACL_Cross(Exp_Basic):
    def __init__(self, args):
        super(Exp_Sleep_ACL_Cross, self).__init__(args)
        self.train_datas, self.train_stfts, self.train_labels, self.valid_datas, self.valid_stfts, self.valid_labels = get_dataset_cross(self.args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args)
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
            model.to(self.args.device)
        if isinstance(model, torch.nn.DataParallel):
            print("Model is wrapped in DataParallel, using multiple GPUs.")
        else:
            print("Model is not wrapped in DataParallel.")
        return model

    def _select_optimizer(self):
        model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    # def _select_criterion(self):
    #     ce_loss = nn.CrossEntropyLoss()
    #     focal_loss = FocalLoss()
    #     cl_loss = ContrastiveLoss()
    #     return {'ce_loss':ce_loss, 'focal_loss': focal_loss, 'cl_loss':cl_loss}

    def _select_criterion(self):
        focal_loss = FocalLoss()
        cl_loss = ContrastiveLoss()
        return {'ce_loss':focal_loss, 'cl_loss':cl_loss}

    def train(self, setting):
        for k in range(10):
            self.path = os.path.join(self.args.checkpoints, setting+str(k))
            if not os.path.exists(self.path):
                os.makedirs(self.path)
            best_result = {'accuracy': 0, 'f1_score': 0, 'kappa': 0, 'wake_f1': 0, 'n1_f1': 0,
                               'n2_f1': 0, 'n3_f1': 0, 'rem_f1': 0, 'cm': 0}
            self.model = self._build_model()
            self.model_optim = self._select_optimizer()
            self.criterion = self._select_criterion()
            early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

            unlabeled = self.get_data(k, is_training=True)
            num_samples = int(len(unlabeled['data'])*self.args.ratio)
            labeled, unlabeled = self.get_labeled_data_by_CL(unlabeled, num_samples) # CL

            self.train_loader = self.get_train_loader(labeled)
            self.valid_loader = self.get_data(k, is_training=False)

            best_result = self.train_model(setting, k, early_stopping, best_result)

            for i in range(self.args.total):
                self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints + setting + str(k), 'checkpoint.pth')))
                labeled, unlabeled = self.get_labeled_data_by_ACL(labeled, unlabeled, num_samples, i) #AL

                self.train_loader = self.get_train_loader(labeled)
                early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, best_score=best_result['f1_score'])

                best_result = self.train_model(setting, k, early_stopping, best_result)

            save_results(self.args.result_path, 'Fold' + str(k + 1) + '_' + setting, best_result, 0)
        return

    def train_model(self, setting, k, early_stopping, best_result):
        path = os.path.join(self.args.checkpoints, setting + str(k))
        if not os.path.exists(path):
            os.makedirs(path)
        train_steps = len(self.train_loader)

        for epoch in range(self.args.epochs):
            torch.cuda.empty_cache()
            start_time = time.time()
            loss = self.train_one_epoch(epoch)
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - start_time))

            train_loss = np.average(loss)
            vali_loss, acc, f1, kappa, cm, wake_f1, n1_f1, n2_f1, n3_f1, rem_f1 = self.vali()
            if best_result['accuracy'] < acc:
                best_result.update(accuracy=acc, f1_score=f1, kappa=kappa, cm=cm,
                                   wake_f1=wake_f1, n1_f1=n1_f1, n2_f1=n2_f1, n3_f1=n3_f1, rem_f1=rem_f1)
            print(
                "Fold: {0}, Epoch: {1}, Steps: {2} | Train Loss: {3:.3f} Vali Loss: {4:.3f} | "
                "Total_Acc: {5:.3f} Total_F1: {6:.3f} Kappa: {7:.3f} | "
                "Wake_F1: {8:.3f} N1_F1: {9:.3f} N2_F1: {10:.3f} N3_F1: {11:.3f} REM_F1: {12:.3f}"
                .format(k + 1, epoch + 1, train_steps, train_loss, vali_loss,
                        acc, f1, kappa,
                        wake_f1, n1_f1, n2_f1, n3_f1, rem_f1))

            early_stopping(-f1, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            if (epoch + 1) % 5 == 0:
                adjust_learning_rate(self.model_optim, epoch + 1, self.args)
            save_results(self.args.result_path, 'Fold' + str(k + 1) + '_' + setting, best_result, 0)
        return best_result

    def train_one_epoch(self, epoch):
        iter_count = 0
        train_loss = []
        train_steps = len(self.train_loader)
        time_now = time.time()
        self.model.train()
        for i, (data, stft, data_mask, label) in enumerate(self.train_loader):
            iter_count += 1

            data = data.to(self.device)
            stft = stft.to(self.device)
            label = label.to(self.device)
            data_mask = data_mask.to(self.device)
            batch = {'data': data, 'stft': stft, 'data_mask': data_mask}

            enc_out, cls_out, recon_out = self.model(batch, is_train=True)

            CELoss = self.criterion['ce_loss']
            CLLoss = self.criterion['cl_loss']

            ce_loss = CELoss(cls_out, label.long().squeeze(-1))
            cl_loss = CLLoss(enc_out, label)
            recon_loss = 1 * torch.sum(torch.pow((data - recon_out) * data_mask, 2)) / (
                                torch.sum(data_mask) + 1e-10) / 2
            weight = self.args.weight
            loss = ce_loss*weight[0] + cl_loss*weight[1] + recon_loss*weight[2]
            train_loss.append(loss.item())

            if (i + 1) % 100 == 0:
                print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((self.args.epochs - epoch) * train_steps - i)
                print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
            self.model_optim.step()
        return train_loss

    def vali(self):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        CELoss = self.criterion['ce_loss']
        with torch.no_grad():
            for i, (data, stft, label) in enumerate(self.valid_loader):
                data = data.to(self.device)
                stft = stft.to(self.device)
                label = label.to(self.device)

                outputs = self.model({'data': data, 'stft': stft})

                pred = outputs.detach().cpu()
                loss = CELoss(pred, label.long().squeeze().cpu())
                total_loss.append(loss)

                preds.append(outputs.detach())
                trues.append(label)

            total_loss = np.average(total_loss)

            preds = torch.cat(preds, 0)
            trues = torch.cat(trues, 0)
            probs = torch.nn.functional.softmax(preds, dim=-1)  # (total_samples, num_classes) est. prob. for each class and sample
            predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
            trues = trues.flatten().cpu().numpy()
            acc, f1, kappa, cm, wake_f1, n1_f1, n2_f1, n3_f1, rem_f1 = get_accuracy(predictions, trues)

            return total_loss, acc, f1, kappa, cm, wake_f1, n1_f1, n2_f1, n3_f1, rem_f1

    def test(self, setting):
        preds = []
        trues = []
        k = self.args.test_fold
        test_loader = self.get_loaders(k, False)
        self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints + setting +str(k), 'checkpoint.pth')))

        with torch.no_grad():
            for i, (data, stft, label) in enumerate(test_loader):

                data = data.to(self.device)
                stft = stft.to(self.device)
                label = label.to(self.device)

                outputs = self.model({'data': data, 'stft': stft})

                preds.append(outputs.detach())
                trues.append(label)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = torch.nn.functional.softmax(preds, dim=-1)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        probs = probs.cpu().numpy()  # 转换为Numpy数组
        trues = trues.flatten().cpu().numpy()
        np.savetxt('prob.txt', probs)
        np.savetxt('predict.txt', predictions, fmt='%d')
        np.savetxt('trues.txt', trues, fmt='%d')

        acc, f1, kappa, cm, wake_f1, n1_f1, n2_f1, n3_f1, rem_f1 = get_accuracy(predictions, trues)
        print(
            "Total_Acc: {0:.3f} Total_F1: {1:.3f} Kappa: {2:.3f} | "
            "Wake_F1: {3:.3f} N1_F1: {4:.3f} N2_F1: {5:.3f} N3_F1: {6:.3f} REM_F1: {7:.3f}"
            .format( acc, f1, kappa, wake_f1, n1_f1, n2_f1, n3_f1,
                    rem_f1))

        return acc, f1, kappa, cm, wake_f1, n1_f1, n2_f1, n3_f1, rem_f1

    def get_data(self, k, is_training: bool = True):
        if is_training:
            print("get fold ", str(k + 1), " train loader ...")
            label = self.train_labels[k]  # batch,
            data = self.train_datas[k]  # batch, length, channel
            data_mask = generate_mask(data, 0.5)
            stft = self.train_stfts[k]  # batch, length, channel
            return {'data': data, 'stft': stft, 'data_mask': data_mask, 'label': label}
        else:
            print("get fold ", str(k + 1), " valid loader ...")
            label = self.valid_labels[k]  # batch,
            data = self.valid_datas[k]  # batch, length, channel
            stft = self.valid_stfts[k]  # batch, length, channel
            datasets = TensorDataset(data, stft, label)
            dataloader = DataLoader(datasets,
                                    batch_size=self.args.batch_size,
                                    shuffle=True,
                                    num_workers=self.args.num_workers,
                                    pin_memory=True,
                                    drop_last=True)
            return dataloader

    def get_train_loader(self, labeled):
        data = labeled['data']
        stft = labeled['stft']
        data_mask = labeled['data_mask']
        label = labeled['label']
        datasets = TensorDataset(data, stft, data_mask, label)
        train_loader = DataLoader(datasets,
                                       batch_size=self.args.batch_size,
                                       shuffle=True,
                                       num_workers=self.amrgs.num_workers,
                                       pin_memory=True,
                                       drop_last=True)
        return train_loader

    def get_labeled_data_by_CL(self, unlabeled, num_samples):
        data = unlabeled['data']
        stft = unlabeled['stft']
        data_mask = unlabeled['data_mask']
        label = unlabeled['label']

        labeled_indices = select_simple_samples(data, num_samples)
        all_indices = set(range(len(data)))
        unlabeled_indices = all_indices - set(labeled_indices)
        labeled_data = data[labeled_indices]
        labeled_stft= stft[labeled_indices]
        labeled_data_mask = data_mask[labeled_indices]
        labeled_label = label[labeled_indices]

        unlabeled_data = data[list(unlabeled_indices)]
        unlabeled_label = label[list(unlabeled_indices)]
        unlabeled_stft = stft[list(unlabeled_indices)]
        unlabeled_data_mask = data_mask[list(unlabeled_indices)]

        counts = [0, 0, 0, 0, 0]
        label_counts = Counter(labeled_label)
        for label, count in label_counts.items():
            counts[label] += 1
        print(counts)

        labeled = {'data': labeled_data, 'data_mask': labeled_data_mask,
                   'stft': labeled_stft, 'label': labeled_label}
        unlabeled = {'data': unlabeled_data, 'data_mask': unlabeled_data_mask,
                     'stft': unlabeled_stft, 'label': unlabeled_label}
        return labeled, unlabeled

    def get_labeled_data_by_ACL(self, labeled, unlabeled, num_samples, epoch, max_epoch):

        labeled_indices_AL = select_most_valuable_samples(self.model, labeled, unlabeled, num_samples)
        labeled_indices_CL = select_simple_samples(unlabeled, num_samples)
        alpha = max(0.0, 1.0 - epoch / max_epoch)
        beta = 1.0 - alpha

        num_CL = int(num_samples * alpha)
        num_AL = num_samples - num_CL

        random.shuffle(labeled_indices_CL)
        random.shuffle(labeled_indices_AL)

        selected_CL = labeled_indices_CL[:num_CL]
        selected_AL = labeled_indices_AL[:num_AL]

        labeled_indices = selected_CL + selected_AL

        all_indices = set(range(len(unlabeled['data'])))
        unlabeled_indices = all_indices - set(labeled_indices)

        labeled_data = torch.cat((unlabeled['data'][labeled_indices],labeled['data']), dim=0)
        labeled_stft = torch.cat((unlabeled['stft'][labeled_indices],labeled['stft']), dim=0)
        labeled_label = torch.cat((unlabeled['label'][labeled_indices],labeled['label']), dim=0)
        labeled_data_mask = torch.cat((unlabeled['data_mask'][labeled_indices],labeled['data_mask']), dim=0)

        unlabeled_data = unlabeled['data'][list(unlabeled_indices)]
        unlabeled_label = unlabeled['label'][list(unlabeled_indices)]
        unlabeled_stft = unlabeled['stft'][list(unlabeled_indices)]
        unlabeled_data_mask = unlabeled['data_mask'][list(unlabeled_indices)]

        counts = [0, 0, 0, 0, 0]
        label_counts = Counter(labeled_label)
        for label, count in label_counts.items():
            counts[label] += 1
        print('labeled: ', len(labeled_label))
        print(counts)

        labeled = {'data': labeled_data, 'data_mask': labeled_data_mask,
                   'stft': labeled_stft, 'label': labeled_label}
        unlabeled = {'data': unlabeled_data, 'data_mask': unlabeled_data_mask,
                     'stft': unlabeled_stft, 'label': unlabeled_label}
        return labeled, unlabeled

