import os
import torch

from models import SleepAC


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'SleepAC': SleepAC,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
            device = torch.device('cuda:{}'.format(self.args.device))
            print('Use GPU: cuda:{}'.format(self.args.device))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
