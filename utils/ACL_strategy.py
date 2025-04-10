import random
import time
from datetime import datetime

from scipy.spatial import cKDTree
from sklearn.metrics import pairwise_distances
import numpy as np
import torch
import pywt
import faiss
from torch import autocast


def select_simple_samples_by_statistical_features(data, num_samples):
    std_devs = torch.std(data, dim=1)
    avg_std_devs = torch.mean(std_devs, dim=1)
    simple_indices = torch.argsort(avg_std_devs)[:num_samples]
    return simple_indices

def select_simple_samples_by_frequency_features(data, num_samples):
    sample_rate = 100
    target_freq_range = [0.5, 60]
    freq_bins = np.fft.rfftfreq(data.shape[1], d=1./sample_rate)
    target_freq_indices = np.where((freq_bins >= target_freq_range[0]) & (freq_bins <= target_freq_range[1]))[0]

    energy_in_target_freq = torch.zeros(data.shape[0])
    fft_result = torch.fft.rfft(data, dim=0)
    # for i in range(data.shape[0]):
    #     energy_in_target_freq[i] = torch.sum(torch.abs(fft_result[target_freq_indices, :])**2)
    energy_in_target_freq = torch.sum(
        torch.pow(torch.abs(fft_result[:, target_freq_indices, :]), 2),
        dim=(1, 2)
    )

    simple_indices = torch.argsort(energy_in_target_freq)[-num_samples:]
    return simple_indices

def denoise_signal_with_wavelet(signal, wavelet='db4', level=5):
    coeffs = pywt.wavedec(signal, wavelet, mode='symmetric')
    threshold = np.median(np.abs(coeffs[-level])) / 0.6745 * np.sqrt(2 * np.log(len(signal)))
    denoised_coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
    return pywt.waverec(denoised_coeffs, wavelet, mode='symmetric')

def select_simple_samples_by_wavelet_denoising(data, num_samples, wavelet='db4', level=5):
    denoised_signals = torch.zeros_like(data)
    # for i in range(data.shape[0]):
    for j in range(data.shape[2]):
        denoised_signals[:, :, j] = torch.tensor(
            denoise_signal_with_wavelet(data[:, :, j].cpu().numpy(), wavelet, level))

    noise_levels = torch.mean(torch.abs(data - denoised_signals), dim=(1, 2))
    simple_indices = torch.argsort(noise_levels)[:num_samples]
    return simple_indices

def select_simple_samples(unlabeled_data, num_samples, method='wavelet'):
    n = num_samples//3
    print(datetime.now().strftime("%H:%M:%S"), 'simple indices1 start ...')
    indices1 = select_simple_samples_by_statistical_features(unlabeled_data, num_samples)
    print(datetime.now().strftime("%H:%M:%S"), 'simple indices1 end')
    print(datetime.now().strftime("%H:%M:%S"), 'simple indices2 start ...')
    indices2 = select_simple_samples_by_frequency_features(unlabeled_data, num_samples)
    print(datetime.now().strftime("%H:%M:%S"), 'simple indices2 end')
    print(datetime.now().strftime("%H:%M:%S"), 'simple indices3 start ...')
    indices3 = select_simple_samples_by_wavelet_denoising(unlabeled_data, num_samples)
    print(datetime.now().strftime("%H:%M:%S"), 'simple indices3 end')
    indices_thrice, indices_twice, indices_once = find_common_and_unique_samples([indices1, indices2, indices3])
    indices_common = indices_thrice.tolist() + indices_twice.tolist()
    indices_unique = indices_once.tolist()
    if len(indices_common) > 2 * n and len(indices_unique) > n:
        simple_sample_indices = random.sample(indices_common, 2 * n) + random.sample(indices_unique, n)
    elif len(indices_common) <= 2 * n:
        simple_sample_indices = indices_common + random.sample(indices_unique, num_samples - len(indices_common))
    elif len(indices_unique) <= n:
        simple_sample_indices = indices_unique + random.sample(indices_common, num_samples - len(indices_unique))
    print('simple_sample_indices:', len(simple_sample_indices))
    return simple_sample_indices

def select_by_uncertainty(output, num_samples):
    uncertainties=[]

    probs = torch.softmax(output, dim=1)
    uncertainty = -torch.max(probs, dim=1)[0]
    uncertainties.append(uncertainty.cpu())
    uncertainties = torch.cat(uncertainties)
    selected_indices = torch.argsort(uncertainties, descending=True)[:num_samples]
    torch.cuda.empty_cache()
    return selected_indices.to('cpu')

def select_by_density5(data, num_samples):
    data_np = data.view(data.shape[0], -1).cpu().numpy()
    distances = pairwise_distances(data_np)
    density = np.mean(distances, axis=1)
    selected_indices = np.argsort(density)[:num_samples]
    return torch.tensor(selected_indices)

def select_by_density(data, num_samples, k=10, gpu_ids=None):
    data_np = data.view(data.shape[0], -1).cpu().numpy().astype(np.float32)
    num_samples_total, feature_dim = data_np.shape

    if gpu_ids is None:
        gpu_ids = faiss.get_num_gpus()
        gpu_ids = list(range(gpu_ids))


    res = [faiss.StandardGpuResources() for _ in gpu_ids]
    gpu_index = faiss.GpuIndexFlatL2(res[0], feature_dim)
    gpu_index.add(data_np)
    distances, _ = gpu_index.search(data_np, k + 1)
    density = np.mean(distances[:, 1:], axis=1)
    selected_indices = np.argsort(density)[:num_samples]
    gpu_index.reset()
    return torch.tensor(selected_indices)

def select_by_density2(data, num_samples, k=10):
    data_np = data.view(data.shape[0], -1).cpu().numpy().astype(np.float32)
    num_samples_total, feature_dim = data_np.shape

    index = faiss.IndexFlatL2(feature_dim)
    index.add(data_np)
    distances, _ = index.search(data_np, k + 1)

    density = np.mean(distances[:, 1:], axis=1)

    selected_indices = np.argsort(density)[:num_samples]

    return torch.tensor(selected_indices)

def select_by_diversity(labeled_data, unlabeled_data, num_samples, gpu_ids=None):

    labeled_data_np = labeled_data.view(labeled_data.shape[0], -1).cpu().numpy().astype(np.float32)
    unlabeled_data_np = unlabeled_data.view(unlabeled_data.shape[0], -1).cpu().numpy().astype(np.float32)

    num_labeled, feature_dim = labeled_data_np.shape

    if gpu_ids is None:
        gpu_ids = faiss.get_num_gpus()
        gpu_ids = list(range(gpu_ids))


    res = [faiss.StandardGpuResources() for _ in gpu_ids]

    gpu_index = faiss.GpuIndexFlatL2(res[0], feature_dim)

    gpu_index.add(labeled_data_np)

    distances, _ = gpu_index.search(unlabeled_data_np, 1)

    selected_indices = np.argsort(distances[:, 0])[-num_samples:]

    return torch.tensor(selected_indices)

def select_by_diversity2(labeled_data, unlabeled_data, num_samples):

    labeled_data_np = labeled_data.view(labeled_data.shape[0], -1).cpu().numpy()
    unlabeled_data_np = unlabeled_data.view(unlabeled_data.shape[0], -1).cpu().numpy()
    distances = pairwise_distances(unlabeled_data_np, labeled_data_np)
    min_distances = np.min(distances, axis=1)
    selected_indices = np.argsort(min_distances)[-num_samples:]
    return torch.tensor(selected_indices)

def select_by_diversity3(labeled_data, unlabeled_data, num_samples):

    labeled_data_np = labeled_data.view(labeled_data.shape[0], -1).cpu().numpy().astype(np.float32)
    unlabeled_data_np = unlabeled_data.view(unlabeled_data.shape[0], -1).cpu().numpy().astype(np.float32)

    tree = cKDTree(labeled_data_np)

    min_distances, _ = tree.query(unlabeled_data_np, k=1)

    selected_indices = np.argsort(min_distances)[-num_samples:]

    return torch.tensor(selected_indices)

def get_data_enc(model, data):
    torch.cuda.empty_cache()
    model.eval()
    batch_size = 512
    encs = []
    outputs = []
    stft = data['stft']
    data = data['data']
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = {'data': data[i:i + batch_size], 'stft': stft[i:i + batch_size]}
            with autocast(device_type='cuda'):  # 启用混合精度
                output, enc = model(batch, is_al=True, is_train=False)
            encs.append(enc)
            outputs.append(output)
        encs = torch.cat(encs)
        outputs = torch.cat(outputs)
        torch.cuda.empty_cache()
    return encs, outputs

def select_most_valuable_samples(model, labeled_data, unlabeled_data, num_samples):

    n = num_samples//3

    unlabeled_enc, unlabeled_output = get_data_enc(model, unlabeled_data)
    labeled_enc, _ = get_data_enc(model, labeled_data)

    print(datetime.now().strftime("%H:%M:%S"), 'most valuable indices1 start ...')
    indices1 = select_by_uncertainty(unlabeled_output, num_samples)
    print(datetime.now().strftime("%H:%M:%S"), 'most valuable indices1 end')

    print(datetime.now().strftime("%H:%M:%S"), 'most valuable indices2 start ...')
    indices2 = select_by_density(unlabeled_enc, num_samples, gpu_ids=[1])
    print(datetime.now().strftime("%H:%M:%S"), 'most valuable indices2 end')

    print(datetime.now().strftime("%H:%M:%S"), 'most valuable indices3 start ...')
    indices3 = select_by_diversity(labeled_enc, unlabeled_enc, num_samples, gpu_ids=[2])
    print(datetime.now().strftime("%H:%M:%S"), 'most valuable indices3 end')

    indices_thrice, indices_twice, indices_once = find_common_and_unique_samples([indices1, indices2, indices3])
    indices_common = indices_thrice.tolist() +indices_twice.tolist()
    indices_unique = indices_once.tolist()
    if len(indices_common) > 2*n and len(indices_unique) > n:
        selected_indices = random.sample(indices_common, 2*n)+random.sample(indices_unique, n)
    elif len(indices_common) <= 2*n:
        selected_indices = indices_common+random.sample(indices_unique, num_samples-len(indices_common))
    elif len(indices_unique) <= n:
        selected_indices = indices_unique+random.sample(indices_common, num_samples-len(indices_unique))
    print('selected_indices:', len(selected_indices))
    return selected_indices

def find_common_and_unique_samples(indices_list):

    all_indices = torch.cat(indices_list)

    unique_indices, counts = torch.unique(all_indices, return_counts=True)

    indices_thrice = unique_indices[counts == 3]

    indices_twice = unique_indices[counts == 2]

    indices_once = unique_indices[counts == 1]

    return indices_thrice, indices_twice, indices_once

