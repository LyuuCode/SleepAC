# SleepAC

This repository contains the official implementation of our paper:

> **SleepAC: Less Dependency on Manual Annotations, More Reliable Sampling for Automatic Sleep Staging**



## Project Structure

```perl
SleepAC/
│  main.py                # Main script to train and evaluate the model
├─exp/                    # Experimental settings and training logic
├─layers/                 # Model building blocks
├─models/                 # SleepAC model architecture
└─utils/                  # Data loading, sampling strategies, helper functions
```



## Environment Setup

```bash
pip install -r requirements.txt
```



## Datasets

The SleepEDF dataset is on https://physionet.org/content/sleep-edfx/1.0.0/

The ISRUC dataset is on https://sleeptight.isr.uc.pt/

The SHHS dataset is on https://sleepdata.org/datasets/shhs



## Training

To train the SleepAC model, run:

```bash
python main.py
```

