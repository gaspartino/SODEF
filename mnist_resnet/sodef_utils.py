import os
import random
import argparse

import numpy as np
import torch

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--data-dir', type=str, default='../cifar-data')
    parser.add_argument('--dataset', type=str, default='lisa')
    parser.add_argument('--normalize', action='store_true', help='Ativa normalização dos dados')
    parser.add_argument('--epsilon', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--total_loops', '-tl', type=int, default=1)

    parser.add_argument('--epochs_phase1', '-ep1', type=int, default=40)
    parser.add_argument('--epochs_phase2', '-ep2', type=int, default=10)
    parser.add_argument('--epochs_phase3', '-ep3', type=int, default=5)

    parser.add_argument('--train-savepath', type=str, default='./data/train_resnet_final.npz')
    parser.add_argument('--test-savepath', type=str, default='./data/test_resnet_final.npz')
    parser.add_argument('--folder-savemodel', type=str, default='./EXP/models')

    return parser.parse_args()

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def inf_generator(iterable):
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()
