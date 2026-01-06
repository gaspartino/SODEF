%%writefile /kaggle/working/SODEF/mnist_resnet/attack.py

import os
import argparse
import logging
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchattacks
import torchvision
import torchvision.transforms as transforms
import kagglehub
import geotorch

from models import *
from art.utils import load_mnist
from types import SimpleNamespace
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.datasets import MNIST, CIFAR10, ImageFolder
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent, CarliniL2Method

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.set_printoptions(threshold=np.inf, suppress=True)

fc_dim = 64
endtime = 5

folder_savemodel = './EXP/MNIST_resnet_final'
folder = './EXP/resnetfct5_15/model.pth'

act = torch.sin 

saved = torch.load(folder, weights_only=False)
print(folder)

statedic = saved['state_dict']
args = saved['args']
args.tol = 1e-5
print(args)
f_coeffi = -1

from torchdiffeq import odeint_adjoint as odeint

class ConcatFC(nn.Module):

    def __init__(self, dim_in, dim_out):
        super(ConcatFC, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        
    def forward(self, t, x):
        return self._layer(x)

class ODEfunc_mlp(nn.Module):

    def __init__(self, dim):
        super(ODEfunc_mlp, self).__init__()
        self.fc1 = ConcatFC(fc_dim, fc_dim)
        self.act1 = act
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = f_coeffi*self.fc1(t, x)
        out = self.act1(out)
        return out 
     
class ODEBlock(nn.Module):

    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, endtime]).float()

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time, rtol=args.tol, atol=args.tol)
        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value

class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)


class RunningAverageMeter(object):
    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val

class newLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(newLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features,out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return F.linear(input, self.weight.T, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class ORTHFC(nn.Module):
    def __init__(self, dimin, dimout, bias):
        super(ORTHFC, self).__init__()
        if dimin >= dimout:
            self.linear = newLinear(dimin, dimout,  bias=bias)
        else:
            self.linear = nn.Linear(dimin, dimout,  bias=bias)
        geotorch.orthogonal(self.linear, "weight")

    def forward(self, x):
        return self.linear(x)

class ORTHFC_NOBAIS(nn.Module):
    def __init__(self, dimin, dimout):
        super(ORTHFC_NOBAIS, self).__init__()
        if dimin >= dimout:
            self.linear = newLinear(dimin, dimout,  bias=False)
        else:
            self.linear = nn.Linear(dimin, dimout,  bias=False)
        geotorch.orthogonal(self.linear, "weight")

    def forward(self, x):
        return self.linear(x)
class MLP_OUT_ORT(nn.Module):
    def __init__(self, out_features=10):
        super(MLP_OUT_ORT, self).__init__()
        self.fc0 = ORTHFC(fc_dim, out_features, False)
    def forward(self, input_):
        h1 = self.fc0(input_)
        return h1

class MLP_OUT_final(nn.Module):

    def __init__(self, out_features=10):
        super(MLP_OUT_final, self).__init__()
        self.fc0 = nn.Linear(fc_dim, out_features)
    def forward(self, input_):
        h1 = self.fc0(input_)
        return h1

class MLP_OUT_BALL(nn.Module):
    def __init__(self, out_features=10):
        super(MLP_OUT_BALL, self).__init__()
        self.fc0 = nn.Linear(64, out_features, bias=False)
    def forward(self, input_):
        h1 = self.fc0(input_)
        return h1  

    
def one_hot(x, K):
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)

def accuracy(model, dataset_loader):
    total_correct = 0
    for x, y in dataset_loader:
        x = x.to(device)
        y = one_hot(np.array(y.numpy()), 4)

        target_class = np.argmax(y, axis=1)
        predicted_class = np.argmax(model(x).cpu().detach().numpy(), axis=1)
        total_correct += np.sum(predicted_class == target_class)
    return total_correct / len(dataset_loader.dataset)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    
device = 'cuda' 
best_acc = 0
start_epoch = 0

# Data
print('==> Preparing data..')

def bstl_loaders(train_batch_size=256, test_batch_size=64, normalize=False):
    transform_list = [transforms.Resize((64, 32)), transforms.ToTensor()]

    if normalize:
        transform_list.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        )

    transform = transforms.Compose(transform_list)
    
    data_dir = '/kaggle/input/bstl-dataset'
    train_dir = f"{data_dir}/train"
    test_dir = f"{data_dir}/test"

    train_data = ImageFolder(root=train_dir, transform=transform)
    test_data = ImageFolder(root=test_dir, transform=transform)

    train_loader = DataLoader(train_data, batch_size=train_batch_size,
                             shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=test_batch_size,
                            shuffle=False, pin_memory=True)
    train_eval_loader = DataLoader(train_data, batch_size=train_batch_size,
                             shuffle=False, pin_memory=True)
    return train_loader, test_loader, train_eval_loader, test_data, 4

def lisa_loaders(train_batch_size=256, test_batch_size=64, normalize=False):
    path = kagglehub.dataset_download("chandanakuntala/cropped-lisa-traffic-light-dataset")
    
    print("Path to dataset files:", path)

    train_dir = f"{path}/cropped_lisa_1/train_1"
    val_dir = f"{path}/cropped_lisa_1/val_1"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform_list = [transforms.Resize((32, 32)), transforms.ToTensor()]

    if normalize:
        transform_list.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        )

    transform = transforms.Compose(transform_list)

    train_dataset = ImageFolder(train_dir, transform=transform)
    test_dataset = ImageFolder(val_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=2)
    train_eval_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader, train_eval_loader, test_dataset, 7

def cifar10_loaders(train_batch_size=256, test_batch_size=64, normalize=False):
    transform_list = [transforms.Resize((32, 32)), transforms.ToTensor()]

    if normalize:
        transform_list.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        )

    transform = transforms.Compose(transform_list)
    
    train_loader = DataLoader(CIFAR10('data', train=True, download=True, transform=transform),
                             batch_size=train_batch_size, shuffle=True, pin_memory=True)
    train_eval_loader = DataLoader(CIFAR10('data', train=True, download=True, transform=transform),
                            batch_size=train_batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(CIFAR10('data', train=False, download=True, transform=transform),
                            batch_size=test_batch_size, shuffle=False, pin_memory=True)

    test_dataset = CIFAR10('data', train=False, download=True, transform=transform)
    
    return train_loader, test_loader, train_eval_loader, test_dataset, 10

if args.dataset == 'lisa':
    trainloader, testloader, train_eval_loader, testset, num_classes = lisa_loaders(
        train_batch_size=args.batch_size,
        test_batch_size=args.batch_size,
        normalize=args.normalize
    )
elif args.dataset == 'cifar10':
    trainloader, testloader, train_eval_loader, testset, num_classes = cifar10_loaders(
        train_batch_size=args.batch_size,
        test_batch_size=args.batch_size,
        normalize=args.normalize
    )
else:
    trainloader, testloader, train_eval_loader, testset, num_classes = bstl_loaders(
        train_batch_size=args.batch_size,
        test_batch_size=1024,
        normalize=args.normalize
    )

class fcs(nn.Module):

    def __init__(self,  in_features=512):
        super(fcs, self).__init__()
        self.dropout = 0.1
        self.merge_net = nn.Sequential(nn.Linear(in_features=in_features,
                                                 out_features=2048),
#                                        nn.ReLU(),
                                       nn.Tanh(),
#                                        nn.Dropout(p=dropout),
                                       nn.Linear(in_features=2048,
                                                 out_features=fc_dim),
                                       nn.Tanh(),
#                                        nn.Sigmoid(),
                                       )

    def forward(self, inputs):
        output = self.merge_net(inputs)
        return output

print('==> Building model..')


net = resnet18(weights=None)

net = net.to(device)

net = nn.Sequential(*list(net.children())[0:-1])

fcs_temp = fcs()

fc_layersa = MLP_OUT_BALL(num_classes)

model_fea = nn.Sequential(*net, fcs_temp, fc_layersa).to(device)
saved_temp = torch.load(folder_savemodel+'/ckpt.pth')
statedic_temp = saved_temp['net']
model_fea.load_state_dict(statedic_temp)


odefunc = ODEfunc_mlp(0)
feature_layers = [ODEBlock(odefunc)] 
fc_layers = [MLP_OUT_final(num_classes)]
model_dense = nn.Sequential( *feature_layers, *fc_layers).to(device)
statedic = saved['state_dict']
model_dense.load_state_dict(statedic)


class tempnn(nn.Module):
    def __init__(self):
        super(tempnn, self).__init__()
    def forward(self, input_):
        h1 = input_[...,0,0]
        return h1
tempnn_ = tempnn()
model = nn.Sequential(*net, tempnn_,fcs_temp,  *model_dense).to(device)


if torch.cuda.device_count() > 1:
    print(f"Usando {torch.cuda.device_count()} GPUs com DataParallel")
    model = nn.DataParallel(model)

model.eval()
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

classifier = PyTorchClassifier(
    model=model,
    clip_values=(0, 1),
    loss=criterion,
    optimizer=optimizer,
    input_shape=(3, 32, 32),
    nb_classes=num_classes,
    device_type="gpu"
)

print(folder, ' time: ', endtime)

class mnist_samples(Dataset):
    def __init__(self, dataset, leng, iid):
        self.dataset = dataset
        self.len = leng
        self.iid = iid
    def __len__(self):
            return self.len

    def __getitem__(self, idx):
        x,y = self.dataset[idx+self.len*self.iid]
        return x,y
        
test_samples = mnist_samples(testset,32,4)
test_loader_samples = DataLoader(test_samples, batch_size=100, shuffle=False, num_workers=2, drop_last=False)

def evaluate_model(model, dataset_loader, attack=None, attack_name="Clean", device=None):
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    y_true_all, y_pred_all = [], []

    for i, (x, y) in enumerate(dataset_loader):
        x, y = x.to(device), y.to(device)

        if attack is not None:
            x = attack(x, y)

        with torch.no_grad():
            outputs = model(x)
            predicted_class = outputs.argmax(dim=1)

        y_true_all.extend(y.cpu().numpy())
        y_pred_all.extend(predicted_class.cpu().numpy())

    acc = (torch.tensor(y_true_all) == torch.tensor(y_pred_all)).float().mean().item()
    precision = precision_score(y_true_all, y_pred_all, average='macro', zero_division=0)
    recall = recall_score(y_true_all, y_pred_all, average='macro', zero_division=0)
    f1 = f1_score(y_true_all, y_pred_all, average='macro', zero_division=0)

    print(f"\n=== {attack_name} RESULTS ===")
    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print(f"F1-score:  {f1 * 100:.2f}%\n")

    return acc, precision, recall, f1

def accuracy_clean(model, dataset_loader, device):
    return evaluate_model(model, dataset_loader, None, "Clean", device)

def accuracy_FGSM(model, dataset_loader, eps, device, normalize):
    attack = torchattacks.FGSM(model, eps=eps)
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    if normalize:
        attack.set_normalization_used(mean=mean, std=std)

    return evaluate_model(model, dataset_loader, attack, f"FGSM (ε={eps})", device)

def accuracy_PGD(model, dataset_loader, eps, device, normalize):
    attack = torchattacks.PGD(model, eps=eps)   
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    if normalize:
        attack.set_normalization_used(mean=mean, std=std)

    return evaluate_model(model, dataset_loader, attack, f"PGD (ε={eps})", device)

def accuracy_MIM(model, dataset_loader, eps, device, normalize):
    attack = torchattacks.MIFGSM(model, eps=eps) 
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    if normalize:
        attack.set_normalization_used(mean=mean, std=std)

    return evaluate_model(model, dataset_loader, attack, f"MIM (ε={eps})", device)
    
def accuracy_AutoAttack(model, dataset_loader, num_classes, eps, device, normalize):
    attack = torchattacks.AutoAttack(model, eps=eps, n_classes=num_classes)
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    if normalize:
        attack.set_normalization_used(mean=mean, std=std)

    return evaluate_model(model, dataset_loader, attack, f"AutoAttack (ε={eps})", device)

acc, prec, rec, f1 = accuracy_clean(model, testloader, device)

all_eps = [0.01, 8/255, 0.04, 0.055, 0.07, 0.085, 0.1, 0.115, 0.13, 0.15, 0.175, 0.2]
    
for eps in all_eps:
    accuracy_FGSM(model, testloader, eps, device, args.normalize)

for eps in all_eps:
    accuracy_PGD(model, testloader, eps, device, args.normalize)

for eps in all_eps:
    accuracy_MIM(model, testloader, eps, device, args.normalize)

for eps in all_eps:
    accuracy_AutoAttack(model, testloader, num_classes, eps, device, args.normalize)
