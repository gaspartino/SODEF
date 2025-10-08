from torch import nn
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import logging
from torch.nn.parameter import Parameter
import geotorch
import math
import random
import numpy as np
import os
import argparse
from torchdiffeq import odeint_adjoint as odeint
from torch.utils.data import Dataset, DataLoader
from model import *
from torchvision.datasets import MNIST, CIFAR10, ImageFolder

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

train_savepath = './data/MNIST_train_resnet_final.npz'
test_savepath = './data/MNIST_test_resnet_final.npz'

fc_dim = 64
folder_savemodel = './EXP/MNIST_resnet_final'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=512, type=int)
    parser.add_argument('--data-dir', default='../cifar-data', type=str)
    parser.add_argument('--dataset', default='lisa', type=str)
    parser.add_argument('--normalize', action='store_true', help='Ativa normalização dos dados')
    parser.add_argument('--is-lip', action='store_true', help='Usa dimensão 32 (LIP) ao invés de 64')
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--out-dir', default='train_fgsm_output', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    return parser.parse_args()


args = get_args()

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
seed_torch()



def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


device = 'cuda' 
best_acc = 0 
start_epoch = 0  

    

def inf_generator(iterable):
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()
            
def get_mnist_loaders(data_aug=False, batch_size=128, test_batch_size=1000, perc=1.0):
    if data_aug:
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.ToTensor(),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_loader = DataLoader(
        datasets.MNIST(root='.data/mnist', train=True, download=True, transform=transform_train), batch_size=batch_size,
        shuffle=True, num_workers=1, drop_last=True
    )

    train_eval_loader = DataLoader(
        datasets.MNIST(root='.data/mnist', train=True, download=True, transform=transform_test),
        batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
    )

    test_loader = DataLoader(
        datasets.MNIST(root='.data/mnist', train=False, download=True, transform=transform_test),
        batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
    )

    return train_loader, test_loader, train_eval_loader

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch
import kagglehub

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

    return train_loader, test_loader, train_eval_loader, 7


def bstl_loaders(train_batch_size=256, test_batch_size=64, normalize=False, is_lip=True):
    dim = 32 if is_lip else 64

    transform_list = [transforms.Resize((32, dim)), transforms.ToTensor()]

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
    return train_loader, test_loader, train_eval_loader, 4

if args.dataset == 'lisa':
    trainloader, testloader, train_eval_loader, num_classes = lisa_loaders(
        train_batch_size=args.batch_size,
        test_batch_size=args.batch_size,
        normalize=args.normalize
    )
else:
    trainloader, testloader, train_eval_loader, num_classes = bstl_loaders(
        train_batch_size=args.batch_size,
        test_batch_size=args.batch_size,
        normalize=args.normalize,
        is_lip=args.is_lip
    )
    
class LipConvExtractor(nn.Module):
    def __init__(self, lip_model):
        super().__init__()
        self.conv1 = lip_model.LipCNNConv1
        self.conv2 = lip_model.LipCNNConv2
        self.conv3 = lip_model.LipCNNConv3
        self.conv4 = lip_model.LipCNNConv4
        self.flatten = nn.Flatten()

    def forward(self, x):
        L = torch.eye(x.shape[1], dtype=torch.float64, device=x.device)

        x, L = self.conv1(x, L)
        x = nn.ReLU()(x)
        x, L = self.conv2(x, L)
        x = nn.ReLU()(x)
        x, L = self.conv3(x, L)
        x = nn.ReLU()(x)
        x, L = self.conv4(x, L)
        x = nn.ReLU()(x)

        return x

print('==> Building model..')
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision.models import resnet34, ResNet34_Weights
from torch.utils.data import DataLoader
from types import SimpleNamespace
from model_lip import *

print('==> Building model..')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.is_lip:
    config = SimpleNamespace(
        model="Lip4C1F",
        in_channels=3,
        img_size=32,
        num_classes=num_classes,
        gamma=1.0,
        layer="Lip2C1F"
    )

    model = getModel(config).to(device)

    model_state = torch.load(f"EXP/{args.dataset}_lipkernel_4c1fc_nat.ckpt.ckpt", map_location=device)

    try:
        model.load_state_dict(model_state)
    except RuntimeError:
        new_state_dict = OrderedDict()
        for k, v in model_state.items():
            new_state_dict[k.replace("module.", "")] = v
        model.load_state_dict(new_state_dict)

    lip_cnn_final = LipConvExtractor(model).to(device)
    for param in lip_cnn_final.parameters():
        param.requires_grad = False

    fcs_temp = fcs(in_features=128)  
    fc_layers = MLP_OUT_BALL(num_classes)
    for param in fc_layers.parameters():
        param.requires_grad = False

    net = nn.Sequential(lip_cnn_final, fcs_temp, fc_layers).to(device)

else:
    net = resnet34(weights=ResNet34_Weights.DEFAULT)
    net = net.to(device)

    net = nn.Sequential(*list(net.children())[:-1])

    fcs_temp = fcs()  
    fc_layers = MLP_OUT_BALL(num_classes)
    for param in fc_layers.parameters():
        param.requires_grad = False

    net = nn.Sequential(net, fcs_temp, fc_layers).to(device)

print(net)
#cure = CURE_Regularizer(net, device, lambda_=4.0)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001, eps=1e-4, amsgrad=True)


def save_feature(model, dataset_loader, save_path):
    x_save, y_save = [], []
    modulelist = list(model)

    for x, y in dataset_loader:
        x = x.to(device)
        y_ = y.numpy()

        for l in modulelist[:-2]:  # Extrator de características
            x = l(x)

        x = net[-2](x[..., 0, 0])  # fcs_temp
        x_ = x.cpu().detach().numpy()

        x_save.append(x_)
        y_save.append(y_)

    np.savez(save_path, x_save=np.concatenate(x_save), y_save=np.concatenate(y_save))


def train(epoch, trainloader):
    net.train()
    train_loss, correct, total = 0, 0, 0
    modulelist = list(net)

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        x = inputs
        for l in modulelist[:-2]:  # Extrator de características
            x = l(x)

        x = net[-2](x[..., 0, 0])  # fcs_temp
        x = net[-1](x)  # Camada final (7 classes)
        outputs = x

        loss = criterion(outputs, targets)
        #reg, grad_norm = cure.compute(inputs, targets)
        #loss = loss + reg
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    print(f"\nTrain in epoch {epoch+1}: accuracy = {round(acc, 2)}%")

def test(epoch, testloader, save_model_path, train_eval_loader):
    global best_acc
    net.eval()
    test_loss, correct, total = 0, 0, 0
    modulelist = list(net)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)

            x = inputs
            for l in modulelist[:-2]:  # Extrator de características
                x = l(x)

            x = net[-2](x[..., 0, 0])  # fcs_temp
            x = net[-1](x)  # Camada final (7 classes)
            outputs = x

            loss = criterion(outputs, targets)
            test_loss += loss.item()

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            
    acc = 100. * correct / total
    print(f"Test in epoch {epoch+1}: accuracy = {round(acc, 2)}%")
    if acc > best_acc:
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(state, folder_savemodel + '/ckpt.pth')
        best_acc = acc

        save_feature(net, train_eval_loader, train_savepath)
        save_feature(net, testloader, test_savepath)


############################################### Phase 1 ################################################
makedirs(folder_savemodel)
makedirs('./data')

for epoch in range(40):
    train(epoch, trainloader)
    test(epoch, testloader, './models/ckpt.pth', train_eval_loader)
################################################ Phase 2 ################################################
weight_diag = 10
weight_offdiag = 10
weight_norm = 0
weight_lossc = 0
weight_f = 0.2

exponent = 1.0
exponent_f = 50
exponent_off = 0.1

endtime = 1

trans = 1.0
transoffdig = 1.0
trans_f = 0.0
numm = 8
timescale = 1
fc_dim = 64
t_dim = 1
act = torch.sin
act2 = torch.nn.functional.relu

class ConcatFC(nn.Module):

    def __init__(self, dim_in, dim_out):
        super(ConcatFC, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)

    def forward(self, t, x):
        return self._layer(x)

class ODEfunc_mlp(nn.Module):  # dense_resnet_relu1,2,7

    def __init__(self, dim):
        super(ODEfunc_mlp, self).__init__()
        self.fc1 = ConcatFC(fc_dim, fc_dim)
        self.act1 = act
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = -1 * self.fc1(t, x)
        out = self.act1(out)
        return out

class ODEBlocktemp(nn.Module):

    def __init__(self, odefunc):
        super(ODEBlocktemp, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, endtime]).float()

    def forward(self, x):
        out = self.odefunc(0, x)
        return out

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


class MLP_OUT(nn.Module):

    def __init__(self, num_classes=10):
        super(MLP_OUT, self).__init__()
        self.fc0 = nn.Linear(fc_dim, num_classes)

    def forward(self, input_):
        h1 = self.fc0(input_)
        return h1


def one_hot(x, K):
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)


def accuracy(model, dataset_loader):
    total_correct = 0
    for x, y in dataset_loader:
        x = x.to(device)
        y = one_hot(np.array(y.numpy()), 7)

        target_class = np.argmax(y, axis=1)
        predicted_class = np.argmax(model(x).cpu().detach().numpy(), axis=1)
        total_correct += np.sum(predicted_class == target_class)
    return total_correct / len(dataset_loader.dataset)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def df_dz_regularizer(f, z):
    #     print("+++++++++++")
    regu_diag = 0.
    regu_offdiag = 0.0
    for ii in np.random.choice(z.shape[0], min(numm, z.shape[0]), replace=False):
        batchijacobian = torch.autograd.functional.jacobian(lambda x: odefunc(torch.tensor(1.0).to(device), x),
                                                            z[ii:ii + 1, ...], create_graph=True)
        batchijacobian = batchijacobian.view(z.shape[1], -1)
        if batchijacobian.shape[0] != batchijacobian.shape[1]:
            raise Exception("wrong dim in jacobian")

        tempdiag = torch.diagonal(batchijacobian, 0)
        regu_diag += torch.exp(exponent * (tempdiag + trans))
        #         print(regu_diag)

        offdiat = torch.sum(
            torch.abs(batchijacobian) * ((-1 * torch.eye(batchijacobian.shape[0]).to(device) + 0.5) * 2), dim=0)
        off_diagtemp = torch.exp(exponent_off * (offdiat + transoffdig))
        #         off_diagtemp = torch.exp(exponent*(torch.sum(torch.abs(batchijacobian)*((-1*torch.eye(batchijacobian.shape[0]).to(device)+0.5)*2), dim=0)+transoffdig))
        regu_offdiag += off_diagtemp

    return regu_diag / numm, regu_offdiag / numm


def f_regularizer(f, z):
    tempf = torch.abs(odefunc(torch.tensor(1.0).to(device), z))
    regu_f = torch.pow(exponent_f * tempf, 2)
    return regu_f


def critialpoint_regularizer(y1):
    regu4 = torch.linalg.norm(y1, dim=1)
    regu4 = regu4.mean()
    regu4 = torch.exp(-0.1 * regu4 + 5)
    return regu4.mean()


class DensemnistDatasetTrain(Dataset):
    def __init__(self):
        """
        """
        npzfile = np.load(train_savepath)

        self.x = npzfile['x_save']
        self.y = npzfile['y_save']

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx, ...]
        y = self.y[idx]

        return x, y


class DensemnistDatasetTest(Dataset):
    def __init__(self):
        """
        """
        npzfile = np.load(test_savepath)

        self.x = npzfile['x_save']
        self.y = npzfile['y_save']

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx, ...]
        y = self.y[idx]

        return x, y

odesavefolder = './EXP/dense_resnet_final'
makedirs(odesavefolder)
odefunc = ODEfunc_mlp(0)

feature_layers = [ODEBlocktemp(odefunc)]
fc_layers = [MLP_OUT(num_classes)]

for param in fc_layers[0].parameters():
    param.requires_grad = False

model = nn.Sequential(*feature_layers, *fc_layers).to(device)
criterion = nn.CrossEntropyLoss().to(device)
regularizer = nn.MSELoss()

train_loader = DataLoader(DensemnistDatasetTrain(),
                          batch_size=32,
                          shuffle=True, num_workers=1
                          )
train_loader__ = DataLoader(DensemnistDatasetTrain(),
                            batch_size=32,
                            shuffle=True, num_workers=1
                            )

test_loader = DataLoader(DensemnistDatasetTest(),
                         batch_size=32,
                         shuffle=True, num_workers=1
                         )

data_gen = inf_generator(train_loader)
batches_per_epoch = len(train_loader)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, eps=1e-3, amsgrad=True)

best_acc = 0
tempi = 0

for itr in range(10 * batches_per_epoch):
    # break

    optimizer.zero_grad()
    x, y = data_gen.__next__()
    x = x.to(device)
    y = y.to(device)
    modulelist = list(model)
    y0 = x
    x = modulelist[0](x)
    y1 = x
    for l in modulelist[1:]:
        x = l(x)
    logits = x
    y00 = y0  # .clone().detach().requires_grad_(True)

    regu1, regu2 = df_dz_regularizer(odefunc, y00)
    regu1 = regu1.mean()
    regu2 = regu2.mean()
    regu3 = f_regularizer(odefunc, y00)
    regu3 = regu3.mean()
    loss = weight_f * regu3 + weight_diag * regu1 + weight_offdiag * regu2
    #         loss = weight_f*regu3

    if itr % 100 == 1:
        torch.save({'state_dict': model.state_dict(), 'args': args},
                   os.path.join(odesavefolder, 'model_diag.pth' + str(itr // 100)))

    loss.backward()
    optimizer.step()
    torch.cuda.empty_cache()

    if itr % batches_per_epoch == 0:
        print(f"Epoch {itr}/{10 * batches_per_epoch}")
        if itr == 0:
            continue
        with torch.no_grad():
            if True:  # val_acc > best_acc:
                torch.save({'state_dict': model.state_dict(), 'args': args},
                           os.path.join(odesavefolder, 'model_' + str(itr // batches_per_epoch) + '.pth'))

################################################ Phase 3, train final FC ################################################

endtime = 5
layernum = 0

folder = './EXP/dense_resnet_final/model_9.pth'
saved = torch.load(folder, weights_only=False)
print('load...', folder)
statedic = saved['state_dict']
args = saved['args']
tol = 1e-5
savefolder_fc = './EXP/resnetfct5_15/'
print('saving...', savefolder_fc, ' endtime... ',endtime)


class ODEBlock(nn.Module):

    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, endtime]).float()

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time, rtol=tol, atol=tol)
        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


makedirs(savefolder_fc)
odefunc = ODEfunc_mlp(0)
feature_layers = [ODEBlock(odefunc)]
fc_layers = [MLP_OUT(num_classes)]
model = nn.Sequential(*feature_layers, *fc_layers).to(device)
model.load_state_dict(statedic)
for param in odefunc.parameters():
    param.requires_grad = False

criterion = nn.CrossEntropyLoss().to(device)
regularizer = nn.MSELoss()

train_loader = DataLoader(DensemnistDatasetTrain(),
                          batch_size=128,
                          shuffle=True, num_workers=1
                          )
train_loader__ = DataLoader(DensemnistDatasetTrain(),
                            batch_size=128,
                            shuffle=True, num_workers=1
                            )
test_loader = DataLoader(DensemnistDatasetTest(),
                         batch_size=128,
                         shuffle=True, num_workers=1
                         )

data_gen = inf_generator(train_loader)
batches_per_epoch = len(train_loader)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, eps=1e-3, amsgrad=True)

best_acc = 0
for itr in range(5 * batches_per_epoch):

    optimizer.zero_grad()
    x, y = data_gen.__next__()
    x = x.to(device)

    y = y.to(device)

    modulelist = list(model)
    
    y0 = x
    x = modulelist[0](x)
    y1 = x
    for l in modulelist[1:]:
        x = l(x)
    logits = x

    loss = criterion(logits, y)

    loss.backward()
    optimizer.step()
    torch.cuda.empty_cache()

    if itr % batches_per_epoch == 0:
        if itr == 0:
            continue
        with torch.no_grad():
            val_acc = accuracy(model, test_loader)
            train_acc = accuracy(model, train_loader__)
            if val_acc > best_acc:
                torch.save({'state_dict': model.state_dict(), 'args': args}, os.path.join(savefolder_fc, 'model.pth'))
                best_acc = val_acc
            print(
                "Epoch {:04d}|Train Acc {:.4f} | Test Acc {:.4f}".format(
                    itr // batches_per_epoch, train_acc, val_acc
                )
            )
