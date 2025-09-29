import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import argparse
import logging
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import geotorch
import math
import torchvision
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent, CarliniL2Method
from art.estimators.classification import PyTorchClassifier
from art.utils import load_mnist
from models import *
from torchvision.datasets import MNIST, CIFAR10, ImageFolder
import os
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
#         self.weight = self.weighttemp.T
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
    def __init__(self):
        super(MLP_OUT_ORT, self).__init__()
        self.fc0 = ORTHFC(fc_dim, 7, False)#nn.Linear(fc_dim, 4)
    def forward(self, input_):
        h1 = self.fc0(input_)
        return h1

class MLP_OUT_final(nn.Module):

    def __init__(self):
        super(MLP_OUT_final, self).__init__()
        self.fc0 = nn.Linear(fc_dim, 7)
    def forward(self, input_):
        h1 = self.fc0(input_)
        return h1

class MLP_OUT_BALL(nn.Module):
    def __init__(self):
        super(MLP_OUT_BALL, self).__init__()
        self.fc0 = nn.Linear(64, 7, bias=False)
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

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    
device = 'cuda' #if torch.cuda.is_available() else 'cpu'
# print(device)
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
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
        shuffle=True, num_workers=2, drop_last=True
    )

    train_eval_loader = DataLoader(
        datasets.MNIST(root='.data/mnist', train=True, download=True, transform=transform_test),
        batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True
    )

    test_loader = DataLoader(
        datasets.MNIST(root='.data/mnist', train=False, download=True, transform=transform_test),
        batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True
    )
    testset = datasets.MNIST(root='.data/mnist', train=False, download=True, transform=transform_test)
    return train_loader, test_loader, train_eval_loader, testset

def get_cars_loaders(data_aug=False, batch_size=64, test_batch_size=1000):
    train_path = '/content/SODEF/Cars Dataset/train'
    test_path = '/content/SODEF/Cars Dataset/test'

    if data_aug:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_loader = DataLoader(
        datasets.ImageFolder(root=train_path, transform=train_transform),
        batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
    )

    train_eval_loader = DataLoader(
        datasets.ImageFolder(root=train_path, transform=test_transform),
        batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
    )

    test_loader = DataLoader(
        datasets.ImageFolder(root=test_path, transform=test_transform),
        batch_size=test_batch_size, shuffle=False, num_workers=1, drop_last=True
    )
    testset = datasets.ImageFolder(root=test_path, transform=test_transform)
    return train_loader, test_loader, train_eval_loader, testset


def get_sign(batch_size=128):
    train_dir = "/content/train"
    test_dir = "/content/test"

    transform = transforms.Compose([
        transforms.Resize((64, 32)),
        transforms.ToTensor()
    ])

    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    train_eval_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"Número de imagens em train: {len(train_dataset)}")
    print(f"Número de imagens em test: {len(test_dataset)}")
    print(f"Classes: {train_dataset.classes}")

    return train_loader, test_loader, train_eval_loader, test_dataset
def get_cifar10_loaders(data_aug=False, batch_size=128, test_batch_size=1000):
    if data_aug:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
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
        datasets.CIFAR10(root='.data/cifar10', train=True, download=True, transform=transform_train),
        batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
    )

    train_eval_loader = DataLoader(
        datasets.CIFAR10(root='.data/cifar10', train=True, download=True, transform=transform_test),
        batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
    )

    test_loader = DataLoader(
        datasets.CIFAR10(root='.data/cifar10', train=False, download=True, transform=transform_test),
        batch_size=test_batch_size, shuffle=False, num_workers=1, drop_last=True
    )
    testset = datasets.CIFAR10(root='.data/cifar10', train=False, download=True, transform=transform_test)

    return train_loader, test_loader, train_eval_loader, testset

def bstl_loaders(train_batch_size=256, test_batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((32, 64)),
        transforms.ToTensor(),
    ])
    
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

def lisa_loaders(train_batch_size=256, test_batch_size=64):
    path = kagglehub.dataset_download("chandanakuntala/cropped-lisa-traffic-light-dataset")
    
    print("Path to dataset files:", path)

    train_dir = f"{path}/cropped_lisa_1/train_1"
    val_dir = f"{path}/cropped_lisa_1/val_1"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                     std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ImageFolder(train_dir, transform=transform)
    test_dataset = ImageFolder(val_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=2)
    train_eval_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader, train_eval_loader, test_dataset, 7

trainloader, testloader, train_eval_loader, testset, num_classes = lisa_loaders()

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

from layer_lip import *

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
from model_lbdn import KWL
from model_lip import *

print('==> Building model..')
'''
scale = "medium"
width = {
    'small': 1,
    'medium': 2,
    'large': 4
}[scale]

config = SimpleNamespace(
    in_channels=3,
    img_size=32,
    num_classes=7,
    gamma=100,
    LLN=False,
    width=width,
    layer='Sandwich_alt' 
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

kwl_cnn = KWL(config).to(device)  # move modelo para GPU
model_state = torch.load("/kaggle/input/kwl_lisa/pytorch/default/2/kwl(1fc)_large_sandwich_lisa_natural.ckpt", map_location=device)

xshape = (1, config.in_channels, config.img_size, config.img_size)
x = (torch.rand(xshape) + 0.3 * torch.randn(xshape)).to(device)  # move entrada para GPU

kwl_cnn.load_state_dict(model_state, strict=False)

kwl_cnn(x)  # agora tudo está no mesmo dispositivo

# Congelar parâmetros
for param in kwl_cnn.parameters():
    param.requires_grad = False

# Criar modelo final
net = nn.Sequential(*list(kwl_cnn.model.children())[:4]).to(device)
fcs_temp = fcs(in_features=64*width) 
'''

config = SimpleNamespace(
    model="Lip4C1F",
    in_channels=3,
    img_size=32,
    num_classes=7,
    gamma=1.0,       
    layer="Lip2C1F"          
)

model = getModel(config).to(device)
model_state = torch.load(f"EXP/lisa_lipkernel_4c1fc.ckpt")

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

net = [lip_cnn_final]

#fcs_temp = fcs(in_features=32*width) #lbdn 
fcs_temp = fcs(in_features=128)  #lip
# fc_layersa = nn.Linear(fc_dim, 4,  bias=True)
# fc_layersa = MLP_OUT_ORT()
fc_layersa = MLP_OUT_BALL()

model_fea = nn.Sequential(*net, fcs_temp, fc_layersa).to(device)
saved_temp = torch.load(folder_savemodel+'/ckpt.pth')
# saved_temp = torch.load(folder_savemodel+'/ckpt-Copy1.pth')
statedic_temp = saved_temp['net']
model_fea.load_state_dict(statedic_temp, strict=False)
    
    
odefunc = ODEfunc_mlp(0)
feature_layers = [ODEBlock(odefunc)] 
fc_layers = [MLP_OUT_final()]
model_dense = nn.Sequential( *feature_layers, *fc_layers).to(device)
statedic = saved['state_dict']
model_dense.load_state_dict(statedic, strict=False)


class tempnn(nn.Module):
    def __init__(self):
        super(tempnn, self).__init__()
    def forward(self, input_):
        h1 = input_[...,0,0]
        return h1
tempnn_ = tempnn()
model = nn.Sequential(*net, tempnn_,fcs_temp,  *model_dense).to(device)
# model = nn.Sequential(*net, tempnn_,fcs_temp,  fc_layersa).to(device)

if torch.cuda.device_count() > 1:
    print(f"Usando {torch.cuda.device_count()} GPUs com DataParallel")
    model = nn.DataParallel(model)
    
model.eval()    
print(model)

# Step 2a: Define the loss function and the optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)



classifier = PyTorchClassifier(
    model=model,
    clip_values=(0, 1),
    loss=criterion,
    optimizer=optimizer,
    input_shape=(3, 32, 32),
    nb_classes=4,
    device_type="gpu"
)
print("model size")
print(model(torch.randn(1, 3, 32, 32).to(device)).shape)

print(folder, ' time: ', endtime)

class mnist_samples(Dataset):
    def __init__(self, dataset, leng, iid):
        self.dataset = dataset
        self.len = leng
        self.iid = iid
    def __len__(self):
#             return 425
            return self.len

    def __getitem__(self, idx):
        x,y = self.dataset[idx+self.len*self.iid]
        return x,y
test_samples = mnist_samples(testset,32,4)
# test_loader_samples = DataLoader(test_samples, batch_size=1, shuffle=False, num_workers=2, drop_last=False)
test_loader_samples = DataLoader(test_samples, batch_size=100, shuffle=False, num_workers=2, drop_last=False)

# Step 4: Evaluate the ART classifier on adversarial test examples

from tqdm import tqdm
import torchattacks

def accuracy_clean(classifier, dataset_loader):
    total_correct = 0
    for x, y in dataset_loader:
#         x = x.to(device)
        predictions = classifier.predict(x)
        y = one_hot(np.array(y.numpy()), 7)
        target_class = np.argmax(y, axis=1)
#         predicted_class = np.argmax(predictions.cpu().detach().numpy(), axis=1)
        predicted_class = np.argmax(predictions, axis=1)
        total_correct += np.sum(predicted_class == target_class)
    return total_correct / len(dataset_loader.dataset)

accuracy_clean = accuracy_clean(classifier, testloader)
print(f"Accuracy on benign: {round(accuracy_clean * 100,2)}%")



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def accuracy_AA(model, dataset_loader, eps, num_classes, device):
    attack = torchattacks.AutoAttack(model, norm='Linf', eps=eps, version='standard', n_classes=num_classes)
    total_correct = 0
    total_samples = 0
    model = model.to(device)

    total_batches = len(dataset_loader)
    for i, (x, y) in enumerate(dataset_loader):
        x, y = x.to(device), y.to(device)

        # Gerar amostras adversariais
        x_adv = attack(x, y)

        # Fazer previsões no modelo adversarial
        with torch.no_grad():
            predictions = model(x_adv)
            predicted_class = predictions.argmax(dim=1)

        # Contar acertos
        total_correct += (predicted_class == y).sum().item()
        total_samples += y.size(0)

        print(f"[{i+1}/{total_batches}] Processado - Acurácia parcial: {(total_correct / total_samples)*100:.2f}%")

    accuracy = total_correct / total_samples
    print(f"\nAccuracy under AutoAttack (eps={eps}): {accuracy*100:.2f}%")

    return accuracy

def accuracy_FGSM_alt(classifier, dataset_loader, eps):
    attack = torchattacks.FGSM(model, eps=eps)
    total_correct = 0
    total_samples = 0

    for x, y in dataset_loader:
        x, y = x.to(device), y.to(device)

        # Gerar amostras adversariais
        x_adv = attack(x, y)

        # Fazer previsões no modelo adversarial
        with torch.no_grad():
            predictions = classifier.model(x_adv)  # Usa o modelo diretamente, mantendo os tensores na GPU
            predicted_class = predictions.argmax(dim=1)  # Obtém a classe prevista diretamente em PyTorch

        # Contar acertos
        total_correct += (predicted_class == y).sum().item()
        total_samples += y.size(0)

    return total_correct / total_samples

def accuracy_PGD_alt(classifier, dataset_loader, eps):
    attack = torchattacks.PGD(model, eps=eps)
    total_correct = 0
    total_samples = 0

    for x, y in dataset_loader:
        x, y = x.to(device), y.to(device)

        # Gerar amostras adversariais
        x_adv = attack(x, y)

        # Fazer previsões no modelo adversarial
        with torch.no_grad():
            predictions = classifier.model(x_adv)  # Usa o modelo diretamente, mantendo os tensores na GPU
            predicted_class = predictions.argmax(dim=1)  # Obtém a classe prevista diretamente em PyTorch

        # Contar acertos
        total_correct += (predicted_class == y).sum().item()
        total_samples += y.size(0)

    return total_correct / total_samples

def accuracy_MIM_alt(classifier, dataset_loader, eps):
    attack = torchattacks.MIFGSM(model, eps=eps)
    total_correct = 0
    total_samples = 0

    for x, y in dataset_loader:
        x, y = x.to(device), y.to(device)

        # Gerar amostras adversariais
        x_adv = attack(x, y)

        # Fazer previsões no modelo adversarial
        with torch.no_grad():
            predictions = classifier.model(x_adv)  # Usa o modelo diretamente, mantendo os tensores na GPU
            predicted_class = predictions.argmax(dim=1)  # Obtém a classe prevista diretamente em PyTorch

        # Contar acertos
        total_correct += (predicted_class == y).sum().item()
        total_samples += y.size(0)

    return total_correct / total_samples
    
#all_eps = [0.01, 0.02, 8/255, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
all_eps = [8/255, 0.1]

# Aplicar AutoAttack para cada valor de epsilon
for eps in all_eps:
    accuracy = accuracy_FGSM_alt(classifier, testloader, eps)
    print(f"Accuracy on FGSM (ε={round(eps,2)}): {round(accuracy * 100, 2)}%")

    accuracy = accuracy_PGD_alt(classifier, testloader, eps)
    print(f"Accuracy on PGD (ε={round(eps,2)}): {round(accuracy * 100, 2)}%")
    
    #accuracy = accuracy_MIM_alt(classifier, testloader, eps)
    #print(f"Accuracy on MIM (ε={round(eps,2)}): {round(accuracy * 100, 2)}%")

    #accuracy = accuracy_AA(model, testloader, eps, 4, device = device)
    #print(f"Accuracy on AA (ε={round(eps,2)}): {round(accuracy * 100, 2)}%")

for eps in all_eps:
    #accuracy = accuracy_FGSM_alt(classifier, testloader, eps)
    #print(f"Accuracy on FGSM (ε={round(eps,2)}): {round(accuracy * 100, 2)}%")

    #accuracy = accuracy_PGD_alt(classifier, testloader, eps)
    #print(f"Accuracy on PGD (ε={round(eps,2)}): {round(accuracy * 100, 2)}%")
    
    #accuracy = accuracy_MIM_alt(classifier, testloader, eps)
    #print(f"Accuracy on MIM (ε={round(eps,2)}): {round(accuracy * 100, 2)}%")

    accuracy = accuracy_AA(model, testloader, eps, 4, device = device)
    print(f"Accuracy on AA (ε={round(eps,2)}): {round(accuracy * 100, 2)}%")
#for ep in eps:
  

# from robustbench.data import _load_dataset

# # from robustbench.data import load_cifar10
# epsilon = 0.3
# batch_size = 50

# # x_test, y_test = load_cifar10(n_examples=500)
# x_test, y_test = _load_dataset(testset,50)


# from autoattack import AutoAttack
# adversary = AutoAttack(model, norm='Linf', eps=epsilon, version='standard')

# x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=batch_size)
