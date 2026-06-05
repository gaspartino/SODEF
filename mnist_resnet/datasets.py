import torch
import kagglehub

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder, CIFAR10

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

    train_dataset = datasets.MNIST(root='.data/mnist', train=False, download=True, transform=transform_train)
    test_dataset = datasets.MNIST(root='.data/mnist', train=True, download=True, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)

    train_eval_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True)

    return train_loader, test_loader, train_eval_loader, test_dataset, 10

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

    train_dataset = ImageFolder(root=train_dir, transform=transform)
    test_dataset = ImageFolder(root=test_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, pin_memory=True)
    train_eval_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=False, pin_memory=True)
    
    return train_loader, test_loader, train_eval_loader, test_dataset, 4

def cifar10_loaders(train_batch_size=256, test_batch_size=64, normalize=False):
    transform_list = [transforms.Resize((32, 32)), transforms.ToTensor()]

    if normalize:
        transform_list.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        )

    transform = transforms.Compose(transform_list)

    train_dataset = CIFAR10('data', train=True, download=True, transform=transform)
    test_dataset = CIFAR10('data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, pin_memory=True)
    train_eval_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, pin_memory=True)

    return train_loader, test_loader, train_eval_loader, test_dataset, 10
