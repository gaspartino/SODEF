import torch 
import torch.nn as nn 
from layer_lip import *

def getModel(config):
    models = {
        'FCModel': FCModel,
        'AllLeNet5': AllLeNet5,
        'LipLeNet5Max': LipLeNet5Max,
        'VanillaLeNet5': VanillaLeNet5,
        'All2C2F': All2C2F,
        'AOL2C2F': AOL2C2F,
        'LipLeNet5': LipLeNet5,
        'Lip2C2FPool': Lip2C2FPool,
        'Lip2C2F': Lip2C2F,
        'Lip2C1F': Lip2C1F,
        'Lip3C1F': Lip3C1F,
        'Lip4C1F': Lip4C1F,
        'Vanilla2C2F': Vanilla2C2F,
        'Vanilla2C2FPool': Vanilla2C2FPool,
    }[config.model]
    return models(config)

#------------------------------------------------------------
class FCModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.in_channels = config.in_channels
        self.img_size = config.img_size
        self.num_classes = config.num_classes
        self.gamma = config.gamma 
        self.layer = config.layer
        #self.scale = config.scale

        self.flatten = nn.Flatten()
        self.FC1 = nn.Linear(32*32,10)
        #self.FC2 = nn.Linear(64,self.num_classes)

    def forward(self,x):
        x = self.flatten(x)
        #x = F.relu(self.FC1(x))
        x = self.FC1(x)

        return x  


class Lip2C2F(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.in_channels = config.in_channels
        self.img_size = config.img_size
        self.num_classes = config.num_classes
        self.gamma = config.gamma 
        self.layer = config.layer
        g = self.gamma ** 0.5
        n = self.img_size // 4

        # Definir camadas
        self.LipCNNConv1 = LipCNNConv(self.in_channels, 32, 4, stride = 2)
        self.LipCNNConv2 = LipCNNConv(32, 512, 4, stride = 2)
        self.LipCNNFc1 = LipCNNFc(512 * n * n, self.num_classes, psi = None)
        #self.LipCNNFc1 = LipCNNFc(32 * n * n, 100)
        #self.LipCNNFc2 = LipCNNFc(100, self.num_classes, psi = None)
        self.flatten = nn.Flatten()

    def forward(self, x):
        device = x.device
        
        n = self.img_size // 4
        L = self.gamma * torch.eye(self.in_channels, dtype = torch.float64, device = device)
        x, L = self.LipCNNConv1(x, L)
        x = F.relu(x)
        x, L = self.LipCNNConv2(x, L)
        x = F.relu(x)
        x = self.flatten(x)
        L = torch.kron(L,torch.eye(n*n).to(L.device))
        x, L = self.LipCNNFc1(x, L)
        #x = F.relu(x)
        #x, _ = self.LipCNNFc2(x,L)
        
        return x  
    
class Lip2C2FPool(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.in_channels = config.in_channels
        self.img_size = config.img_size
        self.num_classes = config.num_classes
        self.gamma = config.gamma 
        self.layer = config.layer
        g = self.gamma ** 0.5
        n = self.img_size // 4
        
        # Define layers
        self.Pool = nn.AvgPool2d(2)
        #self.Pool = nn.AvgPool2d(2, divisor_override=2)
        self.LipCNNConv1 = LipCNNConv(self.in_channels, 16, 4, padding = 0)
        self.LipCNNConv2 = LipCNNConv(16, 32, 4, padding = 0)
        self.LipCNNFc1 = LipCNNFc(32 * n * n, 100)
        self.LipCNNFc2 = LipCNNFc(100, self.num_classes, psi = None)
        #self.fc1 = nn.Linear(64, self.num_classes)
        self.flatten = nn.Flatten()

    def forward(self, x):
        # Forward pass through custom layer
        n = self.img_size // 4

        # Define the asymmetric padding
        padding = (2, 1, 2, 1)  # (left, right, top, bottom)

        L = self.gamma * torch.eye(self.in_channels, dtype = torch.float64)
        x = F.pad(x, padding, mode="constant",value=0)
        x, L = self.LipCNNConv1(x, L)
        L = 2 * L
        x = F.relu(x)
        x = self.Pool(x)
        x = F.pad(x, padding, mode="constant",value=0)
        x, L = self.LipCNNConv2(x, L)
        L = 2 * L
        x = F.relu(x)
        x = self.Pool(x)
        x = self.flatten(x)
        L = torch.kron(L,torch.eye(n * n))
        x, L = self.LipCNNFc1(x, L)
        x = F.relu(x)
        x, _ = self.LipCNNFc2(x,L)
        
        return x



class Lip2C1F(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.in_channels = config.in_channels
        self.img_size = config.img_size
        self.num_classes = config.num_classes
        self.gamma = config.gamma
        self.layer = config.layer
        g = self.gamma ** 0.5

        self.LipCNNConv1 = LipCNNConv(self.in_channels, 32, 4, stride=2)
        self.LipCNNConv2 = LipCNNConv(32, 128, 4, stride=2)
        
        n = self.img_size // 4
        self.LipCNNFc1 = LipCNNFc(128 * n * n, self.num_classes, psi=None)

        self.flatten = nn.Flatten()

    def forward(self, x):
        device = x.device
        n = self.img_size // 4
        L = self.gamma * torch.eye(self.in_channels, dtype=torch.float64, device=device)

        x, L = self.LipCNNConv1(x, L)
        x = F.relu(x)

        x, L = self.LipCNNConv2(x, L)
        x = F.relu(x)

        x = self.flatten(x)
        L = torch.kron(L, torch.eye(n * n).to(L.device))
        x, L = self.LipCNNFc1(x, L)

        return x

class Lip3C1F(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.in_channels = config.in_channels
        self.img_size = config.img_size
        self.num_classes = config.num_classes
        self.gamma = config.gamma
        self.layer = config.layer
        g = self.gamma ** 0.5
        
        self.flatten = nn.Flatten()

        self.LipCNNConv1 = LipCNNConv(self.in_channels, 16, 4, stride=2)
        self.LipCNNConv2 = LipCNNConv(16, 32, 4, stride=2)
        self.LipCNNConv3 = LipCNNConv(32, 64, 4, stride=2)
        
        n = self.img_size // 8
        self.LipCNNFc1   = LipCNNFc(64 * n * n, self.num_classes, psi=None)


    def forward(self, x):
        device = x.device
        n = self.img_size // 8
        L = self.gamma * torch.eye(self.in_channels, dtype=torch.float64, device=device)

        x, L = self.LipCNNConv1(x, L)
        x = F.relu(x)

        x, L = self.LipCNNConv2(x, L)
        x = F.relu(x)

        x, L = self.LipCNNConv3(x, L)
        x = F.relu(x)

        x = self.flatten(x)
        L = torch.kron(L, torch.eye(n * n).to(L.device))
        x, L = self.LipCNNFc1(x, L)

        return x

class Lip4C1F(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.in_channels = config.in_channels
        self.img_size = config.img_size
        self.num_classes = config.num_classes
        self.gamma = config.gamma
        self.layer = config.layer
        g = self.gamma ** 0.5
        
        self.flatten = nn.Flatten()

        self.LipCNNConv1 = LipCNNConv(self.in_channels, 16, 4, stride=2)
        self.LipCNNConv2 = LipCNNConv(16, 32, 4, stride=2)
        self.LipCNNConv3 = LipCNNConv(32, 64, 4, stride=2)
        self.LipCNNConv4 = LipCNNConv(64, 128, 4, stride=2)  # NOVA CAMADA

        # atenção: agora são 4 reduções por stride=2
        n = self.img_size // 16  
        self.LipCNNFc1 = LipCNNFc(128 * n * n, self.num_classes, psi=None)


    def forward(self, x):
        device = x.device
        n = self.img_size // 16  # também precisa mudar aqui!
        L = self.gamma * torch.eye(self.in_channels, dtype=torch.float64, device=device)

        x, L = self.LipCNNConv1(x, L)
        x = F.relu(x)

        x, L = self.LipCNNConv2(x, L)
        x = F.relu(x)

        x, L = self.LipCNNConv3(x, L)
        x = F.relu(x)

        x, L = self.LipCNNConv4(x, L)  # passa pela nova camada
        x = F.relu(x)

        x = self.flatten(x)
        L = torch.kron(L, torch.eye(n * n).to(L.device))
        x, L = self.LipCNNFc1(x, L)

        return x

class AOL2C2F(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.in_channels = config.in_channels
        self.img_size = config.img_size
        self.num_classes = config.num_classes
        self.gamma = config.gamma 
        self.layer = config.layer

        if self.gamma is not None:
            g = self.gamma ** (1.0 / 2)
        n = self.img_size // 4

        # Define the asymmetric padding
        self.padding = (2, 1, 2, 1)  # (left, right, top, bottom) converted to (top, bottom, left, right)

        self.conv1 = AolConv(self.in_channels, 16, kernel_size=4, scale=g)
        self.pool1 = nn.AvgPool2d(2, divisor_override=2)
        self.conv2 = AolConv(16, 32, kernel_size=4)
        self.pool2 = nn.AvgPool2d(2, divisor_override=2)
        self.flatten = nn.Flatten()
        self.fc1 = AolFc(32 * n * n, 100)
        self.fc2 = AolLin(100, self.num_classes, scale=g)

    def forward(self, x):
        x = F.pad(x, self.padding, mode="constant", value=0)
        x = self.conv1(x)
        x = self.pool1(x)
        x = F.pad(x, self.padding, mode="constant", value=0)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class All2C2F(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.in_channels = config.in_channels
        self.img_size = config.img_size
        self.num_classes = config.num_classes
        self.gamma = config.gamma 
        self.layer = config.layer

        if self.gamma is not None:
            g = self.gamma ** (1.0 / 2)
        n = self.img_size // 4

        if self.layer == 'Sandwich':
            self.model = nn.Sequential(
                SandwichConv(self.in_channels, 16, 4, stride = 2, scale=g),
                SandwichConv(16, 32, 4, stride=2),
                nn.Flatten(),
                SandwichFc(32 * n * n, 100),
                SandwichLin(100, self.num_classes, scale=g)
            )
        elif self.layer == 'Orthogon':
            self.model = nn.Sequential(
                OrthogonConv(self.in_channels, 16, 4, stride=2, scale=g), 
                OrthogonConv(16, 32, 4, stride=2), 
                nn.Flatten(),
                OrthogonFc(32 * n * n, 100), 
                OrthogonLin(100, self.num_classes, scale=g)
            ) 
        elif self.layer == 'Aol':
            # Define the asymmetric padding
            padding = (2, 1, 2, 1)  # (left, right, top, bottom)
            self.model = nn.Sequential(
                F.pad(padding, mode="constant",value=0),
                AolConv(self.in_channels, 16, kernel_size=4, padding = 1, scale=g),
                nn.AvgPool2d(2, divisor_override=2),
                F.pad(padding, mode="constant",value=0),
                AolConv(16, 32, kernel_size=4, padding = 1),
                nn.AvgPool2d(2, divisor_override=2),
                #nn.AvgPool2d(4, divisor_override=4),
                nn.Flatten(),
                AolFc(32 * n * n, 100), 
                AolLin(100, self.num_classes, scale=g)
            )  

    def forward(self, x):
        return self.model(x)
    
class Vanilla2C2F(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.in_channels = config.in_channels
        self.img_size = config.img_size
        self.num_classes = config.num_classes
        self.gamma = config.gamma 
        self.layer = config.layer
        
        n = self.img_size // 4  # The size after two Conv2d layers with stride=2
        
        self.model = nn.Sequential(
            nn.Conv2d(self.in_channels, 16, 4, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, padding=1, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(n * n * 32, 100),
            nn.ReLU(),
            nn.Linear(100, self.num_classes)
        )

    def forward(self, x):
        return self.model(x)
    
class Vanilla2C2FPool(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.in_channels = config.in_channels
        self.img_size = config.img_size
        self.num_classes = config.num_classes
        self.layer = config.layer
        self.gamma = config.gamma 

        n = self.img_size // 4

        # Define the asymmetric padding
        self.padding = (2, 1, 2, 1)  # (left, right, top, bottom) converted to (top, bottom, left, right)

        self.conv1 = nn.Conv2d(self.in_channels, 16, kernel_size=4)
        self.pool1 = nn.AvgPool2d(2, divisor_override=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4)
        self.pool2 = nn.AvgPool2d(2, divisor_override=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * n * n, 100)
        self.fc2 = nn.Linear(100, self.num_classes)

    def forward(self, x):
        x = F.pad(x, self.padding, mode="constant", value=0)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.pad(x, self.padding, mode="constant", value=0)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#------------------------------------------------------------
class LipLeNet5(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.in_channels = config.in_channels
        self.img_size = config.img_size
        self.num_classes = config.num_classes
        self.gamma = config.gamma 
        self.layer = config.layer
        g = self.gamma ** 0.5
        # Define layers
        self.LipCNNConv1 = LipCNNConv(self.in_channels, 6, 5, padding = 0)
        self.Pool = nn.AvgPool2d(2)
        self.LipCNNConv2 = LipCNNConv(6, 16, 5, padding = 0)
        self.LipCNNFc1 = LipCNNFc(5*5*16, 120)
        self.LipCNNFc2 = LipCNNFc(120, 84)
        #self.LipCNNFc3 = LipCNNFc(12*12*6, self.num_classes, psi = None)
        self.LipCNNFc3 = LipCNNFc(84, self.num_classes, psi = None)
        self.flatten = nn.Flatten()

    def forward(self, x):
        # Forward pass through custom layer

        L = self.gamma * torch.eye(self.in_channels, dtype = torch.float64)
        x, L = self.LipCNNConv1(x, L)
        x = F.relu(x)
        x = self.Pool(x)
        L = 2 * L # Rescaling by dividing by the Lipschitz constant of the average pooling layer
        x, L = self.LipCNNConv2(x, L)
        x = F.relu(x)
        x = self.Pool(x)
        L = 2 * L # Rescaling by dividing by the Lipschitz constant of the average pooling layer
        x = self.flatten(x)
        L = torch.kron(L,torch.eye(5*5))
        x, L = self.LipCNNFc1(x, L)
        x = F.relu(x)
        x, L = self.LipCNNFc2(x, L)
        x = F.relu(x)
        x, _ = self.LipCNNFc3(x,L)
        
        return x
    
class AllLeNet5(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.in_channels = config.in_channels
        self.img_size = config.img_size
        self.num_classes = config.num_classes
        self.gamma = config.gamma 
        self.layer = config.layer

        if self.gamma is not None:
            g = self.gamma ** (1.0 / 2)
        n = self.img_size // 4

        if self.layer == 'Aol':
            self.model = nn.Sequential(
                AolConv(self.in_channels, 6, kernel_size=5, scale=g),
                nn.AvgPool2d(2),
                AolConv(6, 16, kernel_size=5), 
                nn.AvgPool2d(2),
                nn.Flatten(),
                AolFc(16 * 5 * 5, 120),
                AolFc(120, 84), 
                AolLin(84, self.num_classes, scale=g)
            )  

    def forward(self, x):
        return self.model(x)

class VanillaLeNet5(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.in_channels = config.in_channels
        self.img_size = config.img_size
        self.num_classes = config.num_classes
        self.gamma = config.gamma 
        self.layer = config.layer
        
        n = self.img_size // 4  # The size after two Conv2d layers with stride=2
        
        self.model = nn.Sequential(
            nn.Conv2d(self.in_channels, 6, 5, padding=0, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(6, 16, 5, padding=0, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Flatten(),
            nn.Linear(5*5*16, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, self.num_classes)
        )

    def forward(self, x):
        return self.model(x) 

class LipLeNet5Max(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.in_channels = config.in_channels
        self.img_size = config.img_size
        self.num_classes = config.num_classes
        self.gamma = config.gamma 
        self.layer = config.layer
        g = self.gamma ** 0.5
        
        # Define layers
        self.LipCNNConv1 = LipCNNConvMax(self.in_channels, 6, 5, padding = 0)
        self.Pool = nn.MaxPool2d(2)
        self.LipCNNConv2 = LipCNNConvMax(6, 16, 5, padding = 0)
        self.LipCNNFc1 = LipCNNFc(4*4*16, 120)
        self.LipCNNFc2 = LipCNNFc(120, 84)
        #self.LipCNNFc3 = LipCNNFc(12*12*6, self.num_classes, psi = None)
        self.LipCNNFc3 = LipCNNFc(84, self.num_classes, psi = None)
        self.flatten = nn.Flatten()

    def forward(self, x):
        # Forward pass through custom layer

        L = self.gamma * torch.eye(self.in_channels, dtype = torch.float64)
        x, L = self.LipCNNConv1(x, L)
        x = F.relu(x)
        x = self.Pool(x)
        #L = 2 * L # Rescaling by dividing by the Lipschitz constant of the average pooling layer
        x, L = self.LipCNNConv2(x, L)
        x = F.relu(x)
        x = self.Pool(x)
        #L = 2 * L # Rescaling by dividing by the Lipschitz constant of the average pooling layer
        x = self.flatten(x)
        L = torch.kron(L,torch.eye(4*4))
        x, L = self.LipCNNFc1(x, L)
        x = F.relu(x)
        x, L = self.LipCNNFc2(x, L)
        x = F.relu(x)
        x, _ = self.LipCNNFc3(x,L)
        
        return x
