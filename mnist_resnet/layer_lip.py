import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.linalg import block_diag
import einops

## 1D
def findP(H, Qm):
    n_u = Qm.shape[0]
    n_x = H.shape[0]
    epsilon = 1e-6
    T = np.zeros((n_x, n_x))
    A, B = construct_AB(n_x, n_u) 
    for k in range(n_x - n_u + 1):
        T += np.linalg.matrix_power(A, k) @ (B @ np.linalg.inv(Qm) @ B.T + H.T @ H + epsilon * np.eye(n_x)) @ np.linalg.matrix_power(A, k).T
    P = torch.inverse(T)
    return P

def findF_1D(P, Qm):
    n_u = Qm.shape[0]
    n_x = P.shape[0]
    A, B = construct_AB(n_x, n_u)
    F_upper_left = P - A.T @ P @ A
    F_upper_right = - A.T @ P @ B
    F_lower_left = - B.T @ P @ A
    F_lower_right = Qm - B.T @ P @ B
    F = np.block([[F_upper_left, F_upper_right], [F_lower_left, F_lower_right]])
    return 0.5 * (F + F.T)

def construct_AB(n_x, n_u):
    A_upper_left = torch.zeros((n_x - n_u, n_u))
    A_upper_right = torch.eye(n_x - n_u)
    A = np.block([[A_upper_left, A_upper_right], [torch.zeros((n_u, n_x))]])    
    B_upper = torch.zeros((n_x - n_u, n_u))
    B_lower = torch.eye(n_u)
    B = np.block([[B_upper], [B_lower]])   
    return A, B

def findP1P2(H1, H2, A12, B1, Qm, cin, cout, r, s):

    # Garantir que todos os tensores estão no mesmo dispositivo
    device = B1.device  # Pegamos o device de B1 (pode ser CPU ou CUDA)
    H1 = H1.to(device)
    H2 = H2.to(device)
    A12 = A12.to(device)
    B1 = B1.to(device)
    Qm = Qm.to(device)


    n_u = Qm.shape[0]
    n_x1 = H1.shape[0]
    n_x2 = H2.shape[0]
    n_y = cout
    epsilon = 1e-6
    A11, A22, B2, _ = construct_A11A22B2C1(cin, cout, r, s)
    A11 = A11.to(device)     # Construção das matrizes e movimentação para o mesmo device
    A22 = A22.to(device)
    B2 = B2.to(device)

    B = torch.cat((B1, B2), dim = 0)
    if s > 1:
        Qm = torch.kron(torch.eye(s**2, device=device),Qm)
    Qminv = torch.linalg.inv(Qm)
    # Garantir que B1, B2 e Qminv estão no mesmo device
    B1 = B1.to(device)
    B2 = B2.to(device)
    Qminv = Qminv.to(device)

    Xtilde = torch.cat((torch.cat((B1 @ Qminv @ B1.T, B1 @ Qminv @ B2.T), dim = 1), torch.cat((B2 @ Qminv @ B1.T, B2 @ Qminv @ B2.T), dim = 1)), dim = 0) #+ epsilon * torch.eye(n_x1+n_x2)
    Xtilde = (Xtilde + Xtilde.T) / 2
    Xtilde22 = Xtilde[n_x1:, n_x1:]
    Xtilde11 = Xtilde[:n_x1, :n_x1]
    Xtilde12 = Xtilde[:n_x1, n_x1:]

    # Compute T2
    T2 = torch.zeros((n_x2, n_x2), dtype = torch.float64, device=device)
    for k in range(n_x2 - n_u + 1):
        T2 += torch.matrix_power(A22, k) @ (Xtilde22 + H2.T @ H2 + epsilon * torch.eye(n_x2, device=device)) @ torch.matrix_power(A22, k).T

    # Compute Xhat11
    Xhat11 = A12 @ T2 @ A12.T + Xtilde11 + (Xtilde12 + A12 @ T2 @ A22.T) @ torch.inverse(T2 - A22 @ T2 @ A22.T - Xtilde22) @ (Xtilde12 + A12 @ T2 @ A22.T).T
    
    # Compute T1
    T1 = torch.zeros((n_x1, n_x1), dtype = torch.float64, device=device)
    for k in range(n_x1 - n_y + 1):
        T1 += torch.matrix_power(A11, k) @ (Xhat11 + H1.T @ H1 + epsilon * torch.eye(n_x1, device=device)) @ torch.matrix_power(A11, k).T
    
    P1 = torch.inverse(T1)
    P2 = torch.inverse(T2)

    # Garantir que as inversas estão no mesmo device
    P1 = torch.inverse(T1).to(device)
    P2 = torch.inverse(T2).to(device)

    return P1, P2

def findF_2D(P1, P2, A12, B1, Qm, cin, cout, r, s):
    
    # Obtém o device de um dos tensores (assumindo que pelo menos um está na GPU)
    device = P1.device  

    # Move todos os tensores para o mesmo device
    P1 = P1.to(device)
    P2 = P2.to(device)
    A12 = A12.to(device)
    B1 = B1.to(device)
    Qm = Qm.to(device)
    
    P = block_diagonal(P1,P2)
    n_x1 = P1.shape[0]
    n_x2 = P2.shape[0]
    A11, A22, B2, _ = construct_A11A22B2C1(cin, cout, r, s)
    A11 = A11.to(device)
    A22 = A22.to(device)
    B2  = B2.to(device)

    A21 = torch.zeros((n_x2, n_x1))

    A = torch.cat((torch.cat((A11.to(device), A12.to(device)), dim = 1), torch.cat((A21.to(device), A22.to(device)), dim = 1)), dim = 0)
    B = torch.cat((B1.to(device), B2.to(device)), dim = 0)

    if s > 1:
        Qm = torch.kron(torch.eye(s**2, device=device),Qm)

    F_upper_left = P - A.T @ P @ A
    F_upper_right = - A.T @ P @ B
    F_lower_left = - B.T @ P @ A
    F_lower_right = Qm - B.T @ P @ B
    Fmat = torch.cat((torch.cat((F_upper_left, F_upper_right), dim = 1), torch.cat((F_lower_left, F_lower_right),dim = 1)), dim = 0)
    return 0.5 * (Fmat + Fmat.T)

def flatten_stride(mat, cout):
    len_ = mat.shape[0] // cout
    device = mat.device
    mat2 = torch.empty((cout,0), device=device)
    for ii in range(1, len_ + 1):
        mat2 = torch.cat((mat2, mat[(ii-1)*cout:ii*cout, :].to(device)),dim=1)
    return mat2

def construct_A11A22B2C1(cin, cout, kernel, s):
    n_x1 = int(cout*np.ceil((kernel-s)/s))

    A11C1_identity = torch.eye(n_x1, dtype = torch.float64)
    A11C1_zeros = torch.zeros((cout,n_x1), dtype = torch.float64)

    A11C1 = torch.cat((A11C1_zeros, A11C1_identity), dim = 0)

    A11 = A11C1[:n_x1, :]
    C1 = A11C1[n_x1:,:]

    A22B2_identity = torch.eye((kernel-s)*cin, dtype = torch.float64)
    A22B2_zeros = torch.zeros(((kernel-s)*cin,s*cin), dtype = torch.float64)

    A22B2 = torch.cat((A22B2_zeros, A22B2_identity), dim = 1)

    A22 = A22B2[:, :(kernel-s)*cin]
    B2 = A22B2[:, (kernel-s)*cin:]

    if s > 1:
        A22mat = A22
        B2mat = B2
        for ii in range(1, s):
            A22 = block_diag(A22, A22mat)
            B2 = block_diag(B2, B2mat)

        A22 = torch.from_numpy(A22)
        B2 = torch.from_numpy(B2)

    return A11, A22, B2, C1

def ABCDt2K(A12,B1,C2,D,l,s):
    n_y, n_u = D.shape

    cin = int(n_u / (s**2))
    cout = n_y
    l1 = l
    l2 = l
        
    # Combine A12, B1, C2, and D into a matrix
    mat = torch.cat((torch.cat((A12, B1), dim = 1), torch.cat((C2, D), dim = 1)), dim = 0)

    # Reshape mat into a 4D array and permute its dimensions
    K = torch.reshape(mat, (l1, cout, l2, cin))

    # Reverse the order of kernel dimensions
    K = torch.flip(K, dims=[0, 2])

    # Reshape K back to its original shape
    K = torch.permute(K, (1,3,0,2))

    return K

def ABCD2K(A12,B1,C2,D,l,s):
    n_y, n_u = D.shape

    cin = int(n_u / (s**2))
    cout = n_y
    l1 = l
    l2 = l

    if s > 1:
    # In case stride >= 2, we need to reshape A12, B1, C2, D
    # Initialize arrays
        A12flat = torch.empty((cout, 0))
        B1flat = torch.empty((cout, 0))
        C2flat = C2
        Dflat = D
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # passando para o device correto
        A12flat = A12flat.to(device) 
        B1flat = B1flat.to(device)
        # print("layer.py, 214", device)
        # Loop to concatenate slices of A12 and B1
        for ii in range(1, int(np.ceil((l1 - s) / s)) + 1):
            A12flat = torch.cat((A12flat, A12[(ii-1)*cout:ii*cout, :].to(device)),dim=1)
            B1flat = torch.cat((B1flat, B1[(ii-1)*cout:ii*cout, :].to(device)),dim=1)
            # print("DEVICE ***********", A12flat.device, B1flat.device)


        # Calculate r
        r = s - (l1 - s) % s
    
        # If condition to slice A12flat and B1flat
        if r != s:
            A12flat = A12flat[:, r*cin*(l2-s):]
            B1flat = B1flat[:, r*cin*s:]
        
        # Initialize empty arrays for A12, B1, C2, and D
        A12 = torch.empty((0, (l2-s)*cin))
        B1 = torch.empty((0, s*cin))
        C2 = torch.empty((0, (l2-s)*cin))
        D = torch.empty((0, s*cin))
    
        # Colocando nos dispositivos apropriados:
        device = A12.device  # Obtém o device do tensor A12
        A12flat = A12flat.to(device)  # Move A12flat para o mesmo device de A12
        B1 = B1.to(device)
        C2 = C2.to(device)
        D  = D.to(device)

        # Loop to concatenate slices of A12flat and B1flat
        for ii in range(l1 - s):
            A12 = torch.cat((A12, A12flat[:, ii*(l2-s)*cin:(ii+1)*(l2-s)*cin].to(device)), dim = 0)
            B1 = torch.cat((B1, B1flat[:, ii*s*cin:(ii+1)*s*cin].to(device)), dim = 0)
        
        # Loop to concatenate slices of C2flat and Dflat
        for ii in range(s):
            C2 = torch.cat((C2, C2flat[:, ii*(l2-s)*cin:(ii+1)*(l2-s)*cin].to(device)), dim = 0)
            D = torch.cat((D, Dflat[:, ii*s*cin:(ii+1)*s*cin].to(device)), dim = 0)
        
    # Combine A12, B1, C2, and D into the final matrix
    mat = torch.cat((torch.cat((A12, B1), dim = 1), torch.cat((C2, D), dim = 1)), dim = 0)

    # Reshape mat into a 4D array and permute its dimensions
    K = torch.reshape(mat, (l1, cout, l2, cin))

    # Reverse the order of kernel dimensions
    #K = K[::-1, :, ::-1, :]
    K = torch.flip(K, dims=[0, 2])

    # Reshape K back to its original shape
    K = torch.permute(K, (1,3,0,2))

    return K

def reshape_A12B1(A12t, B1t, cout, cin, r, s):
    # Definir o device com base no tensor de entrada
    device = A12t.device

    # Garantir que os tensores de entrada estão no mesmo device
    A12t = A12t.to(device)
    B1t = B1t.to(device) 

    # Flattening matrices
    A12flat = flatten_stride(A12t, cout).to(device)
    B1flat = flatten_stride(B1t, cout).to(device)
    l1 = r
    l2 = r

    # Calculate r
    r = s - ((l1 - s) % s)
        
    # If condition and padding with zeros
    if r != s:
        A12flat = torch.cat((torch.zeros((cout, r * cin * (l2 - s)),device=device), A12flat), dim = 1)
        B1flat = torch.cat((torch.zeros((cout, r * cin * s),device=device), B1flat), dim = 1)
            
    # Initialize empty arrays
    A12 = torch.empty((0, cin * (l2 - s) * s), device=device)
    B1 = torch.empty((0, cin * s**2), device=device)
        
    # Loop to concatenate slices of A12flat and B1flat
    for ii in range(1, int(np.ceil((l1 - s) / s)) + 1):
        A12 = torch.cat((A12, A12flat[:, (ii-1)*cin*(l2-s)*s : ii*cin*(l2-s)*s]), dim = 0)
        B1 = torch.cat((B1, B1flat[:, (ii-1)*cin*s**2 : ii*cin*s**2]), dim = 0)
    return A12, B1

def K2ABCD(K,s = 1):

    # Extract size of K
    cout, cin, l1, l2 = K.shape

    # Permute and reshape K
    K_permuted = torch.permute(torch.flip(K,dims = (2,3)), (2, 0, 3, 1))
    K_reshaped = torch.reshape(K_permuted, (-1, l2 * cin))

    # Split mat into A12, B1, C2, and D
    A12 = K_reshaped[:cout * (l1 - s), :cin * (l2 - s)]
    B1 = K_reshaped[:cout * (l1 - s), cin * (l2 - s):cin * l2]
    C2 = K_reshaped[cout * (l1 - s):, :cin * (l2 - s)]
    D = K_reshaped[cout * (l1 - s):, (l2 - s) * cin:]

    # Additional reshaping if stride >= 2
    if s>1:
        A12, B1 = reshape_A12B1(A12, B1, cout, cin, l1, s)
        C2 = flatten_stride(C2, cout)
        D = flatten_stride(D, cout)

    return A12,B1,C2,D

def block_diagonal(matrix1, matrix2):
    # Get the shapes of the input matrices
    m1, n1 = matrix1.shape
    m2, n2 = matrix2.shape

    # Compute the size of the resulting block-diagonal matrix
    m = m1 + m2
    n = n1 + n2

    # Create a zero-initialized matrix of the correct size
    result = torch.zeros((m, n), dtype=matrix1.dtype, device=matrix1.device)

    # Copy the input matrices to the appropriate positions
    result[:m1, :n1] = matrix1
    result[m1:, n1:] = matrix2

    return result

## from https://github.com/locuslab/orthogonal-convolutions
def cayley(W):
    if len(W.shape) == 2:
        return cayley(W[None])[0]
    _, cout, cin = W.shape 
    if cin > cout:
        return cayley(W.transpose(1, 2)).transpose(1, 2)
    U, V = W[:, :cin], W[:, cin:]
    I = torch.eye(cin, dtype=W.dtype, device=W.device)[None, :, :]
    A = U - U.conj().transpose(1, 2) + V.conj().transpose(1, 2) @ V
    iIpA = torch.inverse(I + A)
    return torch.cat((iIpA @ (I - A), -2 * V @ iIpA), axis=1)
            
def fft_shift_matrix( n, s):
        shift = torch.arange(0, n).repeat((n, 1))
        shift = shift + shift.T
        return torch.exp(1j * 2 * np.pi * s * shift / n)
        
class StridedConv(nn.Module):
    def __init__(self, *args, **kwargs):
        striding = False
        if 'stride' in kwargs and kwargs['stride'] == 2:
            kwargs['stride'] = 1
            striding = True
        super().__init__(*args, **kwargs)
        downsample = "b c (w k1) (h k2) -> b (c k1 k2) w h"
        if striding:
            self.register_forward_pre_hook(lambda _, x: \
                    einops.rearrange(x[0], downsample, k1=2, k2=2))  

#------------------------------------------------------------
## Sandwich

class SandwichLin(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, scale=1.0, AB=False):
        super().__init__(in_features+out_features, out_features, bias)
        self.alpha = nn.Parameter(torch.ones(1, dtype=torch.float32, requires_grad=True))
        self.alpha.data = self.weight.norm()
        self.scale = scale   
        self.AB = AB
        self.Q = None

    def forward(self, x):
        fout, _ = self.weight.shape
        if self.training or self.Q is None:
            self.Q = cayley(self.alpha * self.weight / self.weight.norm())
        Q = self.Q if self.training else self.Q.detach()
        x = F.linear(self.scale * x, Q[:, fout:]) # B @ x 
        if self.AB:
            x = 2 * F.linear(x, Q[:, :fout].T) # 2 A.T @ B @ x
        if self.bias is not None:
            x += self.bias
        return x
    
class SandwichFc(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, scale=1.0):
        super().__init__(in_features+out_features, out_features, bias)
        self.alpha = nn.Parameter(torch.ones(1, dtype=torch.float32, requires_grad=True))
        self.alpha.data = self.weight.norm() 
        self.scale = scale 
        self.psi = nn.Parameter(torch.zeros(out_features, dtype=torch.float32, requires_grad=True))   
        self.Q = None

    def forward(self, x):
        fout, _ = self.weight.shape
        if self.training or self.Q is None:
            self.Q = cayley(self.alpha * self.weight / self.weight.norm())
        Q = self.Q if self.training else self.Q.detach()
        x = F.linear(self.scale * x, Q[:, fout:]) # B*h 
        if self.psi is not None:
            x = x * torch.exp(-self.psi) * (2 ** 0.5) # sqrt(2) \Psi^{-1} B * h
        if self.bias is not None:
            x += self.bias
        x = F.relu(x) * torch.exp(self.psi) # \Psi z
        x = 2 ** 0.5 * F.linear(x, Q[:, :fout].T) # sqrt(2) A^top \Psi z
        return x

class SandwichConv1(nn.Module):
    def __init__(self,cin, cout, scale=1.0) -> None:
        super().__init__()
        self.scale = scale
        self.kernel = nn.Parameter(torch.empty(cout, cin+cout))
        self.bias = nn.Parameter(torch.empty(cout))
        nn.init.xavier_normal_(self.kernel)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.kernel)
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound) 
        self.alpha = nn.Parameter(torch.ones(1, dtype=torch.float32, requires_grad=True))
        self.alpha.data = self.kernel.norm()
        self.Q = None

    def forward(self, x):
        cout = self.kernel.shape[0]
        if self.training or self.Q is None:
            P = cayley(self.alpha * self.kernel / self.kernel.norm())
            self.Q = 2 * P[:, :cout].T @ P[:, cout:]
        Q = self.Q if self.training else self.Q.detach()
        x = F.conv2d(self.scale * x, Q[:,:, None, None])
        x += self.bias[:, None, None]
        return F.relu(x)

class SandwichConv1Lin(nn.Module):
    def __init__(self,cin, cout, scale=1.0) -> None:
        super().__init__()
        self.scale = scale
        self.kernel = nn.Parameter(torch.empty(cout, cin+cout))
        self.bias = nn.Parameter(torch.empty(cout))
        nn.init.xavier_normal_(self.kernel)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.kernel)
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound) 
        self.alpha = nn.Parameter(torch.ones(1, dtype=torch.float32, requires_grad=True))
        self.alpha.data = self.kernel.norm()
        self.Q = None

    def forward(self, x):
        cout = self.kernel.shape[0]
        if self.training or self.Q is None:
            P = cayley(self.alpha * self.kernel / self.kernel.norm())
            self.Q = 2 * P[:, :cout].T @ P[:, cout:]
        Q = self.Q if self.training else self.Q.detach()
        x = F.conv2d(self.scale * x, Q[:,:, None, None])
        x += self.bias[:, None, None]
        return x
    
class SandwichConvLin(StridedConv, nn.Conv2d):
    def __init__(self, *args, **kwargs):
        args = list(args)
        if 'stride' in kwargs and kwargs['stride'] == 2:
            args = list(args)
            args[0] = 4 * args[0] # 4x in_channels
            if len(args) == 3:
                args[2] = max(1, args[2] // 2) # //2 kernel_size; optional
                kwargs['padding'] = args[2] // 2 # TODO: added maxes recently
            elif 'kernel_size' in kwargs:
                kwargs['kernel_size'] = max(1, kwargs['kernel_size'] // 2)
                kwargs['padding'] = kwargs['kernel_size'] // 2
        scale = 1.0
        if 'scale' in kwargs:
            scale = kwargs['scale']
            del kwargs['scale']
        args[0] += args[1]
        args = tuple(args)
        super().__init__(*args, **kwargs)
        self.scale = scale
        self.register_parameter('alpha', None)
        self.Qfft = None
        
    def forward(self, x):
        x = self.scale * x 
        cout, chn, _, _ = self.weight.shape
        cin = chn - cout
        batches, _, n, _ = x.shape
        if not hasattr(self, 'shift_matrix'):
            s = (self.weight.shape[2] - 1) // 2
            self.shift_matrix = fft_shift_matrix(n, -s)[:, :(n//2 + 1)].reshape(n * (n // 2 + 1), 1, 1).to(x.device)
        
        if self.training or self.Qfft is None or self.alpha is None:
            wfft = self.shift_matrix * torch.fft.rfft2(self.weight, (n, n)).reshape(cout, chn, n * (n // 2 + 1)).permute(2, 0, 1).conj()
            if self.alpha is None:
                self.alpha = nn.Parameter(torch.tensor(wfft.norm().item(), requires_grad=True).to(x.device))
            self.Qfft = cayley(self.alpha * wfft / wfft.norm())
        
        Qfft = self.Qfft if self.training else self.Qfft.detach()
        # Afft, Bfft = Qfft[:,:,:cout], Qfft[:,:,cout:]
        xfft = torch.fft.rfft2(x).permute(2, 3, 1, 0).reshape(n * (n // 2 + 1), cin, batches)
        xfft = 2 * Qfft[:,:,:cout].conj().transpose(1,2) @ Qfft[:,:,cout:] @ xfft 
        x = torch.fft.irfft2(xfft.reshape(n, n // 2 + 1, cout, batches).permute(3, 2, 0, 1))
        if self.bias is not None:
            x += self.bias[:, None, None]

        return x
        
class SandwichConv(StridedConv, nn.Conv2d):
    def __init__(self, *args, **kwargs):
        args = list(args)
        if 'stride' in kwargs and kwargs['stride'] == 2:
            args = list(args)
            args[0] = 4 * args[0] # 4x in_channels
            if len(args) == 3:
                args[2] = max(1, args[2] // 2) # //2 kernel_size; optional
                kwargs['padding'] = args[2] // 2 # TODO: added maxes recently
            elif 'kernel_size' in kwargs:
                kwargs['kernel_size'] = max(1, kwargs['kernel_size'] // 2)
                kwargs['padding'] = kwargs['kernel_size'] // 2
        scale = 1.0
        if 'scale' in kwargs:
            scale = kwargs['scale']
            del kwargs['scale']
        args[0] += args[1]
        args = tuple(args)
        super().__init__(*args, **kwargs)
        self.psi  = nn.Parameter(torch.zeros(args[1]))
        self.scale = scale
        self.register_parameter('alpha', None)
        self.Qfft = None
        
    def forward(self, x):
        x = self.scale * x 
        cout, chn, _, _ = self.weight.shape
        cin = chn - cout
        batches, _, n, _ = x.shape
        if not hasattr(self, 'shift_matrix'):
            s = (self.weight.shape[2] - 1) // 2
            self.shift_matrix = fft_shift_matrix(n, -s)[:, :(n//2 + 1)].reshape(n * (n // 2 + 1), 1, 1).to(x.device)
        
        if self.training or self.Qfft is None or self.alpha is None:
            wfft = self.shift_matrix * torch.fft.rfft2(self.weight, (n, n)).reshape(cout, chn, n * (n // 2 + 1)).permute(2, 0, 1).conj()
            if self.alpha is None:
                self.alpha = nn.Parameter(torch.tensor(wfft.norm().item(), requires_grad=True).to(x.device))
            self.Qfft = cayley(self.alpha * wfft / wfft.norm())
        
        Qfft = self.Qfft if self.training else self.Qfft.detach()
        # Afft, Bfft = Qfft[:,:,:cout], Qfft[:,:,cout:]
        xfft = torch.fft.rfft2(x).permute(2, 3, 1, 0).reshape(n * (n // 2 + 1), cin, batches)
        xfft = 2 ** 0.5 * torch.exp(-self.psi).diag().type(xfft.dtype) @ Qfft[:,:,cout:] @ xfft 
        x = torch.fft.irfft2(xfft.reshape(n, n // 2 + 1, cout, batches).permute(3, 2, 0, 1))
        if self.bias is not None:
            x += self.bias[:, None, None]
        xfft = torch.fft.rfft2(F.relu(x)).permute(2, 3, 1, 0).reshape(n * (n // 2 + 1), cout, batches)
        xfft = 2 ** 0.5 * Qfft[:,:,:cout].conj().transpose(1,2) @ torch.exp(self.psi).diag().type(xfft.dtype) @ xfft
        x = torch.fft.irfft2(xfft.reshape(n, n // 2 + 1, cout, batches).permute(3, 2, 0, 1))

        return x

#------------------------------------------------------------
## LipCNN

class LipCNNConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride = 1, padding = 1, bias = True):
        super().__init__()
        self.n_x1 = out_channels*int(np.ceil((kernel-stride)/stride))
        self.n_x2 = in_channels*(kernel-stride)*stride
        self.n_u = in_channels*stride**2
        self.n_y = out_channels
        self.Q = None

        init2C2F=False

        if init2C2F == True:
            ## This inititialization is used for 2C2F
            self.weight = nn.Parameter(torch.empty((out_channels + self.n_x2 + self.n_u, out_channels), dtype=torch.float64), requires_grad=True)
            self.bias = nn.Parameter(torch.zeros(out_channels), requires_grad=True)
            self.A12 = nn.Parameter(torch.empty((out_channels * (kernel - stride), in_channels * (kernel - stride)), dtype=torch.float64), requires_grad=True)
            self.B1 = nn.Parameter(torch.empty((out_channels * (kernel - stride), in_channels * stride), dtype=torch.float64), requires_grad=True)
            self.H1 = nn.Parameter(torch.empty((self.n_x1, self.n_x1), dtype=torch.float64), requires_grad=True)
            self.H2 = nn.Parameter(torch.empty((self.n_x2, self.n_x2), dtype=torch.float64), requires_grad=True)
            self.psi = nn.Parameter(100*torch.ones((out_channels), dtype=torch.float64), requires_grad=True)
            self.q = nn.Parameter(torch.ones((out_channels), dtype=torch.float64), requires_grad=True)

            nn.init.xavier_normal_(self.weight)
            nn.init.xavier_normal_(self.A12)
            nn.init.xavier_normal_(self.B1)
            nn.init.xavier_normal_(self.H1)
            nn.init.xavier_normal_(self.H2)
        else:
            ## This inititialization is used for LeNet5
            self.weight = nn.Parameter(torch.randn((out_channels + self.n_x2 + self.n_u, out_channels), dtype=torch.float64), requires_grad=True)
            self.bias = nn.Parameter(torch.zeros(out_channels), requires_grad=True)
            self.A12 = nn.Parameter(0.001*torch.randn((out_channels * (kernel - stride), in_channels * (kernel - stride)), dtype=torch.float64), requires_grad=True)
            self.B1 = nn.Parameter(0.001*torch.randn((out_channels * (kernel - stride), in_channels * stride), dtype=torch.float64), requires_grad=True)
            self.H1 = nn.Parameter(torch.randn((self.n_x1, self.n_x1), dtype=torch.float64), requires_grad=True)
            self.H2 = nn.Parameter(torch.randn((self.n_x2, self.n_x2), dtype=torch.float64), requires_grad=True)
            self.psi = nn.Parameter(10*torch.ones((out_channels), dtype=torch.float64), requires_grad=True)
            self.q = nn.Parameter(torch.ones((out_channels), dtype=torch.float64), requires_grad=True)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = kernel
        self.stride = stride
        self.padding = padding

    def LipCNN2CNN(self,L):
        cin = self.in_channels
        cout = self.out_channels
        s = self.stride
        r = self.kernel
        epsilon = 1e-6
        n_x1 = self.n_x1
        n_x2 = self.n_x2
        n_u = self.n_u
        n_y = self.n_y
        if s>1:
            A12, B1 = reshape_A12B1(self.A12, self.B1, cout, cin, r, s)
        else:
            A12 = self.A12
            B1 = self.B1

        if self.training or self.Q is None:
            self.Q = cayley(self.weight)
        Q = self.Q if self.training else self.Q.detach()
        U = Q[:cout, :]
        V = Q[cout:, :]
        Qm = L.T @ L

        P1, P2 = findP1P2(self.H1, self.H2, A12, B1, Qm, cin, cout, r, s)
        Fmat = findF_2D(P1, P2, A12, B1, Qm, cin, cout, r, s)

        F1 = Fmat[:n_x1, :n_x1]
        F2 = Fmat[n_x1:, n_x1:]
        F12 = Fmat[:n_x1, n_x1:] 
        device = F1.device  # Obtém o dispositivo de F1
        #print("DEVICE =", device)
        F1inv = torch.linalg.inv(F1 + 0 * torch.eye(n_x1, device=device))

        _, _, _, C1 = construct_A11A22B2C1(cin, cout, r, s)

        F1inv = F1inv.to(device)  # Move F1inv para o mesmo dispositivo de C1
        C1 = C1.to(device)
        mat = 0.5 * C1 @ (F1inv+F1inv.T) @ C1.T
        #print("DEVICE ==============",C1.device, F1inv.device)

        gamma = epsilon + self.psi ** 2
        gamma += 0.5 * torch.sum(torch.abs(mat), dim=1) # Diagonal dominance
        gamma += 0.5 * torch.max(torch.linalg.eigvalsh(mat)) # Maximum eigenvalue
        q_abs = torch.abs(self.q)
        #q_abs = self.q ** 2
        q = q_abs[None, :]
        q_inv = (1/(q_abs+epsilon))[:, None]
        gamma += 0.5 * torch.abs(q_inv * mat  * q).sum(1)


        try:
            LGamma = torch.linalg.cholesky(2*torch.diag(gamma) - mat)
        except RuntimeError:
            F1inv = torch.linalg.inv(F1 + 1e-12 * torch.eye(n_x1, device = device))

            mat = 0.5 * C1 @ (F1inv+F1inv.T) @ C1.T
            gamma = epsilon + self.psi ** 2
            gamma += 0.5 * torch.sum(torch.abs(mat), dim=1) # Diagonal dominance
            #gamma += 0.5 * torch.max(torch.linalg.eigvalsh(mat)) # Maximum eigenvalue

            LGamma = torch.linalg.cholesky(2*torch.diag(gamma) - mat)

        try:
            arg = F2 - F12.T @ F1inv @ F12
            LF = torch.linalg.cholesky(arg) # + epsilon * torch.eye(n_x2+n_u))
        except RuntimeError:
            F1inv = torch.linalg.inv(F1 + 1e-12 * torch.eye(n_x1, device = device))
            arg = F2 - F12.T @ F1inv @ F12
            try: 
                LF = torch.linalg.cholesky(arg + 1e-12 * torch.eye(n_x2+n_u, device = device))
                #print('cheat1')
            except RuntimeError:
                try:
                    LF = torch.linalg.cholesky(arg + 1e-10 * torch.eye(n_x2+n_u, device = device))
                    print('cheat2')
                except RuntimeError:
                    try:
                        LF = torch.linalg.cholesky(arg + 1e-8 * torch.eye(n_x2+n_u, device = device))
                        print('cheat3')
                    except RuntimeError:
                        try:
                            LF = torch.linalg.cholesky(arg + 1e-6 * torch.eye(n_x2+n_u, device = device))
                            print('cheat4')
                        except RuntimeError:
                            LF = torch.linalg.cholesky(arg + 1e-4 * torch.eye(n_x2+n_u, device = device))
                            print('cheat5')

        C2hat =  C1 @ F1inv @ F12 - LGamma @ V.T @ LF.T
        C2 = C2hat[:,:n_x2]
        D = C2hat[:,n_x2:]

        K = ABCD2K(A12, B1 , C2, D, r, s)

        L = U @ LGamma.T / gamma

        return K, L
    
    def forward(self, x, L):
        K, L = self.LipCNN2CNN(L)
        s = self.stride
        p = self.padding
        r = self.kernel

        #print("DEVICE do x = ", x.device)
        x = F.conv2d(x, K.to(x.device).float(), padding=p, stride=s)
        if self.bias is not None:
            # print("layer.py 718", x.device)
            x += self.bias.to(x.device)[:, None, None]
        return x, L


class LipCNNConvMax(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride = 1, padding = 1):
        super().__init__()
        self.n_x1 = out_channels*int(np.ceil((kernel-stride)/stride))
        self.n_x2 = in_channels*(kernel-stride)*stride
        self.n_u = in_channels*stride**2
        self.n_y = out_channels

        ## This inititialization is used for LeNet5
        #self.weight = nn.Parameter(torch.randn((out_channels + self.n_x2 + self.n_u, out_channels), dtype=torch.float64), requires_grad=True)
        #self.A12 = nn.Parameter(0.001*torch.randn((out_channels * (kernel - stride), in_channels * (kernel - stride)), dtype=torch.float64), requires_grad=True)
        #self.B1 = nn.Parameter(0.001*torch.randn((out_channels * (kernel - stride), in_channels * stride), dtype=torch.float64), requires_grad=True)
        #self.H1 = nn.Parameter(torch.randn((self.n_x1, self.n_x1), dtype=torch.float64), requires_grad=True)
        #self.H2 = nn.Parameter(torch.randn((self.n_x2, self.n_x2), dtype=torch.float64), requires_grad=True)
        #self.psi = nn.Parameter(10*torch.randn((out_channels), dtype=torch.float64), requires_grad=True)

        ## This inititialization is used for 2C2F
        self.weight = nn.Parameter(torch.empty((self.n_x2 + self.n_u, out_channels), dtype=torch.float64), requires_grad=True)
        self.A12 = nn.Parameter(torch.empty((out_channels * (kernel - stride), in_channels * (kernel - stride)), dtype=torch.float64), requires_grad=True)
        self.B1 = nn.Parameter(torch.empty((out_channels * (kernel - stride), in_channels * stride), dtype=torch.float64), requires_grad=True)
        self.H1 = nn.Parameter(torch.empty((self.n_x1, self.n_x1), dtype=torch.float64), requires_grad=True)
        self.H2 = nn.Parameter(torch.empty((self.n_x2, self.n_x2), dtype=torch.float64), requires_grad=True)
        self.delta = nn.Parameter(torch.ones((out_channels), dtype=torch.float64), requires_grad=True)
        self.omega = nn.Parameter(torch.ones((out_channels), dtype=torch.float64), requires_grad=True)

        nn.init.xavier_normal_(self.weight)
        nn.init.xavier_normal_(self.A12)
        nn.init.xavier_normal_(self.B1)
        nn.init.xavier_normal_(self.H1)
        nn.init.xavier_normal_(self.H2)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = kernel
        self.stride = stride
        self.padding = padding

    def LipCNN2CNN(self,L):
        cin = self.in_channels
        cout = self.out_channels
        s = self.stride
        r = self.kernel
        epsilon = 1e-6
        n_x1 = self.n_x1
        n_x2 = self.n_x2
        n_u = self.n_u
        n_y = self.n_y
        if s>1:
            A12, B1 = reshape_A12B1(self.A12, self.B1, cout, cin, r, s)
        else:
            A12 = self.A12
            B1 = self.B1


        if self.training or self.Q is None:
            self.Q = cayley(self.weight)
        Q = self.Q if self.training else self.Q.detach()
        V = Q
        Qm = L.T @ L

        P1, P2 = findP1P2(self.H1, self.H2, A12, B1, Qm, cin, cout, r, s)
        Fmat = findF_2D(P1, P2, A12, B1, Qm, cin, cout, r, s)

        F1 = Fmat[:n_x1, :n_x1]
        F2 = Fmat[n_x1:, n_x1:]
        F12 = Fmat[:n_x1, n_x1:] 
        device = F1.device  # Obtém o dispositivo de F1
        # print("DEVICE =", device)
        F1inv = torch.linalg.inv(F1 + 0 * torch.eye(n_x1, device=device))

        _, _, _, C1 = construct_A11A22B2C1(cin, cout, r, s)

        mat = 0.5 * C1 @ (F1inv+F1inv.T) @ C1.T

        mu = epsilon + self.delta **2
        mu += torch.sum(torch.abs(mat), dim=1) # Diagonal dominance
        #mu += 0.5 * torch.max(torch.linalg.eigvalsh(mat)) # Maximum eigenvalue
        gamma = 0.5 * mu + self.omega ** 2
        q = (2*gamma-mu) / gamma ** 2
        l = q ** 0.5

        try:
            LGamma = torch.linalg.cholesky(torch.diag(mu) - mat)
        except RuntimeError:
            F1inv = torch.linalg.inv(F1 + 1e-12 * torch.eye(n_x1))

            mat = 0.5 * C1 @ (F1inv+F1inv.T) @ C1.T
            mu = epsilon + self.delta **2
            mu += torch.sum(torch.abs(mat), dim=1) # Diagonal dominance
            #mu += 0.5 * torch.max(torch.linalg.eigvalsh(mat)) # Maximum eigenvalue
            gamma = mu + self.omega ** 2
            q = (2*gamma-mu) / gamma ** 2
            l = q ** 0.5
            
            LGamma = torch.linalg.cholesky(torch.diag(mu) - mat)

        
        try:
            arg = F2 - F12.T @ F1inv @ F12
            LF = torch.linalg.cholesky(arg)
        except RuntimeError:
            F1inv = torch.linalg.inv(F1 + 1e-12 * torch.eye(n_x1))
            arg = F2 - F12.T @ F1inv @ F12
            try: 
                LF = torch.linalg.cholesky(arg + 1e-12 * torch.eye(n_x2+n_u))
                #print('cheat1')
            except RuntimeError:
                try:
                    LF = torch.linalg.cholesky(arg + 1e-10 * torch.eye(n_x2+n_u))
                    print('cheat2')
                except RuntimeError:
                    try:
                        LF = torch.linalg.cholesky(arg + 1e-8 * torch.eye(n_x2+n_u))
                        print('cheat3')
                    except RuntimeError:
                        try:
                            LF = torch.linalg.cholesky(arg + 1e-6 * torch.eye(n_x2+n_u))
                            print('cheat4')
                        except RuntimeError:
                            LF = torch.linalg.cholesky(arg + 1e-4 * torch.eye(n_x2+n_u))
                            print('cheat5')

        C2hat =  C1 @ F1inv @ F12 - LGamma @ V.T @ LF.T
        C2 = C2hat[:,:n_x2]
        D = C2hat[:,n_x2:]

        K = ABCD2K(A12, B1 , C2, D, r, s)

        L = torch.diag(l)

        return K, L
    
    def forward(self, x, L):
        K, L = self.LipCNN2CNN(L)
        s = self.stride
        p = self.padding
        r = self.kernel

        x = F.conv2d(x, K.float(), padding = p, stride = s)
        x += self.bias[:, None, None]
        return x, L


class LipCNNFc(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, psi=True):
        super().__init__(in_features+out_features, out_features, bias)
        self.psi = nn.Parameter(torch.ones(out_features, dtype=torch.float64), requires_grad=True) if psi else None
        self.Q = None
        self.cin = in_features
        self.cout = out_features
        self.weight.data = self.weight.data.to(torch.float64)


    def LipFC2FC(self,L):
        cout = self.cout

        if self.training or self.Q is None:
            self.Q = cayley(self.weight.T)

        Q = self.Q if self.training else self.Q.detach()
        U = Q[:cout, :]
        V = Q[cout:, :]
        if self.psi is not None:
            L0 = L
            W = (2 ** 0.5) * L.T @ V / self.psi #* torch.exp(-self.psi) 
            L = (2 ** 0.5) * U * self.psi #* torch.exp(self.psi)

        else:
            W = L.T @ V
            L = 0

        return W.T, L

    def forward(self, x, L):
        W, L = self.LipFC2FC(L)
        # print("layer.py 898", x.device )
        x = F.linear(x, W.to(x.device).float())
        if self.bias is not None:
            x += self.bias.to(x.device)
        return x, L
 
 #------------------------------------------------------------   
## Orthogonal layer, from https://github.com/locuslab/orthogonal-convolutions 
 
class OrthogonLin(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, scale=1.0):
        super().__init__(in_features, out_features, bias)
        self.alpha = nn.Parameter(torch.ones(1, dtype=torch.float32, requires_grad=True))
        self.alpha.data = self.weight.norm()
        self.scale = scale
        self.Q = None
            
    def forward(self, x):
        if self.training or self.Q is None:
            self.Q = cayley(self.alpha * self.weight / self.weight.norm())
        Q = self.Q if self.training else self.Q.detach()
        y = F.linear(self.scale * x, Q, self.bias)
        return y
 
class OrthogonFc(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, scale=1.0):
        super().__init__(in_features, out_features, bias)
        self.activation = nn.ReLU(inplace=False)
        self.alpha = nn.Parameter(torch.ones(1, dtype=torch.float32, requires_grad=True))
        self.alpha.data = self.weight.norm()
        self.scale = scale
        self.Q = None
            
    def forward(self, x):
        if self.training or self.Q is None:
            self.Q = cayley(self.alpha * self.weight / self.weight.norm())
        Q = self.Q if self.training else self.Q.detach()
        y = F.linear(self.scale * x, Q, self.bias)
        y = self.activation(y)
        return y

class OrthogonConvLin(StridedConv, nn.Conv2d):
    def __init__(self, *args, **kwargs):
        args = list(args)
        if 'stride' in kwargs and kwargs['stride'] == 2:
            args = list(args)
            args[0] = 4 * args[0] # 4x in_channels
            if len(args) == 3:
                args[2] = max(1, args[2] // 2) 
                kwargs['padding'] = args[2] // 2 
            elif 'kernel_size' in kwargs:
                kwargs['kernel_size'] = max(1, kwargs['kernel_size'] // 2)
                kwargs['padding'] = kwargs['kernel_size'] // 2
        scale = 1.0
        if 'scale' in kwargs:
            scale = kwargs['scale']
            del kwargs['scale']
        args = tuple(args)
        super().__init__(*args, **kwargs)
        self.scale = scale 
        self.register_parameter('alpha', None)
        self.Qfft = None
    
    def forward(self, x):
        x = self.scale * x 
        cout, cin, _, _ = self.weight.shape
        batches, _, n, _ = x.shape
        if not hasattr(self, 'shift_matrix'):
            s = (self.weight.shape[2] - 1) // 2
            self.shift_matrix = fft_shift_matrix(n, -s)[:, :(n//2 + 1)].reshape(n * (n // 2 + 1), 1, 1).to(x.device)
        xfft = torch.fft.rfft2(x).permute(2, 3, 1, 0).reshape(n * (n // 2 + 1), cin, batches)
        if self.training or self.Qfft is None or self.alpha is None:
            wfft = self.shift_matrix * torch.fft.rfft2(self.weight, (n, n)).reshape(cout, cin, n * (n // 2 + 1)).permute(2, 0, 1).conj()
            if self.alpha is None:
                self.alpha = nn.Parameter(torch.tensor(wfft.norm().item(), requires_grad=True).to(x.device))
            self.Qfft = cayley(self.alpha * wfft / wfft.norm())
        Qfft = self.Qfft if self.training else self.Qfft.detach()
        yfft = (Qfft @ xfft).reshape(n, n // 2 + 1, cout, batches)
        y = torch.fft.irfft2(yfft.permute(3, 2, 0, 1))
        if self.bias is not None:
            y += self.bias[:, None, None]
        return y
    
class OrthogonConv(StridedConv, nn.Conv2d):
    def __init__(self, *args, **kwargs):
        args = list(args)
        if 'stride' in kwargs and kwargs['stride'] == 2:
            args = list(args)
            args[0] = 4 * args[0] # 4x in_channels
            if len(args) == 3:
                args[2] = max(1, args[2] // 2) 
                kwargs['padding'] = args[2] // 2 
            elif 'kernel_size' in kwargs:
                kwargs['kernel_size'] = max(1, kwargs['kernel_size'] // 2)
                kwargs['padding'] = kwargs['kernel_size'] // 2
        scale = 1.0
        if 'scale' in kwargs:
            scale = kwargs['scale']
            del kwargs['scale']
        args = tuple(args)
        super().__init__(*args, **kwargs)
        self.scale = scale 
        self.activation = nn.ReLU(inplace=False)
        self.register_parameter('alpha', None)
        self.Qfft = None
    
    def forward(self, x):
        x = self.scale * x 
        cout, cin, _, _ = self.weight.shape
        batches, _, n, _ = x.shape
        if not hasattr(self, 'shift_matrix'):
            s = (self.weight.shape[2] - 1) // 2
            self.shift_matrix = fft_shift_matrix(n, -s)[:, :(n//2 + 1)].reshape(n * (n // 2 + 1), 1, 1).to(x.device)
        xfft = torch.fft.rfft2(x).permute(2, 3, 1, 0).reshape(n * (n // 2 + 1), cin, batches)
        if self.training or self.Qfft is None or self.alpha is None:
            wfft = self.shift_matrix * torch.fft.rfft2(self.weight, (n, n)).reshape(cout, cin, n * (n // 2 + 1)).permute(2, 0, 1).conj()
            if self.alpha is None:
                self.alpha = nn.Parameter(torch.tensor(wfft.norm().item(), requires_grad=True).to(x.device))
            self.Qfft = cayley(self.alpha * wfft / wfft.norm())
        Qfft = self.Qfft if self.training else self.Qfft.detach()
        yfft = (Qfft @ xfft).reshape(n, n // 2 + 1, cout, batches)
        y = torch.fft.irfft2(yfft.permute(3, 2, 0, 1))
        if self.bias is not None:
            y += self.bias[:, None, None]
        y = self.activation(y)
        return y

#------------------------------------------------------------
# SDP Lipschitz Layer, from https://github.com/araujoalexandre/lipschitz-sll-networks
class SLLBlockConv(nn.Module):
    def __init__(self, cin, cout, kernel_size=3, scale=1.0, epsilon=1e-6):
        super().__init__()

        self.activation = nn.ReLU(inplace=False)
        self.scale = scale
        self.kernel = nn.Parameter(torch.empty(cout, cin, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(cout))
        self.q = nn.Parameter(torch.randn(cout))

        nn.init.xavier_normal_(self.kernel)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.kernel)
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound) 

        self.epsilon = epsilon

    def forward(self, x):
        res = F.conv2d(self.scale * x, self.kernel, bias=self.bias, padding=1)
        res = self.activation(res)
        kkt = F.conv2d(self.kernel, self.kernel, padding=self.kernel.shape[-1] - 1)
        q_abs = torch.abs(self.q)
        T = 2 / (torch.abs(q_abs[None, :, None, None] * kkt).sum((1, 2, 3)) / q_abs)
        res = T[None, :, None, None] * res
        res = F.conv_transpose2d(res, self.kernel, padding=1)
        out = x - res
        return out  
                  
class SLLBlockFc(nn.Module):
    def __init__(self, cin, cout, scale = 1.0, epsilon=1e-6):
        super().__init__()
        self.activation = nn.ReLU(inplace=False)
        self.scale = scale
        self.weights = nn.Parameter(torch.empty(cout, cin))
        self.bias = nn.Parameter(torch.empty(cout))
        self.q = nn.Parameter(torch.rand(cout))

        nn.init.xavier_normal_(self.weights)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init

        self.epsilon = epsilon

    def forward(self, x):
        res = F.linear(self.scale * x, self.weights, self.bias)
        res = self.activation(res)
        q_abs = torch.abs(self.q)
        q = q_abs[None, :]
        q_inv = (1/(q_abs+self.epsilon))[:, None]
        T = 2/torch.abs(q_inv * self.weights @ self.weights.T * q).sum(1)
        res = T * res
        res = F.linear(res, self.weights.t())
        out = x - res
        return out

#------------------------------------------------------------
# almost orthogonal layer (AOL), based on SLL implementation

class AolLin(nn.Module):
    def __init__(self, cin, cout, epsilon=1e-6, scale=1.0):
        super().__init__()
        self.scale = scale
        self.weights = nn.Parameter(torch.empty(cout, cin))
        self.bias = nn.Parameter(torch.empty(cout))

        nn.init.xavier_normal_(self.weights)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  

        self.epsilon = epsilon

    def forward(self, x):
        T = 1/torch.sqrt(torch.abs(self.weights.T @ self.weights).sum(1))
        x = self.scale * T * x 
        res = F.linear(x, self.weights, self.bias)
        return res

class AolFc(nn.Module):
    def __init__(self, cin, cout, epsilon=1e-6, scale=1.0):
        super().__init__()
        self.scale = scale
        self.weights = nn.Parameter(torch.empty(cout, cin))
        self.bias = nn.Parameter(torch.empty(cout))

        nn.init.xavier_normal_(self.weights)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  

        self.epsilon = epsilon

    def forward(self, x):
        T = 1/torch.sqrt(torch.abs(self.weights.T @ self.weights).sum(1))
        x = self.scale * T * x 
        res = F.linear(x, self.weights, self.bias)
        return F.relu(res)
    
class AolConvLin(nn.Module):
    def __init__(self, cin, cout, kernel_size=3, epsilon=1e-6, scale=1.0):
        super().__init__()

        self.scale = scale
        self.kernel = nn.Parameter(torch.empty(cout, cin, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(cout))
        self.padding = (kernel_size - 1) // 2
        nn.init.xavier_normal_(self.kernel)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.kernel)
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound) 

        self.epsilon = epsilon

    def forward(self, x):
        res = F.conv2d(self.scale * x, self.kernel, padding=self.padding)
        kkt = F.conv2d(self.kernel, self.kernel, padding=self.kernel.shape[-1] - 1)
        T = 1 / torch.sqrt(torch.abs(kkt).sum((1, 2, 3)))
        res = T[None, :, None, None] * res + self.bias[:, None, None]
        return res

class AolConv(nn.Module):
    def __init__(self, cin, cout, kernel_size=3, epsilon=1e-6, scale=1.0, padding = 0):
        super().__init__()

        self.scale = scale
        self.kernel = nn.Parameter(torch.empty(cout, cin, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(cout))
        #self.padding = (kernel_size - 1) // 2
        self.padding = padding
        nn.init.xavier_normal_(self.kernel)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.kernel)
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound) 

        self.epsilon = epsilon

    def forward(self, x):
        res = F.conv2d(self.scale * x, self.kernel, padding=self.padding)
        kkt = F.conv2d(self.kernel, self.kernel, padding=self.kernel.shape[-1] - 1)
        T = 1 / torch.sqrt(torch.abs(kkt).sum((1, 2, 3)))
        res = T[None, :, None, None] * res + self.bias[:, None, None]
        return F.relu(res)    
