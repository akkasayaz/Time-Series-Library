# coding=utf-8
# author=maziqing
# email=maziqing.mzq@alibaba-inc.com

import numpy as np
import torch
import torch.nn as nn


def get_frequency_modes(seq_len, modes=64, mode_select_method='random'):
    """
    get modes on frequency domain:
    'random' means sampling randomly;
    'else' means sampling the lowest modes;
    """
    modes = min(modes, seq_len // 2)
    if mode_select_method == 'random':
        index = list(range(0, seq_len // 2))
        np.random.shuffle(index)
        index = index[:modes]
    else:
        index = list(range(0, modes))
    index.sort()
    return index
from PyEMD import EMD

# ########## empirical mode decomposition layer #############
class EMDBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_heads, seq_len, modes=1, mode_select_method='random'):
        super(EMDBlock, self).__init__()
        self.n_heads = n_heads
        self.modes = modes if modes > 0 else seq_len // 2
        self.head_dim = out_channels // n_heads
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.randn(n_heads, self.head_dim, self.head_dim))
        nn.init.xavier_uniform_(self.weight)

    def emd_decomposition(self, X):
        B, L, E = X.shape
        import time

        t1 = time.time()
        print(self.modes)
        emd = EMD(max_imfs=self.modes)
        reconstructed = []
        for i in range(B):
            for j in range(L):
                imfs = emd(X[i, j].detach().cpu().numpy())
                reconstructed.append(torch.tensor(imfs[-1], dtype=torch.float32))
                if i == 0 and j == 0:
                    print("one decomposition time:", time.time() - t1)
        reconstructed = torch.stack(reconstructed)
        
        # move reconstructed to the device of X
        reconstructed = reconstructed.to(X.device)
        print('finish emd decomposition')
        print('emd decomposition time:', time.time() - t1)
        return reconstructed

    def forward(self, q, k, v, mask=None):
        B, L, H, E = q.shape
        x = q.reshape(B, L, H * E)
        
        reconstructed = self.emd_decomposition(x)
        reconstructed = reconstructed.reshape(B, L, H, E)
        
        return reconstructed.contiguous(), None



#  Emperical Mode Decomposition Cross Attention
class EMDCrossAttention(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len_q, seq_len_kv, modes=64, mode_select_method='random', num_heads=8):
        super(EMDCrossAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = out_channels // num_heads
        self.out_channels = out_channels
        self.q_emd = EMDBlock(in_channels, out_channels, num_heads, seq_len_q, modes, mode_select_method)
        self.k_emd = EMDBlock(in_channels, out_channels, num_heads, seq_len_kv, modes, mode_select_method)
        self.v_emd = EMDBlock(in_channels, out_channels, num_heads, seq_len_kv, modes, mode_select_method)
        self.out_proj = nn.Linear(out_channels, out_channels)
        self.scale = self.head_dim ** -0.5

    def forward(self, q, k, v, mask=None):
        # Get EMD representations
        q_emd, _ = self.q_emd(q, None, None)
        k_emd, _ = self.k_emd(k, None, None)
        v_emd, _ = self.v_emd(v, None, None)
        
        # Ensure dimensions match for attention
        B, L_q, H, E = q_emd.shape
        _, L_k, _, _ = k_emd.shape
        
        # Reshape for attention computation
        q_emd = q_emd.reshape(B, L_q, H * E)
        k_emd = k_emd.reshape(B, L_k, H * E)
        v_emd = v_emd.reshape(B, L_k, H * E)

        # Compute attention scores
        attn = torch.matmul(q_emd, k_emd.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(attn, dim=-1)

        # Apply attention to values
        output = torch.matmul(attn, v_emd)
        output = self.out_proj(output)
        
        return output, attn

# ########## dynamic mode decomposition layer #############
import torch
import torch.nn as nn
from pydmd import DMD

import torch
import numpy as np
from pydmd import DMD


class DMDBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_heads, seq_len, modes=0, mode_select_method='random'):
        super(DMDBlock, self).__init__()
        self.n_heads = n_heads
        self.modes = modes if modes > 0 else seq_len // 2
        self.head_dim = out_channels // n_heads
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.randn(n_heads, self.head_dim, self.head_dim))
        nn.init.xavier_uniform_(self.weight)

    def dmd_decomposition(self, X):
        X_numpy = X.detach().cpu().numpy()
        B, L, E = X_numpy.shape
        
        dmd = DMD(svd_rank=self.modes)
        dmd.fit(X_numpy.T)
        reconstruction = dmd.reconstructed_data.T
        reconstructed = torch.tensor(reconstruction, 
                                   device=X.device, 
                                   dtype=X.dtype)
        # all_reconstructions = []
        # for i in range(B):
        #     dmd = DMD(svd_rank=self.modes)
        #     dmd.fit(X_numpy[i].T)
        #     reconstruction = dmd.reconstructed_data
        #     all_reconstructions.append(reconstruction.T)
            
        # reconstructed = torch.tensor(np.stack(all_reconstructions), 
        #                            device=X.device, 
        #                            dtype=X.dtype)
        return reconstructed

    def forward(self, q, k, v, mask=None):
        B, L, H, E = q.shape
        x = q.reshape(B, L, H * E)
        
        reconstructed = self.dmd_decomposition(x)
        reconstructed = reconstructed.reshape(B, L, H, E)
        
        return reconstructed.contiguous(), None
    

    
class DMDCrossAttention(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len_q, seq_len_kv, modes=64, mode_select_method='random', num_heads=8):
        super(DMDCrossAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = out_channels // num_heads
        self.out_channels = out_channels
        self.q_dmd = DMDBlock(in_channels, out_channels, num_heads, seq_len_q, modes, mode_select_method)
        self.k_dmd = DMDBlock(in_channels, out_channels, num_heads, seq_len_kv, modes, mode_select_method)
        self.v_dmd = DMDBlock(in_channels, out_channels, num_heads, seq_len_kv, modes, mode_select_method)
        self.out_proj = nn.Linear(out_channels, out_channels)
        self.scale = self.head_dim ** -0.5

    def forward(self, q, k, v, mask=None):
        # Get DMD representations
        q_dmd, _ = self.q_dmd(q, None, None)
        k_dmd, _ = self.k_dmd(k, None, None)
        v_dmd, _ = self.v_dmd(v, None, None)
        
        # Ensure dimensions match for attention
        B, L_q, H, E = q_dmd.shape
        _, L_k, _, _ = k_dmd.shape
        
        # Reshape for attention computation
        q_dmd = q_dmd.reshape(B, L_q, H * E)
        k_dmd = k_dmd.reshape(B, L_k, H * E)
        v_dmd = v_dmd.reshape(B, L_k, H * E)

        # Compute attention scores
        attn = torch.matmul(q_dmd, k_dmd.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(attn, dim=-1)

        # Apply attention to values
        output = torch.matmul(attn, v_dmd)
        output = self.out_proj(output)
        
        return output, attn
    
# ########## fourier layer #############
class FourierBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_heads, seq_len, modes=0, mode_select_method='random'):
        super(FourierBlock, self).__init__()
        print('fourier enhanced block used!')
        """
        1D Fourier block. It performs representation learning on frequency domain, 
        it does FFT, linear transform, and Inverse FFT.    
        """
        # get modes on frequency domain
        self.index = get_frequency_modes(seq_len, modes=modes, mode_select_method=mode_select_method)
        print('modes={}, index={}'.format(modes, self.index))

        self.n_heads = n_heads
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(self.n_heads, in_channels // self.n_heads, out_channels // self.n_heads,
                                    len(self.index), dtype=torch.float))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(self.n_heads, in_channels // self.n_heads, out_channels // self.n_heads,
                                    len(self.index), dtype=torch.float))

    # Complex multiplication
    def compl_mul1d(self, order, x, weights):
        x_flag = True
        w_flag = True
        if not torch.is_complex(x):
            x_flag = False
            x = torch.complex(x, torch.zeros_like(x).to(x.device))
        if not torch.is_complex(weights):
            w_flag = False
            weights = torch.complex(weights, torch.zeros_like(weights).to(weights.device))
        if x_flag or w_flag:
            return torch.complex(torch.einsum(order, x.real, weights.real) - torch.einsum(order, x.imag, weights.imag),
                                 torch.einsum(order, x.real, weights.imag) + torch.einsum(order, x.imag, weights.real))
        else:
            return torch.einsum(order, x.real, weights.real)

    def forward(self, q, k, v, mask):
        # size = [B, L, H, E]
        B, L, H, E = q.shape
        x = q.permute(0, 2, 3, 1)
        # Compute Fourier coefficients
        x_ft = torch.fft.rfft(x, dim=-1)
        # Perform Fourier neural operations
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=x.device, dtype=torch.cfloat)
        for wi, i in enumerate(self.index):
            if i >= x_ft.shape[3] or wi >= out_ft.shape[3]:
                continue
            out_ft[:, :, :, wi] = self.compl_mul1d("bhi,hio->bho", x_ft[:, :, :, i],
                                                   torch.complex(self.weights1, self.weights2)[:, :, :, wi])
        # Return to time domain
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return (x, None)

# ########## Fourier Cross Former ####################
class FourierCrossAttention(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len_q, seq_len_kv, modes=64, mode_select_method='random',
                 activation='tanh', policy=0, num_heads=8):
        super(FourierCrossAttention, self).__init__()
        print(' fourier enhanced cross attention used!')
        """
        1D Fourier Cross Attention layer. It does FFT, linear transform, attention mechanism and Inverse FFT.    
        """
        self.activation = activation
        self.in_channels = in_channels
        self.out_channels = out_channels
        # get modes for queries and keys (& values) on frequency domain
        self.index_q = get_frequency_modes(seq_len_q, modes=modes, mode_select_method=mode_select_method)
        self.index_kv = get_frequency_modes(seq_len_kv, modes=modes, mode_select_method=mode_select_method)

        print('modes_q={}, index_q={}'.format(len(self.index_q), self.index_q))
        print('modes_kv={}, index_kv={}'.format(len(self.index_kv), self.index_kv))

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(num_heads, in_channels // num_heads, out_channels // num_heads, len(self.index_q), dtype=torch.float))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(num_heads, in_channels // num_heads, out_channels // num_heads, len(self.index_q), dtype=torch.float))

    # Complex multiplication
    def compl_mul1d(self, order, x, weights):
        x_flag = True
        w_flag = True
        if not torch.is_complex(x):
            x_flag = False
            x = torch.complex(x, torch.zeros_like(x).to(x.device))
        if not torch.is_complex(weights):
            w_flag = False
            weights = torch.complex(weights, torch.zeros_like(weights).to(weights.device))
        if x_flag or w_flag:
            return torch.complex(torch.einsum(order, x.real, weights.real) - torch.einsum(order, x.imag, weights.imag),
                                 torch.einsum(order, x.real, weights.imag) + torch.einsum(order, x.imag, weights.real))
        else:
            return torch.einsum(order, x.real, weights.real)

    def forward(self, q, k, v, mask):
        # size = [B, L, H, E]
        B, L, H, E = q.shape
        xq = q.permute(0, 2, 3, 1)  # size = [B, H, E, L]
        xk = k.permute(0, 2, 3, 1)
        xv = v.permute(0, 2, 3, 1)

        # Compute Fourier coefficients
        xq_ft_ = torch.zeros(B, H, E, len(self.index_q), device=xq.device, dtype=torch.cfloat)
        xq_ft = torch.fft.rfft(xq, dim=-1)
        for i, j in enumerate(self.index_q):
            if j >= xq_ft.shape[3]:
                continue
            xq_ft_[:, :, :, i] = xq_ft[:, :, :, j]
        xk_ft_ = torch.zeros(B, H, E, len(self.index_kv), device=xq.device, dtype=torch.cfloat)
        xk_ft = torch.fft.rfft(xk, dim=-1)
        for i, j in enumerate(self.index_kv):
            if j >= xk_ft.shape[3]:
                continue
            xk_ft_[:, :, :, i] = xk_ft[:, :, :, j]

        # perform attention mechanism on frequency domain
        xqk_ft = (self.compl_mul1d("bhex,bhey->bhxy", xq_ft_, xk_ft_))
        if self.activation == 'tanh':
            xqk_ft = torch.complex(xqk_ft.real.tanh(), xqk_ft.imag.tanh())
        elif self.activation == 'softmax':
            xqk_ft = torch.softmax(abs(xqk_ft), dim=-1)
            xqk_ft = torch.complex(xqk_ft, torch.zeros_like(xqk_ft))
        else:
            raise Exception('{} actiation function is not implemented'.format(self.activation))
        xqkv_ft = self.compl_mul1d("bhxy,bhey->bhex", xqk_ft, xk_ft_)
        xqkvw = self.compl_mul1d("bhex,heox->bhox", xqkv_ft, torch.complex(self.weights1, self.weights2))
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=xq.device, dtype=torch.cfloat)
        for i, j in enumerate(self.index_q):
            if i >= xqkvw.shape[3] or j >= out_ft.shape[3]:
                continue
            out_ft[:, :, :, j] = xqkvw[:, :, :, i]
        # Return to time domain
        out = torch.fft.irfft(out_ft / self.in_channels / self.out_channels, n=xq.size(-1))
        return (out, None)
