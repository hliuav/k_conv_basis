import time
import os
import sys

import numpy as np
import torch
from pypapi import events
from pypapi import papi_high as high

from k_conv_basis_utils import recover_k_conv, conv_with_fft_matrix, conv_with_fft
from test import test_recovered_b, test_recovered_b_tilde

import matplotlib.pyplot as plt

import gc
from memory_profiler import profile
gc.collect()

head_idx = 2

def naive_exact_attention_score(Q, K, V):
    QK = Q @ K.T

    mask = np.tril(np.ones_like(QK)).astype(np.float64)
    QK_exp = np.exp(QK)
    QK_exp_mask = mask * QK_exp
    D_stable_inv = np.diagflat(1 / np.sum(QK_exp_mask, axis=1))  # Derived similarly as in original
    return D_stable_inv @ QK_exp_mask @ V
    #return QK_exp_mask @ V

def k_conv_basis_attention_score(Q, K, V, k, T, delta, epsilon):
    n = Q.shape[0]
    b_tilde, m, b= recover_k_conv(Q, K, k=k, T=T, delta=delta, epsilon=epsilon)
    QKV_approx= np.zeros_like(Q, dtype=np.float64)
    for i in range(k):
        QKV_approx += conv_with_fft_matrix(b_tilde[i, :], V, shift=n - m[i])

    D_approx = np.zeros(n, dtype=np.float64)
    for i in range(k):
        D_approx += conv_with_fft(b_tilde[i, :], np.ones(n), shift=n - m[i])

    #return QKV_approx
    QKV_approx = np.expand_dims(D_approx ** -1, axis=1) * QKV_approx

    # test
    #mask = np.tril(np.ones_like(Q@K.T)).astype(np.float64)
    #QK_mask = mask * (Q@K.T)
    #QK_exp_mask = mask * np.exp(Q@K.T)
    #test_recovered_b(QK_mask, b, m)
    #test_recovered_b_tilde(QK_exp_mask, b_tilde, m)


    return QKV_approx
#data_type = torch.float64

#def naive_exact_attention_score(Q, K, V):
#    print("Compute naive exact attention score")
#    device = torch.device(Q.device)
#    
#    Q = Q.to(device)
#    K = K.to(device)
#    V = V.to(device)
#    
#    QK = Q @ K.T
#    mask = torch.tril(torch.ones_like(QK)).to(data_type).to(device)
#    QK_exp = torch.exp(QK)
#    QK_exp_mask = mask * QK_exp
#    D_stable_inv = torch.diag(1 / torch.sum(QK_exp_mask, dim=1)).to(device)  # Derived similarly as in original
#    n = QK_exp_mask.shape[0]
#    # caculcate the matrix by 
#    #for i in range(n):
#    #    QK_exp_mask[:, i] = D_stable_inv @ QK_exp_mask[:, i]
#    QKV = D_stable_inv @ QK_exp_mask @ V
#    # save QKV
#    torch.save(QKV, 'QKV.pth')
#    return QKV
#    #temp = D_stable_inv @ QK_exp_mask
#    #print(D_stable_inv.shape)
#    #return temp @ V
#
#@profile
#def k_conv_basis_attention_score(Q, K, V, k, T, delta, epsilon):
#    device = torch.device(Q.device)
#    print("compute k conv basis attention score")
#    
#    Q = Q.to(device)
#    K = K.to(device)
#    V = V.to(device)
#    
#    n, d= Q.shape
#    b_tilde, m, b = recover_k_conv(Q, K, k=k, T=T, delta=delta, epsilon=epsilon)
#    print(m)
#    
#    QKV_approx = torch.zeros_like(Q, dtype=data_type, device=device)
#    FFTConv = FFTConvolver(n, d, device)
#    for i in range(k):
#        if i % 10 == 0:
#            print(i)
#        QKV_approx += FFTConv.conv_with_fft_matrix(b_tilde[i, :], V, shift=n - m[i])
#        gc.collect()
#        #QKV_approx += conv_with_fft_matrix(b_tilde[i, :], V, shift=n - m[i])
#
#    D_approx = torch.zeros(n, dtype=data_type, device=device)
#    for i in range(k):
#        if i % 100 == 0:
#            print(i)
#        temp = FFTConv.conv_with_fft(b_tilde[i, :], torch.ones(n, device=device), shift=n - m[i])
#        D_approx += temp
#        del temp
#        #D_approx += conv_with_fft(b_tilde[i, :], torch.ones(n, device=device), shift=n - m[i])
#
#    QKV_approx = (D_approx ** -1).unsqueeze(1) * QKV_approx
#
#    return QKV_approx


def main():
    #token_size = 8065
    #token_size = 16130
    #token_size = 25600
    #token_size = 100
    token_size = 32000
    q_value_path = f'q_value_{token_size}_{head_idx}.pth'
    k_value_path = f'k_value_{token_size}_{head_idx}.pth'
    v_value_path = f'v_value_{token_size}_{head_idx}.pth'
    #q_value_path = f'q_value.pth'
    #k_value_path = f'k_value.pth'
    #v_value_path = f'v_value.pth'
    q_value = torch.load(q_value_path)
    k_value = torch.load(k_value_path)
    v_value = torch.load(v_value_path)

    _, _, n, d = q_value.shape

    # visualize Q K
    #A_all = torch.matmul(q_value, k_value.permute(0, 1, 3, 2))
    #mask_all = torch.tril(torch.ones(n, n), diagonal=0).bool().to('cuda')
    ## Normalize the attention matrix along the last two dimensions
    #mean_A1 = A_all.mean(dim=(-2, -1), keepdim=True)  # Correct usage
    #max_abs_A1 = torch.max(torch.abs(A_all), dim=-1, keepdim=True)[0]
    #max_abs_A1 = torch.max(max_abs_A1, dim=-2, keepdim=True)[0]
    #A_all = (A_all - mean_A1) / max_abs_A1
    #A_all = mask_all * A_all
    #A_all = A_all.cpu().detach().numpy()

    #plt.rcParams.update({'font.size': 55, 'legend.fontsize': 50})
    #fig, ax = plt.subplots(figsize=(14, 14))

    #cax = ax.imshow(A_all[0, 0, :, :], cmap="viridis")
    #ax.set_title(f"Heatmap of Layer 0 Head 0", fontweight='bold')
    #ax.set_xlabel("Target Position", fontweight='bold')
    #ax.set_ylabel("Source Position", fontweight='bold')
    ## Add a color bar to each subplot
    #plt.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

    #plt.tight_layout()
    #output_folder = "attention_diagonal"
    #plt.savefig(f"{output_folder}/attention_{n}.pdf")
    



    print(q_value.shape)
    q_value = q_value.detach().cpu().numpy().astype(np.float64)
    k_value = k_value.detach().cpu().numpy().astype(np.float64)
    v_value = v_value.detach().cpu().numpy().astype(np.float64)
    #q_value = q_value.to('cpu')
    #k_value = k_value.to('cpu')
    #v_value = v_value.to('cpu')
    #q_value = q_value.to('cuda')
    #k_value = k_value.to('cuda')
    #v_value = v_value.to('cuda')

    Q = q_value[0, 0, :, :]
    K = k_value[0, 0, :, :]
    V = v_value[0, 0, :, :]

    #n,d = Q.shape
    #print(d)

    Q = Q / np.sqrt(d)
    K = K / np.sqrt(d)
    V = V

    # k conv basis parameters
    #k = int(n ** 0.1)
    k = 32000
    T = 1
    delta = 1e-9#n ** 0.5
    epsilon = 0

    if len(sys.argv) > 1:
        k = int(sys.argv[1])


    high.start_counters(
        [
            events.PAPI_DP_OPS,
        ]
    )
    
    start_time = time.time()
    if os.path.exists(f'QKV_{head_idx}.pth'):
        QKV = torch.load(f'QKV_{head_idx}.pth')
    else:
        QKV = naive_exact_attention_score(Q, K, V)
    end_time = time.time()
    time_naive = end_time - start_time
    flops_naive = high.stop_counters()
    
    high.start_counters(
        [
            events.PAPI_DP_OPS,
        ]
    )
    start_time = time.time()
    #QKV_approx = k_conv_basis_attention_score(Q.detach().numpy(), K.detach().numpy(), V.detach().numpy(), k=k, T=T, delta=delta, epsilon=epsilon)
    QKV_approx = k_conv_basis_attention_score(Q, K, V, k=k, T=T, delta=delta, epsilon=epsilon)
    end_time = time.time()
    time_approx = end_time - start_time
    flops_approx = high.stop_counters()

    #QKV = QKV.to('cuda')
    #relative_diff  = torch.norm(QKV - QKV_approx, p='fro') / torch.norm(QKV, p='fro')
    #QKV = QKV.detach().cpu().numpy()
    relative_diff = np.linalg.norm(QKV - QKV_approx, ord='fro') / np.linalg.norm(QKV, ord='fro')
    print("relative_diff:", relative_diff)
    print("time_naive:", time_naive)
    print("time_approx:", time_approx)
    print("flops_naive:", flops_naive)
    print("flops_approx:", flops_approx)

    # save flops_approx and time_approx with name k
    torch.save(QKV, f'QKV_{head_idx}.pth')
    torch.save(relative_diff, f'relative_diff_{k}_{head_idx}.pth')
    torch.save(QKV_approx, f'QKV_approx_{k}_{head_idx}.pth')
    torch.save(time_approx, f'time_approx_{k}_{head_idx}.pth')
    torch.save(flops_approx, f'flops_approx_{k}_{head_idx}.pth')



if __name__ == '__main__':
    main()


