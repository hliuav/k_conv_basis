import time

import numpy as np
import torch
from pypapi import events
from pypapi import papi_high as high

from k_conv_basis_utils import binary_search, get_b_tilde_from_b, sub_conv_matrix, conv_with_fft_matrix, conv_with_fft
from test import test_recovered_b, test_recovered_b_tilde

def recover_k_conv(Q, K, k, T, delta, epsilon):
    n, d = Q.shape
    v = np.zeros(T, dtype=np.float64)  # Initial vector v
    u = np.zeros(n, dtype=np.float64)  # Initial vector u
    s = -1  # Initial index s
    t = n - T
    

    m = np.zeros(k, dtype=int)
    b = np.zeros((k, n), dtype=np.float64)
    for i in range(k):
        s += 1
        s = binary_search(Q, K, k, T, delta, epsilon, v, s, t)
        m[i] = n - s
        if m[i] <= 0:
            break
        H_s = Q @ (K.T)[:,s]
        b[i, :m[i]] = H_s[s:s + m[i]] - u[:m[i]]
        v += b[i, :T]
        u += b[i, :]
    b_tilde = get_b_tilde_from_b(b)
    # Additional outputs based on Lemma A.2 not provided, assuming direct return
    return b_tilde, m

def naive_exact_attention_score(Q, K, V):
    QK = Q @ K.T

    mask = np.tril(np.ones_like(QK)).astype(np.float64)
    QK_exp = np.exp(QK)
    QK_exp_mask = mask * QK_exp
    D_stable_inv = np.diagflat(1 / np.sum(QK_exp_mask, axis=1))  # Derived similarly as in original
    return D_stable_inv @ QK_exp_mask @ V

def k_conv_basis_attention_score(Q, K, V, k, T, delta, epsilon):
    n = Q.shape[0]
    b_tilde, m = recover_k_conv(Q, K, k=k, T=T, delta=delta, epsilon=epsilon)

    QKV_approx= np.zeros_like(Q, dtype=np.float64)
    for i in range(k):
        print(b_tilde[i, :])
        QKV_approx += conv_with_fft_matrix(b_tilde[i, :], V, shift=n - m[i])

    #D_approx_inv = np.diagflat(1 / np.sum(sum_b, axis=1))  # Derived similarly as in original
    D_approx = np.zeros(n, dtype=np.float64)
    for i in range(k):
        D_approx += conv_with_fft(b_tilde[i, :], np.ones(n), shift=n - m[i])

    QKV_approx = np.expand_dims(D_approx ** -1, axis=1) * QKV_approx
    return QKV_approx


def main():
    q_value_path = 'q_value.pth'
    k_value_path = 'k_value.pth'
    v_value_path = 'v_value.pth'
    q_value = torch.load(q_value_path)
    k_value = torch.load(k_value_path)
    v_value = torch.load(v_value_path)
    q_value = q_value.cpu().numpy()
    k_value = k_value.cpu().numpy()
    v_value = v_value.cpu().numpy()

    Q = q_value[0, 20, :, :]
    K = k_value[0, 20, :, :]
    V = v_value[0, 20, :, :]

    Q = Q.astype(np.float64)
    K = K.astype(np.float64)
    V = V.astype(np.float64)

    # k conv basis parameters
    k = 5
    T = 5
    delta = 20
    epsilon = 1

    high.start_counters(
        [
            events.PAPI_DP_OPS,
        ]
    )
    
    start_time = time.time()
    QKV = naive_exact_attention_score(Q, K, V)
    end_time = time.time()
    flops_naive = high.stop_counters()
    time_naive = end_time - start_time
    
    high.start_counters(
        [
            events.PAPI_DP_OPS,
        ]
    )
    start_time = time.time()
    QKV_approx = k_conv_basis_attention_score(Q, K, V, k=k, T=T, delta=delta, epsilon=epsilon)
    end_time = time.time()
    time_approx = end_time - start_time
    flops_approx = high.stop_counters()

    print("diff:", np.linalg.norm(QKV - QKV_approx, ord='fro'))
    print("time_naive:", time_naive)
    print("time_approx:", time_approx)
    print("flops_naive:", flops_naive)
    print("flops_approx:", flops_approx)



if __name__ == '__main__':
    main()


