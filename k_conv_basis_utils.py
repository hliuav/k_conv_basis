import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from pypapi import events
from pypapi import papi_high as high
from torch import nn

def get_b_tilde_from_b(b):
    b_tilde = np.ones_like(b, dtype=np.float64) # exp(0)
    k, n = b.shape
    sum_b_r = np.zeros(n, dtype=np.float64)
    sum_b_r_minus_1 = np.zeros(n, dtype=np.float64)
    for i in range(k):
        if i == 0:
            sum_b_r += b[i]
            b_tilde[i, :] = np.exp(sum_b_r)
        else:
            sum_b_r += b[i]
            sum_b_r_minus_1 += b[i - 1]
            b_tilde[i, :] = np.exp(sum_b_r) - np.exp(sum_b_r_minus_1)
        # stable exp
        #sum_b_r_max = np.max(sum_b_r, axis=-1, keepdims=True)
        #sum_b_r_minus_1_max = np.max(sum_b_r_minus_1, axis=-1, keepdims=True)
        #b_tilde[i, :] = np.exp(sum_b_r - sum_b_r_max) - np.exp(sum_b_r_minus_1 - sum_b_r_minus_1_max)
        #b_tilde[i, :] = np.exp(sum_b_r) - np.exp(sum_b_r_minus_1)
    return b_tilde


def binary_search(Q, K, k, T, delta, epsilon, v, s, t):
    n, _ = Q.shape
    if s >= t:
        # change 1
        return t
    j = (s + t) // 2

    H_j = Q @ (K.T)[:,j]

    # Norm calculation
    alpha = np.linalg.norm(H_j[j : j + T] - v, ord=1)
    if alpha >= delta - 2 * T * epsilon:
        return binary_search(Q, K, k, T, delta, epsilon, v, s, j)
    else:
        return binary_search(Q, K, k, T, delta, epsilon, v, j + 1, t)


def sub_conv_matrix(a, m):
    n = a.shape[0]
    # Create the convolution matrix for the first m elements
    result_matrix = np.zeros((n, n))
    for i in range(n - m, n):
        result_matrix[i:, i] = a[: n - i]

    return result_matrix


def conv_with_fft(a, x, shift=0):
    n = a.shape[0] 
    n = n - shift
    a_padded = np.zeros(2 * n, dtype=np.float64)
    x_padded = np.zeros(2 * n, dtype=np.float64)
    a_padded[:n] = a[:n]
    x_padded[:n] = x[-n:]

    result = np.zeros_like(a, dtype=np.float64)
    result[-n:] = np.fft.ifft(np.fft.fft(a_padded) * np.fft.fft(x_padded)).real[:n]
    return result


def conv_with_fft_matrix(a, X, shift=0):
    n, d = X.shape
    result_matrix = np.zeros_like(X, dtype=np.float64)
    for i in range(d):
        result_matrix[:, i] = conv_with_fft(a, X[:, i], shift=shift)
    return result_matrix


# PyTorch implementation of the above functions
class NaiveConv(nn.Module):
    def forward(self, a, x):
        n = a.shape[0]
        sub_conv_matrix = torch.zeros((n, n))
        for i in range(n):
            sub_conv_matrix[i:, i] = a[: n - i]
        return sub_conv_matrix @ x


class ConvWithFFT(nn.Module):
    def forward(self, a, x):
        print(a.shape)
        n = a.shape[0]
        a_padded = torch.zeros(2 * n, dtype=torch.float32)
        x_padded = torch.zeros(2 * n, dtype=torch.float32)
        a_padded[:n] = a
        x_padded[:n] = x

        return torch.fft.ifft(
            torch.diag(torch.fft.fft(a_padded)) @ torch.fft.fft(x_padded)
        ).real[:n]


def main():
    # FLOPs and time in numpy
    # Test case for sub_conv_matrix and conv_with_fft
    a = np.array([1, 3, 5, 7, 9])
    x = np.array([1, 2, 3, 4, 5])
    #conv_with_fft = ConvWithFFT()
    print(sub_conv_matrix(a, len(a)) @ x)
    print(conv_with_fft(a, x))
    
    ns = [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 32000]
    flops_naive_list = []
    flops_fft_list = []
    time_naive_list = []
    time_fft_list = []
    for n in ns:
        a = np.random.randn(n).astype(np.double)
        x = np.random.randn(n).astype(np.double)
        # Compute time and FLOPs (double precision) for naive method
        high.start_counters(
            [
                events.PAPI_DP_OPS,
            ]
        )
    
        start_time = time.time()
        sub_conv_matrix(a, n) @ x
        end_time = time.time()
    
        flops_naive = high.stop_counters()
        time_naive = end_time - start_time
    
        high.start_counters(
            [
                events.PAPI_DP_OPS,
            ]
        )
        start_time = time.time()
        conv_with_fft(a, x)
        end_time = time.time()
        flops_fft = high.stop_counters()
        time_fft = end_time - start_time
        flops_fft = flops_fft[0] / n
        flops_naive = flops_naive[0] / n
        time_fft = time_fft / n
        time_naive = time_naive / n
        # print (n, flops_naive, flops_fft)
        print("n: ", n)
        print("Naive FLOPs: ", flops_naive, "FFT FLOPs: ", flops_fft)
        print("Naive time: ", time_naive, "FFT time: ", time_fft)
        flops_fft_list.append(flops_fft)
        flops_naive_list.append(flops_naive)
        time_fft_list.append(time_fft)
        time_naive_list.append(time_naive)
    
    s = 1
    flops_naive_list = flops_naive_list[s:]
    flops_fft_list = flops_fft_list[s:]
    time_fft_list = time_fft_list[s:]
    time_naive_list = time_naive_list[s:]
    ns = ns[s:]
    # Plot naive flops and fft flops versus n
    plt.figure(figsize=(10, 6))
    plt.plot(ns, flops_naive_list, label="Naive FLOPs", marker="o")
    plt.plot(ns, flops_fft_list, label="FFT FLOPs", marker="x")
    plt.xscale("log")
    plt.xlabel("Vector length n")
    plt.ylabel("average FLOPs / token numbers")
    plt.title("Comparison of FLOPs for Convolution Implementations")
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig("conv_flops.png")
    
    # Plot naive time and fft time versus n
    plt.figure(figsize=(10, 6))
    plt.plot(ns, time_naive_list, label="Naive Time", marker="o")
    plt.plot(ns, time_fft_list, label="FFT Time", marker="x")
    plt.xscale("log")
    plt.xlabel("Vector length n")
    plt.ylabel("average Time (s) / token numbers")
    plt.title("Comparison of Time for Convolution Implementations")
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig("conv_time.png")

if __name__ == "__main__":
    main()
