import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from pypapi import events
from pypapi import papi_high as high
from torch import nn


def get_b_tilde_from_b(b):
    b_tilde = np.zeros_like(b)
    k, n = b.shape
    sum_b_r = np.zeros(n)
    sum_b_r_minus_1 = np.zeros(n)
    for i in range(k):
        if i == 0:
            sum_b_r += b[i]
        else:
            sum_b_r += b[i]
            sum_b_r_minus_1 += b[i - 1]
        b_tilde[i, :] = np.exp(sum_b_r) - np.exp(sum_b_r_minus_1)
    return b_tilde


def binary_search(Q, K, k, T, delta, epsilon, v, s, t):
    n, _ = Q.shape
    if s >= t:
        return s
    j = (s + t) // 2

    H = np.tril(np.ones((n, n))) * (Q @ K.T)

    # Norm calculation
    alpha = np.linalg.norm(H[j, j : j + T - 1] - v, ord=1)

    if alpha >= delta - 2 * T * epsilon:
        return binary_search(Q, K, k, T, delta, epsilon, v, s, j)
    else:
        return binary_search(Q, K, k, T, delta, epsilon, v, j + 1, t)


def sub_conv_matrix(a, m):
    n = a.shape[0]
    # Create the convolution matrix for the first m elements
    sub_conv_matrix = np.zeros((n, n))
    for i in range(n - m, n):
        sub_conv_matrix[i:, i] = a[: n - i]

    # Without for-loop implementation
    # n = a.shape[0]
    ## Create an empty matrix of zeros
    # sub_conv_matrix = np.zeros((n, n))
    #
    ## Generate indices for the lower triangle including the diagonal shifted by n-m rows
    # rows, cols = np.tril_indices(n, -n + m)
    #
    ## Calculate corresponding indices in the vector 'a'
    # indices = cols - rows
    #
    ## Assign the values from 'a' using calculated indices
    # sub_conv_matrix[rows, cols] = a[indices]
    return sub_conv_matrix


def conv_with_fft(a, x):
    n = a.shape[0]
    a_padded = np.zeros(2 * n)
    x_padded = np.zeros(2 * n)
    a_padded[:n] = a
    x_padded[:n] = x

    return np.fft.ifft(np.fft.fft(a_padded) * np.fft.fft(x_padded)).real[:n]


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


# FLOPs and time in numpy
# Test case for sub_conv_matrix and conv_with_fft
a = np.array([1, 3, 5, 7, 9])
x = np.array([1, 2, 3, 4, 5])
#a = torch.tensor(a, dtype=torch.float32)
#x = torch.tensor(x, dtype=torch.float32)
#naive_conv = NaiveConv()
#conv_with_fft = ConvWithFFT()
print(sub_conv_matrix(a, len(a)) @ x)
print(conv_with_fft(a, x))

ns = [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 32000]
flops_naive_list = []
flops_fft_list = []
time_naive_list = []
time_fft_list = []
for n in ns:
    a = np.random.randn(n)
    x = np.random.randn(n)
    # Compute time and FLOPs (double precision) for naive method
    start_time = time.time()
    high.start_counters(
        [
            events.PAPI_DP_OPS,
        ]
    )

    sub_conv_matrix(a, n) @ x

    flops_naive = high.stop_counters()
    end_time = time.time()
    time_naive = end_time - start_time

    start_time = time.time()
    high.start_counters(
        [
            events.PAPI_DP_OPS,
        ]
    )
    conv_with_fft(a, x)
    end_time = time.time()
    flops_fft = high.stop_counters()
    time_fft = end_time - start_time
    flops_fft = flops_fft[0]
    flops_naive = flops_naive[0]
    # print (n, flops_naive, flops_fft)
    print("n: ", n)
    print("Naive FLOPs: ", flops_naive, "FFT FLOPs: ", flops_fft)
    print("Naive time: ", time_naive, "FFT time: ", time_fft)
    flops_fft_list.append(flops_fft)
    flops_naive_list.append(flops_naive)
    time_fft_list.append(time_fft)
    time_naive_list.append(time_naive)

# Plot naive flops and fft flops versus n
plt.figure(figsize=(10, 5))
plt.plot(ns, flops_naive_list, label="Naive FLOPs", marker="o")
plt.plot(ns, flops_fft_list, label="FFT FLOPs", marker="x")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Vector length n")
plt.ylabel("FLOPs")
plt.title("Comparison of FLOPs for Convolution Implementations")
plt.legend()
plt.grid(True)
plt.show()
plt.savefig("conv_flops.png")

# Plot naive time and fft time versus n
plt.figure(figsize=(10, 5))
plt.plot(ns, time_naive_list, label="Naive Time", marker="o")
plt.plot(ns, time_fft_list, label="FFT Time", marker="x")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Vector length n")
plt.ylabel("Time (s)")
plt.title("Comparison of Time for Convolution Implementations")
plt.legend()
plt.grid(True)
plt.show()
plt.savefig("conv_time.png")

#
# a = torch.randn(100)
# x = torch.randn(100)
# macs, params = profile(naive_conv, inputs=(a,x,))
# print(macs, params)
# macs, params = profile(conv_with_fft, inputs=(a,x, ))
# print(macs, params)
# macs, params = get_model_complexity_info(conv_with_fft, ((10),(10)), as_strings=True,
#                                           print_per_layer_stat=True, verbose=True)
# print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
