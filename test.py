import os
import time

import matplotlib.pyplot as plt
import numpy as np
from pypapi import events
from pypapi import papi_high as high

from k_conv_basis_utils import sub_conv_matrix, conv_with_fft


def test_recovered_b(QK_mask, b, m):
    # Check if the recovered b is correct
    k = b.shape[0]
    QK_approx = np.zeros_like(QK_mask, dtype=np.float64)
    for i in range(k):
        QK_approx += sub_conv_matrix(b[i, :], m[i])
    print("QK_mask:", QK_mask)
    print("QK_approx:", QK_approx)
    print("diff", QK_mask - QK_approx)
    print(np.max(QK_mask - QK_approx))
    print(np.linalg.norm(QK_mask - QK_approx, ord='fro'))
    print(np.linalg.norm(QK_mask, ord='fro'))
    print(np.linalg.norm(QK_mask - QK_approx, ord='fro') / np.linalg.norm(QK_mask, ord='fro'))

def test_recovered_b_tilde(QK_exp_mask, b_tilde, m):
    # Check if the recovered b_tilde is correct by stable softmax
    k = b_tilde.shape[0]
    QK_exp_mask_approx = np.zeros_like(QK_exp_mask, dtype=np.float64)
    for i in range(k):
        QK_exp_mask_approx += sub_conv_matrix(b_tilde[i, :], m[i])
    print("QK_exp_mask:", QK_exp_mask)
    print("QK_exp_mask_approx:", QK_exp_mask_approx)
    print("diff", QK_exp_mask - QK_exp_mask_approx)

def test_conv_fft_time():
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
