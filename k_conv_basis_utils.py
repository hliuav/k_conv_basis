import numpy as np
import torch
from torch import nn

def recover_k_conv(Q, K, k, T, delta, epsilon):
    n, d = Q.shape
    v = np.zeros(T, dtype=np.float32)  # Initial vector v
    u = np.zeros(n, dtype=np.float32)  # Initial vector u
    s = 0  # Initial index s
    t = n - T
    

    m = np.zeros(k, dtype=int)
    b = np.zeros((k, n), dtype=np.float32)

    # Caculate the first b
    b[0, :] = Q @ K.T[:, 0]
    m[0] = n
    v += b[0, :T]
    u += b[0, :]
    for i in range(1, k):
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
    return b_tilde, m, b

def get_b_tilde_from_b(b):
    b_tilde = np.ones_like(b, dtype=np.float32) # exp(0)
    k, n = b.shape
    sum_b_r = np.zeros(n, dtype=np.float32)
    sum_b_r_minus_1 = np.zeros(n, dtype=np.float32)
    for i in range(k):
        if i == 0:
            sum_b_r += b[i]
            b_tilde[i, :] = np.exp(sum_b_r)
        else:
            sum_b_r += b[i]
            sum_b_r_minus_1 += b[i - 1]
            b_tilde[i, :] = np.exp(sum_b_r) - np.exp(sum_b_r_minus_1)
    return b_tilde


def binary_search(Q, K, k, T, delta, epsilon, v, s, t):
    n, _ = Q.shape
    if s >= t:
        # change 1
        return t
    j = (s + t) // 2

    H_j = Q @ ((K.T)[:,j])

    # Norm calculation
    alpha = np.linalg.norm(H_j[j : j + T] - v, ord=1)
    #print(k)
    #print(alpha)
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
    a_padded = np.zeros(2 * n, dtype=np.float32)
    x_padded = np.zeros(2 * n, dtype=np.float32)
    a_padded[:n] = a[:n]
    x_padded[:n] = x[-n:]

    result = np.zeros_like(a, dtype=np.float32)
    result[-n:] = np.fft.ifft(np.fft.fft(a_padded) * np.fft.fft(x_padded)).real[:n]
    return result


def conv_with_fft_matrix(a, X, shift=0):
    n, d = X.shape
    result_matrix = np.zeros_like(X, dtype=np.float32)
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
