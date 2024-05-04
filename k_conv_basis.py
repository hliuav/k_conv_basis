import numpy as np
import torch

from k_conv_basis_utils import binary_search, get_b_tilde_from_b, sub_conv_matrix, conv_with_fft_matrix, conv_with_fft

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
    return b_tilde, m, b

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

    n = Q.shape[0]
    k = 47
    T = 1
    delta = 1e-2
    epsilon = 1e-3

    Q = Q.astype(np.float64)
    K = K.astype(np.float64)
    b_tilde, m, b = recover_k_conv(Q, K, k=k, T=T, delta=delta, epsilon=epsilon)
    QK = Q @ K.T

    mask = np.tril(np.ones_like(QK)).astype(np.float64)
    QK_exp = np.exp(QK)
    QK_mask = mask * QK
    QK_exp_mask = mask * QK_exp
    D_stable_inv = np.diagflat(1 / np.sum(QK_exp_mask, axis=1))  # Derived similarly as in original
    QKV = D_stable_inv @ QK_exp_mask @ V

    #QK_mask = np.where(QK_mask==0.0, float("-inf"), QK_mask)
    #QK_mask_stable_exp = np.exp(QK_mask - np.max(QK_mask, axis=-1, keepdims=True))
    #D_stable_inv = np.diagflat(1 / np.sum(QK_mask_stable_exp, axis=1))  # Derived similarly as in original
    #QK_mask = D_stable_inv @ QK_mask_stable_exp
    #QK_mask_exp = np.exp(QK_mask)
    #print(QK_mask)
    #QKV = QK_mask_stable_exp @ V

    # Check if the recovered b is correct
    #QK_approx = np.zeros_like(Q @ K.T, dtype=np.float64)
    #for i in range(k):
    #    QK_approx += sub_conv_matrix(b[i, :], m[i])
    #print("QK_mask:", QK_mask)
    #print("QK_approx:", QK_approx)
    #print("diff", QK_mask - QK_approx)
    #print(np.max(QK_mask - QK_approx))
    #print(np.linalg.norm(QK_mask - QK_approx, ord='fro'))
    #print(np.linalg.norm(QK_mask, ord='fro'))
    #print(np.linalg.norm(QK_mask - QK_approx, ord='fro') / np.linalg.norm(QK_mask, ord='fro'))

    # Check if the recovered b_tilde is correct by stable softmax
    
    #sum_b = np.zeros_like(QK_mask, dtype=np.float64)
    #for i in range(k):
    #    sum_b += sub_conv_matrix(b_tilde[i, :], m[i])
    #print("QK_exp_mask:", QK_exp_mask)
    #print("b_tilde:", sum_b)
    #print("diff", QK_exp_mask - sum_b)

    QKV_approx= np.zeros_like(QKV, dtype=np.float64)
    for i in range(k):
        print(b_tilde[i, :])
        QKV_approx += conv_with_fft_matrix(b_tilde[i, :], V, shift=n - m[i])

    #D_approx_inv = np.diagflat(1 / np.sum(sum_b, axis=1))  # Derived similarly as in original
    D_approx = np.zeros(n, dtype=np.float64)
    for i in range(k):
        D_approx += conv_with_fft(b_tilde[i, :], np.ones(n), shift=n - m[i])

    QKV_approx = np.expand_dims(D_approx ** -1, axis=1) * QKV_approx

    print("QKV:", QKV)
    print("QKV_approx:", QKV_approx)
    print("diff:", np.linalg.norm(QKV - QKV_approx, ord='fro'))
    #print(QKV_approx - QKV)
    #print(QKV.sum())
    #print(QKV_approx.sum())

if __name__ == '__main__':
    main()


