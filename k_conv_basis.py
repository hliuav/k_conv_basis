import numpy as np

from k_conv_basis_utils import binary_search, get_b_tilde_from_b

def recover_k_conv(Q, K, T, delta, epsilon):
    n, k = Q.shape
    v = np.zeros(T)  # Initial vector v
    u = np.zeros(n)  # Initial vector u
    s = 1  # Initial index s
    t = n - T + 1
    H = np.tril(np.ones((n, n))) * (Q @ K.T)

    m = np.zeros(k)
    b = np.zeros((k, n))
    b_tilde = np.zeros((k, n))
    for i in range(k):
        s += 1
        s = binary_search(Q, K, k, T, delta, epsilon, v, s, t)
        m[i] = n - s + 1

        b[i, : m[i]] = H[s, s : s + m[i] - 1] - u[: m[i]]
        v += b[i, :T]
        u += b[i]

    b_tilde = get_b_tilde_from_b(b)
    # Additional outputs based on Lemma A.2 not provided, assuming direct return
    return b_tilde, m


