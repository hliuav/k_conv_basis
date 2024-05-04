import numpy as np

# Define example matrices Q, K, V
Q = np.random.randn(10, 5)
K = np.random.randn(10, 5)
V = np.random.randn(10, 5)

# Original computation that may cause overflow
QK_T_orig = Q @ K.T
exp_QK_T_orig = np.exp(QK_T_orig)
D_orig = np.diagflat(1 / np.sum(exp_QK_T_orig, axis=1))  # Assuming D is derived from these sums
result_orig = D_orig @ exp_QK_T_orig @ V

# Improved computation with stabilization
QK_T_max = np.max(QK_T_orig, axis=-1, keepdims=True)  # Max of each row
exp_QK_T_stable = np.exp(QK_T_orig - QK_T_max)
D_stable = np.diagflat(1 / np.sum(exp_QK_T_stable, axis=1))  # Derived similarly as in original
result_stable = D_stable @ exp_QK_T_stable @ V

# Comparing the results
difference = np.linalg.norm(result_orig - result_stable)
print("Norm of the difference between original and stabilized results:", difference)
print("Original result sample:", result_orig[:2])  # Show a sample of the output
print("Stabilized result sample:", result_stable[:2])
