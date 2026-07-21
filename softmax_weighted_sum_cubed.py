import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax

def sample_weighted_sum(n_dims, n_samples, transform='none'):
    x = np.random.randn(n_samples, n_dims)
    if transform == 'squared':
        x = x ** 2
    elif transform == 'cubed':
        x = x ** 3
    weights = softmax(x, axis=1)
    z = np.random.randn(n_samples, n_dims)
    return np.sum(weights * z, axis=1)

# Extended variance plot up to n=10000
fig, ax = plt.subplots(figsize=(10, 6))
dims_extended = [2, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
vars_sq = []
vars_nosq = []
vars_cubed = []

n_samples = 10000

for n in dims_extended:
    print(f"Computing n={n}...")
    samples_sq = sample_weighted_sum(n, n_samples, transform='squared')
    samples_nosq = sample_weighted_sum(n, n_samples, transform='none')
    samples_cubed = sample_weighted_sum(n, n_samples, transform='cubed')
    vars_sq.append(samples_sq.var())
    vars_nosq.append(samples_nosq.var())
    vars_cubed.append(samples_cubed.var())
    print(f"  Sq={samples_sq.var():.4f}, None={samples_nosq.var():.4f}, Cubed={samples_cubed.var():.4f}")

ax.plot(dims_extended, vars_sq, 'o-', label='Squared', markersize=8, linewidth=2)
ax.plot(dims_extended, vars_cubed, 'd-', label='Cubed', markersize=8, linewidth=2)
ax.plot(dims_extended, vars_nosq, 's-', label='None (raw)', markersize=8, linewidth=2)
ax.axhline(1, color='k', linestyle='--', alpha=0.5, label='Var=1')
ax.plot(dims_extended, 1/np.array(dims_extended), 'k:', linewidth=2, label='1/n')
ax.set_xlabel('Context length (n)', fontsize=12)
ax.set_ylabel(r'Variance of $\sum_i S_i Z_i$', fontsize=12)
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend(fontsize=11)
ax.set_title(r'Variance of softmax-weighted sum vs context length', fontsize=14)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('softmax_weighted_sum_cubed.png', dpi=150)
plt.show()
print("\nSaved to softmax_weighted_sum_cubed.png")
