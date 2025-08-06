# %% [markdown]
# # Convergence of Stochastic Approximation Step-Sizes
#
# This notebook explores why sequences of step-size parameters \(\{\alpha_n\}\) 
# converge (or fail to converge) under the Robbins–Monro conditions:
#   1. \(\sum_{n=1}^\infty \alpha_n = \infty\)
#   2. \(\sum_{n=1}^\infty \alpha_n^2 < \infty\)
#
# We will:
# - Visualize different \(\alpha_n\) schedules
# - Plot their cumulative sums and squared sums to see which conditions hold
# - Simulate the update \(Q_{n+1}=Q_n+\alpha_n(X_n-Q_n)\) to see convergence behavior

# %%
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# ## 1. Define Step-Size Sequences
# We compare:
# - Constant:          \(\alpha_n=0.1\)
# - Harmonic:          \(\alpha_n=1/n\)                  (p=1)
# - Subharmonic slow:  \(\alpha_n=1/n^{0.75}\)             (0.5<p<1)
# - Subharmonic fast:  \(\alpha_n=1/n^{1.25}\)            (p>1)

# %%
N = 1000
n = np.arange(1, N+1)
schedules = {
    'constant α=0.1'     : np.full(N, 0.1),
    '1/n (p=1.0)'         : 1.0/n,
    '1/n^0.75 (p=0.75)'   : 1.0/(n**0.75),
    '1/n^1.25 (p=1.25)'   : 1.0/(n**1.25),
}

# %% [markdown]
# ### Plot the Step-Size Sequences

# %%
plt.figure(figsize=(8, 4))
for name, alpha in schedules.items():
    plt.plot(n, alpha, label=name)
plt.title('Step-Size Sequences')
plt.xlabel('Iteration n')
plt.ylabel(r'$\alpha_n$')
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2. Check Convergence Conditions
# Compute partial sums:
# \(
#    S_1(n)=\sum_{i=1}^n\alpha_i,
#    \quad
#    S_2(n)=\sum_{i=1}^n\alpha_i^2
#\)

# %%
plt.figure(figsize=(8, 4))
for name, alpha in schedules.items():
    S1 = np.cumsum(alpha)
    plt.plot(n, S1, label=f'S1: {name}')
plt.title('Cumulative Sum $S_1(n)=\sum_{i=1}^n\alpha_i$')
plt.xlabel('n')
plt.ylabel(r'$S_1(n)$')
plt.legend(); plt.tight_layout(); plt.show()

plt.figure(figsize=(8, 4))
for name, alpha in schedules.items():
    S2 = np.cumsum(alpha**2)
    plt.plot(n, S2, label=f'S2: {name}')
plt.title('Cumulative Sum $S_2(n)=\sum_{i=1}^n\alpha_i^2$')
plt.xlabel('n')
plt.ylabel(r'$S_2(n)$')
plt.legend(); plt.tight_layout(); plt.show()

# %% [markdown]
# **Interpretation**:
# - If S1 grows without bound (diverges) then condition 1 is met.
# - If S2 converges to a finite limit, then condition 2 is met.
#
# From the plots you should observe:
# - **constant** and **1/n^0.75**: S1→∞ but S2→∞ → fails condition 2
# - **1/n**             : S1→∞ and S2→finite → meets both conditions → converges
# - **1/n^1.25**       : S1→finite and S2→finite → fails condition 1  → does not converge

# %% [markdown]
# ## 3. Simulate the Update Rule
# We simulate a single sequence of i.i.d. samples \(X_n\sim\mathcal{N}(1,1)\) and update
# \(Q_{n+1} = Q_n + \alpha_n (X_n - Q_n)\).

# %%
true_mean = 1.0
Q_hist = {name: np.zeros(N) for name in schedules}
for name, alpha in schedules.items():
    Q = 0.0
    for i in range(N):
        x = true_mean + np.random.randn()
        Q += alpha[i] * (x - Q)
        Q_hist[name][i] = Q

# %% [markdown]
# Plot the trajectory of \(Q_n\) for each schedule

# %%
plt.figure(figsize=(8, 4))
for name, Qvals in Q_hist.items():
    plt.plot(n, Qvals, label=name)
plt.axhline(true_mean, linestyle='--', label='true mean')
plt.title('Estimate $Q_n$ over Iterations')
plt.xlabel('n')
plt.ylabel(r'$Q_n$')
plt.legend(); plt.tight_layout(); plt.show()

# %% [markdown]
# **Conclusion:**
# - Only the schedule \(\alpha_n=1/n\) (and any with exponent 0.5<p<1) satisfies both Robbins–Monro conditions and drives \(Q_n\) toward the true mean.
# - Constant or too slowly/too quickly decaying step-sizes either never settle or freeze prematurely.
#
# You can adjust `N` or add more schedules to explore other behaviors.
