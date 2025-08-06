import numpy as np
import matplotlib.pyplot as plt

def run_experiment(eps, runs=2000, steps=10000, k=10):
    '''
    Exercise 2.3 In the comparison shown in Figure 2.2, which method will perform best in the long run
    in terms of cumulative reward and probability of selecting the best action? How much better will it
    be? Express your answer quantitatively.
    Answer:
    the epsilon 0.01 achieves higher rewards than 0.1, 
    aproximatelly 0.9% higher, 
    while the optimal selection get 0.991 while 0.1 get 0.91

    Observations:
    - As larger the variance is  noisier rewards are. Requires more exploration to find the optimal action.
    - Suppose rewards are non-stationary (vary over time): more exploration requires 
    
    '''
    rewards = np.zeros(steps)
    optimal = np.zeros(steps)
    for _ in range(runs):
        q_true = np.random.randn(k)
        best = np.argmax(q_true)
        Q = np.zeros(k)
        N = np.zeros(k)
        for t in range(steps):
            if np.random.rand() < eps:
                a = np.random.randint(k)
            else:
                a = np.argmax(Q)
            r = q_true[a] + np.random.randn()
            N[a] += 1
            Q[a] += (r - Q[a]) / N[a]
            rewards[t] += r
            optimal[t] += (a == best)
    return rewards / runs, optimal / runs * 100

epsilons = [0, 0.01, 0.1]
labels   = ['ε = 0 (greedy)', 'ε = 0.01', 'ε = 0.1']
styles   = ['g-', 'r-', 'k-']
steps    = np.arange(1, 10001)
results  = {}
for eps in epsilons:
    results[eps] = run_experiment(eps)

# ### Quantitative Long-Run Comparison
# Compute final-step values and relative improvements

for eps in epsilons:
    avg_r, pct_opt = results[eps]
    print(f"ε={eps:<5}  avg_reward={avg_r[-1]:.4f}  %optimal={pct_opt[-1]:.2f}%")

r01, p01 = results[0.01]
r1, p1   = results[0.1]
rel_imp  = (r01[-1] - r1[-1]) / r1[-1] * 100
abs_pp   = (p01[-1] - p1[-1])
print(f" Relative reward improvement of ε=0.01 vs ε=0.1: {rel_imp:.2f}%")
print(f"Absolute %-optimal difference: {abs_pp:.2f} percentage points")


plt.figure(figsize=(10, 8))
for eps, lbl, sty in zip(epsilons, labels, styles):
    avg_r, pct_opt = results[eps]
    plt.subplot(2, 1, 1)
    plt.plot(steps, avg_r, sty, label=lbl)
    plt.ylabel('Average reward')
    plt.legend(loc='lower right')
    plt.subplot(2, 1, 2)
    plt.plot(steps, pct_opt, sty, label=lbl)
    plt.xlabel('Steps')
    plt.ylabel('% Optimal action')
    plt.legend(loc='lower right')

plt.tight_layout()
plt.show()
