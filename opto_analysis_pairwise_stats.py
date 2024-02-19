import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

ds_ae = pd.read_csv('J:\\Opto JAWS Data\\ds_ae_pairwise.csv')

colors_plot = ['orange', 'green', 'orange', 'green', 'orange', 'green']
fig, ax = plt.subplots(tight_layout=True, figsize=(7, 5))
for a in range(np.shape(ds_ae)[0]):
    ax.scatter(np.arange(6), np.array(ds_ae.iloc[a, 1:]), c=colors_plot, s=20)
    ax.plot(np.arange(6), np.array(ds_ae.iloc[a, 1:]), color='dimgray', linewidth=0.5)
ax.bar(np.arange(6), ds_ae.mean(), color=colors_plot, alpha=0.5)
ax.set_xticks([0.5, 2.5, 4.5])
ax.set_xticklabels(['tied', 'split right\nfast', 'split left\nfast'], fontsize=20)
ax.set_xlabel('Session', fontsize=20)
ax.set_ylabel('Double support\nafter-effect magnitude', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig('J:\\Opto JAWS Data\\session_comparison.png', dpi=128)
plt.savefig('J:\\Opto JAWS Data\\session_comparison.svg', dpi=128)

colors_plot = ['orange', 'green', 'orange', 'green']
fig, ax = plt.subplots(tight_layout=True, figsize=(7, 5))
for a in range(np.shape(ds_ae)[0]):
    ax.scatter(np.arange(4), np.array(ds_ae.iloc[a, 1:-2]), c=colors_plot, s=20)
    ax.plot(np.arange(4), np.array(ds_ae.iloc[a, 1:-2]), color='dimgray', linewidth=0.5)
ax.bar(np.arange(4), ds_ae.iloc[:, 1:-2].mean(), color=colors_plot, alpha=0.5)
ax.set_xticks([0.5, 2.5])
ax.set_xticklabels(['tied', 'split right\nfast'], fontsize=20)
ax.set_xlabel('Session', fontsize=20)
ax.set_ylabel('Double support\nafter-effect magnitude', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig('J:\\Opto JAWS Data\\session_comparison_tied_split_right.png', dpi=128)
plt.savefig('J:\\Opto JAWS Data\\session_comparison_tied_split_right.svg', dpi=128)