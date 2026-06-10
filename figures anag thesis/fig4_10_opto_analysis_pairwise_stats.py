import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

ds_ae = pd.read_csv('J:\\Opto JAWS Data\\ds_ae_pairwise.csv')
#CHNAGE NAMES OF SESSIONS
colors_plot = ['orange', 'green', 'orange', 'green', 'orange', 'green']
fig, ax = plt.subplots(tight_layout=True, figsize=(7, 5))
for a in range(np.shape(ds_ae)[0]):
    ax.scatter(np.arange(6), np.array(ds_ae.iloc[a, 1:]), c=colors_plot, s=20)
    ax.plot(np.arange(2), np.array(ds_ae.iloc[a, 1:3]), color='dimgray', linewidth=0.5)
    ax.plot(np.array([2, 3]), np.array(ds_ae.iloc[a, 3:5]), color='dimgray', linewidth=0.5)
    ax.plot(np.array([4, 5]), np.array(ds_ae.iloc[a, 5:7]), color='dimgray', linewidth=0.5)
ax.scatter(np.arange(6), ds_ae.mean(), marker='_', s=1000, linewidth=4, alpha=0.5, color=colors_plot)
ax.set_xticks([0.5, 2.5, 4.5])
ax.set_xticklabels(['tied', 'split\ncontralateral\nfast', 'split\nipsilateral\nfast'], fontsize=20)
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
    ax.plot(np.arange(2), np.array(ds_ae.iloc[a, 1:3]), color='dimgray', linewidth=0.5)
    ax.plot(np.array([2, 3]), np.array(ds_ae.iloc[a, 3:5]), color='dimgray', linewidth=0.5)
ax.scatter(np.arange(4), ds_ae.iloc[:, 1:-2].mean(), marker='_', s=1000, linewidth=4, alpha=0.5, color=colors_plot)
ax.set_xticks([0.5, 2.5])
ax.set_xticklabels(['tied', 'split contralateral\nfast'], fontsize=20)
ax.set_xlabel('Session', fontsize=20)
ax.set_ylabel('Double support\nafter-effect magnitude', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig('J:\\Opto JAWS Data\\session_comparison_tied_split_right.png', dpi=128)
plt.savefig('J:\\Opto JAWS Data\\session_comparison_tied_split_right.svg', dpi=128)