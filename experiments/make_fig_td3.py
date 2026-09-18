"""Fig. S1 (supplementary): TD3-RL training curve from data/td3_train_log.npz."""
import os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "legend.fontsize": 6, "xtick.labelsize": 6, "ytick.labelsize": 6,
                     "lines.linewidth": 1.0, "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": ":", "legend.framealpha": 0.85})
d = np.load(os.path.join(HERE, "data", "td3_train_log.npz")); r = d["ep_reward"]; ep = np.arange(1, len(r) + 1)
ma = np.convolve(r, np.ones(200) / 200, mode="valid"); ep_ma = np.arange(200, len(r) + 1)
fig, ax = plt.subplots(figsize=(3.5, 1.6))
ax.plot(ep, r, color="C1", alpha=0.18, lw=0.4, label="per-episode mean reward")
ax.plot(ep_ma, ma, color="C1", lw=1.2, label="200-episode moving average")
ax.set_xlabel("training episode"); ax.set_ylabel("mean reward"); ax.set_xlim(0, len(r)); ax.set_ylim(0.4, 2.8)
ax.legend(loc="lower right", handlelength=1.8)
fig.tight_layout(pad=0.3)
for out in (os.path.join(HERE, "figures", "fig_td3_train.pdf"), os.path.join(HERE, "figures", "fig_td3_train.png")):
    fig.savefig(out, dpi=300)
print("MA200 first/last:", ma[0], ma[-1], "last-500 rel change %:", abs(ma[-1] - ma[-500]) / abs(ma[-500]) * 100)
