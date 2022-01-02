import numpy as np
import matplotlib.pyplot as plt


def plot_history(hists, exp_path):
    x = np.arange(1, len(hists["loss"]["train"]) + 1)
    fig, axes = plt.subplots(nrows=1, ncols=len(hists), figsize=(20, 5))
    for ax, (name, hist) in zip(axes, hists.items()):
        for label, h in hist.items():
            ax.plot(x, h, label=label)

        ax.set_title("Model " + name)
        ax.set_xlabel('epochs')
        ax.set_ylabel(name)
        ax.legend(loc="best")

    plt.savefig(f"{exp_path}/metrics.png", dpi=fig.dpi)
    plt.close(fig)
