# nas_plot.py
import os
import numpy as np
import matplotlib.pyplot as plt

RESULT_FILES = {
    "Random": "results_random_2.txt",
    "TPE": "results_tpe_2.txt",
    "Grid": "results_grid.txt",
}

# Output figure
OUTPUT_FIG = "nas_sampler_comparison.png"

FIG_TITLE = "Best-so-far Accuracy vs Number of Trials"
X_LABEL = "Number of Trials"
Y_LABEL = "Maximum Achieved Accuracy"


def load_results(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Result file not found: {path}")

    data = np.loadtxt(path, comments="#")

    if data.ndim == 1:
        data = data.reshape(1, -1)

    trial = data[:, 0]
    best_so_far = data[:, 2]

    return trial, best_so_far


plt.figure(figsize=(7, 5))

for label, path in RESULT_FILES.items():
    trial, best = load_results(path)

    order = np.argsort(trial)
    best = best[order]

    x = np.arange(1, len(best) + 1)

    plt.plot(x, best, marker="o", label=label)

plt.title(FIG_TITLE)
plt.xlabel(X_LABEL)
plt.ylabel(Y_LABEL)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(OUTPUT_FIG, dpi=200)
plt.show()

print(f"[INFO] Figure saved to {OUTPUT_FIG}")
