import matplotlib.pyplot as plt

from .config import CH06_DIR


def plot_values(epochs_seen, examples_seen, train_values, val_values,
                label="loss"):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(
        epochs_seen, val_values, linestyle="-.",
        label=f"Validation {label}"
    )
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()

    ax2 = ax1.twiny()
    ax2.plot(examples_seen, train_values, alpha=0)  # Align ticks only
    ax2.set_xlabel("Examples seen")

    fig.tight_layout()
    plot_path = CH06_DIR / f"{label}-plot.pdf"
    plt.savefig(plot_path)
    plt.close(fig)
    print(f"Saved {plot_path}")
