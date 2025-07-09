import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator
from scipy.stats import t


def calculate_PI(data: np.ndarray, pct=0.95):
    """Calculate prediction interval along the first axis.

    Expect data of shape (num_samples, num_features)

    Return (lower_bound, upper_bound)
    """
    mu: np.ndarray = data.mean(axis=0)
    se: np.ndarray = data.std(axis=0, ddof=1)
    n = data.shape[0]
    t_val = t.ppf(1 - (1 - pct) / 2, df=n - 1)
    margin: np.ndarray = t_val * se * np.sqrt(1 + 1 / n)
    return (mu - margin, mu + margin)


def format_plot(ticker: str, metrics: str, ax: Axes | None = None):
    if ax is None:
        ax = plt.gca()
    ax.set_title(f"{ticker} w/ similar stocks")
    ax.set_xlabel("Day")
    ax.set_ylabel(f"Standardized {metrics}")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend()
    ax.grid(True)
