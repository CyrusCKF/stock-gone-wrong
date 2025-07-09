import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def to_sliding_time_series(df: pd.DataFrame, window: int):
    """Create a matrix of sliding window time data series of size `window`

    Expect `df` have rows of time and columns of group.

    Return (cleaned_sliding_series, is_row_used)
    """
    matrix = df.to_numpy().T
    sliding_series = np.lib.stride_tricks.sliding_window_view(
        matrix, window, axis=1
    ).reshape((-1, window))

    # filter row if any value is nan
    non_nans = ~np.isnan(sliding_series).any(axis=1)
    sliding_series = sliding_series[non_nans]
    return sliding_series, non_nans


def sliding_metrics_series_view(df: pd.DataFrame, window: int, extra: int = 0):
    """Convert `df` to a mapping of metrics to matrix of sliding view, with size `window`.
    `extra` can be specified to ensure there will be data after the window

    Expect `df` to be have format like `yf.Tickers.history`
    """
    extra_window = window + extra
    metrics_series: dict[str, np.ndarray] = {}
    metrics = list(df.columns.levels[0].values)

    non_nans = None
    for m in metrics:
        sliding_series, new_non_nans = to_sliding_time_series(df[m], extra_window)
        if non_nans is not None:
            assert (non_nans == new_non_nans).all()
        non_nans = new_non_nans
        metrics_series[m] = sliding_series[:, :window]
    return metrics_series, non_nans


def extended_minmax_scale(
    vector: np.ndarray, feature_range: tuple[int, int], fit_window: slice
):
    """Like sklearn.preprocessing.minmax_scale, but params are calculated
    in a subwindow.

    Only support 1-d vector
    """
    scaler = MinMaxScaler(feature_range)
    scaler.fit(vector[fit_window].reshape((-1, 1)))
    vector = scaler.transform(vector.reshape((-1, 1))).squeeze()
    return vector
