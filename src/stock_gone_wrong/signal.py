import heapq

import numpy as np
import scipy.signal


def find_peaks_with_ends(data: np.ndarray, **kwargs) -> np.ndarray:
    """Like scipy.signal.find_peaks but include both ends if valid"""
    minimum = np.min(data)
    extended_x = np.concat([[minimum], data, [minimum]])
    peaks, _ = scipy.signal.find_peaks(extended_x, **kwargs)
    return peaks - 1


def find_stock_peaks(data: np.ndarray, window=2):
    """Custom peak finding algorithm that works well with stock data.

    It considers the neighbouring peaks and decide whether it is still the peak.
    In a way, it tries to find the local maxima of local maxima.

    User may consider other signal algorithms like scipy.signal.find_peaks
    for a more traditional approach.

    Args:
        window: Number of neighbours in each side to compare with
    """
    og_peaks = find_peaks_with_ends(data)
    if window == 0:
        return og_peaks
    peak_data = data[og_peaks]
    filtered = find_peaks_with_ends(peak_data, distance=window)
    return og_peaks[filtered]


def remove_saddle(data: np.ndarray, indices: np.ndarray):
    peaks = find_peaks_with_ends(data[indices])
    troughs = find_peaks_with_ends(-data[indices])
    pois = [e for (i, e) in enumerate(indices) if i in peaks or i in troughs]
    return np.array(pois)


def find_largest_changes(data: np.ndarray, indices: np.ndarray, num: int):
    """Return list of neighbours that have the largest changes"""
    neighbours = list(zip(indices[:-1], indices[1:]))
    changes = [abs(data[n[0]] - data[n[1]]).item() for n in neighbours]
    i_val = heapq.nlargest(num, enumerate(neighbours), key=lambda x: changes[x[0]])
    return np.array([val for (i, val) in sorted(i_val)])
