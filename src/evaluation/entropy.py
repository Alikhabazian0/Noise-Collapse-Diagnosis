import numpy as np
from math import factorial
from collections import Counter

def shannon_entropy(values, n_bins=30, normalize=True):
    values = np.asarray(values)

    counts, _ = np.histogram(values, bins=n_bins)
    probs = counts / counts.sum()
    probs = probs[probs > 0]

    entropy = - np.sum(probs * np.log2(probs))

    if normalize:
        max_entropy = np.log2(n_bins)
        entropy = entropy / max_entropy if max_entropy > 0 else np.nan

    return entropy

def permutation_entropy(values, order=3, delay=1, normalize=True):
    values = np.asarray(values)

    n = len(values)
    if n < order * delay:
        return np.nan
    
    patterns = []

    for i in range(n - delay * (order - 1)):
        window = values[i: i + delay * order:delay]
        pattern = tuple(np.argsort(window))
        patterns.append(pattern)

    counts = Counter(patterns)
    probs = np.array(list(counts.values()), dtype=float)
    probs = probs / probs.sum()

    entropy = - np.sum(probs * np.log2(probs))

    if normalize:
        max_entropy = np.log2(factorial(order))
        entropy = entropy / max_entropy if max_entropy > 0 else np.nan

    return entropy

def entropy_by_sigma(values, sigma_values, n_bins=30, order=3, delay=1):
    rows = []

    unique_sigmas = np.sort(np.unique(sigma_values))

    for sigma in unique_sigmas:
        mask = sigma_values == sigma
        segment_values = values[mask]

        rows.append({
            "sigma": sigma,
            "n_samples": len(segment_values),
            "shannon_entropy": shannon_entropy(
                segment_values,
                n_bins=n_bins,
                normalize=True
            ),
            "permutation_entropy": permutation_entropy(
                segment_values,
                order=order,
                delay=delay,
                normalize=True
            )
        })

    return rows
