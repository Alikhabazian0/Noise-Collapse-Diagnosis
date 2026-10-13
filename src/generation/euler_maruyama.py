import numpy as np

def create_stepwise_sigma_path(
        series_length: int,
        sigma_levels: np.ndarray
) -> np.ndarray:
    """
    Create a stepwise increasing sigma path
    
    Example:
    level 1 for first 10 %,
    level 2 for second 10 %,
    ...
    """
    num_levels = len(sigma_levels)
    segment_length = series_length // num_levels

    sigma_path = np.zeros(series_length)

    for i, sigma in enumerate(sigma_levels):
        start = i * segment_length

        if i == num_levels - 1:
            end = series_length
        else:
            end = (i + 1) * segment_length

        sigma_path[start:end] = sigma

    return sigma_path

def simulate_euler_maruyama(
        trend: np.ndarray,
        noise: np.ndarray,
        sigma_path: np.ndarray,
        dt: float,
        x0: float = 0.0
) -> np.ndarray:
    """
    Euler maruyama simulation:

    X[t+1] = X[t] * dt + sigma[t] * sqrt(dt) * noise[t]
    """
    n = len(trend)

    if len(noise) != n or len(sigma_path) != n:
        raise ValueError("trend, noise, and sigma_path must have same length")
    
    x = np.zeros(n)
    x[0] = x0

    for t in range(1, n):
        x[t] = (
            x[t - 1]
            + trend[t] * dt
            + sigma_path[t] * np.sqrt(dt) * noise[t]
        )

    return x
