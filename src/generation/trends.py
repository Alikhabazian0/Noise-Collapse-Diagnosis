import numpy as np

def linear_trend(t: np.ndarray, drift: float) -> np.ndarray:
    """
    Linear deterministic trend component.
    """
    return drift * np.ones_like(t, dtype=float)

def nonlinear_trend(
        t: np.ndarray,
        drift: float,
        amplitude: float,
        period: int
) -> np.ndarray:
    """
    Nonlinear trend: linear drift + sinusoidal component.
    """
    sinusoidal_part = amplitude * np.sin(2 * np.pi * t / period)
    linear_part =  drift * t

    # Return increments of the trend, not the level itself
    trend_level = linear_part + sinusoidal_part
    trend_increment = np.diff(trend_level, prepend=trend_level[0])

    return trend_increment

def get_trend(
        trend_type: str,
        t: np.ndarray,
        linear_drift: float,
        nonlinear_drift: float,
        nonlinear_amplitude: float,
        nonlinear_period: int
) -> np.ndarray:
    """
    Select trend type.
    """
    if trend_type == "linear":
        return linear_trend(t, linear_drift)
    
    if trend_type == "nonlinear":
        return nonlinear_trend(
            t=t,
            drift=nonlinear_drift,
            amplitude=nonlinear_amplitude,
            period=nonlinear_period
        )
    
    raise ValueError(f"Unknown rend_type: {trend_type}")