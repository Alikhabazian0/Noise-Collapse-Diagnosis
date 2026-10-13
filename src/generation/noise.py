import numpy as np

def gaussian_noise(size: int, rng: np.random.Generator) -> np.ndarray:
    """
    Standard Gaussian noise.
    """
    return rng.normal(loc=0.0, scale=1.0, size=size)

def student_t_noise(
        size: int,
        df: float,
        rng: np.random.Generator
) -> np.ndarray:
    """
    Standardized Student-t noise.

    Student-t has variance df/(df-2), so we standardize it 
    to approximately unit variance.
    """
    raw = rng.standard_t(df=df, size=size)

    if df > 2:
        raw = raw / np.sqrt(df / (df - 2))
        
    return raw

def get_noise(
        noise_type: str,
        size: int,
        rng: np.random.Generator,
        student_t_df: float
) -> np.ndarray:
    """
    Select noise type.
    """
    if noise_type == "gaussian":
        return gaussian_noise(size=size, rng=rng)

    if noise_type == "student_t":
        return student_t_noise(size=size, df=student_t_df, rng=rng)

    raise ValueError(f"Unknown noise_type: {noise_type}")  