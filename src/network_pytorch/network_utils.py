import torch
import numpy as np
from dataclasses import dataclass


def create_connectivity(types: int, n_nodes: int, connectivity_filling_factor: float, device: str = 'cpu'):
    connectivity = torch.randn((n_nodes, n_nodes), dtype=torch.float32, device=device)
    connectivity = connectivity / np.sqrt(n_nodes)
    mask = torch.rand(connectivity.shape) > connectivity_filling_factor
    connectivity[mask] = 0


    pos = np.argwhere(types<2)
    if len(pos) > 0:
        connectivity[pos[:, None], :] = 0.0  # Remove connections between node types < 2

    mask = (connectivity != 0).float()
    edge_index = mask.nonzero().t().contiguous()

    return connectivity, edge_index


@dataclass
class FilterCoefficients:
    """Stores the coefficients for a digital filter in the form:
    y[n] = b[0]*x[n] + b[1]*x[n-1] + ... + b[nb]*x[n-nb]
           - a[1]*y[n-1] - a[2]*y[n-2] - ... - a[na]*y[n-na]
    """
    b: np.ndarray  # Numerator coefficients
    a: np.ndarray  # Denominator coefficients


def create_low_pass(tau: float, dt: float):
    alpha = np.exp(-dt / tau)
    b = np.array([1 - alpha])
    a = np.array([1, -alpha])
    return FilterCoefficients(b, a)


def create_high_pass(tau: float, dt: float):
    alpha = np.exp(-dt / tau)
    b = np.array([alpha])
    a = np.array([1, -alpha])
    return FilterCoefficients(b, a)


def create_moving_average(window_size: int):
    b = np.ones(window_size) / window_size
    a = np.array([1])
    return FilterCoefficients(b, a)


def create_band_pass(center_freq: float, bandwidth: float, dt: float):
    omega_0 = 2 * np.pi * center_freq * dt
    r = np.exp(-np.pi * bandwidth * dt)
    b = np.array([1 - r ** 2])
    a = np.array([1, -2 * r * np.cos(omega_0), r ** 2])
    return FilterCoefficients(b, a)


def create_band_stop(center_freq: float, bandwidth: float, dt: float):
    """Create a band-stop (notch) filter.

    Args:
        center_freq: Center frequency in Hz
        bandwidth: Bandwidth in Hz
        dt: Time step in seconds
    """
    omega_0 = 2 * np.pi * center_freq * dt
    r = np.exp(-np.pi * bandwidth * dt)
    b = np.array([1, -2 * np.cos(omega_0), 1])
    a = np.array([1, -2 * r * np.cos(omega_0), r ** 2])
    return FilterCoefficients(b, a)


def create_all_pass(phase_shift: float, dt: float):
    """Create an all-pass filter that shifts phase without changing magnitude.

    Args:
        phase_shift: Desired phase shift in radians
        dt: Time step in seconds
    """
    alpha = np.tan(phase_shift / 2)
    b = np.array([-alpha, 1])
    a = np.array([1, -alpha])
    return FilterCoefficients(b, a)


def create_butterworth_low_pass(cutoff_freq: float, order: int, dt: float):
    """Create a Butterworth low-pass filter.

    Args:
        cutoff_freq: Cutoff frequency in Hz
        order: Filter order
        dt: Time step in seconds
    """
    # Normalized cutoff frequency
    wc = 2 * np.pi * cutoff_freq * dt
    # Calculate poles
    poles = np.exp(1j * np.pi * (2 * np.arange(order) + order + 1) / (2 * order))
    # Convert to polynomial coefficients
    a = np.poly(poles)
    # Normalize
    a = a / a[0]
    # Calculate numerator for unity gain at DC
    b = np.array([1])
    return FilterCoefficients(b, a)


def create_chebyshev_low_pass(cutoff_freq: float, ripple_db: float, order: int, dt: float):
    """Create a Chebyshev Type I low-pass filter.

    Args:
        cutoff_freq: Cutoff frequency in Hz
        ripple_db: Passband ripple in dB
        order: Filter order
        dt: Time step in seconds
    """
    # Normalized cutoff frequency
    wc = 2 * np.pi * cutoff_freq * dt
    # Calculate ripple factor
    epsilon = np.sqrt(10 ** (ripple_db / 10) - 1)
    # Calculate poles
    beta = np.arcsinh(1 / epsilon) / order
    poles = np.exp(1j * np.pi * (2 * np.arange(order) + order + 1) / (2 * order))
    poles = wc * np.sinh(beta) * np.real(poles) + 1j * wc * np.cosh(beta) * np.imag(poles)
    # Convert to polynomial coefficients
    a = np.poly(poles)
    # Normalize
    a = a / a[0]
    # Calculate numerator for unity gain at DC
    b = np.array([1])
    return FilterCoefficients(b, a)


def create_resonator(freq: float, q_factor: float, dt: float):
    """Create a resonator filter that strongly amplifies signals at a specific frequency.

    Args:
        freq: Resonant frequency in Hz
        q_factor: Quality factor (higher = narrower bandwidth)
        dt: Time step in seconds
    """
    omega_0 = 2 * np.pi * freq * dt
    alpha = np.sin(omega_0) / (2 * q_factor)
    b = np.array([alpha])
    a = np.array([1, -2 * np.cos(omega_0), 1 - alpha])
    return FilterCoefficients(b, a)


def create_delay(delay_samples: int):
    """Create a delay filter that delays the signal by a specified number of samples.

    Args:
        delay_samples: Number of samples to delay
    """
    b = np.zeros(delay_samples + 1)
    b[-1] = 1
    a = np.array([1])
    return FilterCoefficients(b, a)


def process_filter_coefficients(coefs, filter_length_max, device='cpu'):
    """Process filter coefficients by converting to tensors and padding if necessary.

    Args:
        coefs: FilterCoefficients object containing b and a arrays
        filter_length_max: Maximum allowed filter length
        device: Device to place tensors on

    Returns:
        tuple: (a_tensor, b_tensor) padded to filter_length_max
    """
    b = torch.tensor(coefs.b, dtype=torch.float32, device=device)
    a = torch.tensor(coefs.a, dtype=torch.float32, device=device)

    if len(b) > filter_length_max or len(a) > filter_length_max:
        raise ValueError(f"Filter length exceeds maximum allowed length of {filter_length_max}")

    # Pad with zeros if necessary
    if len(b) < filter_length_max:
        b = torch.cat([b, torch.zeros(filter_length_max - len(b), dtype=torch.float32, device=device)])
    if len(a) < filter_length_max:
        a = torch.cat([a, torch.zeros(filter_length_max - len(a), dtype=torch.float32, device=device)])

    return a, b