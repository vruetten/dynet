import torch
import numpy as np
from dataclasses import dataclass

def create_filter_bank(n_node_types, node_type_names, filter_length_max, device):
    Fz = np.zeros((n_node_types, n_node_types, 2, filter_length_max))
    types = {}
    for i in range(n_node_types):
        types[i] = {}

    types[0]['type'] = node_type_names[0]
    types[0]['dt'] = np.ones(n_node_types)*0.01
    types[0]['rate'] = np.random.rand(n_node_types)*0.01

    types[1]['type'] = node_type_names[1]
    types[1]['dt'] = np.ones(n_node_types)*0.01
    types[1]['frequency'] = np.random.rand(n_node_types)

    types[2]['type'] = node_type_names[2]
    types[2]['tau'] = np.random.rand(n_node_types)
    types[2]['dt'] = np.ones(n_node_types)*0.01

    types[3]['type'] = node_type_names[3]
    types[3]['tau'] = np.random.rand(n_node_types)*0.01
    types[3]['dt'] = np.ones(n_node_types)*0.01

    types[4]['type'] = node_type_names[4]
    types[4]['window_size'] = np.random.randint(1, 4, n_node_types)
    types[4]['dt'] = np.ones(n_node_types)*0.01

    for i in range(n_node_types):
        if types[i]['type'] in ['poisson', 'oscillator']:
            continue
        for j in range(n_node_types):
            if types[i]['type'] == 'low_pass':
                coefs = create_low_pass(types[i]['tau'][j], types[i]['dt'][j])
            elif types[i]['type'] == 'high_pass':
                coefs = create_high_pass(types[i]['tau'][j], types[i]['dt'][j])
            elif types[i]['type'] == 'moving_average':
                coefs = create_moving_average(types[i]['window_size'][j])
            else:
                continue
            a, b = process_filter_coefficients(coefs, filter_length_max, device)
            Fz[i, j, 0] = a
            Fz[i, j, 1] = b
    return Fz


def create_poisson_activities(n_nodes, n_frames, device):
    """
    Create a tensor of shape (n_nodes, n_frames) with Poisson distributed values.
    Each node has a random amplitude between 0.1 and 1.0 and a random rate between 0.01 and 0.05.

    Args:
        n_nodes: Number of nodes (first dimension)
        n_frames: Number of time frames (second dimension)
        device: Device to create tensors on

    Returns:
        torch.Tensor: Shape (n_nodes, n_frames) with Poisson activities
    """
    # Random amplitudes between 0.1 and 1.0 for each node
    amplitudes = torch.rand(n_nodes, device=device) * 0.9 + 0.1  # 0.1 to 1.0

    # Random rates between 0.01 and 0.05 for each node
    rates = torch.rand(n_nodes, device=device) * 0.04 + 0.01  # 0.01 to 0.05

    # Create Poisson distributed activities
    # Each node gets its own rate parameter expanded across all frames
    lambda_params = rates.unsqueeze(1).expand(n_nodes, n_frames)  # (n_nodes, n_frames)

    # Generate Poisson samples
    poisson_samples = torch.poisson(lambda_params)

    # Scale by amplitude for each node
    poisson_activities = poisson_samples * amplitudes.unsqueeze(1)

    return poisson_activities


def create_oscillator_activities(n_nodes, n_frames, dt=1.0, amplitude_range=(0.5, 1.5), frequency_range=(0.01, 0.1), phase_randomize=True, device=None):
    """
    Create a tensor of shape (n_nodes, n_frames) with oscillatory activities.
    Each node has a random frequency between 0.1 and 1.0 Hz.

    Args:
        n_nodes: Number of nodes (first dimension)
        n_frames: Number of time frames (second dimension)
        device: Device to create tensors on
        dt: Time step in seconds (default 1.0)
        amplitude_range: Tuple of (min, max) amplitude for each oscillator
        phase_randomize: Whether to randomize initial phases

    Returns:
        torch.Tensor: Shape (n_nodes, n_frames) with oscillatory activities
    """
    # Random frequencies between 0.1 and 1.0 Hz
    freq_min, freq_max = frequency_range
    frequencies = torch.rand(n_nodes, device=device) * (freq_max - freq_min) + freq_min

    # Random amplitudes for each node
    amp_min, amp_max = amplitude_range
    amplitudes = torch.rand(n_nodes, device=device) * (amp_max - amp_min) + amp_min

    # Time vector in seconds
    t = torch.arange(n_frames, device=device, dtype=torch.float32) * dt

    # Random phases if requested
    if phase_randomize:
        phases = torch.rand(n_nodes, device=device) * 2 * np.pi
    else:
        phases = torch.zeros(n_nodes, device=device)

    # Create oscillatory activities
    # frequencies: (n_nodes,) -> (n_nodes, 1)
    # t: (n_frames,) -> (1, n_frames)
    # phases: (n_nodes,) -> (n_nodes, 1)
    freq_expanded = frequencies.unsqueeze(1)  # (n_nodes, 1)
    phase_expanded = phases.unsqueeze(1)  # (n_nodes, 1)
    amp_expanded = amplitudes.unsqueeze(1)  # (n_nodes, 1)

    # Broadcasting: (n_nodes, 1) * (1, n_frames) = (n_nodes, n_frames)
    oscillatory_activities = amp_expanded * torch.sin(2 * np.pi * freq_expanded * t + phase_expanded)

    return oscillatory_activities

def create_connectivity(types: int, n_nodes: int, connectivity_filling_factor: float, device: str = 'cpu'):
    connectivity = torch.randn((n_nodes, n_nodes), dtype=torch.float32, device=device)
    connectivity = connectivity / np.sqrt(n_nodes)
    mask = torch.rand(connectivity.shape) > connectivity_filling_factor
    connectivity[mask] = 0


    pos = np.argwhere(types<2)
    if len(pos) > 0:
        connectivity[pos[:, None], :] = 0.0  # Remove connections between node types < 2

    mask = (connectivity.t() != 0).float()
    edge_index = mask.nonzero().t().contiguous()

    return connectivity, edge_index

def get_equidistant_points(n_points=1024, device=None):
    indices = np.arange(0, n_points, dtype=float) + 0.5
    r = np.sqrt(indices / n_points)
    theta = np.pi * (1 + 5 ** 0.5) * indices
    x, y = r * np.cos(theta), r * np.sin(theta)

    pos = torch.tensor(np.stack((x, y), axis=1), dtype=torch.float32, device=device) / 2
    perm = torch.randperm(pos.size(0))
    pos = pos[perm]

    return pos

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


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a PyTorch tensor to a NumPy array.

    Args:
        tensor (torch.Tensor): The PyTorch tensor to convert.

    Returns:
        np.ndarray: The NumPy array.
    """
    return tensor.detach().cpu().numpy()

