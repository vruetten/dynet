#%%
from dataclasses import dataclass

import torch
import numpy as np
import matplotlib.pyplot as pl
device='cpu'

def create_connectivity(n_nodes: int, connectivity_filling_factor: float, device: str = 'cpu'):
    connectivity = torch.randn((n_nodes, n_nodes), dtype=torch.float32, device=device)
    connectivity = connectivity / np.sqrt(n_nodes)
    mask = torch.rand(connectivity.shape) >  connectivity_filling_factor
    connectivity[mask] = 0
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
    b = np.array([1 - r**2])
    a = np.array([1, -2 * r * np.cos(omega_0), r**2])
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
    a = np.array([1, -2 * r * np.cos(omega_0), r**2])
    return FilterCoefficients(b, a)

def create_all_pass(phase_shift: float, dt: float):
    """Create an all-pass filter that shifts phase without changing magnitude.
    
    Args:
        phase_shift: Desired phase shift in radians
        dt: Time step in seconds
    """
    alpha = np.tan(phase_shift/2)
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
    epsilon = np.sqrt(10**(ripple_db/10) - 1)
    # Calculate poles
    beta = np.arcsinh(1/epsilon) / order
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

# Example usage
np.random.seed(0)

n_node_types = 5
n_node_types_per_type = 2
n_nodes = n_node_types * n_node_types_per_type
filter_length_max = 5

ids = np.arange(n_nodes)

# types = np.random.randint(0, n_node_types, n_nodes)
types = np.repeat(np.arange(n_node_types), n_node_types_per_type)

# types = np.array([0,1,2,0,1,2])
activities = np.random.randn(n_nodes)

X = np.column_stack((ids, types, activities)) # id, type, activity at t0
connectivity_filling_factor = 0.1
connectivity, edge_index = create_connectivity(n_nodes, connectivity_filling_factor)

Fz = np.zeros((n_node_types, n_node_types, 2, filter_length_max))
Fz.shape


# create filter bank for each node type
types = {}
for i in range(n_node_types):
    types[i] = {}

types[0]['type'] = 'poisson'
types[0]['dt'] = np.ones(n_node_types)*0.01
types[0]['rate'] = np.random.rand(n_node_types)*0.01

types[1]['type'] = 'oscillator'
types[1]['dt'] = np.ones(n_node_types)*0.01
types[1]['frequency'] = np.random.rand(n_node_types)

types[2]['type'] = 'low_pass'
types[2]['tau'] = np.random.rand(n_node_types)
types[2]['dt'] = np.ones(n_node_types)*0.01

types[3]['type'] = 'high_pass'
types[3]['tau'] = np.random.rand(n_node_types)*0.01
types[3]['dt'] = np.ones(n_node_types)*0.01

types[4]['type'] = 'moving_average'
types[4]['window_size'] = np.random.randint(1, 4, n_node_types)
types[4]['dt'] = np.ones(n_node_types)*0.01



# i in the receiving node type, j in the sending node type
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
        print(a)
        print(b)
        Fz[i,j,0] = a
        Fz[i,j,1] = b


results = {}
results['Fz'] = Fz
results['connectivity'] = connectivity
results['edge_index'] = edge_index
results['types'] = types
results['activities'] = activities
results['X'] = X

path = '/Users/ruettenv/Documents/code/gnn/data/'
import os
os.makedirs(path, exist_ok=True)
np.save(os.path.join(path, 'results.npy'), results)
#%%



# Visualization
Ftmp = Fz.reshape([n_node_types*n_node_types, -1])
pl.figure(figsize=(10,10))
pl.imshow(Ftmp)
pl.colorbar(shrink=0.5)
# pl.clim([-0.03,0.03])
pl.show()

#%%
Fz.shape
#%%
import matplotlib.pyplot as pl

pl.figure(figsize=(10,10))
pl.imshow(connectivity)
pl.colorbar(shrink=0.5)
pl.clim([-0.03,0.03])
pl.show()
#%%


#%%








#%%