import numpy as np
from typing import List, Optional, Tuple, Union, Dict
import pandas as pd
import json

# === Base Node Class ===
class Node:
    """
    Abstract base class for all dynamic nodes.
    Each node evolves over time based on internal state and weighted inputs from other nodes.
    """
    def __init__(self, name: str, n_state: int = 1, delay: float = 0.0, noise_level: float = 0.0, initial_state: Optional[Union[float, np.ndarray]] = None):
        self.name = name
        self.n_state = n_state
        self.delay = delay
        self.noise_level = noise_level
        if initial_state is None:
            self.initial_state = np.zeros(n_state)
        else:
            if isinstance(initial_state, (int, float)):
                self.initial_state = np.full(n_state, initial_state)
            else:
                assert len(initial_state) == n_state, f"Initial state must have length {n_state}"
                self.initial_state = np.array(initial_state)

    def get_derivative(self, t: float, y: np.ndarray, input_val: float) -> np.ndarray:
        raise NotImplementedError("Each node must implement its own dynamics.")

# === Exponential Node ===
class ExponentialNode(Node):
    def __init__(self, name: str, tau: float, **kwargs):
        super().__init__(name, n_state=1, **kwargs)
        self.tau = tau

    def get_derivative(self, t: float, y: np.ndarray, input_val: float) -> np.ndarray:
        # if input_val >1e-1:
            # print(f"input_val: {input_val}, t: {t}")
        return np.array([(-y[0] / self.tau + input_val)])

# === Oscillator Node ===
class OscillatorNode(Node):
    def __init__(self, name: str, tau: float, freq: float, **kwargs):
        super().__init__(name, n_state=2, **kwargs)
        self.tau = tau
        self.freq = freq

    def get_derivative(self, t: float, y: np.ndarray, input_val: float) -> np.ndarray:
        gamma = 1 / self.tau
        omega = 2 * np.pi * self.freq
        return np.array([y[1] , -2 * gamma * y[1] - omega**2 * y[0] + input_val])

# === FilteredNode Mixin ===
class FilteredNode:
    def __init__(self, filter_kernel: Optional[np.ndarray] = None, dt: float = 0.001):
        self.filter_kernel = filter_kernel
        self.dt = dt
        self.buffer = None
        if filter_kernel is not None:
            self.buffer = np.zeros(len(filter_kernel))

    def reset_buffer(self):
        if self.filter_kernel is not None:
            self.buffer = np.zeros(len(self.filter_kernel))

    def update_buffer(self, new_input: float):
        if self.buffer is not None:
            self.buffer = np.roll(self.buffer, -1)
            self.buffer[-1] = new_input

    def get_filtered_input(self) -> float:
        if self.buffer is None or self.filter_kernel is None:
            return 0.0
        return np.dot(self.filter_kernel[::-1], self.buffer)

# === Filtered Exponential Node ===
class FilteredExponentialNode(ExponentialNode, FilteredNode):
    def __init__(self, name: str, tau: float, filter_kernel: Optional[np.ndarray] = None, dt: float = 0.001, **kwargs):
        ExponentialNode.__init__(self, name, tau, **kwargs)
        FilteredNode.__init__(self, filter_kernel, dt)

    def get_derivative(self, t: float, y: np.ndarray, input_val: float) -> np.ndarray:
        self.update_buffer(input_val)
        filtered_input = self.get_filtered_input()
        return super().get_derivative(t, y, filtered_input)

# === Filtered Oscillator Node ===
class FilteredOscillatorNode(OscillatorNode, FilteredNode):
    def __init__(self, name: str, tau: float, freq: float, filter_kernel: Optional[np.ndarray] = None, dt: float = 0.001, **kwargs):
        OscillatorNode.__init__(self, name, tau, freq, **kwargs)
        FilteredNode.__init__(self, filter_kernel, dt)

    def get_derivative(self, t: float, y: np.ndarray, input_val: float) -> np.ndarray:
        self.update_buffer(input_val)
        filtered_input = self.get_filtered_input()
        return super().get_derivative(t, y, filtered_input)

# === Stochastic Poisson Node ===
class PoissonNode(Node):
    def __init__(self, name: str, firing_rate: float, **kwargs):
        super().__init__(name, n_state=1, **kwargs)
        self.firing_rate = firing_rate
        self.tau = 0.01  # Decay time constant

    def get_derivative(self, t: float, y: np.ndarray, input_val: float) -> np.ndarray:
        # Check if we should spike
        if np.random.rand() < self.firing_rate * self.dt:
            return np.array([(1.0 - y[0]) / self.dt])  # Set to 1.0 immediately
        else:
            return np.array([-y[0] / self.tau])  # Decay to 0
