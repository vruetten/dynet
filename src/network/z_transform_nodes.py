import numpy as np
from typing import Optional, Union
from .nodes import Node
from .utils.z_transform import ZTransformFilter

class ZTransformFilteredNode(Node):
    """Base class for nodes with z-transform filtering."""
    
    def __init__(self, name: str, filter_instance: ZTransformFilter, **kwargs):
        """Initialize a node with z-transform filtering.
        
        Args:
            name: Node name
            filter_instance: ZTransformFilter instance
            **kwargs: Additional arguments passed to Node
        """
        super().__init__(name, **kwargs)
        self.filter = filter_instance
    
    def reset_buffer(self):
        """Reset the filter state."""
        self.filter.reset()
    
    def get_derivative(self, t: float, state: np.ndarray, input_val: Union[float, np.ndarray]) -> np.ndarray:
        """Get the derivative with filtered input.
        
        Args:
            t: Current time
            state: Current state
            input_val: Input value
            
        Returns:
            Derivative of the state
        """
        filtered_input = self.filter.process(input_val)
        return self._get_filtered_derivative(t, state, filtered_input)
    
    def _get_filtered_derivative(self, t: float, state: np.ndarray, filtered_input: float) -> np.ndarray:
        """Get the derivative with pre-filtered input. Must be implemented by subclasses.
        
        Args:
            t: Current time
            state: Current state
            filtered_input: Pre-filtered input value
            
        Returns:
            Derivative of the state
        """
        raise NotImplementedError("Subclasses must implement _get_filtered_derivative")

class ZTransformLowPassNode(ZTransformFilteredNode):
    """Node with first-order low-pass filtering."""
    
    def __init__(self, name: str, tau: float, filter_tau: float, **kwargs):
        """Initialize a low-pass filtered node.
        
        Args:
            name: Node name
            tau: Node time constant
            filter_tau: Filter time constant
            **kwargs: Additional arguments passed to Node
        """
        filter_instance = ZTransformFilter.create_low_pass(filter_tau, kwargs.get('dt', 0.001))
        super().__init__(name, filter_instance, **kwargs)
        self.tau = tau
    
    def _get_filtered_derivative(self, t: float, y: np.ndarray, filtered_input: float) -> np.ndarray:
        return np.array([(-y[0] + filtered_input) / self.tau])

class ZTransformBandPassNode(ZTransformFilteredNode):
    """Node with second-order band-pass filtering."""
    
    def __init__(self, name: str, tau: float, freq: float, 
                 filter_center_freq: float, filter_bandwidth: float, **kwargs):
        """Initialize a band-pass filtered node.
        
        Args:
            name: Node name
            tau: Node time constant
            freq: Node frequency
            filter_center_freq: Filter center frequency
            filter_bandwidth: Filter bandwidth
            **kwargs: Additional arguments passed to Node
        """
        filter_instance = ZTransformFilter.create_band_pass(
            filter_center_freq, filter_bandwidth, kwargs.get('dt', 0.001))
        super().__init__(name, filter_instance, **kwargs)
        self.tau = tau
        self.freq = freq
        self.n_state = 2
        self.initial_state = np.zeros(self.n_state)
    
    def _get_filtered_derivative(self, t: float, y: np.ndarray, filtered_input: float) -> np.ndarray:
        gamma = 1 / self.tau
        omega = 2 * np.pi * self.freq
        dy = np.zeros(2)
        dy[0] = y[1]  # dx/dt = v
        dy[1] = -2 * gamma * y[1] - omega**2 * y[0] + omega**2 * filtered_input
        return dy

class ZTransformMovingAverageNode(ZTransformFilteredNode):
    """Node with moving average filtering."""
    
    def __init__(self, name: str, tau: float, window_size: int, **kwargs):
        """Initialize a moving average filtered node.
        
        Args:
            name: Node name
            tau: Node time constant
            window_size: Number of samples to average
            **kwargs: Additional arguments passed to Node
        """
        filter_instance = ZTransformFilter.create_moving_average(window_size)
        super().__init__(name, filter_instance, **kwargs)
        self.tau = tau
    
    def _get_filtered_derivative(self, t: float, y: np.ndarray, filtered_input: float) -> np.ndarray:
        return np.array([(-y[0] + filtered_input) / self.tau]) 