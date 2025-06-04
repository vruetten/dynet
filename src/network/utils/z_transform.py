import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass

@dataclass
class FilterCoefficients:
    """Stores the coefficients for a digital filter in the form:
    y[n] = b[0]*x[n] + b[1]*x[n-1] + ... + b[nb]*x[n-nb]
           - a[1]*y[n-1] - a[2]*y[n-2] - ... - a[na]*y[n-na]
    """
    b: np.ndarray  # Numerator coefficients
    a: np.ndarray  # Denominator coefficients

class ZTransformFilter:
    """Implements various digital filters using z-transform difference equations."""
    
    def __init__(self, coefficients: 'FilterCoefficients'):
        """Initialize the filter with given coefficients.

        Args:
            coefficients: An object with attributes 'b' and 'a'.
                          'b' is the array of numerator coefficients [b_0, b_1, ...].
                          'a' is the array of denominator coefficients [a_0, a_1, ...].
                          a_0 cannot be zero.
        """
        b_in = np.array(coefficients.b, dtype=float)
        a_in = np.array(coefficients.a, dtype=float)

        if not a_in.size:
            raise ValueError("Denominator 'a' coefficients array cannot be empty.")
        if a_in[0] == 0:
            raise ValueError("The first denominator coefficient a_0 cannot be zero.")

        # Normalize coefficients
        a0 = a_in[0]
        self.norm_b_coeffs = b_in / a0
        # The feedback coefficients are a_in[1:] normalized by a0
        self.norm_a_feedback_coeffs = a_in[1:] / a0

        # Store coefficients in a way that's easy to reference if needed,
        # or just use the normalized versions directly.
        # For clarity, we'll use the normalized versions.
        # self.coefficients = FilterCoefficients(b=self.norm_b_coeffs, 
        #                                        a=self.norm_a_feedback_coeffs) # Storing normalized versions

        self.reset()
    
    def reset(self):
        """Reset the filter state (internal buffers)."""
        # x_buffer stores past inputs for the b_k * x[n-k] sum
        self.x_buffer = np.zeros(len(self.norm_b_coeffs))
        # y_buffer stores past outputs for the a_k * y[n-k] sum (k>=1)
        self.y_buffer = np.zeros(len(self.norm_a_feedback_coeffs))
    
    def process(self, x: float) -> float:
        """Process a single input sample through the filter.

        Args:
            x: Input sample x[n]

        Returns:
            Filtered output sample y[n]
        """
        # Update input buffer (stores x[n-M], ..., x[n-1], x[n])
        self.x_buffer = np.roll(self.x_buffer, -1)
        self.x_buffer[-1] = x

        # Calculate the feedforward term: sum(b_k * x[n-k])
        # self.x_buffer[::-1] provides [x[n], x[n-1], ..., x[n-M]]
        feedforward_sum = np.dot(self.norm_b_coeffs, self.x_buffer[::-1])

        # Calculate the feedback term: sum(a_k * y[n-k]) for k=1 to N
        # self.y_buffer stores [y[n-N], ..., y[n-2], y[n-1]]
        # self.y_buffer[::-1] provides [y[n-1], y[n-2], ..., y[n-N]]
        feedback_sum = 0.0
        if self.y_buffer.size > 0: # Only compute if there are feedback terms
            feedback_sum = np.dot(self.norm_a_feedback_coeffs, self.y_buffer[::-1])

        # Calculate new output: y[n] = (1/a0) * (feedforward - feedback_sum_for_a1_to_aN)
        # Since norm_b_coeffs and norm_a_feedback_coeffs are already divided by a0,
        # the (1/a0) scaling is already incorporated.
        new_output = feedforward_sum - feedback_sum

        # Update output buffer for the next iteration
        if self.y_buffer.size > 0:
            self.y_buffer = np.roll(self.y_buffer, -1)
            self.y_buffer[-1] = new_output

        return new_output
    
    @staticmethod
    def create_low_pass(tau: float, dt: float) -> 'ZTransformFilter':
        """Create a first-order low-pass filter.
        
        Args:
            tau: Time constant in seconds
            dt: Time step in seconds
            
        Returns:
            ZTransformFilter instance
        """
        alpha = np.exp(-dt / tau)
        b = np.array([1 - alpha])
        a = np.array([1, -alpha])
        return ZTransformFilter(FilterCoefficients(b, a))
    
    @staticmethod
    def create_band_pass(center_freq: float, bandwidth: float, dt: float) -> 'ZTransformFilter':
        """Create a second-order band-pass filter.
        
        Args:
            center_freq: Center frequency in Hz
            bandwidth: Bandwidth in Hz
            dt: Time step in seconds
            
        Returns:
            ZTransformFilter instance
        """
        omega_0 = 2 * np.pi * center_freq * dt
        r = np.exp(-np.pi * bandwidth * dt)
        
        b = np.array([1 - r**2])
        a = np.array([1, -2 * r * np.cos(omega_0), r**2])
        return ZTransformFilter(FilterCoefficients(b, a))
    
    @staticmethod
    def create_moving_average(window_size: int) -> 'ZTransformFilter':
        """Create a moving average filter.
        
        Args:
            window_size: Number of samples to average
            
        Returns:
            ZTransformFilter instance
        """
        b = np.ones(window_size) / window_size
        a = np.array([1])
        return ZTransformFilter(FilterCoefficients(b, a))
    
    def get_frequency_response(self, freqs: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate the frequency response of the filter.
        
        Args:
            freqs: Array of frequencies in Hz
            dt: Time step in seconds
            
        Returns:
            Tuple of (magnitude, phase) responses
        """
        z = np.exp(2j * np.pi * freqs * dt)
        
        # Calculate numerator and denominator
        num = np.polyval(self.coefficients.b[::-1], z)
        den = np.polyval(self.coefficients.a[::-1], z)
        
        # Calculate frequency response
        H = num / den
        
        return np.abs(H), np.angle(H) 