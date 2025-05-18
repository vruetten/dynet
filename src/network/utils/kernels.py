import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple

class Kernels:
    """
    Commonly used temporal kernels in systems neuroscience and physiology.
    Each method returns a 1D numpy array representing the kernel in time.
    """

    @staticmethod
    def alpha_function(tau: float, dt: float, duration: Optional[float] = None, normalize: bool = True, delay: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Alpha function: k(t) = (t/τ) * exp(1 - t/τ) * H(t)
        Models post-synaptic currents or simple filter dynamics.
        """
        if duration is None:
            duration = 5 * tau + delay

        t = np.arange(0, duration, dt)
        shifted_t = np.maximum(t - delay, 0.0)
        kernel = (shifted_t / tau) * np.exp(1 - shifted_t / tau)
        if normalize:
            kernel /= np.sum(np.abs(kernel))
        return kernel, t

    @staticmethod
    def exponential_decay(tau: float, dt: float, duration: Optional[float] = None, normalize: bool = True, delay: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Exponential decay kernel: k(t) = exp(-t/τ)
        """
        if duration is None:
            duration = 5 * tau + delay

        t = np.arange(0, duration, dt)
        shifted_t = np.maximum(t - delay, 0.0)
        kernel = np.exp(-shifted_t / tau)
        if normalize:
            kernel /= np.max(kernel)
        return kernel, t

    @staticmethod
    def damped_oscillator(frequency: float, tau: float, dt: float, duration: Optional[float] = None, phase_delay: float = 0.0, delay: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Damped sinusoid: k(t) = (1/ω_d) * exp(-γ t) * sin(ω_d t + φ)
        Models resonant synapses or cortical oscillations.
        """
        if duration is None:
            duration = 5 * tau + delay

        t = np.arange(0, duration, dt)
        shifted_t = np.maximum(t - delay, 0.0)
        gamma = 1 / tau
        omega = 2 * np.pi * frequency
        omega_d = np.sqrt(max(omega**2 - gamma**2, 1e-12))
        phi = -omega_d * phase_delay

        kernel = (1 / omega_d) * np.exp(-gamma * shifted_t) * np.sin(omega_d * shifted_t + phi)
        return kernel, t

    @staticmethod
    def biexponential(tau_rise: float, tau_decay: float, dt: float, duration: Optional[float] = None, normalize: bool = True, delay: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Biexponential function: k(t) = exp(-t/τ_decay) - exp(-t/τ_rise)
        Models calcium indicators, synaptic currents with distinct rise/decay.
        """
        if duration is None:
            duration = 5 * max(tau_rise, tau_decay) + delay

        t = np.arange(0, duration, dt)
        shifted_t = np.maximum(t - delay, 0.0)
        kernel = np.exp(-shifted_t / tau_decay) - np.exp(-shifted_t / tau_rise)
        if normalize:
            kernel /= np.sum(np.abs(kernel))
        return kernel, t

    @staticmethod
    def delay_kernel(delay: float, dt: float, duration: Optional[float] = None, normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pure delay kernel: k[n] = δ[n - delay_steps]
        Used to shift a signal in time by delay seconds.
        """
        delay_steps = int(round(delay / dt))
        if duration is None:
            duration = (delay_steps + 2) * dt

        n = int(duration / dt)
        kernel = np.zeros(n)
        if delay_steps < n:
            kernel[delay_steps] = 1.0
        if normalize:
            kernel /= np.sum(np.abs(kernel))
        return kernel, t

    @staticmethod
    def growing_damped_oscillator(frequency: float, tau_rise: float = 1.0, tau_decay: float = 1.0, dt: float = 0.001, duration: Optional[float] = None, phase_delay: float = 0.0, delay: float = 0.0, normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Oscillatory kernel with both rise and decay: k(t) = [exp(-t/τ_decay) - exp(-t/τ_rise)] * sin(ω t + φ)
        Useful for modeling oscillations that ramp up and decay.
        
        Parameters:
        - frequency: frequency of oscillation (Hz)
        - tau_rise: rising time constant (s)
        - tau_decay: decay time constant (s)
        - dt: time step (s)
        - duration: kernel duration (s). Default is 5 × max(tau_rise, tau_decay)
        - phase_delay: phase shift as time (s)
        - delay: additional delay before onset (s)
        - normalize: if True, normalize peak to 1
        """
        if duration is None:
            duration = 5 * max(tau_rise, tau_decay) + delay

        t = np.arange(0, duration, dt)
        # make a gaussian envelope
        envelope = np.exp(-(t-delay)**2 / (2*tau_decay**2))
        omega = 2 * np.pi * frequency
        phi = -omega * phase_delay
        kernel = envelope * np.sin(omega * t + phi)

        if normalize and np.max(np.abs(kernel)) > 0:
            kernel /= np.sum(np.abs(kernel))
            # kernel /= kernel.shape[0]
            # kernel = kernel/np.sum(kernel)

        return kernel, t

    @staticmethod
    def bump_kernel(tau_rise: float, tau_sustain: float, tau_decay: float, dt: float, delay: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a kernel with exponential rise, sustained period, and exponential decay.
        
        Parameters:
        -----------
        tau_rise : float
            Time constant for the rising phase (seconds)
        tau_sustain : float
            Duration of the sustained period (seconds)
        tau_decay : float
            Time constant for the decay phase (seconds)
        dt : float
            Time step (seconds)
        delay : float
            Initial delay before the kernel starts (seconds)
        """
        # Calculate time points for each phase
        t_rise = np.arange(0, 3 * tau_rise, dt)  # Rise phase
        t_sustain = np.arange(0, tau_sustain, dt)  # Sustain phase
        t_decay = np.arange(0, 3 * tau_decay, dt)  # Decay phase
        
        # Generate each phase
        rise = 1 - np.exp(-t_rise / tau_rise)  # Exponential rise to 1
        sustain = np.ones_like(t_sustain)  # Constant at 1
        decay = np.exp(-t_decay / tau_decay)  # Exponential decay from 1
        
        # Combine phases
        kernel = np.concatenate([rise, sustain, decay])
        t = np.arange(0, len(kernel) * dt, dt)
        
        # Add delay if specified
        if delay > 0:
            delay_steps = int(delay / dt)
            kernel = np.pad(kernel, (delay_steps, 0), mode='constant')
            t = np.arange(0, len(kernel) * dt, dt)
            
        return kernel, t

    @staticmethod
    def gaussian_kernel(sigma: float, dt: float, duration: float = None, delay: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a Gaussian kernel.
        
        Parameters:
        -----------
        sigma : float
            Standard deviation of the Gaussian (seconds)
        dt : float
            Time step (seconds)
        duration : float, optional
            Total duration of the kernel. If None, uses 6*sigma
        delay : float
            Initial delay before the kernel starts (seconds)
        """
        if duration is None:
            duration = 6 * sigma  # Cover 3 standard deviations on each side
            
        t = np.arange(0, duration, dt)
        t_centered = t - duration/2  # Center the Gaussian
        kernel = np.exp(-0.5 * (t_centered / sigma)**2)
        
        # Normalize to have area = 1
        kernel = kernel / (np.sum(kernel) * dt)
        
        if delay > 0:
            delay_steps = int(delay / dt)
            kernel = np.pad(kernel, (delay_steps, 0), mode='constant')
            t = np.arange(0, len(kernel) * dt, dt)
            
        return kernel, t
