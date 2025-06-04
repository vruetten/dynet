import numpy as np
import pytest
from src.network.utils.z_transform import ZTransformFilter
from src.network.z_transform_nodes import (
    ZTransformLowPassNode,
    ZTransformBandPassNode,
    ZTransformMovingAverageNode
)

def test_low_pass_filter():
    # Create a low-pass filter
    dt = 0.001
    tau = 0.1
    filter_instance = ZTransformFilter.create_low_pass(tau, dt)
    
    # Generate test signal (step function)
    t = np.arange(0, 1, dt)
    x = np.zeros_like(t)
    x[t > 0.1] = 1.0
    
    # Process signal
    y = np.zeros_like(x)
    for i, xi in enumerate(x):
        y[i] = filter_instance.process(xi)
    
    # Check steady-state value
    assert np.isclose(y[-1], 1.0, atol=1e-2)
    
    # Check rise time (time to reach 63% of final value)
    rise_time_idx = np.where(y > 0.63)[0][0]
    rise_time = t[rise_time_idx]
    assert np.isclose(rise_time, tau, rtol=0.1)

def test_band_pass_filter():
    # Create a band-pass filter
    dt = 0.001
    center_freq = 10.0  # Hz
    bandwidth = 2.0    # Hz
    filter_instance = ZTransformFilter.create_band_pass(center_freq, bandwidth, dt)
    
    # Generate test signal (sum of two sinusoids)
    t = np.arange(0, 1, dt)
    f1 = 5.0   # Hz (should be attenuated)
    f2 = 10.0  # Hz (should pass through)
    x = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t)
    
    # Process signal
    y = np.zeros_like(x)
    for i, xi in enumerate(x):
        y[i] = filter_instance.process(xi)
    
    # Check that f1 component is attenuated
    f1_power = np.abs(np.fft.rfft(x)[int(f1 * len(t) * dt)])**2
    f1_power_filtered = np.abs(np.fft.rfft(y)[int(f1 * len(t) * dt)])**2
    assert f1_power_filtered < 0.1 * f1_power
    
    # Check that f2 component is preserved
    f2_power = np.abs(np.fft.rfft(x)[int(f2 * len(t) * dt)])**2
    f2_power_filtered = np.abs(np.fft.rfft(y)[int(f2 * len(t) * dt)])**2
    assert f2_power_filtered > 0.5 * f2_power

def test_moving_average_filter():
    # Create a moving average filter
    window_size = 10
    filter_instance = ZTransformFilter.create_moving_average(window_size)
    
    # Generate test signal (random noise)
    x = np.random.randn(1000)
    
    # Process signal
    y = np.zeros_like(x)
    for i, xi in enumerate(x):
        y[i] = filter_instance.process(xi)
    
    # Check that variance is reduced
    assert np.var(y) < np.var(x)
    
    # Check that the filter is causal
    for i in range(window_size):
        assert not np.isclose(y[i], np.mean(x[:i+1]))

def test_z_transform_low_pass_node():
    # Create a low-pass filtered node
    dt = 0.001
    tau = 0.1
    filter_tau = 0.05
    node = ZTransformLowPassNode("test_node", tau, filter_tau, dt=dt)
    
    # Test step response
    t = np.arange(0, 1, dt)
    x = np.zeros_like(t)
    x[t > 0.1] = 1.0
    
    # Simulate node
    y = np.zeros_like(x)
    state = np.zeros(node.n_state)
    for i, xi in enumerate(x):
        dy = node.get_derivative(t[i], state, xi)
        state += dy * dt
        y[i] = state[0]
    
    # Check steady-state value
    assert np.isclose(y[-1], 1.0, atol=1e-2)

def test_z_transform_band_pass_node():
    # Create a band-pass filtered node
    dt = 0.001
    tau = 0.1
    freq = 5.0
    filter_center_freq = 10.0
    filter_bandwidth = 2.0
    node = ZTransformBandPassNode(
        "test_node", tau, freq, filter_center_freq, filter_bandwidth, dt=dt
    )
    
    # Generate test signal
    t = np.arange(0, 1, dt)
    f1 = 5.0   # Hz (should be attenuated)
    f2 = 10.0  # Hz (should pass through)
    x = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t)
    
    # Simulate node
    y = np.zeros_like(x)
    state = np.zeros(node.n_state)
    for i, xi in enumerate(x):
        dy = node.get_derivative(t[i], state, xi)
        state += dy * dt
        y[i] = state[0]
    
    # Check frequency response
    f1_power = np.abs(np.fft.rfft(x)[int(f1 * len(t) * dt)])**2
    f1_power_filtered = np.abs(np.fft.rfft(y)[int(f1 * len(t) * dt)])**2
    assert f1_power_filtered < 0.1 * f1_power

def test_z_transform_moving_average_node():
    # Create a moving average filtered node
    dt = 0.001
    tau = 0.1
    window_size = 10
    node = ZTransformMovingAverageNode("test_node", tau, window_size, dt=dt)
    
    # Generate test signal
    t = np.arange(0, 1, dt)
    x = np.random.randn(len(t))
    
    # Simulate node
    y = np.zeros_like(x)
    state = np.zeros(node.n_state)
    for i, xi in enumerate(x):
        dy = node.get_derivative(t[i], state, xi)
        state += dy * dt
        y[i] = state[0]
    
    # Check that variance is reduced
    assert np.var(y) < np.var(x) 