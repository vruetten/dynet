# Z-Transform Implementation for Neural Network Simulator

## Overview
This document outlines the implementation of z-transform based filtering for the neural network simulator. The z-transform provides a powerful way to describe discrete-time systems and implement various types of filters efficiently.

## Mathematical Background

### Z-Transform Basics
The z-transform of a discrete-time signal x[n] is defined as:
X(z) = Σ x[n]z^(-n) for n = -∞ to ∞

The z-transform has several important properties:
1. Linearity: Z{ax[n] + by[n]} = aX(z) + bY(z)
2. Time shifting: Z{x[n-k]} = z^(-k)X(z)
3. Convolution: Z{x[n] * y[n]} = X(z)Y(z)

### Filter Transfer Functions and Difference Equations

#### 1. First-Order Low-Pass Filter
The continuous-time transfer function of a first-order low-pass filter is:
H(s) = 1/(1 + sτ)

Using the bilinear transform (s = 2(1-z^(-1))/(dt(1+z^(-1)))), we get:
H(z) = (1-α)/(1-αz^(-1))

Where:
- α = exp(-dt/τ)
- dt is the sampling time
- τ is the time constant

The corresponding difference equation is:
y[n] = (1-α)x[n] + αy[n-1]

This can be rewritten in standard form:
y[n] = b₀x[n] + a₁y[n-1]
where:
- b₀ = 1-α
- a₁ = -α

#### 2. Second-Order Band-Pass Filter
The continuous-time transfer function of a second-order band-pass filter is:
H(s) = (2ζω₀s)/(s² + 2ζω₀s + ω₀²)

Using the bilinear transform and some algebraic manipulation, we get:
H(z) = (1-r²)/(1-2rcos(ω₀)z^(-1) + r²z^(-2))

Where:
- r = exp(-πBW·dt) is the pole radius
- ω₀ = 2πf₀·dt is the normalized center frequency
- BW is the bandwidth in Hz
- f₀ is the center frequency in Hz

The corresponding difference equation is:
y[n] = (1-r²)x[n] + 2rcos(ω₀)y[n-1] - r²y[n-2]

This can be rewritten in standard form:
y[n] = b₀x[n] + a₁y[n-1] + a₂y[n-2]
where:
- b₀ = 1-r²
- a₁ = -2rcos(ω₀)
- a₂ = r²

#### 3. Moving Average Filter
The moving average filter is a special case of a finite impulse response (FIR) filter. Its transfer function is:
H(z) = (1/N)(1-z^(-N))/(1-z^(-1))

The corresponding difference equation is:
y[n] = (1/N)Σx[n-k] for k=0 to N-1

This can be rewritten in standard form:
y[n] = (1/N)(x[n] + x[n-1] + ... + x[n-N+1])

### Implementation Details

#### Filter Coefficient Structure
All filters are implemented using the standard form:
y[n] = b₀x[n] + b₁x[n-1] + ... + bₙx[n-nb]
       - a₁y[n-1] - a₂y[n-2] - ... - aₙy[n-na]

Where:
- b = [b₀, b₁, ..., bₙ] are the numerator coefficients
- a = [1, a₁, a₂, ..., aₙ] are the denominator coefficients

#### Frequency Response Calculation
The frequency response of a filter is calculated using:
H(f) = B(e^(j2πf·dt))/A(e^(j2πf·dt))

Where:
- B(z) = b₀ + b₁z^(-1) + ... + bₙz^(-n)
- A(z) = 1 + a₁z^(-1) + ... + aₙz^(-n)
- f is the frequency in Hz
- dt is the sampling time

The magnitude response is |H(f)| and the phase response is ∠H(f).

## Implementation Plan

1. Create a new `ZTransformFilter` class that will:
   - Store filter coefficients
   - Maintain filter state
   - Implement common filter types
   - Provide methods to convert between continuous-time and discrete-time parameters

2. Add new node types:
   - `ZTransformFilteredNode` - Base class for nodes with z-transform filtering
   - `ZTransformLowPassNode` - Node with first-order low-pass filtering
   - `ZTransformBandPassNode` - Node with second-order band-pass filtering
   - `ZTransformMovingAverageNode` - Node with moving average filtering

3. Implementation Steps:
   a. Create filter coefficient calculation utilities
   b. Implement the ZTransformFilter class
   c. Create new node classes
   d. Add unit tests
   e. Update documentation

## Usage Example

```python
# Create a low-pass filtered node
node = ZTransformLowPassNode(
    name="filtered_node",
    tau=0.1,  # Time constant
    dt=0.001  # Time step
)

# Create a band-pass filtered node
node = ZTransformBandPassNode(
    name="bandpass_node",
    center_freq=10.0,  # Hz
    bandwidth=2.0,     # Hz
    dt=0.001          # Time step
)
```

## Benefits of Z-Transform Implementation

1. **Computational Efficiency**: Direct implementation of difference equations is more efficient than convolution-based filtering
2. **Numerical Stability**: Better control over filter stability through pole placement
3. **Flexibility**: Easy to implement various filter types with different characteristics
4. **Real-time Processing**: Suitable for real-time signal processing applications

## Next Steps

1. Implement the core ZTransformFilter class
2. Create the new node types
3. Add comprehensive unit tests
4. Update documentation with examples
5. Add performance benchmarks 