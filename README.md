# Neural Network Simulator

A Python package for simulating neural networks with various node types and kernels.

## Features

- Multiple node types:
  - Poisson nodes for spike generation
  - Exponential nodes for decay dynamics
  - Oscillator nodes for periodic behavior
  - Filtered nodes with various kernels
  - Bump kernel nodes
  - Gaussian kernel nodes

- Flexible network architecture:
  - Customizable connectivity
  - Support for delays
  - Noise injection
  - State tracking and derivatives

- Visualization and data saving:
  - Plot states and derivatives
  - Save network data in CSV and JSON formats
  - Metadata tracking for nodes

## Installation

```bash
pip install -e .
```

## Usage

```python
from src.network.network import FlexibleNetwork
from src.network.nodes import PoissonNode, FilteredExponentialNode
from src.utils.plotting import plot_results

# Create network
net = FlexibleNetwork(dt=0.001)

# Add nodes
input_node = PoissonNode(name="Input", firing_rate=10.0)
net.add_node(input_node)

filtered_node = FilteredExponentialNode(name="Filtered", tau=0.1)
net.add_node(filtered_node)

# Set connectivity
connectivity = np.zeros((2, 2))
connectivity[1, 0] = 1.0  # Input connects to filtered node
net.set_connectivity(connectivity)

# Simulate
y, labels, t = net.simulate(duration=2.0)

# Plot results
plot_results(t, y, labels, net)
```

## Project Structure

```
neural_network_simulator/
├── src/
│   ├── network/
│   │   ├── __init__.py
│   │   ├── network.py
│   │   ├── nodes.py
│   │   └── kernels.py
│   └── utils/
│       ├── __init__.py
│       ├── plotting.py
│       └── saving.py
├── examples/
│   └── test_network.py
├── tests/
├── data/
├── setup.py
└── README.md
```

## License

MIT License 