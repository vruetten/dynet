import matplotlib.pyplot as plt
import numpy as np
from typing import List
from network.nodes import (
    PoissonNode, FilteredExponentialNode, FilteredOscillatorNode,OscillatorNode, ExponentialNode
)

def plot_results(t: np.ndarray, y: np.ndarray, labels: List[str], network: 'FlexibleNetwork'):
    """Plot the simulation results."""
    n_states = y.shape[0]
    fig, axes = plt.subplots(n_states, 2, figsize=(8, 2 * n_states), sharex='col')
    
    # Plot each node's state and derivative

    for i, node in enumerate(network.nodes):
        state_idx = i
        if isinstance(node, PoissonNode):
            title = f'{node.name}: Poisson (rate={node.firing_rate:.1f}Hz)'
        elif isinstance(node, OscillatorNode):
            title = f'{node.name}: Oscillator (τ={node.tau:.2f}s, f={node.freq:.1f}Hz)'
        elif isinstance(node, ExponentialNode):
            title = f'{node.name}: Exponential (τ={node.tau:.2f}s)'
        else:
            title = f'{node.name}: State {i}'
            
        # print(f"title: {title}")
        # Plot state
        axes[state_idx, 0].plot(t, y[state_idx], label=f'State {i}')
        axes[state_idx, 0].set_title(title)
        axes[state_idx, 0].set_ylabel('Activity')
        axes[state_idx, 0].legend()

    axes[-1, 0].set_xlabel('Time (s)')
    axes[-1, 1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.show()