import matplotlib.pyplot as plt
import numpy as np
from typing import List
from network.nodes import (
    PoissonNode, FilteredExponentialNode, FilteredOscillatorNode,OscillatorNode, ExponentialNode
)

def plot_results(t: np.ndarray, y: np.ndarray, labels: List[str], network) -> None:
    """Plot the results of a network simulation."""
    n_nodes = len(network.nodes)
    fig, axes = plt.subplots(n_nodes, 2, figsize=(15, 3*n_nodes))
    if n_nodes == 1:
        axes = axes.reshape(1, -1)

    for i, (node, title) in enumerate(zip(network.nodes, labels)):
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

        # Plot kernel if it exists
        if hasattr(node, 'filter_kernel') and node.filter_kernel is not None:
            kernel = node.filter_kernel
            kernel_t = np.arange(len(kernel)) * node.dt
            axes[state_idx, 1].plot(kernel_t, kernel)
            axes[state_idx, 1].set_title(f'Kernel for {title}')
            axes[state_idx, 1].set_xlabel('Time')
            axes[state_idx, 1].set_ylabel('Amplitude')
        else:
            axes[state_idx, 1].set_visible(False)

    # Set x labels only for the bottom row
    for ax in axes[-1]:
        ax.set_xlabel('Time')

    plt.tight_layout()
    plt.show()