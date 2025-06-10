"""
Plotting utilities for neural network connectivity visualization.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_connectivity_with_types(connectivity, X, types, n_node_types, n_node_types_per_type,
                                 node_type_names=None, figsize=(12, 10), clim=None):
    """
    Plot connectivity matrix with node types labeled on Y-axis.

    Args:
        connectivity: Connectivity matrix (n_nodes x n_nodes)
        X: Node data array with columns [id, type, activity]
        types: Dictionary containing node type information
        n_node_types: Number of different node types
        n_node_types_per_type: Number of nodes per type
        node_type_names: List of node type names (optional)
        figsize: Figure size tuple
        clim: Color limits for the plot
    """
    n_nodes = connectivity.shape[0]

    # Default node type names if not provided
    if node_type_names is None:
        node_type_names = [types[i]['type'] for i in range(n_node_types)]

    # Create type labels for each node
    type_labels = [node_type_names[int(X[i, 1])] for i in range(n_nodes)]

    plt.figure(figsize=figsize)
    im = plt.imshow(connectivity, cmap='bwr', aspect='equal')
    plt.title('connectivity matrix', fontsize=18)
    plt.colorbar(im, shrink=0.5, label='weight')

    if clim is not None:
        plt.clim(clim)

    # Set custom tick labels
    plt.xticks(range(n_nodes), [f"N{i}" for i in range(n_nodes)], rotation=45, ha='right')
    plt.yticks(range(n_nodes), [f"N{i} ({type_labels[i]})" for i in range(n_nodes)])

    plt.xlabel('source', fontsize=18)
    plt.ylabel('target', fontsize=18)

    # Add grid for better readability
    # plt.grid(True, alpha=0.1, linestyle='--')

    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.show()


def plot_filter_coefficients(Fz, n_node_types, node_type_names=None, figsize=(10, 10)):
    """
    Plot filter coefficients matrix.

    Args:
        Fz: Filter coefficients tensor (n_node_types x n_node_types x 2 x filter_length_max)
        n_node_types: Number of different node types
        node_type_names: List of node type names (optional)
        figsize: Figure size tuple
    """
    if node_type_names is None:
        node_type_names = [f'Type {i}' for i in range(n_node_types)]

    # Reshape for visualization
    Ftmp = Fz.reshape([n_node_types * n_node_types, -1])

    plt.figure(figsize=figsize)
    plt.imshow(Ftmp, cmap='bwr')
    plt.title('filter coefficients', fontsize=18)
    plt.colorbar(shrink=0.5, label='coefficient value')
    plt.xlabel('filter coefficient index', fontsize=18)
    plt.ylabel('node type pair (receiver × sender)', fontsize=18)

    # Add tick labels for node type pairs
    pair_labels = []
    for i in range(n_node_types):
        for j in range(n_node_types):
            pair_labels.append(f'{node_type_names[i]}←{node_type_names[j]}')

    plt.yticks(range(len(pair_labels)), pair_labels)
    plt.tight_layout()
    plt.show()

