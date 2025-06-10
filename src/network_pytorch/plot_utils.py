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


import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
from matplotlib.patches import Rectangle


def visualize_poisson_activities(activities, title="Neural Activities",
                                 node_names=None, time_axis=None,
                                 style='comprehensive', figsize=(15, 10)):
    """
    Enhanced visualization for Poisson neural activities.

    Args:
        activities: torch.Tensor of shape (n_nodes, n_frames)
        title: Plot title
        node_names: List of node names (optional)
        time_axis: Time values for x-axis (optional)
        style: 'comprehensive', 'heatmap', 'raster', or 'traces'
        figsize: Figure size tuple
    """

    # Convert to numpy if needed
    if torch.is_tensor(activities):
        data = activities.detach().cpu().numpy()
    else:
        data = activities

    n_nodes, n_frames = data.shape

    # Create time axis if not provided
    if time_axis is None:
        time_axis = np.arange(n_frames)

    # Create node names if not provided
    if node_names is None:
        node_names = [f'Node {i + 1}' for i in range(n_nodes)]

    if style == 'comprehensive':
        fig = plt.figure(figsize=figsize)

        ax1 = plt.subplot(2,1,1)
        im = ax1.imshow(data, cmap='plasma', aspect='auto', interpolation='nearest')
        ax1.set_ylabel('nodes')
        ax1.set_xlabel('frames')
        ax1.set_title(f'{title} - activity')

        # cbar = plt.colorbar(im, ax=ax1, shrink=0.8)
        # cbar.set_label('activity')
        if n_nodes <= 20:
            ax1.set_yticks(range(n_nodes))
            ax1.set_yticklabels(node_names, fontsize=8)

        ax5 = plt.subplot(2, 1, 2)
        n_show = min(5, n_nodes)
        colors = plt.cm.Set1(np.linspace(0, 1, n_show))

        for i in range(n_show):
            ax5.plot(time_axis, data[i] + i * 0.5, color=colors[i],
                     linewidth=1.5, label=node_names[i])

        ax5.set_xlabel('frame')
        ax5.set_ylabel('activity (offset)')
        ax5.set_title(f'sample traces (first {n_show} nodes)')
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3)

        plt.tight_layout()



    plt.show()
    return fig


def compare_poisson_activities(activities_list, labels, figsize=(15, 8)):
    """
    Compare multiple Poisson activity matrices side by side.

    Args:
        activities_list: List of activity tensors
        labels: List of labels for each activity matrix
        figsize: Figure size
    """
    n_matrices = len(activities_list)
    fig, axes = plt.subplots(2, n_matrices, figsize=figsize)

    if n_matrices == 1:
        axes = axes.reshape(-1, 1)

    for i, (activities, label) in enumerate(zip(activities_list, labels)):
        if torch.is_tensor(activities):
            data = activities.detach().cpu().numpy()
        else:
            data = activities

        # Heatmap
        im = axes[0, i].imshow(data, cmap='viridis', aspect='auto')
        axes[0, i].set_title(f'{label}\nActivity Heatmap')
        axes[0, i].set_xlabel('Time')
        axes[0, i].set_ylabel('Nodes')
        plt.colorbar(im, ax=axes[0, i], shrink=0.8)

        # Statistics
        node_means = np.mean(data, axis=1)
        axes[1, i].barh(range(len(node_means)), node_means, alpha=0.7)
        axes[1, i].set_title(f'{label}\nMean Activity per Node')
        axes[1, i].set_xlabel('Mean Activity')
        axes[1, i].set_ylabel('Nodes')
        axes[1, i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    return fig


import matplotlib.pyplot as plt
import numpy as np
import torch


def visualize_oscillator_activities(activities, dt=1.0, title="Oscillator Activities",
                                    node_names=None, style='comprehensive', figsize=(15, 10)):
    """
    Simple visualization for oscillatory neural activities.

    Args:
        activities: torch.Tensor of shape (n_nodes, n_frames)
        dt: Time step in seconds
        title: Plot title
        node_names: List of node names (optional)
        style: 'comprehensive', 'heatmap', 'traces', or 'overlay'
        figsize: Figure size tuple
    """

    # Convert to numpy if needed
    if torch.is_tensor(activities):
        data = activities.detach().cpu().numpy()
    else:
        data = activities

    n_nodes, n_frames = data.shape

    # Create time axis
    time_axis = np.arange(n_frames) * dt

    # Create node names if not provided
    if node_names is None:
        node_names = [f'Oscillator {i + 1}' for i in range(n_nodes)]

    if style == 'comprehensive':
        fig = plt.figure(figsize=figsize)

        # 1. Main heatmap
        ax1 = plt.subplot(2,1,1)
        im = ax1.imshow(data, cmap='RdBu_r', aspect='auto', interpolation='bilinear')
        ax1.set_ylabel('oscillator')
        ax1.set_xlabel('frame')

        n_ticks = min(8, n_frames // 10)
        if n_ticks > 1:
            tick_positions = np.linspace(0, n_frames - 1, n_ticks)
            tick_labels = [f'{time_axis[int(pos)]:.1f}' for pos in tick_positions]
            ax1.set_xticks(tick_positions)
            ax1.set_xticklabels(tick_labels)

        # cbar = plt.colorbar(im, ax=ax1, shrink=0.8)
        # cbar.set_label('Activity Level')

        if n_nodes <= 20:
            ax1.set_yticks(range(n_nodes))
            ax1.set_yticklabels(node_names, fontsize=9)

        # 2. Individual traces (stacked)
        ax2 = plt.subplot(2, 1, 2)
        n_show = min(6, n_nodes)
        colors = plt.cm.tab10(np.linspace(0, 1, n_show))

        spacing = (np.max(data) - np.min(data)) * 1.2
        for i in range(n_show):
            offset = i * spacing
            ax2.plot(time_axis, data[i] + offset, color=colors[i],
                     linewidth=1.5, label=node_names[i])

            # Add horizontal line at zero for each trace
            ax2.axhline(y=offset, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)

        ax2.set_xlabel('frame')
        ax2.set_ylabel('activity (stacked)')
        ax2.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()


    plt.show()
    return fig

