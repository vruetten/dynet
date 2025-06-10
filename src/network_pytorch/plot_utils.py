"""
Plotting utilities for neural network connectivity visualization.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
import torch

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

    fig = plt.figure(figsize=figsize)
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

    return fig


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

    fig = plt.figure(figsize=figsize)
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

    return fig




def visualize_poisson_activities(activities,
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
        ax1.set_ylabel('nodes', fontsize=18)
        ax1.set_xlabel('frame', fontsize=18)


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

        ax5.set_xlabel('frame', fontsize=18)
        ax5.set_ylabel('activity (offset)', fontsize=18)
        ax5.set_title(f'sample traces (first {n_show} nodes)', fontsize=18)
        ax5.grid(True, alpha=0.3)

        plt.tight_layout()

    return fig



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
        ax1.set_xlabel('time')

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

        ax2.set_xlabel('time')
        ax2.set_ylabel('activity (stacked)')
        ax2.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()


    plt.show()
    return fig


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation


# Assuming x_list is your list of arrays with length 100
# Each element in x_list should be a 2D array where column 6 contains the activities

def plot_neuron_activities(x_list, n_node_types_per_type, dt=0.01):
    """
    Plot neuron activities from column 6 of x_list over time

    Parameters:
    x_list: list of numpy arrays, each containing neuron data
    dt: time step (default 0.01)
    """

    # Extract column 6 activities from all time frames
    activities = []
    for x in x_list:
        activities.append(x[:, 6])  # Column 6 activities

    activities = np.array(activities)  # Shape: (n_frames, n_neurons)
    n_frames, n_neurons = activities.shape

    excitation_neurons =  activities[:, 0:n_node_types_per_type[0]+n_node_types_per_type[1]]
    activities =  activities[:, n_node_types_per_type[0]+n_node_types_per_type[1]:]


    time = np.arange(n_frames) * dt
    time = time[:activities.shape[0]]  # Ensure time matches activities length
    fig = plt.figure(figsize=(15,15))

    ax1 = plt.subplot(3, 1, 1)
    im = ax1.imshow(excitation_neurons.T, aspect='auto', cmap='viridis',
                    extent=[time[0], time[-1], 0, n_neurons])
    ax1.set_xlabel('time',fontsize=18)
    ax1.set_ylabel('excitation neurons',fontsize=18)

    ax2 = plt.subplot(3, 1, 2)
    for i in range(min(10, n_neurons)):  # Plot first 10 neurons to avoid clutter
        ax2.plot(time, activities[:, i], alpha=0.7, label=f'Neuron {i}')
    ax2.set_xlabel('time',fontsize=18)
    ax2.set_ylabel('activity',fontsize=18)
    plt.xlim([time[0], time[-1]])
    ax2.grid(True, alpha=0.3)

    ax3 = plt.subplot(3, 1, 3)
    im = ax3.imshow(activities.T, aspect='auto', cmap='viridis',
                    extent=[time[0], time[-1], 0, n_neurons])
    ax3.set_xlabel('time',fontsize=18)
    ax3.set_ylabel('neuron',fontsize=18)

    plt.tight_layout()
    return fig


def plot_activity_evolution(x_list, neuron_indices=None, dt=0.01):
    """
    Plot activity evolution for specific neurons

    Parameters:
    x_list: list of numpy arrays
    neuron_indices: list of neuron indices to plot (default: first 5)
    dt: time step
    """

    activities = np.array([x[:, 6] for x in x_list])
    n_frames, n_neurons = activities.shape
    time = np.arange(n_frames) * dt

    if neuron_indices is None:
        neuron_indices = list(range(min(5, n_neurons)))

    plt.figure(figsize=(12, 8))

    for i, idx in enumerate(neuron_indices):
        plt.subplot(len(neuron_indices), 1, i + 1)
        plt.plot(time, activities[:, idx], 'b-', linewidth=1.5)
        plt.ylabel(f'Neuron {idx}\nActivity')
        plt.grid(True, alpha=0.3)

        if i == 0:
            plt.title('Individual Neuron Activity Evolution')
        if i == len(neuron_indices) - 1:
            plt.xlabel('Time')

    plt.tight_layout()
    return plt.gcf()


def create_activity_animation(x_list, dt=0.01, interval=100):
    """
    Create an animated plot showing activity evolution

    Parameters:
    x_list: list of numpy arrays
    dt: time step
    interval: animation interval in milliseconds
    """

    activities = np.array([x[:, 6] for x in x_list])
    n_frames, n_neurons = activities.shape

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Setup for line plot
    line, = ax1.plot([], [], 'b-', linewidth=2)
    ax1.set_xlim(0, n_neurons)
    ax1.set_ylim(np.min(activities), np.max(activities))
    ax1.set_xlabel('Neuron Index')
    ax1.set_ylabel('Activity')
    ax1.set_title('Current Frame Activities')
    ax1.grid(True, alpha=0.3)

    # Setup for time series
    time = np.arange(n_frames) * dt
    ax2.set_xlim(time[0], time[-1])
    ax2.set_ylim(np.min(activities), np.max(activities))
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Mean Activity')
    ax2.set_title('Population Mean Over Time')
    ax2.grid(True, alpha=0.3)

    mean_line, = ax2.plot([], [], 'r-', linewidth=2)
    current_time_line = ax2.axvline(x=0, color='black', linestyle='--', alpha=0.7)

    def animate(frame):
        # Update current frame activities
        line.set_data(range(n_neurons), activities[frame, :])

        # Update time series up to current frame
        current_time = time[:frame + 1]
        current_mean = np.mean(activities[:frame + 1, :], axis=1)
        mean_line.set_data(current_time, current_mean)
        current_time_line.set_xdata([time[frame]])

        ax1.set_title(f'Frame {frame} - Time: {time[frame]:.3f}')

        return line, mean_line, current_time_line

    anim = FuncAnimation(fig, animate, frames=n_frames, interval=interval, blit=True)
    return fig, anim

