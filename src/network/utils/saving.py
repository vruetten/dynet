import pandas as pd
import numpy as np
import json
from typing import Dict, Any

def get_node_parameters(node) -> Dict[str, Any]:
    """Extract parameters from a node for metadata."""
    params = {
        'name': node.name,
        'type': node.__class__.__name__,
        'initial_state': node.initial_state.tolist() if hasattr(node, 'initial_state') else [0.0],
        'noise_level': node.noise_level
    }
    
    # Add specific parameters based on node type
    if hasattr(node, 'tau'):
        params['tau'] = node.tau
    if hasattr(node, 'freq'):
        params['freq'] = node.freq
    if hasattr(node, 'firing_rate'):
        params['firing_rate'] = node.firing_rate
    if hasattr(node, 'filter_kernel') and node.filter_kernel is not None:
        params['filter_kernel_length'] = len(node.filter_kernel)
    
    return params

def save_network_data(network, y, t, shape_descriptor_std=0.1):
    """Create metadata and activity dataframes."""
    # Create metadata dataframe
    metadata = []
    for i, node in enumerate(network.nodes):
        params = get_node_parameters(node)
        # Add coordinates (using random positions for now)
        params['x'] = np.random.normal(0, shape_descriptor_std)
        params['y'] = np.random.normal(0, shape_descriptor_std)
        params['z'] = np.random.normal(0, shape_descriptor_std)
        params['node_id'] = i
        metadata.append(params)
    
    metadata_df = pd.DataFrame(metadata)
    
    # Create activity dataframe
    activity_data = []
    for i, node in enumerate(network.nodes):
        for t_idx, t_val in enumerate(t[1:]):
            activity_data.append({
                'node_id': i,
                'time': t_val,
                'activity': y[i, t_idx],
                'activity_derivative': y[i, t_idx]-y[i, t_idx-1],
                # 'activity_derivative': network.derivatives[i, t_idx] if hasattr(network, 'derivatives') else 0.0
            })
    
    activity_df = pd.DataFrame(activity_data)
    
    return metadata_df, activity_df

def save_network_to_files(network, y, t, base_filename):
    """Save network data to files."""
    # Create dataframes
    metadata_df, activity_df = save_network_data(network, y, t)
    
    # Save to CSV files
    metadata_df.to_csv(f'{base_filename}_metadata.csv', index=False)
    activity_df.to_csv(f'{base_filename}_activity.csv', index=False)
    
    # Save network structure to JSON
    network_info = {
        'dt': network.dt,
        'connectivity': network.connectivity.tolist(),
        'n_nodes': network.n_nodes
    }
    
    with open(f'{base_filename}_network.json', 'w') as f:
        json.dump(network_info, f, indent=2) 