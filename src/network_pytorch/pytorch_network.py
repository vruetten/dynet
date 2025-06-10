#%%


import torch
import numpy as np
import matplotlib.pyplot as plt
from plot_utils import *
from network_utils import *
from tqdm import trange
import os


if __name__ == '__main__':
    device='cpu'

    path = '../graphs_data/'
    os.makedirs(path, exist_ok=True)


    n_node_types = 5
    node_type_names = ['poisson', 'oscillator', 'low_pass', 'high_pass', 'moving_avg']
    n_node_types_per_type = [5, 5, 4, 5, 6]
    n_nodes = sum(n_node_types_per_type)
    filter_length_max = 5
    n_frames = 100

    ids = np.arange(n_nodes)
    types = np.repeat(np.arange(n_node_types), n_node_types_per_type)

    # types = np.array([0,1,2,0,1,2])
    activities = np.random.randn(n_nodes)

    connectivity_filling_factor = 0.9
    connectivity, edge_index = create_connectivity(types, n_nodes, connectivity_filling_factor)
    # X = np.column_stack((ids, types, activities)) # id, type, activity at t0
    plot_connectivity_with_types(
        connectivity.numpy(), np.column_stack((ids, types, activities)), types, n_node_types, n_node_types_per_type,
        node_type_names=node_type_names, clim=[-0.03, 0.03]
    )

    Fz = create_filter_bank(n_node_types, node_type_names, filter_length_max, device)
    plot_filter_coefficients(Fz, n_node_types, node_type_names)

    poisson_activities = create_poisson_activities(n_node_types_per_type[0], n_frames, device)
    visualize_poisson_activities(poisson_activities, style='comprehensive')

    oscillator_activities = create_oscillator_activities(n_nodes=n_node_types_per_type[1], n_frames=n_frames, dt=1.0, amplitude_range=(0.5, 1.5), frequency_range = (0.01, 0.1), phase_randomize=True, device=device)
    visualize_oscillator_activities(activities=oscillator_activities, dt=1, style='comprehensive')



    N1 = torch.arange(n_nodes, dtype=torch.float32, device=device)
    X1 = get_equidistant_points(n_nodes, device)
    V1 = torch.zeros_like(X1)
    T1 = torch.tensor(types, dtype=torch.float32, device=device)
    H1 = torch.zeros_like(X1)
    x_list = []
    y_list = []

    for it in trange(n_frames):

        x = torch.concatenate((N1[:,None], X1, V1, T1[:,None], H1), dim=1)

        pos = np.argwhere(T1 == 0)
        if len(pos) > 0:
            pos = pos.flatten()
            x[pos, 6] = poisson_activities[:, it]
        pos = np.argwhere(T1 == 1)
        if len(pos) > 0:
            pos = pos.flatten()
            x[pos, 6] = oscillator_activities[:, it]











#%%