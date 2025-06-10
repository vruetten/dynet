#%%


import torch
import numpy as np
import matplotlib.pyplot as plt
import torch_geometric.data as data

from tqdm import trange
import os

from plot_utils import *
from utils import *
from PDE_Z1 import *
from config import *

if __name__ == '__main__':
    device='cpu'

    path = '../graphs_data/'
    os.makedirs(path, exist_ok=True)


    config_file = '../config/net_0.yaml'
    config = NeuronConfig.from_yaml(config_file)
    print(config.pretty())

    n_node_types = config.n_node_types
    node_type_names =  config.node_type_names
    n_node_types_per_type = config.n_node_types_per_type
    n_nodes = sum(n_node_types_per_type)
    filter_length_max = config.filter_length_max
    n_frames = config.n_frames
    dt = config.dt

    folder_name = config.folder_name

    path = '../graphs_data/' + folder_name
    os.makedirs(path, exist_ok=True)

    ids = np.arange(n_nodes)
    types = np.repeat(np.arange(n_node_types), n_node_types_per_type)

    # types = np.array([0,1,2,0,1,2])
    activities = np.random.randn(n_nodes)

    connectivity_filling_factor = 0.9
    connectivity, edge_index = create_connectivity(types, n_nodes, connectivity_filling_factor)
    fig = plot_connectivity_with_types(connectivity.numpy(), np.column_stack((ids, types, activities)), types, n_node_types, n_node_types_per_type,node_type_names=node_type_names, clim=[-0.03, 0.03])
    plt.savefig(path + '/connectivity.png', dpi=300)
    plt.close(fig)

    Fz = create_filter_bank(n_node_types, node_type_names, filter_length_max, device)
    fig = plot_filter_coefficients(Fz, n_node_types, node_type_names)
    plt.savefig(path + '/filter_coefficients.png', dpi=300)
    plt.close(fig)

    if n_node_types_per_type[0]>0:
        poisson_activities = create_poisson_activities(n_node_types_per_type[0], n_frames, device)
        fig = visualize_poisson_activities(poisson_activities, style='comprehensive')
        plt.savefig(path + '/poisson_activities.png', dpi=300)
        plt.close(fig)

    if n_node_types_per_type[1]>0:
        oscillator_activities = create_oscillator_activities(n_nodes=n_node_types_per_type[1], n_frames=n_frames, dt=dt, amplitude_range=(0.5, 1.5), frequency_range = (0.01, 0.1), phase_randomize=True, device=device)
        fig = visualize_oscillator_activities(activities=oscillator_activities, dt=1, style='comprehensive')
        plt.savefig(path + '/poisson_activities.png', dpi=300)
        plt.close(fig)


    N1 = torch.arange(n_nodes, dtype=torch.float32, device=device)
    X1 = get_equidistant_points(n_nodes, device)
    V1 = torch.zeros_like(X1)
    T1 = torch.tensor(types, dtype=torch.float32, device=device)
    H1 = torch.zeros_like(X1)
    x_list = []
    y_list = []

    model_generator = PDE_Z1(aggr_type='add', W=connectivity, Fz=Fz, g=5E-2).to(device)

    for it in trange(n_frames):

        x = torch.concatenate((N1[:,None], X1, V1, T1[:,None], H1), dim=1)

        if n_node_types_per_type[0] > 0:
            pos = np.argwhere(T1 == 0)
            if len(pos) > 0:
                pos = pos.flatten()
                x[pos, 6] = poisson_activities[:, it]
        if n_node_types_per_type[1] > 0:
            pos = np.argwhere(T1 == 1)
            if len(pos) > 0:
                pos = pos.flatten()
                x[pos, 6] = oscillator_activities[:, it]

        if it > filter_length_max:

            dataset = data.Data(x=x, pos=x[:, 1:3], edge_index=edge_index)
            y =model_generator(dataset)

            x_list.append(x.detach().cpu().numpy())
            y_list.append(y.detach().cpu().numpy())

            H1[:, 1] = y.squeeze()
            H1[:, 0] = H1[:, 0] + H1[:, 1] * dt


    fig = plot_neuron_activities(x_list=x_list, n_node_types_per_type=n_node_types_per_type, dt=dt)
    plt.savefig(path + '/neuron_activities.png', dpi=300)
    plt.show()















#%%