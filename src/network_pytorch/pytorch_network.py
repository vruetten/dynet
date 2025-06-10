#%%


import torch
import numpy as np
import matplotlib.pyplot as plt
from plot_utils import *
from network_utils import *


device='cpu'


# Example usage
np.random.seed(0)

n_node_types = 5
n_node_types_per_type = [2, 3, 4, 5, 6]
n_nodes = sum(n_node_types_per_type)
filter_length_max = 5

ids = np.arange(n_nodes)

types = np.repeat(np.arange(n_node_types), n_node_types_per_type)


# types = np.array([0,1,2,0,1,2])
activities = np.random.randn(n_nodes)

X = np.column_stack((ids, types, activities)) # id, type, activity at t0
connectivity_filling_factor = 0.9
connectivity, edge_index = create_connectivity(types, n_nodes, connectivity_filling_factor)

Fz = np.zeros((n_node_types, n_node_types, 2, filter_length_max))
Fz.shape


node_type_names = ['poisson', 'oscillator', 'low_pass', 'high_pass', 'moving_avg']

# create filter bank for each node type
types = {}
for i in range(n_node_types):
    types[i] = {}

types[0]['type'] = node_type_names[0]
types[0]['dt'] = np.ones(n_node_types)*0.01
types[0]['rate'] = np.random.rand(n_node_types)*0.01

types[1]['type'] = node_type_names[1]
types[1]['dt'] = np.ones(n_node_types)*0.01
types[1]['frequency'] = np.random.rand(n_node_types)

types[2]['type'] = node_type_names[2]
types[2]['tau'] = np.random.rand(n_node_types)
types[2]['dt'] = np.ones(n_node_types)*0.01

types[3]['type'] = node_type_names[3]
types[3]['tau'] = np.random.rand(n_node_types)*0.01
types[3]['dt'] = np.ones(n_node_types)*0.01

types[4]['type'] = node_type_names[4]
types[4]['window_size'] = np.random.randint(1, 4, n_node_types)
types[4]['dt'] = np.ones(n_node_types)*0.01


# i in the receiving node type, j in the sending node type
for i in range(n_node_types):
    if types[i]['type'] in ['poisson', 'oscillator']:
        continue
    for j in range(n_node_types):
        if types[i]['type'] == 'low_pass':
            coefs = create_low_pass(types[i]['tau'][j], types[i]['dt'][j])
        elif types[i]['type'] == 'high_pass':
            coefs = create_high_pass(types[i]['tau'][j], types[i]['dt'][j])
        elif types[i]['type'] == 'moving_average':
            coefs = create_moving_average(types[i]['window_size'][j])
        else:
            continue
        a, b = process_filter_coefficients(coefs, filter_length_max, device)
        print(a)
        print(b)
        Fz[i,j,0] = a
        Fz[i,j,1] = b


results = {}
results['Fz'] = Fz
results['connectivity'] = connectivity
results['edge_index'] = edge_index
results['types'] = types
results['activities'] = activities
results['X'] = X

path = '../graphs_data/'
import os
os.makedirs(path, exist_ok=True)
np.save(os.path.join(path, 'results.npy'), results)
#%%

connectivity_np = connectivity.numpy()



plot_connectivity_with_types(
    connectivity_np, X, types, n_node_types, n_node_types_per_type,
    node_type_names=node_type_names, clim=[-0.03, 0.03]
)

plot_filter_coefficients(Fz, n_node_types, node_type_names)








#%%