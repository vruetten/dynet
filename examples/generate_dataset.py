#%%
import numpy as np
from network import network
from network.nodes import (
    PoissonNode, FilteredExponentialNode, FilteredOscillatorNode, OscillatorNode
)
from network.utils.plotting import plot_results
from network.utils import saving, kernels
import matplotlib.pyplot as pl
import os

# Parameters for scaling
N = 2  # Number of nodes per kernel (same param and different param)
M = 2  # Number of Poisson nodes
T = 20  # Simulation duration
DT = 0.01  # Time step
np.random.seed(42)

net = network.FlexibleNetwork(dt=DT)

# --- Create M Poisson nodes ---
poisson_nodes = []
for i in range(M):
    node = PoissonNode(f"Poisson_{i+1}", firing_rate=0.3 + 0.1 * i)
    net.add_node(node)
    poisson_nodes.append(node)

# --- Define various kernels and add N nodes for each ---
kernels_to_use = [
    # ("bump", kernels.Kernels.bump_kernel, dict(tau_rise=0.1, tau_sustain=3, tau_decay=0.2, delay=0.1)),
    ("gaussian", kernels.Kernels.gaussian_kernel, dict(sigma=0.2, delay=0.1)),
    # ("alpha", kernels.Kernels.alpha_function, dict(tau=0.3, delay=0.1)),
]

node_indices_by_kernel = []

for kernel_name, kernel_func, kernel_params in kernels_to_use:
    # Same parameter nodes
    kernel, _ = kernel_func(dt=DT, **kernel_params)
    indices = []
    for i in range(N):
        if "delay" in kernel_params:
            node = FilteredExponentialNode(f"{kernel_name}_same_{i+1}", tau=0.3, filter_kernel=kernel, dt=DT, delay=kernel_params["delay"])
        else:
            node = FilteredExponentialNode(f"{kernel_name}_same_{i+1}", tau=0.3, filter_kernel=kernel, dt=DT)
        net.add_node(node)
        indices.append(net.n_nodes - 1)
    # Different parameter nodes (change one param)
    diff_params = kernel_params.copy()
    if 'tau_rise' in diff_params:
        diff_params['tau_rise'] *= 1.5
    if 'sigma' in diff_params:
        diff_params['sigma'] *= 1.5
    if 'tau' in diff_params:
        diff_params['tau'] *= 1.5
    kernel_diff, _ = kernel_func(dt=DT, **diff_params)
    for i in range(N):
        if "delay" in kernel_params:
            node = FilteredExponentialNode(f"{kernel_name}_diff_{i+1}", tau=0.45, filter_kernel=kernel_diff, dt=DT, delay=kernel_params["delay"])
        else:
            node = FilteredExponentialNode(f"{kernel_name}_diff_{i+1}", tau=0.45, filter_kernel=kernel_diff, dt=DT)
        net.add_node(node)
        indices.append(net.n_nodes - 1)
    node_indices_by_kernel.append(indices)

# --- Build connectivity matrix ---
n_total = net.n_nodes
conn = np.zeros((n_total, n_total))

# For each kernel group, connect all its nodes to all Poisson nodes
for indices in node_indices_by_kernel:
    for idx in indices:
        for pidx in range(M):
            conn[idx, pidx] = 1.0

# Optionally, connect Poisson nodes to themselves (or not)
# for pidx in range(M):
#     conn[pidx, pidx] = 1.0

net.set_connectivity(conn)

# --- Simulate ---
duration = T
y, labels, t = net.simulate(duration=duration)

# --- Save results ---
output_dir = "data/generated_dataset"
os.makedirs(output_dir, exist_ok=True)
base_filename = os.path.join(output_dir, "network")
metadata_df, activity_df = saving.save_network_data(net, y, t)
metadata_df.to_csv(f"{base_filename}_metadata.csv", index=False)
activity_df.to_csv(f"{base_filename}_activity.csv", index=False)

# --- Plot results ---
plot_results(t, y, labels, net)

print(f"Simulation completed. Results saved in: {output_dir}")
#%%







