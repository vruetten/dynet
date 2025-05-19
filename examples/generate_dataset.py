#%%
import numpy as np
from network import network
from network import nodes
from importlib import reload
from network.utils import plotting
from network.utils import saving, kernels
import matplotlib.pyplot as pl
import os
reload(nodes)
reload(kernels)
reload(plotting)
reload(saving)
reload(network)

# Parameters for scaling
N = 1  # Number of nodes per kernel (same param and different param)
N_poisson = 1  # Number of Poisson nodes
N_exponential = 5  # Number of exponential nodes
N_oscillator = 1  # Number of oscillator nodes
T = 60  # Simulation duration
DT = 0.01  # Time step
np.random.seed(42)

net = network.FlexibleNetwork(dt=DT)

node_types = []

# --- Create M Poisson nodes ---
for i in range(N_poisson):
    node = nodes.PoissonNode(f"Poisson_{i+1}", firing_rate=0.1 + 0.04 * i)
    net.add_node(node)
    node_types.append("Poisson")
# for i in range(N_exponential):
#     node = nodes.ExponentialNode("Exponential", tau=0.3)
#     net.add_node(node)
#     node_types.append("Exponential")
for i in range(N_oscillator):
    node = nodes.OscillatorNode("Oscillator", tau=5+0.01*i, freq=1, initial_state=1)
    net.add_node(node)
    node_types.append("Oscillator")

# --- Define various kernels and add N nodes for each ---
kernels_to_use = [
    # ("damped_oscillation", kernels.Kernels.growing_damped_oscillator, dict(frequency=1, tau_rise=0.2, tau_decay=2, delay=3), dict(noise_level=0.01, tau=0.3)),
    # ("bump", kernels.Kernels.bump_kernel, dict(tau_rise=0.1, tau_sustain=3, tau_decay=0.2, delay=0.1), dict(noise_level=0.01, tau=0.3)),
    # ("gaussian", kernels.Kernels.gaussian_kernel, dict(sigma=0.3,
    #  delay=0), dict(noise_level=0.01, tau=0.3)),
    ("exponential", kernels.Kernels.exponential_decay, dict(tau=0.3, delay=0.1), dict(noise_level=0.01, tau=3)),
    # ("alpha", kernels.Kernels.alpha_function, dict(tau=0.3, delay=0.1), dict(noise_level=0.01, tau=0.3)),
]


for kernel_name, kernel_func, kernel_params, node_params in kernels_to_use:
    # Same parameter nodes
    kernel, xax = kernel_func(dt=DT, **kernel_params)
    indices = []
    for i in range(1):
        node = nodes.FilteredExponentialNode(f"{kernel_name}_same_{i+1}", tau=node_params["tau"], filter_kernel=kernel, dt=DT, noise_level=node_params["noise_level"])
        node.initial_state = np.array([0.3])
        net.add_node(node)
        indices.append(net.n_nodes - 1)
        node_types.append(kernel_name)
    for i in range(N):
        diff_params = kernel_params.copy()
        diff_node_params = node_params.copy()
        if "noise_level" in diff_node_params:
            diff_node_params["noise_level"] += 0.03
        if "sigma" in diff_params:
            diff_params["sigma"] += 0.01
        if 'tau' in diff_params:
            diff_params['tau'] += 0.01
        if "delay" in diff_params:
            diff_params['delay'] += 0.5
        if "sigma" in diff_params:
            diff_params['sigma'] += 0.1
        kernel_diff, _ = kernel_func(dt=DT, **diff_params)
        node = nodes.FilteredExponentialNode(f"{kernel_name}_diff_{i+1}", tau=diff_node_params["tau"], filter_kernel=kernel_diff, dt=DT, noise_level=diff_node_params["noise_level"])
        net.add_node(node)
        node_types.append(kernel_name)

# --- Build connectivity matrix ---
n_total = net.n_nodes

# build a connectivity matrix based on node_types - all nodes of the same type are connected to one poisson node
conn = np.zeros((n_total, n_total))
n_poisson = node_types.count("Poisson")
n_exponential = node_types.count("Exponential")
n_damped_oscillation = node_types.count("damped_oscillation")
n_bump = node_types.count("bump")
n_gaussian = node_types.count("gaussian")
n_exponential = node_types.count("exponential")
n_alpha = node_types.count("alpha")
for i, node_type in enumerate(np.unique(node_types)):
     print(node_type)
     if node_type == "Poisson" or node_type == "Oscillator":
         pass
     else:
         # find all nodes of the same type
         same_type_nodes = np.where(np.array(node_types) == node_type)[0]
         #  print(same_type_nodes)
         poisson_nodes = np.where(np.array(node_types) == "Poisson")[0]
         poisson_node = np.random.choice(poisson_nodes)
         #  print(poisson_node)
         poisson_or_oscillator_nodes = np.where(np.logical_or(
             np.array(node_types) == "Poisson",
             np.array(node_types) == "Oscillator"
         ))[0]
         poisson_or_oscillator_node = np.random.choice(poisson_or_oscillator_nodes)
         for node in same_type_nodes:
             conn[node, poisson_or_oscillator_node] = 1.0

# pl.figure()
# pl.imshow(conn)
# pl.colorbar()
# pl.show()





# conn = np.zeros((n_total, n_total))
# conn[1,0] = 1.0
# # For each kernel group, connect all its nodes to all Poisson nodes
# for indices in node_indices_by_kernel:
#     for idx in indices:
#         for pidx in range(M):
#             conn[idx, pidx] = 1.0

# theta = np.pi/2
# cos_theta = np.cos(theta)
# sin_theta = np.sin(theta)
# rotation_matrix2d = np.array([[cos_theta, -sin_theta],
#                               [sin_theta, cos_theta]])
# conn = np.zeros((n_total, n_total))
# conn[2:4,0] = 1.0
# conn[3:5, 3:5] = rotation_matrix2d



# Optionally, connect Poisson nodes to themselves (or not)
# for pidx in range(M):
#     conn[pidx, pidx] = 1.0

net.set_connectivity(conn)

# --- Simulate ---
duration = T
y, labels, t = net.simulate(duration=duration)

# --- Plot results ---
plotting.plot_results(t, y, labels, net)
#%%
# make a raster plot of the activity
# raster_plot = plotting.raster_plot(t, y, labels, net)
# raster_plot.savefig(f"{base_filename}_raster.png")
y_norm = (y-y.mean(axis=0))/y.std(axis=0)
pl.figure()
pl.imshow(y_norm, extent=[0, T, 0, n_total], cmap="gray")
# pl.colorbar()
pl.show()
#%%
# --- Save results ---
output_dir = "data/generated_dataset"
os.makedirs(output_dir, exist_ok=True)
base_filename = os.path.join(output_dir, "network")
metadata_df, activity_df = saving.save_network_data(net, y, t)
metadata_df.to_csv(f"{base_filename}_metadata.csv", index=False)
activity_df.to_csv(f"{base_filename}_activity.csv", index=False)



print(f"Simulation completed. Results saved in: {output_dir}")


#%%

### FORMAT DATA
import pandas as pd
# merge metadata_df and activity_df on "node_id" and index of activity_df
merged_df = pd.merge(activity_df, metadata_df, left_on="node_id", right_on="node_id", how="left")
mylist = [group for _, group in merged_df.groupby("time")]









