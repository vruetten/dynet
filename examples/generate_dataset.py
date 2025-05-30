#%%
import numpy as np
from network import network
from network import nodes
from importlib import reload
from network.utils import plotting
from network.utils import saving, kernels
from network.utils import connectivity
import matplotlib.pyplot as pl
import os
from config import SimulationConfig

if __name__ == '__main__':


    reload(connectivity)
    reload(nodes)
    reload(kernels)
    reload(plotting)
    reload(saving)
    reload(network)

    np.random.seed(0)

    # Parameters for scaling


    config_file = './config/test.yaml'  # Insert path to config file
    config = SimulationConfig.from_yaml(config_file)

    output_dir = config.output_dir
    os.makedirs(output_dir, exist_ok=True)
    verbose = config.verbose


    N_poisson = config.simulation.N_poisson # Number of Poisson nodes

    N_exponential = config.simulation.N_exponential  # Number of exponential nodes
    N_exponential_groups = config.simulation.N_exponential_groups # Number of groups of exponential nodes

    N_oscillator = config.simulation.N_oscillator  # Number of oscillator nodes
    N_oscillator_groups = config.simulation.N_oscillator_groups # Number of groups of oscillator nodes

    N_gaussian_kernel = config.simulation.N_gaussian_kernel  # Number of gaussian kernel nodes
    N_gaussian_kernel_groups = config.simulation.N_gaussian_kernel_groups # Number of groups of gaussian kernel nodes

    N_damped_oscillation = config.simulation.N_damped_oscillation  # Number of damped oscillation nodes
    N_damped_oscillation_groups = config.simulation.N_damped_oscillation_groups # Number of groups of damped oscillation nodes

    N_bump = config.simulation.N_bump  # Number of bump nodes

    N_alpha = config.simulation.N_alpha  # Number of alpha nodes
    N_alpha_groups = config.simulation.N_alpha_groups # Number of groups of alpha nodes
    N_biexponential = config.simulation.N_biexponential  # Number of biexponential nodes

    kernel_duration = config.simulation.kernel_duration

    T = config.simulation.T  # Simulation duration
    DT = config.simulation.DT  # Time step

    if N_poisson > 0:
        poisson_firing_rates = np.linspace(0.05, 0.2, N_poisson)

    if N_exponential > 0:
        # exponential_taus = np.linspace(0.4, 10, N_exponential)*0+0.5
        exponential_tau_groups = np.linspace(0.1, 5, N_exponential_groups)  # smaller the faster the decay
        exponential_taus = np.random.choice(exponential_tau_groups, size=N_exponential)
        # exponential_taus = np.linspace(0.1, 10, N_exponential)
        exponential_taus_A = exponential_taus.copy()
        # exponential_taus = exponential_taus*0 + 1
        exponential_noise_levels = np.linspace(0.01, 0.02, N_exponential)*0 + 0.00

    if N_oscillator > 0:
        oscillator_taus_groups = np.linspace(0.25, 0.35, N_oscillator_groups)
        # *0 + 0.2 # smaller the faster decay
        oscillator_types = np.random.choice(np.arange(N_oscillator_groups), size=N_oscillator)
        oscillator_taus = oscillator_taus_groups[oscillator_types]

        oscillator_noise_levels = np.linspace(0.01, 0.05, N_oscillator)*0
        oscillator_freqs_groups = np.linspace(1, 3, N_oscillator_groups)*0 + 1
        oscillator_freqs = oscillator_freqs_groups[oscillator_types]

    if N_gaussian_kernel > 0:
        gaussian_noise_levels = np.linspace(0.01, 0.05, N_gaussian_kernel)*0 + 0.0
        gaussian_taus = np.linspace(0.1, 0.5, N_gaussian_kernel)*0 + 0.2
        gaussian_delay_groups = np.linspace(0.7, 2.5, N_gaussian_kernel_groups)
        gaussian_delays = np.random.choice(gaussian_delay_groups, size=N_gaussian_kernel)
        gaussian_kernels = [kernels.Kernels.gaussian_kernel(sigma=gaussian_taus[i], delay=gaussian_delays[i], dt=DT, duration=kernel_duration)[0] for i in range(N_gaussian_kernel)]

    if N_damped_oscillation > 0:
        damped_oscillation_taus = np.linspace(0.1, 1, N_damped_oscillation)*0 + 0.1
        damped_oscillation_noise_levels = np.linspace(0.01, 0.05, N_damped_oscillation)*0 + 0.0
        damped_oscillation_freq_groups = np.linspace(0.5, 0.6, N_damped_oscillation_groups)
        damped_oscillation_freqs = np.random.choice(damped_oscillation_freq_groups, size=N_damped_oscillation)

        damped_oscillation_delays = np.linspace(0.1, 0.5, N_damped_oscillation)

        damped_oscillation_kernels = [kernels.Kernels.growing_damped_oscillator(frequency=damped_oscillation_freqs[i], tau_rise=damped_oscillation_taus[i], tau_decay=damped_oscillation_taus[i], delay=damped_oscillation_delays[i], dt=DT, duration=kernel_duration)[0] for i in range(N_damped_oscillation)]

    if N_bump > 0:
        bump_taus = np.linspace(0.1, 0.5, N_bump)
        bump_delays = np.linspace(0.1, 0.5, N_bump)
        bump_noise_levels = np.linspace(0.01, 0.05, N_bump)
        bump_kernels = [kernels.Kernels.bump_kernel(tau_rise=bump_taus[i], tau_sustain=bump_taus[i], tau_decay=bump_taus[i], delay=bump_delays[i], dt=DT, duration=kernel_duration)[0] for i in range(N_bump)]

    if N_alpha > 0:
        alpha_taus_groups = np.linspace(0.1, 0.5, N_alpha_groups)
        alpha_types = np.random.choice(np.arange(N_alpha_groups), size=N_alpha)
        alpha_taus = alpha_taus_groups[alpha_types]
        # alpha_taus_A = alpha_taus.copy()
        alpha_taus = alpha_taus
        alpha_noise_levels = np.linspace(0.01, 0.05, N_alpha)*0
        alpha_delays_groups = np.linspace(0.1, 0.5, N_alpha_groups)
        alpha_delays = alpha_delays_groups[alpha_types]
        alpha_kernels = [kernels.Kernels.alpha_function(tau=alpha_taus[i], delay=alpha_delays[i], dt=DT, duration=kernel_duration)[0] for i in range(N_alpha)]

    if N_biexponential > 0:
        biexponential_taus = np.linspace(0.1, 0.5, N_biexponential)
        biexponential_noise_levels = np.linspace(0.01, 0.05, N_biexponential)
        biexponential_delays = np.linspace(0.1, 0.5, N_biexponential)
        biexponential_kernels = [kernels.Kernels.biexponential(tau_rise=biexponential_taus[i], tau_decay=biexponential_taus[i], delay=biexponential_delays[i], dt=DT, duration=kernel_duration)[0] for i in range(N_biexponential)]


    # plot the kernels

    # for i in range(N_damped_oscillation):
    #     pl.figure()
    #     xax = np.arange(len(damped_oscillation_kernels[i]))*DT
    #     pl.plot(xax, damped_oscillation_kernels[i])
    #     pl.show()

    # for i in range(N_gaussian_kernel):
    #     pl.figure()
    #     xax = np.arange(len(gaussian_kernels[i]))*DT
    #     pl.plot(xax, gaussian_kernels[i])
    #     pl.show()



    net = network.FlexibleNetwork(dt=DT)
    node_types = []

    # --- Create M Poisson nodes ---
    for i in range(N_poisson):
        node = nodes.PoissonNode(f"Poisson_{i+1}", firing_rate=poisson_firing_rates[i])
        net.add_node(node)
        node_types.append("Poisson")
        if verbose:
            print(f"adding poisson node {i+1} of {N_poisson} - total nodes: {net.n_nodes}")

    # --- Create M Exponential nodes ---
    for i in range(N_exponential):
        node = nodes.ExponentialNode("Exponential", tau=exponential_taus[i], noise_level=exponential_noise_levels[i])
        if i == 0:
            node.initial_state = np.random.randn()*0
        else:
            node.initial_state = np.random.randn()*0
        net.add_node(node)
        node_types.append("Exponential")
        if verbose:
            print(f"adding exponential node {i+1} of {N_exponential} - total nodes: {net.n_nodes}")

    # --- Create M Oscillator nodes ---
    for i in range(N_oscillator):
        if i == 0:
            initial_state = np.random.randn()
            initial_state = initial_state*0
        else:
            initial_state = np.random.randn()*0
        node = nodes.OscillatorNode("Oscillator", tau=oscillator_taus[i], freq=oscillator_freqs[i], initial_state=initial_state)
        net.add_node(node)
        node_types.append("Oscillator")
        if verbose:
            print(f"adding oscillator node {i+1} of {N_oscillator} - total nodes: {net.n_nodes}")

    # --- Create M Gaussian kernel nodes ---
    for i in range(N_gaussian_kernel):
        node = nodes.FilteredExponentialNode(f"Gaussian_{i+1}", tau=gaussian_taus[i], filter_kernel=gaussian_kernels[i], dt=DT, noise_level=gaussian_noise_levels[i])
        node.initial_state = np.random.randn()*0
        net.add_node(node)
        node_types.append("Gaussian")
        if verbose:
            print(f"adding gaussian node {i+1} of {N_gaussian_kernel} - total nodes: {net.n_nodes}")


    # --- Create M Damped oscillation nodes ---
    for i in range(N_damped_oscillation):
        node = nodes.FilteredExponentialNode(f"Damped_Oscillation_{i+1}", tau=damped_oscillation_taus[i], filter_kernel=damped_oscillation_kernels[i], dt=DT, noise_level=damped_oscillation_noise_levels[i])
        net.add_node(node)
        node_types.append("Damped_Oscillation")
        if verbose:
            print(f"adding damped oscillation node {i+1} of {N_damped_oscillation} - total nodes: {net.n_nodes}")


    for i in range(N_bump):
        node = nodes.FilteredExponentialNode(f"Bump_{i+1}", tau=bump_taus[i], filter_kernel=bump_kernels[i], dt=DT, noise_level=bump_noise_levels[i])
        net.add_node(node)
        node_types.append("Bump")
        if verbose:
            print(f"adding bump node {i+1} of {N_bump} - total nodes: {net.n_nodes}")


    for i in range(N_alpha):
        node = nodes.FilteredExponentialNode(f"Alpha_{i+1}", tau=alpha_taus[i], filter_kernel=alpha_kernels[i], dt=DT, noise_level=alpha_noise_levels[i])
        net.add_node(node)
        node_types.append("Alpha")
        if verbose:
            print(f"adding alpha node {i+1} of {N_alpha} - total nodes: {net.n_nodes}")


    for i in range(N_biexponential):
        node = nodes.FilteredExponentialNode(f"Biexponential_{i+1}", tau=biexponential_taus[i], filter_kernel=biexponential_kernels[i], dt=DT, noise_level=biexponential_noise_levels[i])
        net.add_node(node)
        node_types.append("Biexponential")
        if verbose:
            print(f"adding biexponential node {i+1} of {N_biexponential} - total nodes: {net.n_nodes}")


    # --- Build connectivity matrix ---
    n_total = net.n_nodes
    if verbose:
        print(f"n_total: {n_total}")

    # build a connectivity matrix based on node_types - all nodes of the same type are connected to one poisson node
    conn = np.zeros((n_total, n_total))
    n_poisson = node_types.count("Poisson")
    n_exponential = node_types.count("Exponential")
    n_damped_oscillation = node_types.count("damped_oscillation")
    n_bump = node_types.count("bump")
    n_gaussian = node_types.count("gaussian")
    n_exponential = node_types.count("exponential")
    n_alpha = node_types.count("alpha")

    poisson_nodes_idx = np.where(np.array(node_types) == "Poisson")[0]
    oscillator_nodes_idx = np.where(np.array(node_types) == "Oscillator")[0]
    exponential_nodes_idx = np.where(np.array(node_types) == "Exponential")[0]
    damped_oscillation_nodes_idx = np.where(np.array(node_types) == "Damped_Oscillation")[0]
    bump_nodes_idx = np.where(np.array(node_types) == "Bump")[0]
    gaussian_nodes_idx = np.where(np.array(node_types) == "Gaussian")[0]
    alpha_nodes_idx = np.where(np.array(node_types) == "Alpha")[0]
    biexponential_nodes_idx = np.where(np.array(node_types) == "Biexponential")[0]
    if verbose:
        print(node_types)

    for i, node_type in enumerate(np.unique(node_types)):
         same_type_nodes = np.where(np.array(node_types) == node_type)[0]
         if node_type == "Poisson":
             pass
         elif node_type == "Oscillator":
             i = 0
             for i in range(3):
                 poisson_node_idx = np.random.choice(poisson_nodes_idx)
                 conn[same_type_nodes[i], poisson_node_idx] = 1
         if node_type != "Poisson":
             # find all nodes of the same type
            for i in range(1):
                oscillator_node_idx = np.random.choice(oscillator_nodes_idx)
                if oscillator_node_idx != same_type_nodes[i]:
                    conn[same_type_nodes[i], oscillator_node_idx] = 1
                poisson_node_idx = np.random.choice(poisson_nodes_idx)
            for i in range(1):
                conn[same_type_nodes[np.random.randint(0, len(same_type_nodes))], poisson_node_idx] = 1
            #
            # Create random stable connectivity matrix
            n_nodes = len(same_type_nodes)
            # M_oscillatory_components = 0
            # oscillation_decay_rate = 10
            # osc_frequencies_b = [3, 0.2, 0.3, 0.4]
            A = np.random.randn(n_nodes, n_nodes)/np.sqrt(n_total)
            A = A - np.diag(np.diag(A))
            conn[np.ix_(same_type_nodes, same_type_nodes)] = A

            # A = A - np.diag(np.sum(A, axis=1))
            # real_eigenvalues = -np.linspace(0.1, 1, n_nodes)
            # if node_type == "Exponential":
            #     real_eigenvalues = -exponential_taus_A
            # else:
            #     real_eigenvalues = -np.linspace(0.1, 1, n_nodes)
            # A, Q, D = connectivity.create_matrix_with_oscillations(n_nodes, M_oscillatory_components=M_oscillatory_components, oscillation_decay_rate=oscillation_decay_rate, osc_frequencies_b = osc_frequencies_b, real_eigenvalues = real_eigenvalues)


    pl.figure()
    pl.imshow(conn)
    pl.colorbar()
    pl.tight_layout()
    pl.savefig(output_dir+"connectivity_matrix.png")
    pl.show()
    pl.close()


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
    pl.imshow(y_norm, extent=[0, T, 0, n_total], cmap="gray", aspect="auto")
    pl.savefig(output_dir+"activity.png")
    # pl.colorbar()
    pl.show()
    pl.close()
    # %%

    # pl.figure()
    # pl.matshow(y_norm[:,:100])
    # pl.show()
    # pl.savefig(output_dir+"y_norm.png")
    # pl.close()
    # #%%



    # --- Save results ---
    # output_dir = "data/generated_dataset"

    base_filename = os.path.join(output_dir, "network")
    metadata_df, activity_df = saving.save_network_data(net, y, t)
    metadata_df.to_csv(f"{base_filename}_metadata.csv", index=False)
    activity_df.to_csv(f"{base_filename}_activity.csv", index=False)
    print(f"Simulation completed. Results saved in: {output_dir}")


    #%%

    ### FORMAT DATA for GNN
    import pandas as pd
    # merge metadata_df and activity_df on "node_id" and index of activity_df
    merged_df = pd.merge(activity_df, metadata_df, left_on="node_id", right_on="node_id", how="left")
    mylist = [group for _, group in merged_df.groupby("time")]

    # %%

    #%%











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