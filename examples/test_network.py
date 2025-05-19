
#%%
import numpy as np
from network import network
from network import nodes
from network.utils.plotting import plot_results
from network.utils import saving, plotting, kernels
from importlib import reload
import matplotlib.pyplot as pl
reload(saving)
reload(kernels)
reload(plotting)
reload(nodes)
reload(network)


np.random.seed(123)
# === Demo Usage ===


dt = 0.01
net = network.FlexibleNetwork(dt=dt)

firing_rate = 0.3
firing_rate = 1
net.add_node(nodes.PoissonNode("Poisson", firing_rate=firing_rate))
tau = 0.1
# net.add_node(nodes.ExponentialNode("Exp1", tau=tau))

# Add a node with bump kernel (rise, sustain, decay)
# tau_rise = 0.1
# tau_sustain = 1
# tau_decay = 0.1
# kernel, t_filter = kernels.Kernels.bump_kernel(
#     tau_rise=tau_rise,
#     tau_sustain=tau_sustain,
#     tau_decay=tau_decay,
#     dt=dt
# )
# noise_level = 0
# net.add_node(nodes.FilteredExponentialNode("BumpNode", tau=0.3, filter_kernel=kernel, dt=dt, noise_level=noise_level))

# Add a node with gaussian kernel
# sigma = 0.2
# kernel, t_filter = kernels.Kernels.gaussian_kernel(
#     sigma=sigma,
#     dt=dt
# )
# net.add_node(nodes.FilteredExponentialNode("GaussianNode", tau=0.3, filter_kernel=kernel, dt=dt))


# ####### WAVELET KERNEL
# frequency = 1.0
# tau_decay = 0.9
# tmax = 10
# delay = 0
# kernel, t_filter = kernels.Kernels.growing_damped_oscillator(frequency=frequency, dt=dt, duration=tmax, delay=delay, tau_decay=tau_decay)
# tau = 0.2
# net.add_node(nodes.FilteredExponentialNode("FiltExp", tau=tau, filter_kernel=kernel, dt=dt, noise_level=0.01))

# # add a node with alpha function kernel
# tau = 0.3
# delay = 0
# kernel, t_filter = kernels.Kernels.alpha_function(tau=tau, dt=dt, delay=delay)
# time_constant = 0.3 # exponential decay
# net.add_node(nodes.FilteredExponentialNode("FiltExp", tau=tau, filter_kernel=kernel, dt=dt, noise_level=0.01))

tau = 6
freq = 1.0
initial_state = 1

net.add_node(nodes.OscillatorNode("Osc1", tau=tau, freq=freq, initial_state=initial_state))

# net.add_node(nodes.FilteredOscillatorNode("FiltOsc", tau=2.0, freq=5.0, filter_kernel=np.sin(np.linspace(0, np.pi, 100)), dt=dt))

n_nodes = net.n_nodes
conn = np.zeros((n_nodes, n_nodes))
conn[1:,0] = 0.0
# conn[0,0] = 0
# conn[-1,:] =0.3

net.set_connectivity(conn)

duration = 8
method = 'rk2'
y, labels, t = net.simulate(duration=duration, method=method)

plotting.plot_results(t, y, labels, net)



metadata_df, activity_df = saving.save_network_data(net, y, t,)
# %%
