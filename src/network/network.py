import numpy as np
from typing import List, Tuple
from .nodes import Node, PoissonNode, OscillatorNode
from importlib import reload
from .nodes import FilteredNode



# === Flexible Network ===
class FlexibleNetwork:
    def __init__(self, dt: float):
        self.dt = dt
        self.nodes: List[Node] = []
        self.connectivity: np.ndarray = np.zeros((0, 0))
        self.n_nodes = 0
        self.derivatives = []

    def add_node(self, node: Node):
        node.dt = self.dt
        if isinstance(node, FilteredNode):
            node.reset_buffer()
        self.nodes.append(node)
        n = len(self.nodes)
        self.n_nodes = n
        self.connectivity = np.pad(self.connectivity, ((0, 1), (0, 1)), mode='constant')

    def set_connectivity(self, matrix: np.ndarray):
        assert matrix.shape == (len(self.nodes), len(self.nodes)), "Connectivity shape mismatch."
        self.connectivity = matrix

    def noise_fn(self, i, y_next):
        y_idx = 0
        for node in self.nodes:
            print(f"node: {node.name}, y_idx: {y_idx}")
            if isinstance(node, PoissonNode):
                y_idx += 0
            elif isinstance(node, OscillatorNode):
                if node.noise_level > 0:
                    y_next[y_idx] += node.noise_level * np.random.randn() * np.sqrt(self.dt)
                y_idx += 2
            else:
                if node.noise_level > 0:
                    noise = node.noise_level * np.random.randn() * np.sqrt(self.dt)
                    # print(f"noise: {noise}, y_idx: {y_idx}, y_next: {y_next.shape}")
                    y_next[y_idx] += noise
                y_idx += 1
        return y_next

    def simulate(self, duration: float, method: str = 'rk2') -> Tuple[np.ndarray, List[str], np.ndarray]:
        t = np.arange(0, duration, self.dt)
        state_sizes = [node.n_state for node in self.nodes]
        total_state = sum(state_sizes)
        y = np.zeros((total_state, len(t)))
        indices = np.cumsum([0] + state_sizes)
        self.derivatives = []

        # First, generate all Poisson spikes
        poisson_states = {}
        for i, node in enumerate(self.nodes):
            if isinstance(node, PoissonNode):
                poisson_states[i] = np.zeros(len(t))
                for j in range(len(t)):
                    if np.random.rand() < node.firing_rate * self.dt:
                        poisson_states[i][j] = 1.0

        def system_derivative(t_now: float, y_vec: np.ndarray) -> np.ndarray:
            dydt = np.zeros_like(y_vec)
            t_idx = int(t_now / self.dt)
            
            # Get outputs from all nodes
            outputs = np.zeros(len(self.nodes))
            state_idx = 0
            for i, node in enumerate(self.nodes):
                if isinstance(node, PoissonNode):
                    outputs[i] = poisson_states[i][t_idx]/self.dt # a hack to make the other receive 1
                else:
                    outputs[i] = y_vec[state_idx]
                    state_idx += 1

            # Calculate inputs
            inputs = self.connectivity @ outputs

            # Update derivatives for non-Poisson nodes
            state_idx = 0
            for i, node in enumerate(self.nodes):
                if not isinstance(node, PoissonNode):
                    local_state = y_vec[state_idx:state_idx + node.n_state]
                    input_val = inputs[i]
                    dydt[state_idx:state_idx + node.n_state] = node.get_derivative(t_now, local_state, input_val)
                    state_idx += node.n_state

            self.derivatives.append(dydt.copy())
            return dydt

        def rk2(y0: np.ndarray):
            traj = np.zeros((total_state, len(t)))
            traj[:, 0] = y0
            for i in range(len(t)-1):
                k1 = system_derivative(t[i], traj[:, i])
                k2 = system_derivative(t[i] + 0.5*self.dt, traj[:, i] + 0.5*self.dt*k1)
                traj[:, i+1] = traj[:, i] + self.dt * k2
                traj[:, i+1] = self.noise_fn(i, traj[:, i+1])
            return traj

        # Initialize states for non-Poisson nodes
        y0 = np.zeros(total_state)
        state_idx = 0
        for node in self.nodes:
            if not isinstance(node, PoissonNode):
                y0[state_idx:state_idx + node.n_state] = node.initial_state
                state_idx += node.n_state

        y = rk2(y0) if method == 'rk2' else NotImplemented

        # Combine Poisson and continuous states
        final_y = np.zeros((len(self.nodes), len(t)))
        state_idx = 0
        for i, node in enumerate(self.nodes):
            if isinstance(node, PoissonNode):
                final_y[i] = poisson_states[i]
            else:
                final_y[i] = y[state_idx]
                state_idx += 1

        self.derivatives = np.array(self.derivatives).T
        return final_y, [node.name for node in self.nodes], t
