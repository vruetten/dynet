import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch


class PDE_Z1(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    """
    Compute network signaling, the transfer functions are neuron-neuron-dependent

    Inputs
    ----------
    data : a torch_geometric.data object

    Returns
    -------
    du : float
    the update rate of the signals (dim 1)

    """

    def __init__(self, aggr_type=[], W=None, Fz = None, g=2.0)  :
        super(PDE_Z1, self).__init__(aggr=aggr_type)

        self.W = W
        self.Fz = Fz
        self.g = g

    def forward(self, data=[]):
        x, edge_index = data.x, data.edge_index

        node_type = x[:, 5].long()
        u = x[:, 6:7]

        msg = self.propagate(edge_index, u=u, node_type=node_type[:,None])

        du = self.g * msg - u/5

        return du

    def message(self, edge_index_i, edge_index_j, u_j, node_type_i, node_type_j):

        T = self.W
        return T[edge_index_i, edge_index_j][:, None] * u_j

        T = self.W
        pos = torch.argwhere(edge_index_i == 10).flatten()
        neighbor_id = edge_index_j[pos]
        sum = torch.sum(T[10, neighbor_id][:, None] * u_j[pos])
        print(sum)
        matplotlib.use("Qt5Agg")
        plt.imshow(self.W.detach().cpu().numpy(), cmap='hot', interpolation='nearest')

