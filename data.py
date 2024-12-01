import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_dense_adj



# print("Entered data.py")
def load_dataset(data_name):
    if data_name == "cora":
        dataset = Planetoid(root='/tmp/Cora', name='Cora')

    data = dataset[0]
    X = data.x  # Node features
    X[X != 0] = 1.  # Binarize the features

    # Remove columns with constant values
    non_full_zero_feat_mask = X.sum(dim=0) != 0
    X = X[:, non_full_zero_feat_mask]

    non_full_one_feat_mask = X.sum(dim=0) != X.size(0)
    X = X[:, non_full_one_feat_mask]

    data.x = X
    return data, dataset  # Return both data and dataset for number of classes

def preprocess(data, dataset):
    """Prepare data for GraphMaker.

    Parameters
    ----------
    data : Data
        Graph to be preprocessed.
    dataset : Dataset
        The dataset object to get the number of classes.

    Returns
    -------
    X_one_hot : torch.Tensor of shape (F, N, 2)
        X_one_hot[f, :, :] is the one-hot encoding of the f-th node attribute.
        N = |V|.
    Y : torch.Tensor of shape (N)
        Categorical node labels.
    E_one_hot : torch.Tensor of shape (N, N, 2)
        - E_one_hot[:, :, 0] indicates the absence of an edge.
        - E_one_hot[:, :, 1] is the original adjacency matrix.
    X_marginal : torch.Tensor of shape (F, 2)
        X_marginal[f, :] is the marginal distribution of the f-th node attribute.
    Y_marginal : torch.Tensor of shape (C)
        Marginal distribution of the node labels.
    E_marginal : torch.Tensor of shape (2)
        Marginal distribution of the edge existence.
    X_cond_Y_marginals : torch.Tensor of shape (F, C, 2)
        X_cond_Y_marginals[f, k] is the marginal distribution of the f-th node
        attribute conditioned on the node label being k.
    """
    X = data.x
    Y = data.y
    N = data.num_nodes
    edge_index = data.edge_index
    src, dst = edge_index

    # One-hot encoding of node attributes
    X_one_hot_list = []
    for f in range(X.size(1)):
        # (N, 2)
        X_f_one_hot = F.one_hot(X[:, f].long(), num_classes=2).float()
        X_one_hot_list.append(X_f_one_hot)
    # (F, N, 2)
    X_one_hot = torch.stack(X_one_hot_list, dim=0)

    # Adjacency matrix in one-hot encoding
    E = to_dense_adj(edge_index)[0]  # Convert edge_index to dense adjacency matrix
    E_one_hot = F.one_hot(E.long(), num_classes=2).float()

    # (F, 2)
    X_one_hot_count = X_one_hot.sum(dim=1)
    # (F, 2)
    X_marginal = X_one_hot_count / X_one_hot_count.sum(dim=1, keepdim=True)
    # print("X_marginal", X_marginal)
    # print("X_marginal shape", X_marginal.shape)

    # (N, C)
    # Y_one_hot = F.one_hot(Y, num_classes=dataset.num_classes).float()
    Y_one_hot = F.one_hot(Y, num_classes= 7).float()
    # (C)
    Y_one_hot_count = Y_one_hot.sum(dim=0)
    # (C)
    Y_marginal = Y_one_hot_count / Y_one_hot_count.sum()

    # (2)
    E_one_hot_count = E_one_hot.sum(dim=0).sum(dim=0)
    E_marginal = E_one_hot_count / E_one_hot_count.sum()

    # P(X_f | Y)
    X_cond_Y_marginals = []
    num_classes = Y_marginal.size(-1)
    for k in range(num_classes):
        nodes_k = Y == k
        X_one_hot_k = X_one_hot[:, nodes_k]
        # (F, 2)
        X_one_hot_k_count = X_one_hot_k.sum(dim=1)
        # (F, 2)
        X_marginal_k = X_one_hot_k_count / X_one_hot_k_count.sum(dim=1, keepdim=True)
        
        X_cond_Y_marginals.append(X_marginal_k)
    # print("X_marginal_k", X_marginal_k)
    # print("X_marginal_k shape",X_marginal_k.shape)
    # (F, C, 2)
    X_cond_Y_marginals = torch.stack(X_cond_Y_marginals, dim=1)

    return X_one_hot, Y, E, E_one_hot, X_marginal, Y_marginal, E_marginal, X_cond_Y_marginals
# print("Exiting Data.py")
# # Example usage
# data, dataset = load_dataset("cora")
# X_one_hot, Y, E_one_hot, X_marginal, Y_marginal, E_marginal, X_cond_Y_marginals = preprocess(data, dataset)
