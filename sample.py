import torch
import torch.nn.functional as F
import networkx as nx  # Import NetworkX instead of DGL

from data import load_dataset, preprocess
# from eval_utils import Evaluator
from setup_utils import set_seed

def main(args):
    state_dict = torch.load(args.model_path)
    dataset = state_dict["dataset"]
    print(dataset)

    train_yaml_data = state_dict["train_yaml_data"]
    model_name = train_yaml_data["meta_data"]["variant"]

    print(f"Loaded GraphMaker-{model_name} model trained on {dataset}")
    print(f"Val Nll {state_dict['best_val_nll']}")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    g_real_full = load_dataset(dataset)
    g_real = g_real_full[0]
    X_one_hot_3d_real, Y_real, E_one_hot_real, \
        X_marginal, Y_marginal, E_marginal, X_cond_Y_marginals = preprocess(g_real, dataset)
    Y_one_hot_real = F.one_hot(Y_real)

    # evaluator = Evaluator(dataset,
    #                       g_real,
    #                       X_one_hot_3d_real,
    #                       Y_one_hot_real)

    X_marginal = X_marginal.to(device)
    Y_marginal = Y_marginal.to(device)
    E_marginal = E_marginal.to(device)
    X_cond_Y_marginals = X_cond_Y_marginals.to(device)
    num_nodes = Y_real.size(0)

    if model_name == "Sync":
        from model import ModelSync

        model = ModelSync(X_marginal=X_marginal,
                          Y_marginal=Y_marginal,
                          E_marginal=E_marginal,
                          gnn_X_config=train_yaml_data["gnn_X"],
                          gnn_E_config=train_yaml_data["gnn_E"],
                          num_nodes=num_nodes,
                          **train_yaml_data["diffusion"]).to(device)

        model.graph_encoder.pred_X.load_state_dict(state_dict["pred_X_state_dict"])
        model.graph_encoder.pred_E.load_state_dict(state_dict["pred_E_state_dict"])

    elif model_name == "Async":
        from model import ModelAsync

        model = ModelAsync(X_marginal=X_marginal,
                           Y_marginal=Y_marginal,
                           E_marginal=E_marginal,
                           mlp_X_config=train_yaml_data["mlp_X"],
                           gnn_E_config=train_yaml_data["gnn_E"],
                           num_nodes=num_nodes,
                           **train_yaml_data["diffusion"]).to(device)

        model.graph_encoder.pred_X.load_state_dict(state_dict["pred_X_state_dict"])
        model.graph_encoder.pred_E.load_state_dict(state_dict["pred_E_state_dict"])

    model.eval()

    # Set seed for better reproducibility.
    set_seed()

    sampled_graphs = []
    X_list = []
    Y_list = []

    for _ in range(args.num_samples):
        # Sample the graph and node features
        X_0_one_hot, Y_0_one_hot, E_0 = model.sample()
        src, dst = E_0.nonzero().T

        # Create a NetworkX graph
        g_sample = nx.Graph()
        g_sample.add_edges_from(zip(src.cpu().numpy(), dst.cpu().numpy()))  # Add edges from the sampled data
        
        # Ensure all nodes are added to the graph
        num_nodes = X_0_one_hot.shape[0]
        for i in range(num_nodes):
            g_sample.add_node(i)
        
        # Append the generated graph and corresponding node features to the lists
        sampled_graphs.append(g_sample)
        X_list.append(X_0_one_hot)
        Y_list.append(Y_0_one_hot)

    # Return lists of graphs, X and Y for each sample
    return sampled_graphs, X_list, Y_list

        # evaluator.add_sample(g_sample,
        #                      X_0_one_hot.cpu(),
        #                      Y_0_one_hot.cpu())

    # evaluator.summary()

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, help="Path to the model.")
    parser.add_argument("--num_samples", type=int, default=2,
                        help="Number of samples to generate.")
    args = parser.parse_args()

    sampled_graphs, X, Y = main(args)
    print("Sampled graphs: We can sample graph after each training instance as well.")
    print(sampled_graphs[0])
    print(sampled_graphs[1])
   
    # adj_matrix = nx.adjacency_matrix(sampled_graphs)
    # # Convert to a dense numpy array (optional)
    # adj_matrix_dense = adj_matrix.todense()

    # # Print the adjacency matrix
    # print("Adjacency Matrix:")
    # print(adj_matrix_dense)
    # print(sampled_graphs)
    # print(X.shape)
    # print(Y.shape)
