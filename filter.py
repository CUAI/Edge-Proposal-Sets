import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.utils import negative_sampling
import scipy.sparse as ssp
import numpy as np
from train_and_eval import resource_allocation
import torch_sparse
from adamic_utils import get_A, AA
import torch_geometric.transforms as T

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator

from rank import get_data, add_edges
from models import build_model, default_model_configs
import pandas as pd
from torch_sparse import SparseTensor
from torch_sparse import sum as sparse_sum

from pathlib import Path
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description='ddi (GNN)')
    # experiment configs
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    
    # model configs; overwrite defaults if specified
    parser.add_argument('--num_layers', type=int)
    parser.add_argument('--hidden_channels', type=int)
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--use_feature', type=bool)
    parser.add_argument('--use_learnable_embedding', type=bool)
#     parser.add_argument('--use_node_embedding', action="store_true", default=False)
    
    # other settings
    parser.add_argument('--device', type=int, default=0)
    
    args = parser.parse_args()
    args = default_model_configs(args)
    print(args)
    
    Path("filtered_edges").mkdir(exist_ok=True)
   
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    ##############
    ## load data and model
    ##############
    
    edge_index, edge_weight, split_edge, data = get_data(args)
    data = data.to(device)
    model = build_model(args, data, device)
    print(f'using model {model}')
    
        
    use_params = sum(p.numel() for p in model.parameters() if p.requires_grad) > 0
    print('using params?', use_params)
    if use_params:
        model.load_state_dict(torch.load(f'models/{args.checkpoint}'))
        
        
    parts = args.checkpoint.split("|")
    spec = parts[0]
    sorted_edge_path = parts[1]
    num_sorted_edge = int(parts[2])
    run = parts[3].split(".")[0]
    
    if sorted_edge_path:
        print("Loading corresponding extra edges from ", sorted_edge_path)
        print(f'Using {num_sorted_edge} highest scoring edges')
        # adding extra edges if needed (e.g. the original saved model uses additional edges)
        # should be sorted of shape [E, 2] or [E, 3], where the 3rd index is possibly a score
        extra_edges = torch.load(f"filtered_edges/{sorted_edge_path}.pt")[:num_sorted_edge,:2].t().long()       
        assert extra_edges.size(0) == 2
        assert extra_edges.size(1) == num_sorted_edge
        data.adj_t = add_edges(args.dataset, edge_index, edge_weight, extra_edges, data.num_nodes).to(device)
    else:
        data.adj_t = add_edges(args.dataset, edge_index, edge_weight, torch.zeros([2,0], dtype=int) , data.num_nodes).to(device)
        
    model.eval()
    
    all_scores = []
    # restrict to edges that have at least one common neighbor for relevant models
    if True:
#     if args.model in ['mlpcos', 'simplecos', 'adamic', 'simple']:
        adj_t = data.adj_t.cpu()
#         print("before")
        A2 = adj_t @ adj_t
#         print("after")
        A2 = torch_sparse.remove_diag(A2)
        A2 = A2.to_scipy("csc")
        # dont compute for edges that we are know positive
        A2[adj_t.to_scipy("csc")>0] = 0
        
        indices, values = torch_sparse.from_scipy(A2)
        selected = values.nonzero().squeeze(1)

        m = torch.cat([indices[:, selected].t(), values[selected].unsqueeze(1)], 1).long()
        all_edges = m[:,:2]
        
        print(f'using {all_edges.size()} edges')

        if args.model not in ["adamic_ogb", "resource_allocation"]:
            all_edges = all_edges.t().to(device)
            with torch.no_grad():
                for perm in tqdm(DataLoader(range(all_edges.size(1)), args.batch_size)):
                    edges = all_edges[:, perm]
                    score = model(data.x, edges, data.adj_t).squeeze()
                    edge_score = torch.cat([edges.t(), score.unsqueeze(1)], dim=1).cpu()
                    all_scores.append(edge_score)
            all_scores = torch.cat(all_scores, 0)
        elif args.model == "adamic_ogb":
            all_edges = all_edges.t()
            A = get_A(data.adj_t, data.num_nodes)
            pred, edge = eval('AA')(A, all_edges.cpu())
            all_scores = torch.cat((edge, pred.unsqueeze(0)), 0).T
        else:
#             print("here")
            assert args.model == "resource_allocation"
            train_edges_raw = np.array(split_edge['train']['edge'])
            train_edges_reverse = np.array(
                [train_edges_raw[:, 1], train_edges_raw[:, 0]]).transpose()
            train_edges = np.concatenate(
                [train_edges_raw, train_edges_reverse], axis=0)
            edge_weight = torch.ones(train_edges.shape[0], dtype=int)
            A = ssp.csr_matrix(
                (edge_weight, (train_edges[:, 0], train_edges[:, 1])), shape=(
                    data.num_nodes, data.num_nodes)
            )
#             print("here")
            pred = resource_allocation(A, all_edges.cpu(), batch_size=1024*8)
            all_scores = torch.cat((all_edges.t(), pred.unsqueeze(0)), 0).T

         
    # construct edges on fly. too memory-intensive :/
#     else:
#         N = data.num_nodes
#         col = torch.arange(N)
#         with torch.no_grad():
#             all_scores = []
#             for i in tqdm(range(N)):
#                 row = torch.tensor(i).repeat(N)
#                 edges = torch.stack((row, col), dim=0).to(device)
#                 score = model(x, edges, adj_t).squeeze()
#                 edge_score = torch.cat([edges.t(), score.unsqueeze(1)], dim=1).cpu()
#                 all_scores.append(edge_score)
        

    
    _, indices = all_scores[:,2].sort(descending=True)
    sorted_edges = all_scores[indices].cpu()
    
    print(sorted_edges)
    filename = f'filtered_edges/{spec}_{sorted_edge_path}_{num_sorted_edge}_{run}_sorted_edges.pt'
    torch.save(sorted_edges, filename)
    print("Saving to ", filename)
    
if __name__ == "__main__":
    main()
