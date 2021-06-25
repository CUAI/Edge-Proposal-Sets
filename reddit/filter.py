import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.utils import negative_sampling

import torch_sparse

import torch_geometric.transforms as T

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from data import RedditDataset

from general import GCN, SAGE, LinkPredictor, LinkGNN, MLP, CommonNeighborsPredictor

from pathlib import Path
from tqdm import tqdm        

def main():
    parser = argparse.ArgumentParser(description='Reddit(GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--model', type=str, default='gcn')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=300)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=8 * 16 * 1024)
    parser.add_argument('--name', type=str, default='')
    args = parser.parse_args()
    print(args)
    
    Path("filtered_edges").mkdir(exist_ok=True)
    
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = RedditDataset()
    data = dataset[0]
    
    edge_index = data.edge_index
    split_edge = dataset.get_edge_split()
    data = T.ToSparseTensor()(data)
    
    adj_t = data.adj_t.to_symmetric()
    adj_t = adj_t.to(device)
    data.adj_t = adj_t

    x = data.x.to(device)
    data.x = x

    data = data.to(device)
    
    if args.model == 'sage':
        gnn = SAGE(
            data.num_features, args.hidden_channels, 
            args.hidden_channels, args.num_layers, 
            args.dropout).to(device)
        linkpred = LinkPredictor(
            args.hidden_channels, args.hidden_channels, 
            1, args.num_layers, 
            args.dropout).to(device)
        model = LinkGNN(gnn, linkpred).to(device)
    elif args.model == 'gcn':
        gnn = GCN(
            data.num_features, args.hidden_channels, 
            args.hidden_channels, args.num_layers, 
            args.dropout).to(device)
        linkpred = LinkPredictor(
            args.hidden_channels, args.hidden_channels, 
            1, args.num_layers, 
            args.dropout).to(device)
        model = LinkGNN(gnn, linkpred).to(device)
    elif args.model in ['mlpcos', 'simplecos', 'adamic', 'simple']:
        model = CommonNeighborsPredictor(
            data.num_features, args.hidden_channels, 
            args.hidden_channels, args.num_layers, 
            args.dropout, model_type=args.model).to(device)
    else:
        raise NotImplemented
    print(f'using model {model}')
    
    model_name = args.name if args.name else args.model
    
    use_params = sum(p.numel() for p in model.parameters() if p.requires_grad) > 0
    print('using params?', use_params)
    if use_params:
        model.load_state_dict(torch.load(f'models/{model_name}.pt'))
        
    model.eval()
    
    all_scores = []
    # restrict to edges that have at least one common neighbor for relevant models
    if True:
#     if args.model in ['mlpcos', 'simplecos', 'adamic', 'simple']:
        A2 = adj_t.cpu() @ adj_t.cpu()
        A2 = torch_sparse.remove_diag(A2)
        A2 = A2.to_scipy("csc")
        # dont compute for edges that we are know positive
        A2[adj_t.to_scipy("csc")>0] = 0
        
        indices, values = torch_sparse.from_scipy(A2)
        selected = values.nonzero().squeeze(1)
        
        m = torch.cat([indices[:, selected].t(), values[selected].unsqueeze(1)], 1).long()
        all_edges = m[:,:2].t().to(device)
        
        print(f'using {all_edges.size()} edges')

        with torch.no_grad():
            for perm in tqdm(DataLoader(range(all_edges.size(1)), args.batch_size)):
                edges = all_edges[:, perm]
                score = model(x, edges, adj_t).squeeze()
                edge_score = torch.cat([edges.t(), score.unsqueeze(1)], dim=1).cpu()
                all_scores.append(edge_score)
                
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
        

    all_scores = torch.cat(all_scores, 0)
    _, indices = all_scores[:,2].sort(descending=True)
    sorted_edges = all_scores[indices].cpu()
    
    print(sorted_edges)
    
    torch.save(sorted_edges, f'filtered_edges/{model_name}_sorted_edges.pt')
    
if __name__ == "__main__":
    main()