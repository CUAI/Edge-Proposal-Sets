import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from torch_sparse import sum as sparse_sum

from tqdm import tqdm
from logger import Logger
from torch_sparse import SparseTensor
from data import RedditDataset

class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(MLP, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for lin in self.bns:
            lin.reset_parameters()

    def forward(self, x, x_j = None):
        if x_j is not None:
            x = x * x_j
        for idx, lin in enumerate(self.lins[:-1]):
            x = lin(x)
#             x = self.bns[idx](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x

class MLP_cos(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(MLP_cos, self).__init__()
        self.mlp1 = MLP(in_channels, hidden_channels, out_channels, num_layers,
                 dropout)        
    def reset_parameters(self):
        self.mlp1.reset_parameters()

    def forward(self, x, edges, adj): #
        degrees = adj.sum(-1)
        x =  x + (adj @ x) / degrees.unsqueeze(1)
        x = self.mlp1(x) 
        return torch.sigmoid(torch.nn.CosineSimilarity()(x[edges[0]],x[edges[1]])) , x[edges]
# 
    

class WeightedCommonNeighborsPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(WeightedCommonNeighborsPredictor, self).__init__()
        self.mlp1 = MLP(in_channels, hidden_channels, out_channels, num_layers,
                 dropout)
#         self.mlp2 = MLP(in_channels, hidden_channels, out_channels, num_layers,
#                  dropout)
#         self.weight_bias = torch.nn.Parameter(torch.randn(1))
        
    def reset_parameters(self):
        self.mlp1.reset_parameters()
#         self.mlp2.reset_parameters()
#         self.weight_bias.data = torch.randn(1).to(self.weight_bias.data.device)

    def forward(self, x, edges, adj): 
        common_neighbors = adj[edges[0]].to_torch_sparse_coo_tensor().mul(adj[edges[1]].to_torch_sparse_coo_tensor()).indices()  
#         sparse_sizes = adj[edges[0]].sparse_sizes()
#         degrees = adj.sum(-1)+1e-6
        
#         weights = SparseTensor.from_edge_index(common_neighbors, 1./torch.log(degrees[common_neighbors[1]]), sparse_sizes = sparse_sizes) # sparse(Q, N)
#         weights = sparse_sum(weights, 1)
#         return torch.sigmoid(weights) , None
        
        left_neighbors = common_neighbors.clone()
        left_neighbors[0] = edges[0][common_neighbors[0]]
        
        right_neighbors = common_neighbors.clone()
        right_neighbors[0] = edges[1][common_neighbors[0]]

        sparse_sizes = adj[edges[0]].sparse_sizes()
        
        degrees = adj.sum(-1)+1e-6
        x =  x +(adj @ x) / degrees.unsqueeze(1)
#         x =  torch.cat([x ,(adj @ x) / degrees.unsqueeze(1)], 1)
        x = self.mlp1(x)
        left_edge_features = x[left_neighbors] # (2, Q * sparse(N), F)
        right_edge_features = x[right_neighbors] # (2, Q * sparse(N), F)
        
        B = left_edge_features.shape[1]
        S = left_edge_features.shape[2]
        
        left_edge_weights = torch.nn.CosineSimilarity()(left_edge_features[0], left_edge_features[1])  # (Q * sparse(N))
        right_edge_weights = torch.nn.CosineSimilarity()(right_edge_features[0], right_edge_features[1])                                   
                
        weights = SparseTensor.from_edge_index(common_neighbors, left_edge_weights * right_edge_weights, sparse_sizes = sparse_sizes) # sparse(Q, N)
        
        

        weights = sparse_sum(weights, 1)
        return torch.sigmoid(weights) , None
# + torch.nn.CosineSimilarity()(x[edges[0]], x[edges[1]])

def train(model, data, split_edge, optimizer, batch_size):
    model.train()
    pos_train_edge = split_edge['train']['edge'].to(data.x.device)

    total_loss = total_examples = 0
    running_loss = None
    alpha = 0.99
    running_acc = None
    for idx, perm in enumerate(DataLoader(range(pos_train_edge.size(0)), batch_size,
                           shuffle=True)):

        optimizer.zero_grad()


        pos_edge = pos_train_edge[perm].t()
        neg_edge = torch.randint(0, data.num_nodes, (1,pos_edge.size(1)),dtype=torch.long, device=pos_edge.device)
        
        neg_edge = torch.stack([pos_edge[0],neg_edge[0]])
        out, rep = model(data.x, torch.cat([pos_edge, neg_edge], 1), data.adj_t)
        
        pos_out = out[:pos_edge.size(1)]
        pos_loss = -torch.log(pos_out + 1e-8).mean() 

        # Just do some trivial random sampling.
        

        neg_out = out[pos_edge.size(1):]
        neg_loss = -torch.log(1 - neg_out + 1e-8).mean() 
        loss = pos_loss + neg_loss
        loss.backward()
        del neg_edge
        acc = ((neg_out < 0.5).sum() + (pos_out>0.5).sum()).item()/(0.+out.size(0))
        if running_loss is None:
            running_loss = loss.item()
            running_acc = acc
        running_loss = (1-alpha)*loss.item() + alpha*running_loss
        running_acc = (1-alpha)*acc + alpha*running_acc
        print(running_loss, running_acc, pos_loss.item(), neg_loss.item())

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


@torch.no_grad()
def test(model, data, split_edge, evaluator, batch_size):
    model.eval()


    pos_train_edge = split_edge['train']['edge'].to(data.x.device)
    pos_valid_edge = split_edge['valid']['edge'].to(data.x.device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(data.x.device)
    pos_test_edge = split_edge['test']['edge'].to(data.x.device)
    neg_test_edge = split_edge['test']['edge_neg'].to(data.x.device)

    pos_train_preds = []
    for perm in tqdm(DataLoader(range(pos_train_edge.size(0)), batch_size)):
        edge = pos_train_edge[perm].t()
        pos_train_preds.append(model(data.x, edge, data.adj_t)[0].cpu())
    pos_train_pred = torch.cat(pos_train_preds, dim=0)
    
    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
        edge = pos_valid_edge[perm].t()
        pos_valid_preds.append(model(data.x, edge, data.adj_t)[0].cpu())
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
        edge = neg_valid_edge[perm].t()
        neg_valid_preds.append(model(data.x, edge, data.adj_t)[0].cpu())
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

#     import pdb;pdb.set_trace()
    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        pos_test_preds.append(model(data.x, edge, data.adj_t)[0].cpu())
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        neg_test_preds.append(model(data.x, edge, data.adj_t)[0].cpu())
    neg_test_pred = torch.cat(neg_test_preds, dim=0)
    


    results = {}
    for K in [10, 50, 100]:
        evaluator.K = K
        train_hits = evaluator.eval({
            'y_pred_pos': pos_train_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits)

    pos_loss = -torch.log(pos_valid_pred + 1e-8).mean() 
    neg_loss = -torch.log(1 - neg_valid_pred + 1e-8).mean() 
    print(pos_loss.item(), neg_loss.item())

    return results


def main():
    parser = argparse.ArgumentParser(description='Reddit(GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=300)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=16 * 1024)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=20)
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = RedditDataset()
    data = dataset[0]
    
    edge_index = data.edge_index
    split_edge = dataset.get_edge_split()
    data = T.ToSparseTensor()(data)

    split_edge = dataset.get_edge_split()

    # We randomly pick some training samples that we want to evaluate on:
    torch.manual_seed(12345)
    idx = torch.randperm(split_edge['train']['edge'].size(0))
    idx = idx[:split_edge['valid']['edge'].size(0)]
    split_edge['eval_train'] = {'edge': split_edge['train']['edge'][idx]}
    data = data.to(device)
    
#     data.adj_t = data.adj_t.coalesce().fill_value(1.)
#     model = WeightedCommonNeighborsPredictor(data.num_features, args.hidden_channels, args.hidden_channels, args.num_layers, args.dropout).to(device)

#     model = MLP_cos(data.num_features, args.hidden_channels, args.hidden_channels, args.num_layers, args.dropout).to(device)
#     data.adj_t = data.adj_t.coalesce().fill_value(1.)

    # Pre-compute GCN normalization.
#     adj_t = data.adj_t.set_diag()
#     deg = adj_t.sum(dim=1).to(torch.float)
#     deg_inv_sqrt = deg.pow(-0.5)
#     deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
#     adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
#     data.adj_t = adj_t
    
    evaluator = Evaluator(name='ogbl-collab')
    loggers = {
        'Hits@10': Logger(args.runs, args),
        'Hits@50': Logger(args.runs, args),
        'Hits@100': Logger(args.runs, args),
    }
    sorted_test_edges = torch.load("gnn_sorted_edges.pt")
    curve = []
    for run in range(0, 20):
        model = WeightedCommonNeighborsPredictor(data.num_features, args.hidden_channels, args.hidden_channels, args.num_layers, args.dropout).to(device)
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(),
            lr=args.lr)
        
#         index_end = int(torch.tensor(run).float().exp())
        index_end = int(run * 10000)
        print('---------------------')
        print(f'Using {index_end} highest Adamic-Adar edges')
        print('---------------------')
        extra_edges = sorted_test_edges[:index_end,:2].t().long()
        full_edge_index = torch.cat([edge_index, extra_edges], dim=-1)
        data.full_adj_t = SparseTensor.from_edge_index(full_edge_index, sparse_sizes = data.adj_t.sparse_sizes()).t()
        data.full_adj_t = data.full_adj_t.to_symmetric().cuda()

        adj_t = data.full_adj_t.to(device)
        data.adj_t = adj_t
   

        for epoch in range(1, 1 + args.epochs):
            loss = train(model, data, split_edge, optimizer,
                         args.batch_size)

            if epoch % args.eval_steps == 0:
                results = test(model, data, split_edge, evaluator,
                               args.batch_size)
                for key, result in results.items():
                    loggers[key].add_result(run, result)

                if epoch % args.log_steps == 0:
                    for key, result in results.items():
                        train_hits, valid_hits, test_hits = result
                        print(key)
                        print(f'Run: {run + 1:02d}, '
                              f'Epoch: {epoch:02d}, '
#                               f'Loss: {loss:.4f}, '
                              f'Train: {100 * train_hits:.2f}%, '
                              f'Valid: {100 * valid_hits:.2f}%, '
                              f'Test: {100 * test_hits:.2f}%')

        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics(run)
            if key == "Hits@50":
                result = 100 * torch.tensor(loggers[key].results[run])
                argmax = result[:, 1].argmax().item()
                curve.append(result[argmax, 2])
    print(curve)
    torch.save(curve, "gnn_mlpcommon_curve.pt")
#         torch.save(model.state_dict(), f"mlp_cos_{str(run)}.pt")

    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics()


if __name__ == "__main__":
    main()
