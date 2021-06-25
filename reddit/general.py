import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.utils import negative_sampling, to_undirected

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from data import RedditDataset
from torch_sparse import SparseTensor
from torch_sparse import sum as sparse_sum
from logger import Logger

from datetime import datetime
from tqdm import tqdm

from pathlib import Path

class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(MLP, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x, x_j = None):
        if x_j is not None:
            x = x * x_j
        for idx, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x
    
class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)
    
class LinkGNN(torch.nn.Module):
    def __init__(self, gnn, linkpred):
        super(LinkGNN, self).__init__()
        self.gnn = gnn
        self.linkpred = linkpred
    
    def reset_parameters(self):
        self.gnn.reset_parameters()
        self.linkpred.reset_parameters()
    
    def forward(self, x, edges, adj):
        h = self.gnn(x, adj)
        return self.linkpred(h[edges[0]], h[edges[1]])

class CommonNeighborsPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, model_type='weighted'):
        super(CommonNeighborsPredictor, self).__init__()
        assert model_type in ['mlpcos', 'simplecos', 'adamic', 'simple']
        self.type = model_type
        if self.type == 'mlpcos':
            self.mlp = MLP(in_channels, hidden_channels, out_channels, num_layers,
                     dropout)
        else:
            self.mlp = torch.nn.Identity()
        
    def reset_parameters(self):
        if self.type == 'mlpcos':
            self.mlp.reset_parameters()

    def forward(self, x, edges, adj): 
        common_neighbors = adj[edges[0]].to_torch_sparse_coo_tensor().mul(adj[edges[1]].to_torch_sparse_coo_tensor())
        
        if self.type == 'simple':
            return torch.sparse.sum(common_neighbors, 1).to_dense()
        
        common_neighbors = common_neighbors.indices()  
        
        sparse_sizes = adj[edges[0]].sparse_sizes()
        degrees = adj.sum(-1) + 1e-6
        
        if self.type == 'adamic':
            weights = SparseTensor.from_edge_index(common_neighbors, 
                                                   1./torch.log(degrees[common_neighbors[1]]), 
                                                   sparse_sizes = sparse_sizes) # sparse(Q, N)
            weights = sparse_sum(weights, 1)
            return torch.sigmoid(weights)
        
        left_neighbors = common_neighbors.clone()
        left_neighbors[0] = edges[0][common_neighbors[0]]
        
        right_neighbors = common_neighbors.clone()
        right_neighbors[0] = edges[1][common_neighbors[0]]

        x =  x + (adj @ x) / degrees.unsqueeze(1)
        
        x = self.mlp(x)
        left_edge_features = x[left_neighbors] # (2, Q * sparse(N), F)
        right_edge_features = x[right_neighbors] # (2, Q * sparse(N), F)
        
        left_edge_weights = F.cosine_similarity(left_edge_features[0], left_edge_features[1], dim=1)  # (Q * sparse(N))
        right_edge_weights = F.cosine_similarity(right_edge_features[0], right_edge_features[1], dim=1)                          
                
        weights = SparseTensor.from_edge_index(common_neighbors, 
                                               left_edge_weights * right_edge_weights, 
                                               sparse_sizes = sparse_sizes) # sparse(Q, N)
        weights = sparse_sum(weights, 1)
        return torch.sigmoid(weights) 


def train(model, data, split_edge, optimizer, batch_size, use_params, model_str):
    model.train()
    pos_train_edge = split_edge['train']['edge'].to(data.x.device)

    total_loss = total_examples = 0
    running_loss = None
    alpha = 0.99
    running_acc = None
    for idx, perm in enumerate(DataLoader(range(pos_train_edge.size(0)), batch_size,
                           shuffle=True)):

        if use_params:
            optimizer.zero_grad()

        pos_edge = to_undirected(pos_train_edge[perm].t(), data.num_nodes)
        
        if model_str in ['gcn', 'sage']:
            neg_edge = torch.randint(0, data.num_nodes, pos_edge.size(),dtype=torch.long, device=pos_edge.device)
        else:
            neg_edge = torch.randint(0, data.num_nodes, (1,pos_edge.size(1)),dtype=torch.long, device=pos_edge.device)
            neg_edge = torch.stack([pos_edge[0],neg_edge[0]])
        
        out = model(data.x, torch.cat([pos_edge, neg_edge], 1), data.adj_t).squeeze()
        
        pos_out = out[:pos_edge.size(1)]
        pos_loss = -torch.log(pos_out + 1e-8).mean() 

        neg_out = out[pos_edge.size(1):]
        neg_loss = -torch.log(1 - neg_out + 1e-8).mean() 
        
        loss = pos_loss + neg_loss
        if use_params:
            loss.backward()
        
        acc = ((neg_out < 0.5).sum() + (pos_out>0.5).sum()).item()/(0.+out.size(0))
        if running_loss is None:
            running_loss = loss.item()
            running_acc = acc
        running_loss = (1-alpha)*loss.item() + alpha*running_loss
        running_acc = (1-alpha)*acc + alpha*running_acc
        print(running_loss, running_acc, pos_loss.item(), neg_loss.item())

        if use_params:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        if use_params:
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
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size):
        edge = pos_train_edge[perm].t()
        pos_train_preds.append(model(data.x, edge, data.adj_t).cpu().squeeze())
    pos_train_pred = torch.cat(pos_train_preds, dim=0)
    
    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
        edge = pos_valid_edge[perm].t()
        pos_valid_preds.append(model(data.x, edge, data.adj_t).cpu().squeeze())
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
        edge = neg_valid_edge[perm].t()
        neg_valid_preds.append(model(data.x, edge, data.adj_t).cpu().squeeze())
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        pos_test_preds.append(model(data.x, edge, data.adj_t).cpu().squeeze())
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        neg_test_preds.append(model(data.x, edge, data.adj_t).cpu().squeeze())
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
    parser.add_argument('--model', type=str, default='gcn')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=300)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=16 * 1024)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=20)
    parser.add_argument('--sorted_edge_path', type=str, default='')
    parser.add_argument('--name', type=str, default='')
    args = parser.parse_args()
    print(args)
    
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = RedditDataset()
    data = dataset[0]
    
    edge_index = data.edge_index
    split_edge = dataset.get_edge_split()
    data = T.ToSparseTensor()(data)
    
    Path("curves").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)

    # We randomly pick some training samples that we want to evaluate on:
    torch.manual_seed(12345)
    idx = torch.randperm(split_edge['train']['edge'].size(0))
    idx = idx[:split_edge['valid']['edge'].size(0)]
    split_edge['eval_train'] = {'edge': split_edge['train']['edge'][idx]}
    data = data.to(device)
#     data.adj_t = data.adj_t.coalesce().fill_value(1.)

    if args.model == 'sage':
        gnn = SAGE(
            data.num_features, args.hidden_channels, 
            args.hidden_channels, args.num_layers, 
            args.dropout).to(device)
        linkpred = LinkPredictor(
            args.hidden_channels, args.hidden_channels, 
            1, args.num_layers, 
            args.dropout).to(device)
        model = LinkGNN(gnn, linkpred)
    elif args.model == 'gcn':
        gnn = GCN(
            data.num_features, args.hidden_channels, 
            args.hidden_channels, args.num_layers, 
            args.dropout).to(device)
        linkpred = LinkPredictor(
            args.hidden_channels, args.hidden_channels, 
            1, args.num_layers, 
            args.dropout).to(device)
        model = LinkGNN(gnn, linkpred)
    elif args.model in ['mlpcos', 'simplecos', 'adamic', 'simple']:
        model = CommonNeighborsPredictor(
            data.num_features, args.hidden_channels, 
            args.hidden_channels, args.num_layers, 
            args.dropout, model_type=args.model).to(device)
    else:
        raise NotImplemented
    print(f'using model {model}')

    evaluator = Evaluator(name='ogbl-collab')
    loggers = {
        'Hits@10': Logger(args.runs, args),
        'Hits@50': Logger(args.runs, args),
        'Hits@100': Logger(args.runs, args),
    }

    model_name = args.name if args.name else args.model
    
    save_weights = args.sorted_edge_path == ''
    if args.sorted_edge_path:
        # should be sorted of shape [E, 2] or [E, 3], where the 3rd index is possibly a score
        sorted_test_edges = torch.load(args.sorted_edge_path)
        print('sorted test edges', sorted_test_edges.size())
    else:
        # fake [E, 2], will throw error on 2nd run
        sorted_test_edges = torch.zeros(42, 2)
        
    curve = []
    for run in range(args.runs):
        model.reset_parameters()
        use_params = sum(p.numel() for p in model.parameters() if p.requires_grad) > 0
        if use_params:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        else:
            optimizer = None
        
        index_end = int(run * 10000)
        print('---------------------')
        print(f'Using {index_end} highest scoring edges')
        print('---------------------')
        
        extra_edges = sorted_test_edges[:index_end,:2].t().long()
        assert extra_edges.size(0) == 2
        assert extra_edges.size(1) == index_end
        full_edge_index = torch.cat([edge_index, extra_edges], dim=-1)
        data.full_adj_t = SparseTensor.from_edge_index(full_edge_index, sparse_sizes = data.adj_t.sparse_sizes())
        data.full_adj_t = data.full_adj_t.to_symmetric().to(device)
        data.adj_t = data.full_adj_t
        
        for epoch in range(1, 1 + args.epochs):
            if use_params:
                loss = train(model, data, split_edge, optimizer,
                             args.batch_size, use_params, args.model)
            else:
                loss = -1
                
            best_val = 0.0
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
                              f'Loss: {loss:.4f}, '
                              f'Train: {100 * train_hits:.2f}%, '
                              f'Valid: {100 * valid_hits:.2f}%, '
                              f'Test: {100 * test_hits:.2f}%')
                    print('---')
                    
            if not use_params:
                break
        
        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics(run)
                        
            if key == "Hits@50":
                result = 100 * torch.tensor(loggers[key].results[run])
                argmax = result[:, 1].argmax().item()
                curve.append(result[argmax, 2])
                
        if save_weights:
            torch.save(model.state_dict(), f'models/{model_name}.pt')
            break
            
    print(curve)
    
    time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    
    filename = f"curves/{model_name}_{time}.pt"
    
    torch.save(curve, filename)
    
    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics()


if __name__ == "__main__":
    main()