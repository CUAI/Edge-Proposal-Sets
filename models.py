import argparse
import random
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv, TAGConv, JumpingKnowledge

from torch_sparse import SparseTensor
from torch_sparse import sum as sparse_sum
import pandas as pd
from datetime import datetime
from tqdm import tqdm

from pathlib import Path

class DEA_GNN_JK(torch.nn.Module):
    def __init__(self, num_nodes, embed_dim, 
                 gnn_in_dim, gnn_hidden_dim, gnn_out_dim, gnn_num_layers, 
                 mlp_in_dim, mlp_hidden_dim, mlp_out_dim=1, mlp_num_layers=2, 
                 dropout=0.5, gnn_batchnorm=False, mlp_batchnorm=False, K=2, jk_mode='max'):
        super(DEA_GNN_JK, self).__init__()
        
        assert jk_mode in ['max','sum','mean','lstm','cat']
        # Embedding
        self.emb = torch.nn.Embedding(num_nodes, embedding_dim=embed_dim)

        # GNN 
        convs_list = [TAGConv(gnn_in_dim, gnn_hidden_dim, K)]
        for i in range(gnn_num_layers-2):
            convs_list.append(TAGConv(gnn_hidden_dim, gnn_hidden_dim, K))
        convs_list.append(TAGConv(gnn_hidden_dim, gnn_out_dim, K))
        self.convs = torch.nn.ModuleList(convs_list)

        # MLP
        lins_list = [torch.nn.Linear(mlp_in_dim, mlp_hidden_dim)]
        for i in range(mlp_num_layers-2):
            lins_list.append(torch.nn.Linear(mlp_hidden_dim, mlp_hidden_dim))
        lins_list.append(torch.nn.Linear(mlp_hidden_dim, mlp_out_dim))
        self.lins = torch.nn.ModuleList(lins_list)

        # Batchnorm
        self.gnn_batchnorm = gnn_batchnorm
        self.mlp_batchnorm = mlp_batchnorm
        if self.gnn_batchnorm:
            self.gnn_bns = torch.nn.ModuleList([torch.nn.BatchNorm1d(gnn_hidden_dim) for i in range(gnn_num_layers)])
        
        if self.mlp_batchnorm:
            self.mlp_bns = torch.nn.ModuleList([torch.nn.BatchNorm1d(mlp_hidden_dim) for i in range(mlp_num_layers-1)])

        self.jk_mode = jk_mode
        if self.jk_mode in ['max', 'lstm', 'cat']:
            self.jk = JumpingKnowledge(mode=self.jk_mode, channels=gnn_hidden_dim, num_layers=gnn_num_layers)

        self.dropout = dropout
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.emb.weight)  
        for conv in self.convs:
            conv.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        if self.gnn_batchnorm:
            for bn in self.gnn_bns:
                bn.reset_parameters()
        if self.mlp_batchnorm:
            for bn in self.mlp_bns:
                bn.reset_parameters()
        if self.jk_mode in ['max', 'lstm', 'cat']:
            self.jk.reset_parameters()

    def forward(self, x_feature, edge_label_index, adj_t):
        out = x_feature
        if out is None:
            out = self.emb.weight
        elif self.emb is not None:
            out = torch.cat([self.emb.weight, out], dim=1)

        out_list = []
        for i in range(len(self.convs)):
            out = self.convs[i](out, adj_t.clone())
            if self.gnn_batchnorm:
                out = self.gnn_bns[i](out)
            out = F.relu(out)
            out = F.dropout(out, p=self.dropout, training=self.training)
            out_list += [out]

        if self.jk_mode in ['max', 'lstm', 'cat']:
            out = self.jk(out_list)
        elif self.jk_mode == 'mean':
            out_stack = torch.stack(out_list, dim=0)
            out = torch.mean(out_stack, dim=0)
        elif self.jk_mode == 'sum':
            out_stack = torch.stack(out_list, dim=0)
            out = torch.sum(out_stack, dim=0)

        gnn_embed = out[edge_label_index,:]
        embed_product = gnn_embed[0, :, :] * gnn_embed[1, :, :]
        out = embed_product

        for i in range(len(self.lins)-1):
            out = self.lins[i](out)
            if self.mlp_batchnorm:
                out = self.mlp_bns[i](out)
            out = F.relu(out)
            out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.lins[-1](out).squeeze(1)

        return out
    
    def loss(self, y_pred, y_true):
        return self.loss_fn(y_pred, y_true)


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
    def __init__(self, emb, gnn, linkpred):
        super(LinkGNN, self).__init__()
        self.gnn = gnn
        self.linkpred = linkpred
        self.emb = emb
    
    def reset_parameters(self):
        self.gnn.reset_parameters()
        self.linkpred.reset_parameters()
        if self.emb is not None:
            self.emb.reset_parameters()    
        
    def forward(self,x , edges, adj):
        if x is None:
            x = self.emb.weight
        elif self.emb is not None:
            x = torch.cat([self.emb.weight, x], dim=1)
        h = self.gnn(x, adj)
        return self.linkpred(h[edges[0]], h[edges[1]])

class CommonNeighborsPredictor(torch.nn.Module):
    def __init__(self, emb, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, model_type='weighted'):
        super(CommonNeighborsPredictor, self).__init__()
        assert model_type in ['mlpcos', 'simplecos', 'adamic', 'simple', 'adamic_ogb', 'katz']
        self.type = model_type
        if self.type == 'mlpcos':
            self.mlp = MLP(in_channels, hidden_channels, out_channels, num_layers,
                     dropout)
            self.emb = emb
        else:
            self.mlp = torch.nn.Identity()
            self.emb = emb
        
    def reset_parameters(self):
        if self.type == 'mlpcos':
            self.mlp.reset_parameters()
        if self.emb is not None:
            self.emb.reset_parameters()

    def forward(self, x, edges, adj):
        if (x is None) and (self.type in ['mlpcos', 'simplecos']):
            x = self.emb.weight
        elif self.emb is not None:
            x = torch.cat([self.emb.weight, x], dim=1)
            
        if self.type in ['adamic_ogb', 'katz']:
            return None
        common_neighbors = adj[edges[0]].to_torch_sparse_coo_tensor().mul(adj[edges[1]].to_torch_sparse_coo_tensor())
        
        
        if self.type == 'simple':
            if common_neighbors._nnz() == 0: 
                return torch.zeros((common_neighbors.shape[0])).to(common_neighbors.device)
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
        
#         x = self.mlp(x)
        left_edge_features = x[left_neighbors] # (2, Q * sparse(N), F)
        right_edge_features = x[right_neighbors] # (2, Q * sparse(N), F)
        
        left_edge_weights = F.cosine_similarity(left_edge_features[0], left_edge_features[1], dim=1)  # (Q * sparse(N))
        right_edge_weights = F.cosine_similarity(right_edge_features[0], right_edge_features[1], dim=1)                          
                
        weights = SparseTensor.from_edge_index(common_neighbors, 
                                               left_edge_weights * right_edge_weights, 
                                               sparse_sizes = sparse_sizes) # sparse(Q, N)
        weights = sparse_sum(weights, 1)
        return torch.sigmoid(weights) 


def build_model(args, data, device):   
    assert args.model in ['sage', 'gcn', 'dea', 'mlpcos', 'simplecos', 'adamic', 'simple', 'adamic_ogb', 'katz']
    emb = None
       
    if args.use_learnable_embedding:
        emb = torch.nn.Embedding(data.num_nodes, args.hidden_channels).to(device)
    
    input_dim = 0
    if args.use_learnable_embedding:
        input_dim += args.hidden_channels
    if args.use_feature:
        input_dim += data.x.shape[1]

     
    if args.model == 'sage':
        gnn = SAGE(
            input_dim, args.hidden_channels, 
            args.hidden_channels, args.num_layers, 
            args.dropout).to(device)
        linkpred = LinkPredictor(
            args.hidden_channels, args.hidden_channels, 
            1, args.num_layers, 
            args.dropout).to(device)
        model = LinkGNN(emb, gnn, linkpred)
    elif args.model == 'gcn':
        gnn = GCN(
            input_dim, args.hidden_channels,
            args.hidden_channels, args.num_layers, 
            args.dropout).to(device)
        linkpred = LinkPredictor(
            args.hidden_channels, args.hidden_channels, 
            1, args.num_layers, 
            args.dropout).to(device)
        model = LinkGNN(emb, gnn, linkpred)
    elif args.model == "dea":
        model = DEA_GNN_JK(num_nodes=data.num_nodes, embed_dim=args.hidden_channels,
                   gnn_in_dim=input_dim, gnn_hidden_dim=args.hidden_channels, gnn_out_dim=args.hidden_channels,
                   gnn_num_layers=3, mlp_in_dim=args.hidden_channels, mlp_hidden_dim=args.hidden_channels,
                   mlp_out_dim=1, mlp_num_layers=2,
                   dropout=args.dropout, gnn_batchnorm=True,
                   mlp_batchnorm=True,
                   K=2, jk_mode="max").to(device)
    elif args.model in ['mlpcos', 'simplecos', 'adamic', 'simple', 'adamic_ogb', 'katz']:
        # 'adamic', 'simple' should have 0 input dim
        # 'adamic_ogb' refers to the ogb implementation; in this case the model is not used
        model = CommonNeighborsPredictor(emb,
            input_dim, args.hidden_channels, 
            args.hidden_channels, args.num_layers, 
            args.dropout, model_type=args.model).to(device)

    return model


def default_model_configs(args):
    default_dict = {
        "num_layers": None,
        "hidden_channels": None,
        "dropout": None,
        "batch_size": None,
        "lr": None,
        "epochs": None,
        "use_feature": None,
        "use_learnable_embedding": None,
    }
    
    if args.dataset == 'ddi':
        default_dict["use_feature"] = False
        default_dict["use_learnable_embedding"] = True
        default_dict["batch_size"] = 64 * 1024
        if args.model in ['sage', 'gcn', 'dea']:
            default_dict["num_layers"] = 2 
            default_dict["hidden_channels"] = 256
            default_dict["dropout"] = 0.5 
            default_dict["lr"] = 0.005 
            default_dict["epochs"] = 200
            if args.model in ['dea']:
                default_dict["epochs"] = 400
        if args.model in ['mlpcos', 'simplecos']:
            default_dict["num_layers"] = 2 
            default_dict["hidden_channels"] = 256
            default_dict["dropout"] = 0.5 
            default_dict["lr"] = 0.005 
            default_dict["epochs"] = 200
        if args.model in ['simple', 'simplecos']:
            default_dict["batch_size"] = 1024
            if args.model == 'simplecos':
                default_dict["use_feature"] = True
            # tofix        
            
    if args.dataset == "collab":
        default_dict["use_feature"] = True
        default_dict["use_learnable_embedding"] = True
        default_dict["batch_size"] = 16 * 1024
        if args.model in ['sage', 'gcn']:
            default_dict["num_layers"] = 3
            default_dict["hidden_channels"] = 256
            default_dict["dropout"] = 0.0
            default_dict["lr"] = 0.001
            default_dict["epochs"] = 200
        if args.model in ['mlpcos', 'simplecos']:
            default_dict["num_layers"] = 3
            default_dict["hidden_channels"] = 256
            default_dict["dropout"] = 0.0
            default_dict["lr"] = 0.00001
            default_dict["epochs"] = 400
            # to_fix  
            
    if args.dataset in ["reddit", "twitch", "fb"]:
        default_dict["use_feature"] = True
        default_dict["use_learnable_embedding"] = True
        default_dict["batch_size"] = 64 * 1024
        if args.model in ['sage', 'gcn']:
            default_dict["num_layers"] = 3
            default_dict["hidden_channels"] = 256
            default_dict["dropout"] = 0.0
            default_dict["lr"] = 0.005
            default_dict["epochs"] = 200
        if args.model in ['mlpcos', 'simplecos']:
            default_dict["batch_size"] = 1024
            default_dict["num_layers"] = 3
            default_dict["hidden_channels"] = 256
            default_dict["dropout"] = 0.0
            default_dict["lr"] = 0.001
            default_dict["epochs"] = 10

            
    if args.dataset == "email":
        default_dict["use_feature"] = False
        default_dict["use_learnable_embedding"] = True
        default_dict["batch_size"] = 16*1024
        if args.model in ['sage', 'gcn']:
            default_dict["num_layers"] = 3
            default_dict["hidden_channels"] = 300
            default_dict["dropout"] = 0.0
            default_dict["lr"] = 0.001
            default_dict["epochs"] = 200
        if args.model in ['mlpcos', 'simplecos']:
            default_dict["num_layers"] = 3
            default_dict["hidden_channels"] = 256
            default_dict["dropout"] = 0.0
            default_dict["batch_size"] = 1024 # need extra small batch size?
            default_dict["lr"] = 0.00004
            default_dict["epochs"] = 30

            
            
    over_write_list = ["num_layers", "hidden_channels", "dropout", "batch_size", "lr", "epochs", "use_feature", "use_learnable_embedding"]
    
    for attr in over_write_list:
        if getattr(args, attr) is None:
            setattr(args, attr, default_dict[attr])
    
#     if args.dataset in ['ddi']:
#         args.runs = 20
# probably too long to run ...            
    if args.model in ['adamic', 'simple', 'adamic_ogb', 'katz']:
        args.use_feature = False
        args.use_learnable_embedding = False
    if args.model == 'simplecos' and args.dataset != "email":
        args.use_learnable_embedding = False
        
    
    return args
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='models')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    
    parser.add_argument('--num_layers', type=int)
    parser.add_argument('--hidden_channels', type=int)
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--use_feature', type=bool)
    parser.add_argument('--use_learnable_embedding', type=bool)
    args = parser.parse_args()
    
    args = default_model_configs(args)
    print(args)
    
    
    