import argparse
import random
import os

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.utils import to_undirected
import numpy as np

from ogb.linkproppred import PygLinkPropPredDataset
from reddit.data import RedditDataset
from email_data.data import EmailDataset
from twitch.data import TwitchDataset
from fb.data import FBDataset
from models import build_model, default_model_configs
from train_and_eval import train, test, hits, evaluators, test_adamic, test_katz, test_resource_allocation

from torch_sparse import SparseTensor
from torch_sparse import sum as sparse_sum
from logger import Logger
import pandas as pd
from datetime import datetime
from tqdm import tqdm

from pathlib import Path

def add_edges(dataset, edge_index, edge_weight, extra_edges, num_nodes):
    full_edge_index = torch.cat([edge_index.clone(), extra_edges], dim=-1)
    new_edge_weight = torch.ones(extra_edges.shape[1])
    full_edge_weights = torch.cat([edge_weight, new_edge_weight], 0) 
    adj_t = SparseTensor.from_edge_index(full_edge_index, full_edge_weights, sparse_sizes = [num_nodes,num_nodes])
    adj_t = adj_t.to_symmetric() 
    if dataset != "collab":
        adj_t = adj_t.fill_value(1.)
    return adj_t

def get_dataset(dataset):

    if dataset == "ddi":
        dataset = PygLinkPropPredDataset(name='ogbl-ddi')
    elif dataset == "ppa":
        dataset = PygLinkPropPredDataset(name='ogbl-ppa')        
    elif dataset == "collab":
        dataset = PygLinkPropPredDataset(name='ogbl-collab')
    elif dataset == "email":
        dataset = EmailDataset()
    elif dataset == "reddit":
        dataset = RedditDataset()
    elif dataset == "twitch":
        dataset = TwitchDataset()
    elif dataset == "fb":
        dataset = FBDataset() 
    else:
        raise NotImplemented
    return dataset

def spectral(data, edge_index, dataset_name):
    try:
        x = torch.load(f'embeddings/spectral_{dataset_name}.pt')
        print('Using cache')
        return x
    except:
        print(f'embeddings/spectral_{dataset_name}.pt not found or not enough iterations! Regenerating it now')
    from julia.api import Julia
    jl = Julia(compiled_modules=False)
    from julia import Main
    Main.include("./norm_spec.jl")
    print('Setting up spectral embedding')
    edge_index = to_undirected(edge_index)

    
    N = data.num_nodes
    row, col = edge_index
    adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
    adj = adj.to_scipy(layout='csr')
    result = torch.tensor(Main.main(adj, 128)).float()
    torch.save(result, f'embeddings/spectral_{dataset_name}.pt')
        
    return result


def get_data(args):
    dataset = get_dataset(args.dataset)
    data = dataset[0]
    
    edge_index = data.edge_index
    edge_weight = torch.ones(data.edge_index.size(1))
    if "edge_weight" in data:
        edge_weight = data.edge_weight.view(-1)
    
    split_edge = dataset.get_edge_split()
    idx = torch.randperm(split_edge['train']['edge'].size(0))
    idx = idx[:split_edge['valid']['edge'].size(0)]
    split_edge['eval_train'] = {'edge': split_edge['train']['edge'][idx]}
 
    data = T.ToSparseTensor()(data)
    data.adj_t = data.adj_t.to_symmetric()
    
    # features
    if not args.use_feature:
        data.x = None
#     if args.dataset == "ddi" and args.use_feature:
        
# #         data.x = spectral(data, edge_index, "ddi")
# #             if args.use_node_embedding:
#         if args.use_node_embedding:
#             print('load node2vec features:')
#             data.x = torch.load('ddi_embedding.pt', map_location='cpu')

# #         print('load extra features:')
# #         x_df = pd.read_csv('ddi/x_feature.csv')
# #         x_feature_numpy = x_df.to_numpy()
# #         x_feature = torch.Tensor(x_feature_numpy)
# #         print('extra feature shape:', x_feature.shape)

# #         # Normalize to 0-1
# #         x_max = torch.max(x_feature, dim=0, keepdim=True)[0]
# #         x_min = torch.min(x_feature, dim=0, keepdim=True)[0]
# #         data.x = (x_feature - x_min) / (x_max - x_min + 1e-6)
    
#     if args.dataset == "collab" and args.use_node_embedding:
#         print('load node2vec features:')
#         data.x = torch.cat([data.x, torch.load('collab_embedding.pt', map_location='cpu')], dim = 1)
        
    return edge_index, edge_weight, split_edge, data


def main():
    parser = argparse.ArgumentParser(description='General Experiment')
    # experiment configs
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--model', type=str)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--sorted_edge_path', type=str, default= "")
    parser.add_argument('--num_sorted_edge', type=int)
    parser.add_argument('--sweep_max', type=int)
    parser.add_argument('--sweep_min', type=int)
    parser.add_argument('--sweep_num', type=int)
    parser.add_argument('--only_supervision', action="store_true", default=False)
    parser.add_argument('--also_supervision', action="store_true", default=False)
    parser.add_argument('--gen_dataset_only', action="store_true", default=False)
    parser.add_argument('--valid_proposal', action="store_true", default=False)
#     parser.add_argument('--use_node_embedding', action="store_true", default=False)
    
    # save results
    parser.add_argument('--out_name', type=str)
    parser.add_argument('--save_models', action="store_true", default=False)
    
    # model configs; overwrite defaults if specified
    parser.add_argument('--num_layers', type=int)
    parser.add_argument('--hidden_channels', type=int)
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--use_feature', type=bool)
    parser.add_argument('--use_learnable_embedding', type=bool)
        
    # other settings
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)   
    parser.add_argument('--eval_steps', type=int, default=1)
    
    args = parser.parse_args()
    
    args = default_model_configs(args)
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    Path("curves").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True) 
    
    assert not (args.only_supervision and args.also_supervision)
    
    if args.out_name is None:
        args.out_name = args.dataset + "_" + args.model 
        if args.only_supervision:
            args.out_name += "_onlys"
        elif args.also_supervision:
            args.out_name += "_alsos"
        elif args.valid_proposal:
            args.out_name += "_validproposal"
#         elif args.use_node_embedding:
#             args.out_name += "_node2vec"
    ##############
    ## load data and model
    ##############
    
    edge_index, edge_weight, split_edge, data = get_data(args)
    if args.gen_dataset_only:
        return
    
    if args.model is None:
        raise "Model not specified"
    
    data = data.to(device)
    model = build_model(args, data, device)
    print(f'using model {model}')

    
    ##############
    ## test setting
    ##############
    
    evaluator = evaluators[args.dataset]
    K = hits[args.dataset]
    print("Evaluating at hits: ", K)
    
    
    ##############
    ## prepare adding extra edges
    ##############
    
    if args.sorted_edge_path:
        # should be sorted of shape [E, 2] or [E, 3], where the 3rd index is possibly a score
        sorted_test_edges = torch.load(f"filtered_edges/{args.sorted_edge_path}")
        print('sorted test edges', sorted_test_edges.size())
        
        if args.valid_proposal:
            # concat to top 
            valid_pos = split_edge['valid']['edge']
            valid_pos_set = set()
            for e in valid_pos.numpy():
                valid_pos_set.add(tuple(e))
                valid_pos_set.add(tuple(reversed(e)))          
            sorted_test_edges_after = [np.array([u,v,100000.0]) for u,v in valid_pos_set]
            
               
            d = set()
            for t in split_edge['valid']['edge'].numpy():
                d.add(tuple(sorted(t)))
            
            # remove validaiton edges                   

            for t in sorted_test_edges.numpy():
                u, v, score = tuple(t)
                if (int(u),int(v)) in d or (int(v),int(u)) in d:
                    continue
                sorted_test_edges_after.append(t)
            sorted_test_edges_after = torch.tensor(sorted_test_edges_after)
            
            
            new_proposal_set = set()
            for e in sorted_test_edges_after.numpy()[:len(valid_pos_set)]:
                u, v, score = tuple(e)
                new_proposal_set.add((int(u),int(v)))
            assert len(new_proposal_set &  valid_pos_set) == len(valid_pos_set)
            sorted_test_edges = sorted_test_edges_after
        
    else:
        # fake [E, 2]
        sorted_test_edges = torch.zeros(42, 2)
        
    curve = []
    index_ends = []
    
    if args.sweep_num:
        if args.sweep_min is None:
            args.sweep_min = 0
        if args.sweep_max is None:
            args.sweep_max = (args.sweep_num -1)* 1000
        for i in range(args.sweep_num + 1):
            index_end = int(i * (args.sweep_max - args.sweep_min)/(args.sweep_num))
            index_ends.append(args.sweep_min + index_end)
    elif args.num_sorted_edge :
        index_ends.append(args.num_sorted_edge)
    else:
        index_ends.append(0)
    print(f"Scheduled extra edges sweep: {index_ends} x {args.runs}")

        
    ##############
    ## sweeps
    ##############

    for index_end in index_ends: 
        curve_point = []
        loggers = {
            f'Hits@{K[0]}': Logger(args.runs, args),
            f'Hits@{K[1]}': Logger(args.runs, args),
            f'Hits@{K[2]}': Logger(args.runs, args),
        }
        print('---------------------')
        print(f'Using {index_end} highest scoring edges')
        print('---------------------')
        
        ##############
        ## adding edges
        ##############
        
        extra_edges = sorted_test_edges[: int(index_end),:2].t().long()
        
        assert extra_edges.size(0) == 2
        assert extra_edges.size(1) == index_end
        
        if not args.only_supervision:
            data.adj_t = add_edges(args.dataset, edge_index, edge_weight, extra_edges, data.num_nodes).to(device)  

     
            
        if args.only_supervision or args.also_supervision:
            split_edge['train']['edge'] = torch.cat((split_edge['train']['edge'], extra_edges.t()))
            
        if args.dataset in ["collab", "email", "reddit"]:
            # use eval edges only during evaluation
            val_edge_index = split_edge['valid']['edge'].t()
            val_edge_index = to_undirected(val_edge_index)
            full_extra_edges = torch.cat([extra_edges, val_edge_index], dim=-1)            
            data.full_adj_t =  add_edges(args.dataset, edge_index, edge_weight, full_extra_edges, data.num_nodes).to(device)       
        else:
            data.full_adj_t = data.adj_t
    
        
        for run in range(args.runs):
            model.reset_parameters()
            use_params = sum(p.numel() for p in model.parameters() if p.requires_grad) > 0
            print(sum(p.numel() for p in model.parameters() if p.requires_grad))
            if use_params:
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            else:
                optimizer = None
                
            if not use_params:
                args.epochs = 1

            highest_eval = 0
            for epoch in tqdm(range(1, 1 + args.epochs)):
                if use_params:
                    loss = train(model, data, args.dataset, split_edge, optimizer,
                                 args.batch_size, use_params, args.model, device)
                else:
                    loss = -1

                if epoch % args.eval_steps == 0:
                    if args.model not in ["adamic_ogb", "resource_allocation","katz"]:
                        results = test(model, data, split_edge, evaluator,
                                   args.batch_size, args, device)
                    elif args.model == "adamic_ogb":
                        results = test_adamic(model, data, split_edge, evaluator,
                                   args.batch_size, args, device)
                    elif args.model == "resource_allocation":
                        results = test_resource_allocation(model, data, split_edge, evaluator,
                                   args.batch_size, args, device)    
                    elif args.model == "katz":
                        results = test_katz(model, data, split_edge, evaluator,
                                   args.batch_size, args, device)
                    for key, result in results.items():
                        loggers[key].add_result(run, result)

                    if epoch % args.log_steps == 0:
                        for key, result in results.items():
                            train_hits, valid_hits, test_hits = result
                            if key == f"Hits@{K[1]}":
                                if valid_hits >= highest_eval:
                                    highest_eval = valid_hits
                                    filename = f'{args.out_name}|{args.sorted_edge_path.split(".")[0]}|{index_end}|{run}.pt'
                                    if args.save_models and use_params:
                                        torch.save(model.state_dict(), os.path.join('models', filename))

                            print(key)
                            print(f'Run: {run + 1:02d}, '
                                  f'Epoch: {epoch:02d}, '
                                  f'Loss: {loss:.4f}, '
                                  f'Train: {100 * train_hits:.2f}%, '
                                  f'Valid: {100 * valid_hits:.2f}%, '
                                  f'Test: {100 * test_hits:.2f}%')
                        print('---')

            for key in loggers.keys():
                print(key)
                loggers[key].print_statistics(run)

                if key == f"Hits@{K[1]}":
                    result = 100 * torch.tensor(loggers[key].results[run])
                    argmax = result[:, 1].argmax().item()
                    curve_point = [index_end, result[argmax, 1], result[argmax, 2]]

            time = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
            filename = f'{args.out_name}|{args.sorted_edge_path.split(".")[0]}|{index_end}|{time}.pt'    
            print(curve_point)
            print("Saving curve to ", filename)
            torch.save(curve_point, os.path.join('curves', filename))
        

        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics()


if __name__ == "__main__":
    main()
