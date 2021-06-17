import torch
from ogb.linkproppred import Evaluator
from torch.utils.data import DataLoader
from torch_geometric.utils import negative_sampling, to_undirected
from adamic_utils import get_A, AA, get_pos_neg_edges
from scipy.sparse.linalg import inv
from scipy.sparse import eye
import numpy as np

evaluators = {
    "collab": Evaluator(name='ogbl-collab'),
    "reddit": Evaluator(name='ogbl-collab'),
    "ddi": Evaluator(name='ogbl-ddi'),
    "email": Evaluator(name='ogbl-ddi'),
    "twitch": Evaluator(name='ogbl-ddi'),
    "fb": Evaluator(name='ogbl-collab'),
}
hits = {
    "collab": [10,50,100],
    "reddit":[10, 50, 100],
    "ddi": [10,20,30],
#     "email": [10, 50, 100], # this should be fixed
    "email": [10, 20, 30],
    "twitch": [10, 50, 100],
    "fb": [10, 20, 30],
}

def train(model, data, dataset_name, split_edge, optimizer, batch_size, use_params, model_str, device):
    model.train()
    pos_train_edge = split_edge['train']['edge'].to(device)
    
    row, col, _ = data.adj_t.coo()
    edge_index = torch.stack([col, row], dim=0)
    
    total_loss = total_examples = 0
    running_loss = None
    alpha = 0.99
    running_acc = None
    for idx, perm in enumerate(DataLoader(range(pos_train_edge.size(0)), batch_size,
                           shuffle=True)):

        if use_params:
            optimizer.zero_grad()
        pos_edge = to_undirected(pos_train_edge[perm].t(), data.num_nodes)
        
        if model_str in ['gcn', 'sage', "dea"]:
            if dataset_name in ["collab"] :
                # maybe it is need for replicating email results??
                neg_edge = torch.randint(0, data.num_nodes, pos_edge.size(),dtype=torch.long, device=pos_edge.device)
            else:
                neg_edge = negative_sampling(edge_index, num_nodes=data.num_nodes, num_neg_samples=pos_edge.size(1), method='dense')
        else:
            neg_edge = torch.randint(0, data.num_nodes, (1,pos_edge.size(1)),dtype=torch.long, device=pos_edge.device)
            neg_edge = torch.stack([pos_edge[0],neg_edge[0]])

        
        out = model( data.x, torch.cat([pos_edge, neg_edge], 1), data.adj_t).squeeze()
        
        pos_out = out[:pos_edge.size(1)]
        pos_loss = -torch.log(pos_out + 1e-8).mean() 

        neg_out = out[pos_edge.size(1):]
        neg_loss = -torch.log(1 - neg_out + 1e-8).mean() 
        
        loss = pos_loss + neg_loss
        if model_str in ["dea"]:
            pos_label = torch.ones(pos_edge.shape[1], )
            neg_label = torch.zeros(neg_edge.shape[1], )
            edge_label = torch.cat([pos_label, neg_label], dim=0).to(device)

            loss = model.loss(out, edge_label.type_as(out))
            
        if use_params:
            loss.backward()
        
        acc = ((neg_out < 0.5).sum() + (pos_out>0.5).sum()).item()/(0.+out.size(0))
        if running_loss is None:
            running_loss = loss.item()
            running_acc = acc
        running_loss = (1-alpha)*loss.item() + alpha*running_loss
        running_acc = (1-alpha)*acc + alpha*running_acc
#         print(running_loss, running_acc, pos_loss.item(), neg_loss.item())

        if use_params:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        if use_params:
            optimizer.step()
        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples

@torch.no_grad()
def test(model, data, split_edge, evaluator, batch_size, args, device):
    model.eval()

    pos_train_edge = split_edge['eval_train']['edge'].to(device)
    pos_valid_edge = split_edge['valid']['edge'].to(device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(device)
    pos_test_edge = split_edge['test']['edge'].to(device)
    neg_test_edge = split_edge['test']['edge_neg'].to(device)

    pos_train_preds = []
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size):
        edge = pos_train_edge[perm].t()
        pos_train_preds.append(model( data.x,edge, data.adj_t).cpu().squeeze())
    pos_train_pred = torch.cat(pos_train_preds, dim=0)
    
    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
        edge = pos_valid_edge[perm].t()
        pos_valid_preds.append(model(  data.x,edge, data.adj_t).cpu().squeeze())
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
        edge = neg_valid_edge[perm].t()
        neg_valid_preds.append(model( data.x,edge, data.adj_t).cpu().squeeze())
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        pos_test_preds.append(model( data.x,edge, data.full_adj_t).cpu().squeeze())
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        neg_test_preds.append(model( data.x,edge, data.full_adj_t).cpu().squeeze())
    neg_test_pred = torch.cat(neg_test_preds, dim=0)
    
    results = {}
    for K in hits[args.dataset]:
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

    return results

def test_adamic(model, data, split_edge, evaluator, batch_size, args, device):
    assert args.model == "adamic_ogb"
    A = get_A(data.full_adj_t, data.num_nodes)
    pos_val_edge, neg_val_edge = get_pos_neg_edges('valid', split_edge, 
                                                   data.edge_index, 
                                                   data.num_nodes)
    pos_test_edge, neg_test_edge = get_pos_neg_edges('test', split_edge, 
                                                 data.edge_index, 
                                                 data.num_nodes)
    pos_train_edge = split_edge['train']['edge'].to(device)
    pos_train_pred = torch.ones(pos_train_edge.size(0))
    pos_valid_pred, pos_valid_edge = eval('AA')(A, pos_val_edge)
    neg_valid_pred, neg_valid_edge = eval('AA')(A, neg_val_edge)
    pos_test_pred, pos_test_edge = eval('AA')(A, pos_test_edge)
    neg_test_pred, neg_test_edge = eval('AA')(A, neg_test_edge)
    
    results = {}
    for K in hits[args.dataset]:
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

    return results

def test_katz(model, data, split_edge, evaluator, batch_size, args, device):
    assert args.model == "katz"
    A = get_A(data.full_adj_t, data.num_nodes)
    A_train = get_A(data.adj_t, data.num_nodes)
    pos_train_edge = split_edge['eval_train']['edge']
    pos_valid_edge = split_edge['valid']['edge']
    neg_valid_edge = split_edge['valid']['edge_neg']
    pos_test_edge = split_edge['test']['edge']
    neg_test_edge = split_edge['test']['edge_neg']

    beta = 0.05
    if args.dataset == "collab":
        H_train = beta * A_train
        for _ in range(2):
            H_train += beta * (A_train @ H_train)

        H = beta * A
        for _ in range(2):
            H += beta * (A @ H)
    else:
        H_train = inv(eye(data.num_nodes) - beta * A_train) - eye(data.num_nodes) 
        H = inv(eye(data.num_nodes) - beta * A) - eye(data.num_nodes) 
    
    pos_train_preds = []
    for perm in DataLoader(range(pos_train_edge.size(0)), 100):
        edge = pos_train_edge[perm].t()
        pos_train_preds.append(torch.tensor(H_train[np.array(edge[0]), np.array(edge[1])]).squeeze(0))
    pos_train_pred = torch.cat(pos_train_preds, dim=0)    
    
    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
        edge = pos_valid_edge[perm].t()
        pos_valid_preds.append(torch.tensor(H_train[np.array(edge[0]), np.array(edge[1])]).squeeze(0))
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
        edge = neg_valid_edge[perm].t()
        neg_valid_preds.append(torch.tensor(H_train[np.array(edge[0]), np.array(edge[1])]).squeeze(0))
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        pos_test_preds.append(torch.tensor(H[np.array(edge[0]), np.array(edge[1])]).squeeze(0))
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        neg_test_preds.append(torch.tensor(H[np.array(edge[0]), np.array(edge[1])]).squeeze(0))
    neg_test_pred = torch.cat(neg_test_preds, dim=0)
    
    results = {}
    for K in hits[args.dataset]:
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

    return results

if __name__ == "__main__":
    main()