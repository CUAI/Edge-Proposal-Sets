from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import to_undirected
import torch_geometric.transforms as T
import torch_geometric
import pandas as pd
import shutil, os
import os.path as osp
import torch
import numpy as np
import networkx as nx
import random
from copy import deepcopy

class EmailDataset(InMemoryDataset):
    def __init__(self, root = 'dataset', transform=None, pre_transform=None):
        '''
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
        '''          
        self.root = root
        self.edge_file = "email-Eu-core-temporal.txt"
        self._num_nodes = 986
        self._num_static_edges = 24929
        self._num_temporal_edges = 332334
        
        super(EmailDataset, self).__init__(self.root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.edge_set = torch.load(self.processed_paths[1])
        self.edge_split = torch.load(self.processed_paths[2])

    @property
    def raw_file_names(self):
        return [self.edge_file]

    @property
    def processed_file_names(self):
        return ['email_geometric_data_processed.pt', 'email_edge_set.pt', 'email_edge_split.pt']

    def process(self):
        np.random.seed(42)
        edge_df = pd.read_csv(self.edge_file, sep=' ', header=None, names=['src', 'dst', 'unixts']).sort_values('unixts')
        uniq_edge_df = edge_df[~edge_df.duplicated(subset=['src', 'dst'], keep='last')]
        
        G = nx.OrderedDiGraph()
        for idx, row in uniq_edge_df.iterrows():
            src, dst, timestamp = row.loc['src'], row.loc['dst'], row['unixts']
            G.add_edge(src, dst, time=timestamp)
            
        nodes = np.unique(np.concatenate([edge_df['src'], edge_df['dst']]))
        for node in nodes:
            src_times = np.array(edge_df[edge_df['src'] == node]['unixts'])
            dst_times = np.array(edge_df[edge_df['dst'] == node]['unixts'])
            all_times = np.concatenate([src_times, dst_times])
            # randomly select 256 or length of total times
            num_times = min(len(all_times), 256)
            feature = np.concatenate([np.random.permutation(all_times)[:num_times], np.zeros(256-num_times)])
            mask = np.array([True] * num_times + [False] * (256-num_times))
            
            G.nodes[node]["feature"] = feature
            G.nodes[node]["mask"] = mask
        
        G = nx.convert_node_labels_to_integers(G, label_attribute='original')
        
        data_edges = list(G.edges(data=True))
        data_edges = [(u, v, t['time']) for u, v, t in data_edges]
        data_edges = sorted(data_edges, key=lambda x: x[2])
        edges = np.array([(u, v) for u, v, t in data_edges])
        
        self.edge_set = set(tuple(e) for e in edges)
        assert len(self.edge_set) == self._num_static_edges
        torch.save(self.edge_set, self.processed_paths[1])
        
        edge_split = self.make_edge_split(self.edge_set, edges)
        torch.save(edge_split, self.processed_paths[2])
        
        edge_index = edge_split['train']['edge'].t()
        data = {}
        
        for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
            for key, value in feat_dict.items():
                data[key] = [value] if i == 0 else data[key] + [value]
                
        for key, item in data.items():
            item = np.array(item).astype(np.float32)
            data[key] = torch.tensor(item)

        data['edge_index'] = edge_index.view(2, -1)
        data['num_nodes'] = G.number_of_nodes()
        data['x'] = data['feature']
        del data['feature']
        assert data['num_nodes'] == self._num_nodes
        data = torch_geometric.data.Data.from_dict(data)
        
        data = data if self.pre_transform is None else self.pre_transform(data)

        print('Saving...')
        torch.save(self.collate([data]), self.processed_paths[0])
    
    def get_edge_split(self):
        return self.edge_split
       
    def make_edge_split(self, edge_set, edges):
        np.random.seed(42)
        random.seed(42)
        n = self._num_static_edges
        edges = torch.tensor(edges)
        
        train = edges[:int(0.8*n)]
        valid = edges[int(0.8*n):int(0.9*n)]
        test  = edges[int(0.9*n):]
        
        exist = deepcopy(self.edge_set)
        
        valid_neg = []
        for i, p in enumerate(pair_generator(range(self._num_nodes), exist)):
            if i == 2500:
                break
            valid_neg.append(p)
            
        exist |= set(valid_neg)
        
        valid_neg = torch.from_numpy(np.array(valid_neg))
        
        test_neg = []
        for i, p in enumerate(pair_generator(range(self._num_nodes), exist)):
            if i == 2500:
                break
            test_neg.append(p)
            
        test_neg = torch.from_numpy(np.array(test_neg))
        
        return {"train": {"edge": train}, 
                "valid": {"edge": valid, "edge_neg": valid_neg}, 
                "test": {"edge": test, "edge_neg": test_neg}}

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

def pair_generator(numbers, existing_pairs=set()): 
    """Return an iterator of random pairs from a list of numbers.""" 
    # Keep track of already generated pairs 
    used_pairs = existing_pairs

    while True: 
        pair = tuple(random.sample(numbers, 2))
        if pair not in used_pairs:
            used_pairs.add(pair)
            yield pair

if __name__ == "__main__":
    dataset = EmailDataset()
    print(dataset.get_edge_split())
    data = dataset[0]
    print(data)
    print(len(dataset.edge_set))
    breakpoint()