from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import to_undirected
import torch_geometric.transforms as T
import torch_geometric
import pandas as pd
import shutil, os
import os.path as osp
import requests
import torch
import numpy as np
import networkx as nx
import random
from copy import deepcopy
from collections import defaultdict

import urllib.request

class RedditDataset(InMemoryDataset):
    def __init__(self, root = 'dataset/reddit', transform=None, pre_transform=None):
        '''
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
        '''          
        self.root = root
        self.edge_body_file = "reddit-body.tsv"
        self.edge_title_file = "reddit-title.tsv"
        self.embeddings_file = "reddit-embeddings.csv"
        self._num_nodes = 30744
        self._num_edges = 277041
        
        super(RedditDataset, self).__init__(self.root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.edge_set = torch.load(self.processed_paths[1])
        self.edge_split = torch.load(self.processed_paths[2])
        self.node_split = torch.load(self.processed_paths[3])

    @property
    def raw_file_names(self):
        return [self.edge_body_file, self.edge_title_file, self.embeddings_file]

    @property
    def processed_file_names(self):
        return ['reddit_geometric_data_processed.pt', 'reddit_edge_set.pt', 'reddit_edge_split.pt', 'reddit_node_split.pt']
    
    def download(self):
        print('Downloading...')
        download_url('http://snap.stanford.edu/data/soc-redditHyperlinks-body.tsv', osp.join(self.raw_dir, self.edge_body_file))
        download_url('http://snap.stanford.edu/data/soc-redditHyperlinks-title.tsv', osp.join(self.raw_dir, self.edge_title_file))
        download_url('http://snap.stanford.edu/data/web-redditEmbeddings-subreddits.csv', osp.join(self.raw_dir, self.embeddings_file))

    def process(self):
        np.random.seed(42)
        
        body_df = pd.read_csv(osp.join(self.raw_dir, self.edge_body_file), sep='\t')
        body_df['TIMESTAMP'] = pd.to_datetime(body_df['TIMESTAMP'])
        title_df = pd.read_csv(osp.join(self.raw_dir, self.edge_title_file), sep='\t')
        title_df['TIMESTAMP'] = pd.to_datetime(title_df['TIMESTAMP'])
        master_df = pd.concat([body_df, title_df], ignore_index=True)
        
        emb_df = pd.read_csv(osp.join(self.raw_dir, self.embeddings_file), header=None)
        emb_subs = set(np.array(emb_df.loc[:,0]))
        
        to_remove = []
        for idx, row in master_df.iterrows():
            src = row.loc['SOURCE_SUBREDDIT']
            dst = row.loc['TARGET_SUBREDDIT']
            if src not in emb_subs or dst not in emb_subs:
                to_remove.append(idx)
        
        clean_df = master_df.drop(to_remove)
        clean_df = clean_df.sort_values("TIMESTAMP")
        
        G = nx.OrderedDiGraph()
        for idx, row in clean_df.iterrows():
            src, dst = row.loc['SOURCE_SUBREDDIT'], row.loc['TARGET_SUBREDDIT']
            timestamp = row['TIMESTAMP']
            G.add_edge(src, dst, time=timestamp)
        
        for idx, row in emb_df.iterrows():
            subreddit = row.iloc[0]
            feature = row.iloc[1:]
            if subreddit in G.nodes:
                G.nodes[subreddit]['feature'] = feature
            
        G = nx.convert_node_labels_to_integers(G)
        
        data_edges = list(G.edges(data=True))
        data_edges = [(u, v, t['time']) for u, v, t in data_edges]
        data_edges = sorted(data_edges, key=lambda x: x[2])
        
        node_times = defaultdict(list)
        for u, v, t in data_edges:
            node_times[u].append(t)
            node_times[v].append(t)
        
        node_split = dict()
        for node, times in node_times.items():
            assert node not in node_split
            node_split[node] = max(times)
        
        node_time_list = [(n, t) for n, t in node_split.items()]
        node_time_list = sorted(node_time_list, key=lambda x: x[1])
        node_time_list = [n[0] for n in node_time_list]
        assert len(node_time_list) == self._num_nodes
        m = self._num_nodes
        
        train_nodes = node_time_list[:int(0.4*m)]
        valid_nodes = node_time_list[int(0.4*m):int(0.5*m)]
        test_nodes = node_time_list[int(0.5*m):]
        
        train_nodes, valid_nodes, test_nodes = [torch.tensor(nodes) for nodes in [train_nodes, valid_nodes, test_nodes]]
        
        node_split = {'train': train_nodes, 'valid': valid_nodes, 'test': test_nodes}
        torch.save(node_split, self.processed_paths[3])
        
        edges = np.array([(u, v) for u, v, t in data_edges])
        
        self.edge_set = set(tuple(e) for e in edges)
        assert len(self.edge_set) == self._num_edges
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
    
    def get_node_split(self):
        return self.node_split
       
    def make_edge_split(self, edge_set, edges):
        np.random.seed(42)
        random.seed(42)
        n = self._num_edges
        edges = torch.tensor(edges)
        
        train = edges[:int(0.8*n)]
        valid = edges[int(0.8*n):int(0.9*n)]
        test  = edges[int(0.9*n):]
        
        exist = deepcopy(self.edge_set)
        
        valid_neg = []
        for i, p in enumerate(pair_generator(range(self._num_nodes), exist)):
            if i == 80000:
                break
            valid_neg.append(p)
            
        exist |= set(valid_neg)
        
        valid_neg = torch.from_numpy(np.array(valid_neg))
        
        test_neg = []
        for i, p in enumerate(pair_generator(range(self._num_nodes), exist)):
            if i == 80000:
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

def download_url(url, filename):
    get_response = requests.get(url,stream=True)
    with open(filename, 'wb') as f:
        for chunk in get_response.iter_content(chunk_size=1024):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
            

if __name__ == "__main__":
    dataset = RedditDataset(transform=T.ToSparseTensor())
    print(dataset.get_edge_split())
    data = dataset[0]
    print(data)
    print(len(dataset.edge_set))
    print(dataset.get_node_split())