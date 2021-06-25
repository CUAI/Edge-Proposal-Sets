from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import to_undirected
import torch_geometric.transforms as T
import torch_geometric
import torch_sparse
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
import math
import json
import urllib.request

class TwitchDataset(InMemoryDataset):
    def __init__(self, root = 'dataset/twitch', transform=None, pre_transform=None):
        '''
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
        '''
        self.root = root
        self._num_nodes = 9498
        super(TwitchDataset, self).__init__(self.root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.edge_set = torch.load(self.processed_paths[1])
        self.edge_split = torch.load(self.processed_paths[2])

    @property
    def processed_file_names(self):
        files = ['twitch_data_processed.pt', 'twitch_edge_set.pt', 'twitch_split.pt']
        return files

    def process(self):
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

        data = pd.read_csv('twitch/musae_DE_edges.csv')
        edges = data.values.tolist()
        print(len(edges))
        edges = [list(sorted([int(edge[0]), int(edge[1])])) for edge in edges]
        print(len(edges))
        edges = [edge for edge in edges if edge[0] < edge[1]] # remove self loops
        print(len(edges))

        assert np.all([ab[0] < ab[1] for ab in edges])

        num_edges = len(edges)
        print('total amount of edges:', num_edges)

        random.shuffle(edges)

        n = num_edges
        train_edges = edges[:int(0.8*n)]
        print('train amount:', len(train_edges))
        val_edges = edges[int(0.8*n):int(0.9*n)]
        print('val amount:', len(val_edges))
        test_edges = edges[int(0.9*n):]
        print('test amount:', len(test_edges))

        data = torch_geometric.data.Data(edge_index=torch.tensor(train_edges).t())
        assert data.edge_index.size(0) == 2
        assert data.edge_index.size(1) == len(train_edges)
        print('edge index size', data.edge_index.size())

        data.num_nodes = self._num_nodes

        with open(f"twitch/musae_DE_features.json", 'r') as f:
            j = json.load(f)

        n = self._num_nodes
        features = np.zeros((n,3170))
        for node, feats in j.items():
            if int(node) >= n:
                continue
                print('continued')
            features[int(node), np.array(feats, dtype=int)] = 1
        features = features[:, np.sum(features, axis=0) != 0] # remove zero cols
        data.x = torch.tensor(features).float()

        data = data if self.pre_transform is None else self.pre_transform(data)
        val_edges = torch.tensor(val_edges).t()
        print('val edges size', val_edges.size())
        test_edges = torch.tensor(test_edges).t()
        print('test edges size', test_edges.size())

        train_set = set(tuple(e.numpy()) for e in data.edge_index.t())
        assert len(train_set) != 2
        print('train set size', len(train_set))

        val_set = set(tuple(e.numpy()) for e in val_edges.t())
        assert len(val_set) != 2
        print('val set size', len(val_set))

        test_set = set(tuple(e.numpy()) for e in test_edges.t())
        assert len(test_set) != 2
        print('test set size', len(test_set))

        self.edge_set = train_set | val_set | test_set
        torch.save(self.edge_set, self.processed_paths[1])

        edge_split = self.make_edge_split(self.edge_set, data.edge_index, val_edges, test_edges)
        assert np.array([a < b for a, b in edge_split["train"]["edge"]]).all()
        assert np.array([a < b for a, b in edge_split["valid"]["edge"]]).all()
        assert np.array([a < b for a, b in edge_split["test"]["edge"]]).all()
        assert np.array([a < b for a, b in edge_split["valid"]["edge_neg"]]).all()
        assert np.array([a < b for a, b in edge_split["test"]["edge_neg"]]).all()
        torch.save(edge_split, self.processed_paths[2])

        print('Saving...')
        data.edge_index = torch_geometric.utils.to_undirected(data.edge_index, self._num_nodes)
        torch.save(self.collate([data]), self.processed_paths[0])

    def get_edge_split(self):
        return self.edge_split

    def make_edge_split(self, edge_set, train_edges, val_edges, test_edges):
        np.random.seed(42)
        random.seed(42)
        torch.manual_seed(42)

        exist = deepcopy(edge_set)

        edges_to_generate = val_edges.size(1)
        print('generating', edges_to_generate, 'negative edges')

        valid_neg = []
        for i, p in enumerate(pair_generator(range(self._num_nodes), exist)):
            if i == edges_to_generate:
                break
            valid_neg.append(p)

        exist |= set(valid_neg)

        valid_neg = torch.from_numpy(np.array(valid_neg))

        test_neg = []
        for i, p in enumerate(pair_generator(range(self._num_nodes), exist)):
            if i == edges_to_generate:
                break
            test_neg.append(p)

        test_neg = torch.from_numpy(np.array(test_neg))

        return {"train": {"edge": train_edges.t()},
                "valid": {"edge": val_edges.t(), "edge_neg": valid_neg},
                "test": {"edge": test_edges.t(), "edge_neg": test_neg}}

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

def pair_generator(numbers, existing_pairs=set()):
    """Return an iterator of random pairs from a list of numbers."""
    # Keep track of already generated pairs
    used_pairs = existing_pairs

    while True:
        pair = tuple(sorted(random.sample(numbers, 2)))
        if pair not in used_pairs:
            used_pairs.add(pair)
            used_pairs.add(tuple(reversed(pair)))
            yield pair

if __name__ == "__main__":
    dataset = TwitchDataset(transform=T.ToSparseTensor())
#     dataset = TwitchDataset()
    print(dataset.get_edge_split())
    data = dataset[0]
    print(data)
    print(len(dataset.edge_set))
