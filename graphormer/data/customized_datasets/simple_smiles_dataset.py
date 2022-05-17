# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os.path as osp

import numpy as np
import pandas as pd
import torch
from dgl.data import QM9
from graphormer.data import register_dataset
from ogb.utils.mol import smiles2graph
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.data.in_memory_dataset import InMemoryDataset
from tqdm import tqdm


@register_dataset("customized_qm9_dataset")
def create_customized_dataset():
    dataset = QM9(label_keys=["mu"])
    num_graphs = len(dataset)

    # customized dataset split
    train_valid_idx, test_idx = train_test_split(
        np.arange(num_graphs), test_size=num_graphs // 10, random_state=0
    )
    train_idx, valid_idx = train_test_split(
        train_valid_idx, test_size=num_graphs // 5, random_state=0
    )
    return {
        "dataset": dataset,
        "train_idx": train_idx,
        "valid_idx": valid_idx,
        "test_idx": test_idx,
        "source": "dgl"
    }


class SimpleSMILESDataset(InMemoryDataset):
    """
    csv file with "id,smiles,prop" as columns
    ref: ogb/lsc/pcqm4mv2_pyg.py and https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html
    """
    def __init__(self, dataset_name):
        self.smiles2graph = smiles2graph
        # self.data_frame = pd.read_csv(csv_filename, names=['mid', 'smiles', 'prop'])
        root = './data'  # original root dir for all datasets
        # dataset_name = 'sampled_pcqm4mv2'
        self.folder = osp.join(root, dataset_name)
        self.root = self.folder
        super().__init__(self.folder, transform=None, pre_transform=None)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # locate in: ./data/dataset_name/raw/mysmiles.csv
        return 'raw/mysmiles.csv'

    @property
    def processed_file_names(self):
        # locate in: ./data/dataset_name/processed/...
        return 'geometric_data_processed.pt'

    def process(self):
        data_df = pd.read_csv(osp.join(self.folder, self.raw_file_names), names=['mid', 'smiles', 'prop'], header=0)
        smiles_list = data_df['smiles']
        prop_list = data_df['prop']
        mid_list = data_df['mid']

        print('Converting SMILES strings into graphs...')
        data_list = []
        for i in tqdm(range(len(smiles_list))):
            data = Data()
            smiles = smiles_list[i]
            prop = prop_list[i]
            mid = mid_list[i]
            graph = self.smiles2graph(smiles)
            
            assert(len(graph['edge_feat']) == graph['edge_index'].shape[1])
            assert(len(graph['node_feat']) == graph['num_nodes'])

            data.__num_nodes__ = int(graph['num_nodes'])
            data.edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
            data.edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.int64)
            data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
            data.y = torch.FloatTensor([prop])
            data.mid = torch.LongTensor([mid])
            data.smiles = smiles.strip()

            data_list.append(data)

        # double-check prediction target
        split_dict = self.get_idx_split(len(data_list))
        assert(all([not torch.isnan(data_list[i].y)[0] for i in split_dict['train']]))
        assert(all([not torch.isnan(data_list[i].y)[0] for i in split_dict['valid']]))
        assert(all([not torch.isnan(data_list[i].y)[0] for i in split_dict['test']]))

        data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self, num_data):
        """idx split to train/valid/test"""
        # split_dict = replace_numpy_with_torchtensor(torch.load(osp.join(self.root, 'split_dict.pt')))
        seed = 42
        train_valid_idx, test_idx = train_test_split(
                np.arange(num_data),
                test_size=num_data // 10,
                random_state=seed,
            )
        train_idx, valid_idx = train_test_split(
            train_valid_idx, 
            test_size=num_data // 5, 
            random_state=seed)
        return {
            'train': torch.from_numpy(train_idx),
            'valid': torch.from_numpy(valid_idx),
            'test': torch.from_numpy(test_idx)
        }


def get_splits(n_samples):
    seed = 42
    train_valid_idx, test_idx = train_test_split(np.arange(n_samples), test_size=n_samples // 10, random_state=seed)
    train_idx, valid_idx = train_test_split(train_valid_idx, test_size=n_samples // 5, random_state=seed)
    return train_idx, valid_idx, test_idx
        

@register_dataset("LogS_dataset")
def create_simple_smiles_dataset():
    dataset = SimpleSMILESDataset('LogS')
    num_graphs = len(dataset)
    train_idx, valid_idx, test_idx = get_splits(num_graphs)
    return { "dataset": dataset, "train_idx": train_idx, "valid_idx": valid_idx, "test_idx": test_idx, "source": "pyg" }


@register_dataset("LogS_all_dataset")
def create_simple_smiles_dataset():
    dataset = SimpleSMILESDataset('LogS_all')
    num_graphs = len(dataset)
    train_idx, valid_idx, test_idx = get_splits(num_graphs)
    return { "dataset": dataset, "train_idx": train_idx, "valid_idx": valid_idx, "test_idx": test_idx, "source": "pyg" }


@register_dataset("CL_dataset")
def create_simple_smiles_dataset():
    dataset = SimpleSMILESDataset('CL')
    num_graphs = len(dataset)
    train_idx, valid_idx, test_idx = get_splits(num_graphs)
    return { "dataset": dataset, "train_idx": train_idx, "valid_idx": valid_idx, "test_idx": test_idx, "source": "pyg" }


@register_dataset("CL_all_dataset")
def create_simple_smiles_dataset():
    dataset = SimpleSMILESDataset('CL_all')
    num_graphs = len(dataset)
    train_idx, valid_idx, test_idx = get_splits(num_graphs)
    return { "dataset": dataset, "train_idx": train_idx, "valid_idx": valid_idx, "test_idx": test_idx, "source": "pyg" }


@register_dataset("LD50_dataset")
def create_simple_smiles_dataset():
    dataset = SimpleSMILESDataset('LD50')
    num_graphs = len(dataset)
    train_idx, valid_idx, test_idx = get_splits(num_graphs)
    return { "dataset": dataset, "train_idx": train_idx, "valid_idx": valid_idx, "test_idx": test_idx, "source": "pyg" }


@register_dataset("LD50_all_dataset")
def create_simple_smiles_dataset():
    dataset = SimpleSMILESDataset('LD50_all')
    num_graphs = len(dataset)
    train_idx, valid_idx, test_idx = get_splits(num_graphs)
    return { "dataset": dataset, "train_idx": train_idx, "valid_idx": valid_idx, "test_idx": test_idx, "source": "pyg" }


@register_dataset("papp_dataset")
def create_simple_smiles_dataset():
    dataset = SimpleSMILESDataset('papp')
    num_graphs = len(dataset)
    train_idx, valid_idx, test_idx = get_splits(num_graphs)
    return { "dataset": dataset, "train_idx": train_idx, "valid_idx": valid_idx, "test_idx": test_idx, "source": "pyg" }


@register_dataset("papp_all_dataset")
def create_simple_smiles_dataset():
    dataset = SimpleSMILESDataset('papp_all')
    num_graphs = len(dataset)
    train_idx, valid_idx, test_idx = get_splits(num_graphs)
    return { "dataset": dataset, "train_idx": train_idx, "valid_idx": valid_idx, "test_idx": test_idx, "source": "pyg" }


@register_dataset("PPB_dataset")
def create_simple_smiles_dataset():
    dataset = SimpleSMILESDataset('PPB')
    num_graphs = len(dataset)
    train_idx, valid_idx, test_idx = get_splits(num_graphs)
    return { "dataset": dataset, "train_idx": train_idx, "valid_idx": valid_idx, "test_idx": test_idx, "source": "pyg" }


@register_dataset("PPB_all_dataset")
def create_simple_smiles_dataset():
    dataset = SimpleSMILESDataset('PPB_all')
    num_graphs = len(dataset)
    train_idx, valid_idx, test_idx = get_splits(num_graphs)
    return { "dataset": dataset, "train_idx": train_idx, "valid_idx": valid_idx, "test_idx": test_idx, "source": "pyg" }


def main():
    # dataset = SimpleSMILESDataset('logS')
    for ds in ['CL', 'LD50', 'LogS', 'papp', 'PPB']:
        for suffix in ['', '_all']:
            dataset = SimpleSMILESDataset(ds+suffix)
            print(dataset)
            print(dataset.data.edge_index)
            print(dataset.data.edge_index.shape)
            print(dataset.data.x.shape)
            print(dataset[10])
            print(dataset[10].y)
            print('splits for train/valid/test:', list(map(len, get_splits(len(dataset)))))
            # print(dataset.get_idx_split())


if __name__ == '__main__':
    main()
