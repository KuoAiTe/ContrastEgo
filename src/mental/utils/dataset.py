import networkx as nx
import numpy as np
import torch
import torch_geometric
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData

def merge_all_user_graphs(user_graphs):
    node_embeddings = {}
    graphs = []

    # Aggregate embeddings
    i = 0
    for graph in user_graphs:
        for node_id, data in graph.nodes(data = True):
            if node_id not in node_embeddings:
                node_embeddings[node_id] = []
            node_embeddings[node_id].append(data['features'])
            nx.set_node_attributes(graph, {node_id: {'group': i}})
        i += 1
        
        #nx.set_node_attributes(graph, {node_id: {'group': i}})
        graphs.append(graph)
    
    # Pooling
    for node_id in node_embeddings.keys():
        tweet_embeddings = np.stack(node_embeddings[node_id])
        tweet_embeddings = np.mean(tweet_embeddings, axis = 0)
        node_embeddings[node_id] = tweet_embeddings
    G = nx.compose_all(graphs)
    for node_id, data in G.nodes(data=True):
        features_pooling = node_embeddings[node_id]
        nx.set_node_attributes(graph, {node_id: {'features': features_pooling}})
    
    G = torch_geometric.utils.from_networkx(G)
    return G

def merge_hyper_graphs(graphs):
    i = 0
    for graph in graphs:
        for node_id, data in graph.nodes(data = True):
            nx.set_node_attributes(graph, {node_id: {'group': i}})
        i += 1
    G = nx.compose_all(graphs)
    G = torch_geometric.utils.from_networkx(G)
    hetero_data = HeteroData()
    hetero_data['user'].x = G.features
    edge_index = G.edge_index
    edge_weight = G.weight
    hetero_data['period_id'] = G.period_id
    hetero_data['group'] = G.group
    hetero_data['label'] = G.label
    hetero_data['node_id'] = G.node_id
    hetero_data['edge_index'] = edge_index
    hetero_data['edge_weight'] = edge_weight
    edge_relation = np.array(G.edge_label)
    relations = np.unique(edge_relation)
    for relation in relations:
        indices = np.where(edge_relation == relation, True, False)
        hetero_data[relation].edge_index = edge_index[:, indices]
        hetero_data[relation].edge_weight = edge_weight[indices].float()

    return hetero_data

def merge_hetero_graphs(graphs):
    i = 0
    for graph in graphs:
        for node_id, data in graph.nodes(data = True):
            nx.set_node_attributes(graph, {node_id: {'group': i}})
        i += 1
    G = nx.compose_all(graphs)
    G = torch_geometric.utils.from_networkx(G)
    hetero_data = HeteroData()
    hetero_data['user'].x = G['features']
    hetero_data['period_id'] = G.period_id
    hetero_data['group'] = G.group
    hetero_data['label'] = G.label
    hetero_data['edge_index'] = G.edge_index
    edge_index = G.edge_index
    edge_weight = G.weight
    edge_relation = np.array(G.edge_label)
    relations = np.unique(edge_relation)
    for relation in relations:
        indices = np.where(edge_relation == relation, True, False)
        relation_edge_index = edge_index[:, indices]
        hetero_data[relation].edge_index = relation_edge_index
        hetero_data[relation].edge_weight = edge_weight[indices].float()
    return hetero_data


def to_hetero_graphs(graphs, group):
    hetero_graphs = []
    for graph in graphs:
        hetero_data = to_hetero_graph(graph, group)
    return hetero_graphs

def to_hetero_graph(G, group):
    graph = torch_geometric.utils.from_networkx(G) 
    hetero_data = HeteroData()
    hetero_data['user'].x = graph['features']
    hetero_data['group'] = torch.full(graph.period_id.shape, group)
    hetero_data['period_id'] = graph.period_id
    edge_index = graph.edge_index
    if edge_index.shape[1] > 0:
        edge_weight = graph.weight
        edge_relation = np.array(graph.edge_label)
        relations = np.unique(edge_relation)
        for relation in relations:
            indices = np.where(edge_relation == relation, True, False)
            hetero_data[relation].edge_index = edge_index[:, indices]
            hetero_data[relation].edge_weight = edge_weight[indices].float()
    return hetero_data

class MedDataset(Dataset):
    def __init__(self, raw_data):
        super(MedDataset, self).__init__()
        allowed_keys = ['label', 'user_id', 'user_node_id', 'period_id', 'graph', 'graphs', 'static_graph', 'hyper_graph', 'hyper_graph_augmentation', 'hyper_graphs', 'hetero_graph', 'hetero_graphs', 'hetero_graph_by_user']
        self.users = []
        for row in raw_data:
            inputs = {}
            keys = list(row.keys())
            for key in keys:
                if key in allowed_keys:
                    inputs[key] = row[key]
            
            self.users.append(inputs)

        #self.__createitems__()

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        return self.users[index]

    @staticmethod
    def collate_fn(samples):
        batch_dict = {}
        for key in ['user_id', 'label', 'graph', 'graphs', 'hyper_graph', 'hyper_graphs', 'hyper_graph_augmentation', 'hetero_graphs', 'hetero_graph_by_user', 'static_graph', 'mention_graph', 'quote_graph', 'reply_graph', 'hetero_graph', 'period_id', 'timestamp']:
            data_list = []
            for sample in samples:
                if key in sample:
                    data_list.append(sample[key])
            if len(data_list) > 0:
                assert(len(data_list) == len(samples))
                batch_dict[key] = data_list
            
        # merge static_graph
        
        if 'static_graph' in batch_dict:
            batch_dict['static_graph'] = merge_all_user_graphs(batch_dict['static_graph'])
        
        if 'hyper_graph' in batch_dict:
            batch_dict['hyper_graph'] = merge_hyper_graphs(batch_dict['hyper_graph'])
        if 'hetero_graph' in batch_dict:
            batch_dict['hetero_graph'] = merge_hetero_graphs(batch_dict['hetero_graph'])
        if 'hetero_graphs' in batch_dict:
            hetero_graphs = []
            for i in range(len(batch_dict['hetero_graphs'])):
                user_graphs = [to_hetero_graph(graph, i) for graph in batch_dict['hetero_graphs'][i]]
                hetero_graphs.append(user_graphs)
            
            batch_dict['hetero_graphs'] = hetero_graphs
        if 'hetero_graph_by_user' in batch_dict:
            batch_dict['hetero_graph_by_user'] = [to_hetero_graph(batch_dict['hetero_graph_by_user'][i], i) for i in range(len(batch_dict['hetero_graph_by_user']))]
            
        """
        if 'hyper_graph' in batch_dict:
            graphs = batch_dict['hyper_graph']
            G = nx.compose_all(graphs)
            G = torch_geometric.utils.from_networkx(G)
            batch_dict['hyper_graph'] = G
        """
        return batch_dict
