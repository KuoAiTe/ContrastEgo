import numpy as np
import networkx as nx
import torch
import re
import copy
import glob
import random
import torch_geometric
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from mental.utils.dataclass import BaselineModel, GraphType, TrainingArguments, ModelArguments, EvaluationResult
from torch_geometric.utils import add_self_loops
from torch_geometric.data import HeteroData

def get_model(baseline, data_info):
    model = None
    args = TrainingArguments()
    model_args = ModelArguments()

    if baseline == BaselineModel.MLP:
        model_args.allowed_input_keys = ['static_graph']
    elif baseline in [BaselineModel.GCN, BaselineModel.GraphSAGE, BaselineModel.GAT]:
        args.train_batch_size = 64
        args.test_batch_size = 64
        model_args.allowed_input_keys = ['static_graph']
    elif baseline == BaselineModel.DySAT:
        args.train_batch_size = 32
        args.test_batch_size = 32
        model_args.allowed_input_keys = ['label', 'period_id', 'graphs']
    elif baseline in [BaselineModel.MentalNet, BaselineModel.MentalNet_Original, BaselineModel.MentalNet_SAGE, BaselineModel.MentalNet_GAT, BaselineModel.MentalNet_GAT2]:
        args.train_batch_size = 64
        args.test_batch_size = 64
        model_args.allowed_input_keys = ['label', 'hetero_graph']
    elif baseline in [BaselineModel.MentalNet_DySAT, BaselineModel.MentalNetDySAT_SimSiam]:
        args.train_batch_size = 64
        args.test_batch_size = 64
        model_args.allowed_input_keys = ['label', 'period_id', 'hetero_graphs']
    elif baseline in [
        BaselineModel.MentalPlus,
        BaselineModel.MentalPlus_MEAN_POOLING,
        BaselineModel.MentalPlus_USE_NODE,
        BaselineModel.MentalPlus_USE_GRAPH,
        BaselineModel.MentalPlus_USE_GNN,
        BaselineModel.MentalPlus_NO_HGNN,
        BaselineModel.MentalPlus_Without_Transformer]:
        args.train_batch_size = 64
        args.test_batch_size = 64
        model_args.allowed_input_keys = ['label', 'period_id', 'hyper_graph', 'hyper_graph_augmentation']
    elif baseline in [BaselineModel.EvolveGCN]:
        args.train_batch_size = 99999
        args.test_batch_size = 99999
        model_args.allowed_input_keys = ['label', 'period_id', 'static_graph']
    else:
        args.train_batch_size = 64
        args.test_batch_size = 64

    model = baseline(model_args, data_info)
    return model, args, model_args

def split_graph(graphs, train_user_node_ids, test_user_node_ids):
    train_graphs = []
    test_graphs = []
    for graph in graphs:
        for node, data in graph.nodes(data = True):
            if node in train_user_node_ids:
                train_graphs.append(graph)
                break
            elif node in test_user_node_ids:
                test_graphs.append(graph)
                break
            else:
                print("something wrong")
                exit()
    return train_graphs, test_graphs

def get_train_test_data(data, test_size, random_state = 44):
    stratify = [_['label'] for _ in data]
    train_data, test_data = train_test_split(data, test_size = test_size, stratify = stratify, random_state = random_state)
    return train_data, test_data

def merge_graphs_by_user(user_data):
    result = []
    for user in user_data:
        label = user['label']
        graphs = user['graphs']
        # Aggregate embeddings
        node_embeddings = {}
        weights = {}
        #
        #graphs = [graph for graph in graphs if 'interpolation' not in graph.graph]
        i = 0
        for graph in graphs:
            for node_id, data in graph.nodes(data = True):
                if node_id not in node_embeddings:
                    node_embeddings[node_id] = []
                node_embeddings[node_id].append(data['features'])
                
            for src, dst, data in graph.edges(data = True):
                edge_label, weight = data['label'], data['weight']
                if edge_label not in weights:
                    weights[edge_label] = 0
                weights[edge_label] += weight
                data['time'] = i
            i += 1
        # Normalize weight
        for graph in graphs:
            for _, _, data in graph.edges(data = True):
                edge_label = data['label']
                if weights[edge_label] > 0:
                    data['weight'] = data['weight'] / weights[edge_label]
        
        # Pooling
        for node_id in node_embeddings.keys():
            tweet_embeddings = np.stack(node_embeddings[node_id])
            tweet_embeddings = np.mean(tweet_embeddings, axis = 0)
            node_embeddings[node_id] = tweet_embeddings
        
        G = nx.compose_all(graphs)
        for node_id, data in G.nodes(data = True):
            features_pooling = node_embeddings[node_id]
            nx.set_node_attributes(G, {node_id: {'features': features_pooling}})
        result.append({'label': label, 'graph': G})
    return result
    
def prepare_static_graph(data):
    data = merge_graphs_by_user(data)
    for row in data:
        row['static_graph'] = row['graph']
        del row['graph']
    return data

def prepare_static_hetero_graph(data):
    data = merge_graphs_by_user(data)
    for row in data:
        row['hetero_graph'] = row['graph']
        del row['graph']
    return data

def prepare_static_hetero_graph_by_user(data):
    data = merge_graphs_by_user(data)
    for row in data:
        row['hetero_graph_by_user'] = row['graph']
        del row['graph']
    return data

def prepare_dynamic_homo_graph(data, merge = False):
    for row in data:
        row['graphs'] = [torch_geometric.utils.from_networkx(graph) for graph in row['graphs']]
    return data
    
def prepare_dynamic_homo_graph(data, merge_node_embeddings = False):
    if merge_node_embeddings == True:
        merge_data = merge_graphs_by_user(data)
        node_embeddings = {}
        for graph_info in merge_data:
            graph = graph_info['graph']
            for node_id, attr in graph.nodes(data=True):
                node_embeddings[node_id] = attr['features']
            
    for row in data:
        graphs = []
        for graph in row['graphs']:
            if merge_node_embeddings == True:
                for node_id in graph.nodes():
                    nx.set_node_attributes(graph, {node_id: {'features': node_embeddings[node_id]}})
            graphs.append(torch_geometric.utils.from_networkx(graph))
        row['graphs'] = graphs
    return data

def normalized_graph_weight(graph):
    weights = {}
    for src, dst, data in graph.edges(data=True):
        edge_label, weight = data['label'], data['weight']
        if edge_label not in weights:
            weights[edge_label] = 0
        weights[edge_label] += weight

    for src, dst, data in graph.edges(data=True):
        edge_label = data['label']
        if weights[edge_label] > 0:
            data['weight'] = data['weight'] / weights[edge_label]
    return graph

def prepare_dynamic_hetero_graphs(data):
    for row in data:
        row['hetero_graphs'] = [normalized_graph_weight(graph) for graph in row['graphs']]
        del row['graphs']
    return data

def prepare_dynamic_hyper_graphs(data):
    graphs = []
    i = 0
    for row in data:
        row['hyper_graph'] = graphs_to_hyper_graph(row['graphs'], i)
        del row['graphs']
        i += 1
    return data
    
def get_db_location(data_info, pretrain):
    if pretrain:
        base = f'{data_info.dataset_location}/data/{data_info.tweet_processing_model_name}'
        return f'{base}/pretrain/*/*/*.pickle'
    else:
        base = f'{data_info.dataset_location}/data/{data_info.tweet_processing_model_name}/{data_info.dataset_name}/ut_{data_info.num_tweets_per_period}_mnf_{data_info.max_num_friends}_p_{data_info.periods_in_months}_l_{data_info.max_period_length}'
        return f'{base}/*/*/*.pickle'
    
    keys = list(row.keys())
    for row in user_data:
        for key in keys:
            if key not in allowed_input_keys:
                del row[key]
    return user_data

def graphs_to_hyper_graph(graphs, user_counter):
    hyper_graph = nx.MultiGraph()
    i = 0
    same_node_mapping = {}
    counter = 0
    new_node_mapping = {}
    user_embeddings = []
    graphs.reverse()
    for graph in graphs:
        mapping = {}
        for old_node_id, data in graph.nodes(data = True):
            if old_node_id not in new_node_mapping:
                new_node_mapping[old_node_id] = counter
                counter += 1
            if data['label'] != -100:
                user_embeddings.append(data['features'])
            node_id = new_node_mapping[old_node_id]
            new_node_id = f'{user_counter}_{node_id}_{i}'
            mapping[old_node_id] = new_node_id
            if node_id not in same_node_mapping:
                same_node_mapping[node_id] = []
            same_node_mapping[node_id].append(new_node_id)
            nx.set_node_attributes(graph, {old_node_id: {'node_id': node_id, 'period_id': i}})
            
        graph = nx.relabel_nodes(graph, mapping)
        hyper_graph.add_nodes_from(graph.nodes(data = True))
        weights = {}
        for src, dst, data in graph.edges(data=True):
            edge_label, weight = data['label'], data['weight']
            if edge_label not in weights:
                weights[edge_label] = 0
            weights[edge_label] += weight

        for src, dst, data in graph.edges(data=True):
            edge_label = data['label']
            if weights[edge_label] > 0:
                data['weight'] = data['weight'] / weights[edge_label]
    
        hyper_graph.add_edges_from(graph.edges(data = True))

        i += 1
    nx.set_node_attributes(hyper_graph, {f'{user_counter}': {'features': np.mean(user_embeddings, axis = 0)}})
    
    return hyper_graph

def load_data(data_info, pretrain = False):
    from platform import python_version
    from packaging.version import Version
    if Version(python_version()) < Version('3.8.0'):
        import pickle5 as pickle
    else:
        import pickle
    file_location = get_db_location(data_info, pretrain = pretrain)
    print(file_location)
    pattern = re.compile('(control_group|depressed_group)\/(\w+)\/(\w{1,2})\.pickle')
    file_paths = sorted(glob.glob(file_location, recursive = True))
    files = []
    static_files = []
    for file_path in file_paths:
        m = pattern.findall(file_path)
        if len(m) != 0:
            group, user_id, period_id = m[0]
            label = 0 if group == 'control_group' else 1
            files.append({'file': file_path, 'label': label, 'user_id': int(user_id), 'period_id': int(period_id)})

    graphs_by_user = {}
    count = 0
    for row in files:
        count += 1
        file, label, user_id, period_id = row['file'], row['label'], row['user_id'], row['period_id']
        if user_id not in graphs_by_user:
            graphs_by_user[user_id] = {'user_id': user_id, 'label': label, 'period_id': [], 'graphs': []}
        
        period_indices = graphs_by_user[user_id]['period_id']
        graphs = graphs_by_user[user_id]['graphs']
        with open(file, 'rb') as f:
            G = pickle.load(f)
        for node_id, data in G.nodes(data = True):
            if data['label'] == 'friend':
                nx.set_node_attributes(G, {node_id: {'period_id': period_id, 'label': -100}})
            elif data['label'] == 'depressed_group':
                nx.set_node_attributes(G, {node_id: {'period_id': period_id, 'label': 1}})
            elif data['label'] == 'control_group':
                nx.set_node_attributes(G, {node_id: {'period_id': period_id, 'label': 0}})
        remove_edges = []
        for src, dst, key, data in G.edges(data=True, keys=True):
            if data['weight'] == 0:
                remove_edges.append((src, dst, key))
        G.remove_edges_from(remove_edges)
        G = nx.relabel_nodes(G, lambda x: f'{x}_{user_id}')
        period_indices.append(period_id)
        graphs.append(G)

    return np.array(list(graphs_by_user.values()))

def to_device(feed_dict, device):
    labels = []
    for key, value in feed_dict.items():
        if feed_dict[key] == None:
            continue
        if key == 'label':
            labels.append(feed_dict['label'])
        if key == 'graph':
            feed_dict[key] = [_.to(device) for _ in value]
        if key == 'hetero_graph':
            feed_dict[key] = feed_dict[key].to(device)
        if key == 'hyper_graphs':
            feed_dict[key] = [_.to(device) for _ in value]
        if key == 'hyper_graph':
            feed_dict[key] = feed_dict[key].to(device)
        if key == 'hyper_graph_augmentation':
            feed_dict[key] = feed_dict[key].to(device)
        if key == 'period_id':
            feed_dict[key] = [torch.tensor(_, device = device) for _ in value]
        if key == 'static_graph':
            feed_dict[key] = feed_dict[key].to(device)
        if key == 'graphs':
            for user_graphs in feed_dict['graphs']:
                user_graphs = [_.to(device) for _ in user_graphs]
        if key == 'hetero_graphs':
            for user_graphs in feed_dict['hetero_graphs']:
                user_graphs = [_.to(device) for _ in user_graphs]

        if key == 'hetero_graph_by_user':
            feed_dict[key] = [_.to(device) for _ in value]


    feed_dict['labels'] = torch.tensor(labels, dtype = torch.long, device = device)
    feed_dict['device'] = device
    return feed_dict

def compute_metrics_from_results(y_true, y_pred):
    depressed_indices = (y_true == 1)
    control_indices = ~depressed_indices
    precision = precision_score(y_true, y_pred, pos_label = 1, average = None)
    recall = recall_score(y_true, y_pred, pos_label = 1, average = None)
    f1 = f1_score(y_true, y_pred, pos_label = 1, average = None)
    auc_roc_macro = roc_auc_score(y_true, y_pred, average = 'macro')
    auc_roc_micro = roc_auc_score(y_true, y_pred, average = 'micro')
    acc_depressed = accuracy_score(y_true[depressed_indices], y_pred[depressed_indices])
    acc_control = accuracy_score(y_true[control_indices], y_pred[control_indices])
    num_depressed = np.count_nonzero(depressed_indices)
    num_control = np.count_nonzero(control_indices)
    result = EvaluationResult(
        labels = y_true,
        predictions = y_pred,
        num_depressed = num_depressed,
        num_control = num_control,
        precision_depressed = precision[1],
        recall_depressed = recall[1],
        f1_depressed = f1[1],
        acc_depressed = acc_depressed,
        precision_control = precision[0],
        recall_control = recall[0],
        f1_control = f1[0],
        acc_control = acc_control,
        auc_roc_macro = auc_roc_macro,
        auc_roc_micro = auc_roc_micro
    )
    return result

