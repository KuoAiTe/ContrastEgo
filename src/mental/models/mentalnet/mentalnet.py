import numpy as np

import torch
import torch.nn as nn
from torch_geometric.nn import MLP
from torch_geometric.nn import SAGEConv, GATConv, GCNConv, HeteroConv
from torch_geometric.nn import aggr
from torch_geometric.nn.conv.gatv2_conv import GATv2Conv

import mental

class MentalNet(nn.Module):
    def __init__(self, args, data_info):
        super(MentalNet, self).__init__()
        self.args = args
        config = args.mental_net
        self.config = config
        self.support_edge_weight = True
        heterognn_config = config.heterognn
        self.convs = nn.ModuleDict({
            relation: self.init_convs(heterognn_config) for relation in heterognn_config.hetero_relations
        })
        layer_size = heterognn_config.gnn_hidden_channel * heterognn_config.gnn_num_layer + 1
        
        self.conv1 = nn.Conv1d(1, config.conv1_channel, layer_size, layer_size)
        self.conv2 = nn.Conv1d(config.conv1_channel, config.conv2_channel, config.kernel_size, 1)
        self.maxpool = nn.MaxPool1d(2, 2)
        self.depression_prediction_head = nn.Linear(self.config.out_dim(), 1)
        self.loss_fct = nn.BCEWithLogitsLoss()
        self.sort_aggr = aggr.SortAggregation(k = config.k)
    
    def init_convs(self, heterognn_config):
        convs = torch.nn.ModuleList()
        for i in range(heterognn_config.gnn_num_layer):
            conv = self.init_conv(in_channels = -1, out_channels = heterognn_config.gnn_hidden_channel)
            convs.append(conv)
        convs.append(self.init_conv(in_channels = -1, out_channels = 1))
        return convs

    def init_conv(self, in_channels, out_channels):
        return GCNConv(in_channels = in_channels, out_channels = out_channels)
        
    def forward(self, graph):
        x_dict, edge_index_dict, edge_weight_dict, group = graph.x_dict, graph.edge_index_dict, graph.edge_weight_dict, graph.group
        features = x_dict['user']
        outputs = {}
        for relation, edge_index in edge_index_dict.items():
            y = []
            x = features
            edge_weight = edge_weight_dict[relation]
            for conv in self.convs[relation]:
                if self.support_edge_weight:
                    x = conv(x, edge_index, edge_weight = edge_weight).tanh()
                else:
                    x = conv(x, edge_index).tanh()
                y.append(x)
            x = torch.cat(y, dim = -1)
            outputs[relation] = x
        result = []
        group_unique = torch.unique(graph.group).detach().cpu().numpy()
        for group_id in group_unique:
            indices = (group == group_id)
            out = []
            for relation in edge_index_dict.keys():
                x = outputs[relation][indices]
                x = self.sort_aggr(x)
                x = x.view(x.size(0), 1, x.size(-1))
                x = self.conv1(x).relu()
                x = self.maxpool(x)
                x = self.conv2(x).relu()
                x = x.view(x.size(0), -1)
                out.append(x)
            
            out = torch.cat(out, dim = 0)
            out = torch.sum(out, dim = 0).unsqueeze(dim = 0)
            result.append(out)
        result = torch.cat(result)
        return result

    def predict(self, feed_dict):
        hetero_graph = feed_dict['hetero_graph']
        indices = (hetero_graph.label != -100)
        logits = self.forward(hetero_graph)
        labels = hetero_graph.label[indices]
        prediction_scores = self.depression_prediction_head(logits).flatten()

        is_depressed = torch.zeros_like(labels)
        is_depressed[prediction_scores >= 0] = 1
        return labels, is_depressed

    def compute_loss(self, feed_dict):
        hetero_graph = feed_dict['hetero_graph']
        indices = (hetero_graph.label != -100)
        logits = self.forward(hetero_graph)
        labels = hetero_graph.label[indices].float()
        prediction_scores = self.depression_prediction_head(logits).flatten()
        loss = self.loss_fct(prediction_scores, labels)
        return logits, loss

    def prepare_data(self, data):
        return mental.utils.utilities.prepare_static_hetero_graph(data)

class MentalNet_Original(MentalNet):
    def __init__(self, args, data_info):
        super(MentalNet_Original, self).__init__(args, data_info)

    def forward(self, graph):
        x_dict, edge_index_dict, edge_weight_dict = graph.x_dict, graph.edge_index_dict, graph.edge_weight_dict
        features = x_dict['user']
        outputs = {}
        for relation, edge_index in edge_index_dict.items():
            y = []
            x = features
            edge_weight = edge_weight_dict[relation]
            for conv in self.convs[relation]:
                if self.support_edge_weight:
                    x = conv(x, edge_index, edge_weight = edge_weight).tanh()
                else:
                    x = conv(x, edge_index).tanh()
                y.append(x)
            x = torch.cat(y, dim = -1)
            outputs[relation] = x
        out = []
        for relation in edge_index_dict.keys():
            x = outputs[relation]
            x = self.sort_aggr(x)
            x = x.view(x.size(0), 1, x.size(-1))
            x = self.conv1(x).relu()
            x = self.maxpool(x)
            x = self.conv2(x).relu()
            x = x.view(x.size(0), -1)
            out.append(x)
        
        out = torch.cat(out, dim = 0)
        out = torch.sum(out, dim = 0).unsqueeze(dim = 0)
        return out
    
    def predict(self, feed_dict):
        hetero_graphs = feed_dict['hetero_graph_by_user']
        labels = feed_dict['labels'].flatten()
        logits = []
        for hetero_graph in hetero_graphs:
            logits.append(self.forward(hetero_graph))
        logits = torch.cat(logits, dim = 0)
        prediction_scores = self.depression_prediction_head(logits).flatten()

        is_depressed = torch.zeros_like(labels)
        is_depressed[prediction_scores >= 0] = 1
        return labels, is_depressed

    def compute_loss(self, feed_dict):
        hetero_graphs = feed_dict['hetero_graph_by_user']
        labels = feed_dict['labels'].flatten().float()
        logits = []
        for hetero_graph in hetero_graphs:
            logits.append(self.forward(hetero_graph))
        logits = torch.cat(logits, dim = 0)
        prediction_scores = self.depression_prediction_head(logits).flatten()
        loss = self.loss_fct(prediction_scores, labels)
        return logits, loss

    def prepare_data(self, data):
        return mental.utils.utilities.prepare_static_hetero_graph_by_user(data)

class MentalNetNaive(nn.Module):
    def __init__(self, args, data_info):
        super(MentalNetNaive, self).__init__()
        self.args = args
        config = args.mental_net
        self.config = config
        self.support_edge_weight = True
        self.convs = nn.ModuleDict({
            relation: self.init_convs() for relation in self.config.hetero_relations
        })
        self.depression_prediction_head = nn.Linear(64 * 3, 1)
        self.loss_fct = nn.BCEWithLogitsLoss()

    def init_conv(self, in_channels, out_channels):
        return GATConv(in_channels = in_channels, out_channels = out_channels)

    def init_convs(self, **kwargs):
        convs = torch.nn.ModuleList()
        for i in range(3):
            conv = self.init_conv(in_channels = -1, out_channels = self.config.gnn_hidden_channel)
            convs.append(conv)
        return convs

    def forward(self, inputs):
        x_dict, edge_index_dict, edge_weight_dict = inputs.x_dict, inputs.edge_index_dict, inputs.edge_weight_dict
        features = x_dict['user']
        outputs = []
        for relation, edge_index in edge_index_dict.items():
            x = features
            #edge_index = inputs.edge_index
            edge_weight = edge_weight_dict[relation]
            i = 0
            for conv in self.convs[relation]:
                i += 1
                x = conv(x, edge_index, edge_weight = edge_weight)
                if i != len(self.convs[relation]):
                    x = x.relu()
            outputs.append(x)
        x = torch.cat(outputs, dim = 1)

        return x

    def predict(self, feed_dict):
        hetero_graph = feed_dict['hetero_graph']
        indices = (hetero_graph.label != -100)
        logits = self.forward(hetero_graph)[indices]
        labels = hetero_graph.label[indices]
        prediction_scores = self.depression_prediction_head(logits).flatten()

        is_depressed = torch.zeros_like(labels)
        is_depressed[prediction_scores >= 0] = 1
        return labels, is_depressed

    def compute_loss(self, feed_dict):
        hetero_graph = feed_dict['hetero_graph']
        indices = (hetero_graph.label != -100)
        logits = self.forward(hetero_graph)[indices]
        labels = hetero_graph.label[indices].float()
        prediction_scores = self.depression_prediction_head(logits).flatten()

        loss = self.loss_fct(prediction_scores, labels)
        return logits, loss

    def prepare_data(self, data):
        return mental.utils.utilities.prepare_static_hetero_graph(data)

class MentalNet_GAT2(MentalNet):
    def __init__(self, args, data_info):
        super(MentalNet_GAT2, self).__init__(args, data_info)
        
    def init_conv(self, in_channels, out_channels):
        self.support_edge_weight = False
        return GATv2Conv(in_channels = in_channels, out_channels = out_channels, heads = 1, concat = False, residual = True)
        
class MentalNet_GAT(MentalNet):
    def __init__(self, args, data_info):
        super(MentalNet_GAT, self).__init__(args, data_info)

    def init_conv(self, in_channels, out_channels):
        self.support_edge_weight = False
        return GATConv(in_channels = in_channels, out_channels = out_channels)

class MentalNet_SAGE(MentalNet):
    def __init__(self, args, data_info):
        super(MentalNet_SAGE, self).__init__(args, data_info)

    def init_conv(self, in_channels, out_channels):
        self.support_edge_weight = False
        return SAGEConv(in_channels = in_channels, out_channels = out_channels)
        