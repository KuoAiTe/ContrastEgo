import torch
import torch.nn as nn
from torch_geometric.nn.models import GAT, GCN, GraphSAGE
import mental

class GNNWrapper(nn.Module):
    def __init__(self, args, data_info):
        super(GNNWrapper, self).__init__()
        self.conv_model = None
        self.loss_fct = nn.BCEWithLogitsLoss()

    def forward(self, x, edge_index):
        return self.conv_model(x, edge_index)

    @torch.no_grad()
    def predict(self, inputs):
        assert(self.conv_model != None)
        graph = inputs['static_graph']
        outputs = self.forward(graph.features, graph.edge_index) # [N, T, F]
        indices = (graph.label != -100)
        prediction_scores = outputs[indices].flatten()
        labels = graph.label[indices].detach().cpu()
        is_depressed = torch.zeros_like(labels)
        is_depressed[prediction_scores >= 0] = 1
        return labels, is_depressed

    def compute_loss(self, inputs, pretrain = False):
        assert(self.conv_model != None)
        graph = inputs['static_graph']
        outputs = self.forward(graph.features, graph.edge_index) # [N, T, F]
        indices = (graph.label != -100)
        prediction_scores = outputs[indices].flatten()
        labels = graph.label[indices].float()
        loss = self.loss_fct(prediction_scores, labels)
        return outputs, loss

    def prepare_data(self, data):
        return mental.utils.utilities.prepare_static_graph(data)

class GATWrapper(GNNWrapper):
    def __init__(self, args, data_info):
        super(GATWrapper, self).__init__(args, data_info)
        self.conv_model = GAT(in_channels = data_info.num_features, hidden_channels = args.gnn_hidden_channels, num_layers = args.gnn_num_layers, out_channels = args.gnn_out_channels)

class GCNWrapper(GNNWrapper):
    def __init__(self, args, data_info):
        super(GCNWrapper, self).__init__(args, data_info)
        self.conv_model = GCN(in_channels = data_info.num_features, hidden_channels = args.gnn_hidden_channels, num_layers = args.gnn_num_layers, out_channels = args.gnn_out_channels)

class GraphSAGEWrapper(GNNWrapper):
    def __init__(self, args, data_info):
        super(GraphSAGEWrapper, self).__init__(args, data_info)
        self.conv_model = GraphSAGE(in_channels = data_info.num_features, hidden_channels = args.gnn_hidden_channels, num_layers = args.gnn_num_layers, out_channels = args.gnn_out_channels)
