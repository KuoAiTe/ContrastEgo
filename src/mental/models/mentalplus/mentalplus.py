import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from torch_geometric.nn import SAGEConv, GCNConv, GATConv
from torch_geometric.nn.models import GAT, GCN, GraphSAGE
from torch_geometric.nn.models.mlp import MLP
from .layers import TransformerEncoder
from .heterognn import HeteroGNN
from .supcon import SupConLoss
from .attnconv import TorchAttentionBlock

import random
import mental

@dataclass
class LossOutput:
    pretrain_loss: torch.FloatTensor = None
    train_loss: torch.FloatTensor = None

@dataclass
class MentalPlusOutput:
    logits: str =  None
    cls_logits: torch.FloatTensor = None
    user_node_logits: torch.FloatTensor = None
    user_graph_logits: torch.FloatTensor = None
    sup_con_logits: torch.FloatTensor = None
    sup_labels: torch.FloatTensor = None
    
@dataclass
class PredictionType:
    NODE_ONLY: int = 1
    GRAPH_ONLY: int = 2
    BOTH: int = 3

@dataclass
class GNNType:
    HETERO: int = 1
    GNN: int = 2
    NONE: int = 3

import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class AttentionPooling(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Linear(in_dim, 1),
        )
    def forward(self, last_hidden_state, attention_mask):
        w = self.attention(last_hidden_state).float()
        w[attention_mask==0]=float('-inf')
        w = torch.softmax(w,1)
        attention_embeddings = torch.sum(w * last_hidden_state, dim=1)
        return attention_embeddings

class MentalPlus(nn.Module):
    def __init__(self, args, dataset_info):
        super(MentalPlus, self).__init__()
        self.args = args
        self.config = args.mental_net
        config = self.config
        self.mentalnet_out_dim = config.heterognn.gnn_hidden_channel
        self.heteroGNN = HeteroGNN(config.heterognn)
        self.gnn = GCN(in_channels = dataset_info.num_features, hidden_channels = config.heterognn.gnn_hidden_channel, num_layers = config.heterognn.gnn_num_layer)
        self.hetero_downsampling = nn.Linear(dataset_info.num_features, self.mentalnet_out_dim)

        self.num_time_steps = int(dataset_info.period_length)
        self_attention_config = self.args.self_attention_config
        self_attention_config.hidden_size = self.mentalnet_out_dim 
        self.transformer_encoder = TransformerEncoder(self_attention_config)
        #encoder_layer = nn.TransformerEncoderLayer(d_model = config.heterognn.gnn_hidden_channel, nhead=1, dim_feedforward = config.heterognn.gnn_hidden_channel, batch_first = True)
        #self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.LayerNorm = nn.LayerNorm(self.mentalnet_out_dim, eps=1e-12)

        self.position_ids = torch.arange(self.num_time_steps + 1)
        self.position_embeddings = nn.Embedding(self.num_time_steps + 1, self.mentalnet_out_dim)
        self.positional_encoder = PositionalEncoding(d_model = self.mentalnet_out_dim)
        self.pooling = AttentionPooling(self.mentalnet_out_dim)
        self.dropout = nn.Dropout(0.5)
        self.loss_fct = nn.BCEWithLogitsLoss()

        self.depression_prediction_head = nn.Linear(self.mentalnet_out_dim, 1)
        self.depression_prediction_head_2 = MLP([self.mentalnet_out_dim * 2, self.mentalnet_out_dim, 1], norm = None)
        
        self.mask_prediction_head = nn.Linear(self.mentalnet_out_dim, 1)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.mentalnet_out_dim), requires_grad = True)
        self.pad_token = nn.Parameter(torch.randn(1, self.mentalnet_out_dim), requires_grad = True)

        self.cls_pooler = nn.Linear(self.mentalnet_out_dim, self.mentalnet_out_dim)
        self.cls_activation = nn.Tanh()
        self.gnn_type = GNNType.HETERO
        self.use_transformer = True
        self.use_supervised_learning = True
        self.use_cls_token = False
        self.use_position_embeddings = True
        self.sup_loss = SupConLoss()
        self.projection_head = nn.Sequential(
            nn.Linear(self.mentalnet_out_dim, self.mentalnet_out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.mentalnet_out_dim, 32)
        )
        self.prediction_type = PredictionType.NODE_ONLY
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.cls_token)
        nn.init.xavier_uniform_(self.pad_token)
    
    def forward(self, graph):
        user_graph_logits = None
        sup_con_logits = None
        x_dict, edge_index_dict, edge_weight_dict, group, label = graph.x_dict, graph.edge_index_dict, graph.edge_weight_dict, graph.group, graph.label
        if self.gnn_type == GNNType.HETERO:
            f = self.heteroGNN(x_dict['user'], edge_index_dict, edge_weight_dict)
        elif self.gnn_type == GNNType.GNN:
            f = self.gnn(x_dict['user'], graph.edge_index, edge_weight = graph.edge_weight.float())
        else:
            f = self.hetero_downsampling(x_dict['user'])
            
        group_unique = torch.unique(graph.group).detach().cpu().numpy()
        batch_size = len(group_unique)    
        user_nodes = (graph.label != -100)
        x = self.pad_token.repeat(batch_size, self.num_time_steps, 1)
        y = self.pad_token.repeat(batch_size, self.num_time_steps, 1)
        attention_mask = torch.full((batch_size, self.num_time_steps), False).to(x.device)
        labels = torch.full((batch_size, self.num_time_steps), 0).to(x.device)
    
        for i in range(len(group_unique)):
            group_id = group_unique[i]
            index = (group == group_id) & user_nodes
            user_embeddings = f[index]
            labels[i, :] = graph.label[index][0]
            x[i, :user_embeddings.size(0)] = user_embeddings
            attention_mask[i, :user_embeddings.size(0)] = True
            j = 0
            periods = graph.period_id[index]
            for period in periods:
                index = (group == group_id) & (graph.period_id == period) 
                y[i, j] = torch.mean(f[index], dim = 0)
                j += 1
            
        if self.use_transformer:
            attention_scores = torch.full(attention_mask.shape, -10000.0).to(x.device)
            attention_scores[attention_mask == True] = 0
            
            position_embeddings = self.position_embeddings(self.position_ids[:-1])
            # add position encoding to input_tensor
            #x = self.positional_encoder(x)
            #y = self.positional_encoder(y)
            x += position_embeddings
            x = self.LayerNorm(x)
            y += position_embeddings
            y = self.LayerNorm(x)
            x = self.transformer_encoder(x, attention_scores[:, None, None, :])
            y = self.transformer_encoder(y, attention_scores[:, None, None, :])
            #x = self.transformer_encoder(x, src_key_padding_mask = ~attention_mask)#attention_scores[:, None, None, :])
            #y = self.transformer_encoder(y, src_key_padding_mask = ~attention_mask)##attention_scores[:, None, None, :])
            
        if self.use_cls_token:
            user_node_logits = x[:, 0, :]
            user_graph_logits = y[:, 0, :]
        else:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(x.size()).float()
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min = 1e-12)
            user_node_logits = torch.sum(x * input_mask_expanded, 1) / sum_mask
            user_graph_logits = torch.sum(y * input_mask_expanded, 1) / sum_mask

        if self.use_supervised_learning:
            logits = torch.stack([user_node_logits, user_graph_logits], dim = 1)
            projection = self.projection_head(logits)
            sup_con_logits = F.normalize(projection, dim = -1)

        return MentalPlusOutput(
            user_node_logits = user_node_logits,
            user_graph_logits = user_graph_logits,
            sup_con_logits = sup_con_logits,
            #sup_labels = sup_labels,
        )

    def predict(self, feed_dict):
        depression_prediction_head = self.depression_prediction_head if self.prediction_type != PredictionType.BOTH else self.depression_prediction_head_2
        graph = feed_dict['hyper_graph']
        indices = (graph.node_id == 0) & (graph.period_id == 0)
        labels = graph.label[indices]
        output = self.forward(graph)
        if self.prediction_type == PredictionType.BOTH:
            cls_logits = torch.cat([output.user_node_logits, output.user_graph_logits], dim = -1)
        elif self.prediction_type == PredictionType.NODE_ONLY:
            cls_logits = output.user_node_logits
        elif self.prediction_type == PredictionType.GRAPH_ONLY:
            cls_logits = output.user_graph_logits

        prediction_scores = depression_prediction_head(cls_logits).flatten()
        is_depressed = torch.zeros_like(labels).to(labels.device)
        is_depressed[prediction_scores >= 0] = 1
        return labels, is_depressed

    def compute_loss(self, feed_dict, pretrain = False):
        depression_prediction_head = self.depression_prediction_head if self.prediction_type != PredictionType.BOTH else self.depression_prediction_head_2
        graph = feed_dict['hyper_graph']
        indices = (graph.node_id == 0) & (graph.period_id == 0)
        labels = graph.label[indices]
        output = self.forward(graph)

        if self.prediction_type == PredictionType.BOTH:
            cls_logits = torch.cat([output.user_node_logits, output.user_graph_logits], dim = -1)
        elif self.prediction_type == PredictionType.NODE_ONLY:
            cls_logits = output.user_node_logits
        elif self.prediction_type == PredictionType.GRAPH_ONLY:
            cls_logits = output.user_graph_logits
        
        prediction_scores = depression_prediction_head(cls_logits).flatten()
        pred_loss = self.loss_fct(prediction_scores, labels.flatten().float())
        con_loss = self.sup_loss(output.sup_con_logits, labels = labels) if self.use_supervised_learning else 0
        loss = con_loss + pred_loss
        return None, loss

    def prepare_data(self, data):
        return mental.utils.utilities.prepare_dynamic_hyper_graphs(data)

class MentalPlus_CLS_POOLING(MentalPlus):
    def __init__(self, args, data_info):
        super(MentalPlus_CLS_POOLING, self).__init__(args, data_info)
        self.use_cls_token = True

class MentalPlus_NO_POSITION(MentalPlus):
    def __init__(self, args, data_info):
        super(MentalPlus_NO_POSITION, self).__init__(args, data_info)
        self.use_position_embeddings = False

class MentalPlus_NO_POSITION_CLS_POOLING(MentalPlus):
    def __init__(self, args, data_info):
        super(MentalPlus_NO_POSITION_CLS_POOLING, self).__init__(args, data_info)
        self.use_position_embeddings = False
        self.use_cls_token = True

class MentalPlus_USE_NODE(MentalPlus):
    def __init__(self, args, data_info):
        super(MentalPlus_USE_NODE, self).__init__(args, data_info)
        self.prediction_type = PredictionType.NODE_ONLY

class MentalPlus_USE_GRAPH(MentalPlus):
    def __init__(self, args, data_info):
        super(MentalPlus_USE_GRAPH, self).__init__(args, data_info)
        self.prediction_type = PredictionType.GRAPH_ONLY

class MentalPlus_NO_HGNN(MentalPlus):
    def __init__(self, args, data_info):
        super(MentalPlus_NO_HGNN, self).__init__(args, data_info)
        self.gnn_type = GNNType.NONE

class MentalPlus_NO_SUPERVISED(MentalPlus):
    def __init__(self, args, data_info):
        super(MentalPlus_NO_SUPERVISED, self).__init__(args, data_info)
        self.use_supervised_learning = False

class MentalPlus_USE_GNN(MentalPlus):
    def __init__(self, args, data_info):
        super(MentalPlus_USE_GNN, self).__init__(args, data_info)
        self.gnn_type = GNNType.GNN

class MentalPlus_Without_Transformer(MentalPlus):
    def __init__(self, args, data_info):
        super(MentalPlus_Without_Transformer, self).__init__(args, data_info)
        self.use_cls_token = False
        self.use_transformer = False
        
class MentalPlus_NO_GRAPH_AGGREGATION(MentalPlus):
    def __init__(self, args, data_info):
        super(MentalPlus_NO_GRAPH_AGGREGATION, self).__init__(args, data_info)
    def forward(self, hetero_graphs):
        batch_size = len(hetero_graphs)
        x = self.pad_token.repeat(batch_size, self.num_time_steps, 1)
        y = self.pad_token.repeat(batch_size, self.num_time_steps, 1)
        attention_mask = torch.full((batch_size, self.num_time_steps), False).to(x.device)
        i = 0
        for user_graphs in hetero_graphs:
            j = 0
            for graph in user_graphs:
                if self.gnn_type == GNNType.HETERO:
                    f = self.heteroGNN(graph['user']['x'], graph.edge_index_dict, graph.edge_weight_dict)
                else:
                    f = self.hetero_downsampling(graph['user']['x'])
                x[i, j] = f[0]
                y[i, j] = torch.mean(f, dim = 0)
                attention_mask[i, j] = True
                j += 1
            i += 1
    
        attention_scores = torch.full(attention_mask.shape, -10000.0).to(x.device)
        attention_scores[attention_mask == True] = 0

        if self.use_transformer:
            # self-attention , self-attention postion enconding
            x = self.positional_encoder(x)
            y = self.positional_encoder(y)
            x = self.transformer_encoder(x, src_key_padding_mask = ~attention_mask)
            y = self.transformer_encoder(y, src_key_padding_mask = ~attention_mask)
            
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(x.size()).float()
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min = 1e-12)
        user_node_logits = torch.sum(x * input_mask_expanded, 1) / sum_mask
        user_graph_logits = torch.sum(y * input_mask_expanded, 1) / sum_mask
        if self.use_supervised_learning:
            logits = torch.stack([user_node_logits, user_graph_logits], dim = 1)
            projection = self.projection_head(logits)
            sup_con_logits = F.normalize(projection, dim = -1)

        return MentalPlusOutput(
            user_node_logits = user_node_logits,
            user_graph_logits = user_graph_logits,
            sup_con_logits = sup_con_logits,
        )

    def predict(self, feed_dict):
        depression_prediction_head = self.depression_prediction_head if self.prediction_type != PredictionType.BOTH else self.depression_prediction_head_2
        hetero_graphs = feed_dict['hetero_graphs']
        labels = feed_dict['labels'].flatten()
        output = self.forward(hetero_graphs)
        if self.prediction_type == PredictionType.BOTH:
            cls_logits = torch.cat([output.user_node_logits, output.user_graph_logits], dim = -1)
        elif self.prediction_type == PredictionType.NODE_ONLY:
            cls_logits = output.user_node_logits
        elif self.prediction_type == PredictionType.GRAPH_ONLY:
            cls_logits = output.user_graph_logits
        outputs = depression_prediction_head(cls_logits).flatten()
        is_depressed = torch.zeros_like(labels).to(labels.device)
        is_depressed[outputs >= 0] = 1
        return labels, is_depressed

    def compute_loss(self, feed_dict, pretrain = False):
        depression_prediction_head = self.depression_prediction_head if self.prediction_type != PredictionType.BOTH else self.depression_prediction_head_2
        
        hetero_graphs = feed_dict['hetero_graphs']
        labels = feed_dict['labels'].flatten()
        output = self.forward(hetero_graphs)
        if self.prediction_type == PredictionType.BOTH:
            cls_logits = torch.cat([output.user_node_logits, output.user_graph_logits], dim = -1)
        elif self.prediction_type == PredictionType.NODE_ONLY:
            cls_logits = output.user_node_logits
        elif self.prediction_type == PredictionType.GRAPH_ONLY:
            cls_logits = output.user_graph_logits
            
        prediction_scores = depression_prediction_head(cls_logits).flatten()
        pred_loss = self.loss_fct(prediction_scores, labels.flatten().float())
        con_loss = self.sup_loss(output.sup_con_logits, labels = output.sup_labels) if self.use_supervised_learning else 0
        return None, con_loss + pred_loss
 
    def prepare_data(self, data):
        return mental.utils.utilities.prepare_dynamic_hetero_graphs(data)

