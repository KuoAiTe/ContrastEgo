import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import add_self_loops

class AttentionPooling(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Linear(in_dim, 1),
        )
    def forward(self, last_hidden_state):
        w = self.attention(last_hidden_state).float()
        w = torch.softmax(w,1)
        attention_embeddings = torch.sum(w * last_hidden_state, dim=1)
        return attention_embeddings

class HeteroGNN(nn.Module):
    supports_edge_weight = True
    def __init__(self, config):
        super(HeteroGNN, self).__init__()
        self.config = config
        self.convs = nn.ModuleDict({
            relation: self.init_convs() for relation in config.hetero_relations
        })
        self.norms = nn.ModuleList(
            [nn.LayerNorm(config.gnn_hidden_channel, eps=1e-12) for _ in range(config.gnn_num_layer - 1)] + 
            [nn.LayerNorm(config.gnn_out_channel, eps =1e-12)]
        )
        self.attention_pooling = AttentionPooling(config.gnn_hidden_channel)
        self.act = nn.GELU()
        encoder_layer = nn.TransformerEncoderLayer(d_model = config.gnn_hidden_channel, nhead = 1, dim_feedforward = config.gnn_hidden_channel, batch_first = True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.reset_parameters()

    def reset_parameters(self):
        for convs in self.convs.values():
            for conv in convs:
                conv.reset_parameters()
    
        for norm in self.norms:
            norm.reset_parameters()

    def init_conv(self, in_channels, out_channels):
        return GCNConv(in_channels = in_channels, out_channels = out_channels)

    def init_convs(self):
        convs = torch.nn.ModuleList()
        for _ in range(self.config.gnn_num_layer - 1):
            convs.append(self.init_conv(in_channels = -1, out_channels = self.config.gnn_hidden_channel))
        
        convs.append(self.init_conv(in_channels = -1, out_channels = self.config.gnn_out_channel))
        return convs

    def forward(self, init_x, edge_index_dict, edge_weight_dict, graph = None):
        last_x = init_x
        for i in range(self.config.gnn_num_layer):
            y = []
            for relation in self.config.hetero_relations:
                conv = self.convs[relation][i]
                if relation in edge_index_dict:
                    edge_index = edge_index_dict[relation]
                    edge_weight = edge_weight_dict[relation]
                else:
                    edge_index = add_self_loops(torch.empty((2, 0), dtype=torch.int64), num_nodes = last_x.shape[0])[0].to(last_x.device)
                    edge_weight = torch.full((edge_index.shape[1],), 1.0, device = last_x.device)
                if self.supports_edge_weight:
                    x = conv(last_x, edge_index, edge_weight = edge_weight)
                else:
                    x = conv(last_x, edge_index)
                y.append(x)
            x = torch.stack(y, dim = 1)
            #x = self.transformer_encoder(x)
            #print(last_x.shape)
            #print(x.shape)
            x = torch.sum(x, dim = 1)
            x = self.norms[i](x)
            last_x = self.act(x)
        
        return last_x
