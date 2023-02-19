import torch
import torch.nn as nn
from torch.nn.modules.loss import BCEWithLogitsLoss
from .layers import SelfAttentionLayer, StructuralAttentionLayer, TemporalAttentionLayer2 as TemporalAttentionLayer
from torch_geometric.nn.models import GAT, GCN, GraphSAGE
import mental
class DySAT(nn.Module):
    def __init__(self, args, dataset_info):
        """[summary]

        Args:
            args ([type]): [description]
            time_length (int): Total timesteps in dataset.
        """
        super(DySAT, self).__init__()
        self.args = args
        self.dataset_info = dataset_info
        self.out_dim = args.gnn_hidden_channels
        self.depression_prediction_head = nn.Linear(self.out_dim, 1)
        self.num_time_steps = int(dataset_info.period_length)
        #self.structural_attn = GCN(in_channels = self.dataset_info.num_features, hidden_channels = args.gnn_hidden_channels, num_layers = args.gnn_num_layers)

        self.structural_attention_layers = nn.Sequential()
        for i in range(1):
            layer = StructuralAttentionLayer(input_dim=768,
                                             output_dim=self.out_dim,
                                             n_heads=4,
                                             attn_drop=0.1,
                                             ffd_drop=0.1,
                                             residual=True)
            self.structural_attention_layers.add_module(name="structural_layer_{}".format(i), module=layer)
        
        self.temporal_attention_layers = nn.Sequential()
        for i in range(1):
            layer = TemporalAttentionLayer(input_dim=self.out_dim,
                                           n_heads=4,
                                           num_time_steps=self.num_time_steps,
                                           attn_drop=0.1,
                                           residual=True)
            self.temporal_attention_layers.add_module(name="temporal_layer_{}".format(i), module=layer)
        
        self.loss_fct = BCEWithLogitsLoss()

    def forward(self, batch_graphs):
        # Structural Attention forward
        final_outputs = []
        for graphs in batch_graphs:
            structural_out = []
            for graph in graphs:
                structural_out.append(self.structural_attention_layers((graph.features, graph.edge_index, graph.weight))[0])
            structural_outputs = [g[:,None,:] for g in structural_out] 

            maximum_node_num = structural_outputs[-1].shape[0]
            out_dim = structural_outputs[-1].shape[-1]
            structural_outputs_padded = []
            for out in structural_outputs:
                zero_padding = torch.zeros(maximum_node_num-out.shape[0], 1, out_dim).to(out.device)
                padded = torch.cat((out, zero_padding), dim=0)
                structural_outputs_padded.append(padded)
            
            structural_outputs_padded = torch.cat(structural_outputs_padded, dim=1) 
            temporal_out = self.temporal_attention_layers(structural_outputs_padded)
            graph_representation = torch.sum(temporal_out, dim = 0)
            final_outputs.append(graph_representation[-1, :])
        final_outputs = torch.stack(final_outputs, dim = 0)
        return final_outputs

    def predict(self, feed_dict):
        graphs = feed_dict['graphs']
        labels = feed_dict['labels'].flatten()
        batch_hidden_states = self.forward(graphs)
        logits = batch_hidden_states
        outputs = self.depression_prediction_head(logits).flatten()
        is_depressed = torch.zeros_like(labels).to(labels.device)
        is_depressed[outputs >= 0] = 1
        return labels, is_depressed


    def compute_loss(self, feed_dict):
        graphs = feed_dict['graphs']
        labels = feed_dict['labels'].float().flatten()
        batch_hidden_states = self.forward(graphs)
        logits = batch_hidden_states

        prediction_scores = self.depression_prediction_head(logits).flatten()
        loss = self.loss_fct(prediction_scores, labels)
        return logits, loss
        
    def prepare_data(self, data):
        return mental.utils.utilities.prepare_dynamic_homo_graph(data)
