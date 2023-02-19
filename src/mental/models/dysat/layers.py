import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch_geometric.utils import softmax
from torch_scatter import scatter

import math
BertLayerNorm = torch.nn.LayerNorm



class StructuralAttentionLayer(nn.Module):
    def __init__(self, 
                input_dim, 
                output_dim, 
                n_heads, 
                attn_drop, 
                ffd_drop,
                residual):
        super(StructuralAttentionLayer, self).__init__()
        self.out_dim = output_dim // n_heads
        self.n_heads = n_heads
        self.act = nn.ELU()

        self.lin = nn.Linear(input_dim, n_heads * self.out_dim, bias=False)
        self.att_l = nn.Parameter(torch.Tensor(1, n_heads, self.out_dim))
        self.att_r = nn.Parameter(torch.Tensor(1, n_heads, self.out_dim))
        
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

        self.attn_drop = nn.Dropout(attn_drop)
        self.ffd_drop = nn.Dropout(ffd_drop)

        self.residual = residual
        if self.residual:
            self.lin_residual = nn.Linear(input_dim, n_heads * self.out_dim, bias=False)

        self.xavier_init()

    def forward(self, inputs):
        features, edge_index, edge_weight = inputs
        edge_weight = edge_weight.reshape(-1, 1)
        H, C = self.n_heads, self.out_dim
        x = self.lin(features).view(-1, H, C) # [N, heads, out_dim]
        # attention
        alpha_l = (x * self.att_l).sum(dim=-1).squeeze() # [N, heads]
        alpha_r = (x * self.att_r).sum(dim=-1).squeeze()
        alpha_l = alpha_l[edge_index[0]] # [num_edges, heads]
        alpha_r = alpha_r[edge_index[1]]
        alpha = alpha_r + alpha_l
        alpha = edge_weight * alpha
        alpha = self.leaky_relu(alpha)
        coefficients = softmax(alpha, edge_index[1]) # [num_edges, heads]

        # dropout
        if self.training:
            coefficients = self.attn_drop(coefficients)
            x = self.ffd_drop(x)
        x_j = x[edge_index[0]]  # [num_edges, heads, out_dim]

        # output
        out = self.act(scatter(x_j * coefficients[:, :, None], edge_index[1], dim=0, reduce="sum"))
        out = out.reshape(-1, self.n_heads*self.out_dim) #[num_nodes, output_dim]
        if self.residual:
            out = out + self.lin_residual(features)
        return (out, edge_index, edge_weight)

    def xavier_init(self):
        nn.init.xavier_uniform_(self.att_l)
        nn.init.xavier_uniform_(self.att_r)

        
class TemporalAttentionLayer2(nn.Module):
    def __init__(self, 
                input_dim, 
                n_heads, 
                num_time_steps, 
                attn_drop, 
                residual):
        super(TemporalAttentionLayer2, self).__init__()
        self.n_heads = n_heads
        self.num_time_steps = num_time_steps
        self.residual = residual

        # define weights
        self.position_embeddings = nn.Parameter(torch.Tensor(num_time_steps, input_dim))
        self.Q_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.K_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.V_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        # ff
        self.lin = nn.Linear(input_dim, input_dim, bias=True)
        # dropout 
        self.attn_dp = nn.Dropout(attn_drop)
        self.xavier_init()


    def forward(self, inputs):
        """In:  attn_outputs (of StructuralAttentionLayer at each snapshot):= [N, T, F]"""
        # 1: Add position embeddings to input
        num_time_steps = inputs.size(1)
        position_inputs = torch.arange(0,num_time_steps).reshape(1, -1).repeat(inputs.shape[0], 1).long().to(inputs.device)
        temporal_inputs = inputs + self.position_embeddings[position_inputs] # [N, T, F]

        # 2: Query, Key based multi-head self attention.
        q = torch.tensordot(temporal_inputs, self.Q_embedding_weights, dims=([2],[0])) # [N, T, F]
        k = torch.tensordot(temporal_inputs, self.K_embedding_weights, dims=([2],[0])) # [N, T, F]
        v = torch.tensordot(temporal_inputs, self.V_embedding_weights, dims=([2],[0])) # [N, T, F]

        # 3: Split, concat and scale.
        split_size = int(q.shape[-1]/self.n_heads)
        q_ = torch.cat(torch.split(q, split_size_or_sections=split_size, dim=2), dim=0) # [hN, T, F/h]
        k_ = torch.cat(torch.split(k, split_size_or_sections=split_size, dim=2), dim=0) # [hN, T, F/h]
        v_ = torch.cat(torch.split(v, split_size_or_sections=split_size, dim=2), dim=0) # [hN, T, F/h]
        
        outputs = torch.matmul(q_, k_.permute(0,2,1)) # [hN, T, T]
        outputs = outputs / (self.num_time_steps ** 0.5)
        # 4: Masked (causal) softmax to compute attention weights.
        diag_val = torch.ones_like(outputs[0])
        tril = torch.tril(diag_val)
        masks = tril[None, :, :].repeat(outputs.shape[0], 1, 1) # [h*N, T, T]
        padding = torch.ones_like(masks) * (-2**32+1)
        outputs = torch.where(masks==0, padding, outputs)
        outputs = F.softmax(outputs, dim=2)
        self.attn_wts_all = outputs # [h*N, T, T]
                
        # 5: Dropout on attention weights.
        if self.training:
            outputs = self.attn_dp(outputs)
        outputs = torch.matmul(outputs, v_)  # [hN, T, F/h]
        outputs = torch.cat(torch.split(outputs, split_size_or_sections=int(outputs.shape[0]/self.n_heads), dim=0), dim=2) # [N, T, F]
        
        # 6: Feedforward and residual
        outputs = self.feedforward(outputs)
        if self.residual:
            outputs = outputs + temporal_inputs
        return outputs

    def feedforward(self, inputs):
        outputs = F.relu(self.lin(inputs))
        return outputs + inputs


    def xavier_init(self):
        nn.init.xavier_uniform_(self.position_embeddings)
        nn.init.xavier_uniform_(self.Q_embedding_weights)
        nn.init.xavier_uniform_(self.K_embedding_weights)
        nn.init.xavier_uniform_(self.V_embedding_weights)

class SelfAttentionLayer(nn.Module):
    def __init__(self, config):
        super(SelfAttentionLayer, self).__init__()
        self.layers = nn.ModuleList([SelfAttention(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask):
        for i, layer_module in enumerate(self.layers):
            layer_outputs = layer_module(hidden_states, attention_mask)
            hidden_states = layer_outputs

        return hidden_states
        
class SelfAttention(nn.Module):
    def __init__(self, config):
        super(SelfAttention, self).__init__()
        self.n_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.attention_reign_per_head = int(self.hidden_size / self.n_heads)

        self.query = nn.Linear(self.hidden_size, self.hidden_size)
        self.key = nn.Linear(self.hidden_size, self.hidden_size)
        self.value = nn.Linear(self.hidden_size, self.hidden_size)
        # ff
        self.lin = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        # dropout
        self.attn_dp = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.n_heads, self.attention_reign_per_head)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
        
    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_reign_per_head)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask
        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim = -1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.attn_dp(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        # residual
        outputs = context_layer + hidden_states
        # 6: Feedforward
        outputs = self.feedforward(outputs)
        #
        return outputs

    def feedforward(self, inputs):
        outputs = F.relu(self.lin(inputs))
        return outputs + inputs



class TemporalAttentionLayer(nn.Module):
    def __init__(self,
                input_dim,
                n_heads,
                num_time_steps,
                attn_drop):
        super(TemporalAttentionLayer, self).__init__()
        self.n_heads = n_heads
        self.num_time_steps = num_time_steps
        self.input_dim = input_dim
        self.attention_reign_per_head = int(input_dim / self.n_heads)

        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        
        self.lin = nn.Linear(input_dim, input_dim, bias = True)
        self.attn_dp = nn.Dropout(attn_drop)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.n_heads, self.attention_reign_per_head)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
        
    def forward(self, tuple):
        hidden_states, attention_mask = tuple
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_reign_per_head)

        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask
        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim = -1)
        attention_probs = self.attn_dp(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.input_dim,)
        context_layer = context_layer.view(*new_context_layer_shape)
        # residual
        outputs = context_layer + hidden_states
        # 6: Feedforward
        outputs = self.feedforward(outputs)
        #
        return outputs, attention_mask

    def feedforward(self, inputs):
        outputs = F.relu(self.lin(inputs))
        return outputs + inputs
