import torch
import torch.nn as nn
import torch.nn.functional as F
import math
BertLayerNorm = torch.nn.LayerNorm

class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask):
        for i, layer_module in enumerate(self.layers):
            layer_outputs = layer_module(hidden_states, attention_mask)
            hidden_states = layer_outputs

        return hidden_states
        
class Feedforward(nn.Module):
    def __init__(self, config):
        super(Feedforward, self).__init__()
        self.lin_1 = nn.Linear(config.hidden_size, config.hidden_size * 2)
        self.lin_2 = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    def forward(self, x):
        x = self.lin_1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.lin_2(x)
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super(TransformerEncoderLayer, self).__init__()
        self.multihead_attention = MultiheadSelfAttention(config)
        self.feedforward = Feedforward(config)
        self.layernorm1 = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.layernorm2 = nn.LayerNorm(config.hidden_size, eps=1e-12)
    def forward(self, hidden_states, attention_mask):
        attention_outputs = self.multihead_attention(hidden_states, attention_mask)

        # Add & Norm (Residual)
        norm_output = self.layernorm1(hidden_states + attention_outputs)
        
        #feedfoward
        feedfoward_output = self.feedforward(attention_outputs)
        
        # Add & Norm (Residual)
        block_output = self.layernorm2(norm_output + feedfoward_output)

        return block_output

class MultiheadSelfAttention(nn.Module):
    def __init__(self, config):
        super(MultiheadSelfAttention, self).__init__()
        self.n_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.attention_reign_per_head = self.hidden_size // self.n_heads

        self.query = nn.Linear(self.hidden_size, self.hidden_size)
        self.key = nn.Linear(self.hidden_size, self.hidden_size)
        self.value = nn.Linear(self.hidden_size, self.hidden_size)
        # ff
        # dropout
        self.attn_dp = nn.Dropout(config.attention_probs_dropout_prob)
        self.layernorm2 = nn.LayerNorm(self.hidden_size, eps=1e-12)

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
    
        return context_layer


