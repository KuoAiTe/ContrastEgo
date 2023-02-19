import torch
import torch.nn as nn
import torch.nn.init as init

import torch.nn.functional as F
import math

class TorchAttentionBlock(nn.Module):
    def __init__(self, in_dim, num_head = 4):
        super(TorchAttentionBlock, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(in_dim, num_head, batch_first = True)
        self.attention_mode = 2
    def forward(self, x, attention_mask):
        x = torch.sum(x, dim = 1)
        attn_output, attn_output_weights = self.multihead_attn(x, x, x)
        pooled_output = torch.mean(attn_output, dim = 1)
        return pooled_output


class BertSelfOutput(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob = 0.1, layer_norm_eps = 1e-12):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states):
        outputs = self.dense(hidden_states).relu()
        outputs = self.dropout(hidden_states)
        outputs += hidden_states
        return outputs

class AttentionBlock(nn.Module):
    def __init__(self, num_attention_heads = 1):
        super(AttentionBlock, self).__init__()
        self.att1 = AttentionConv(8, 8, kernel_size = 3, num_attention_heads = num_attention_heads, stride = (1, 1), padding = (1, 0))
        self.att2 = AttentionConv(8, 16, kernel_size = 3, num_attention_heads = num_attention_heads, stride = (2, 2), padding = (1, 0))
        self.att3 = AttentionConv(16, 32, kernel_size = 3, num_attention_heads = num_attention_heads, stride = (2, 2), padding = (1, 0))
        self.att4 = AttentionConv(32, 64, kernel_size = 2, num_attention_heads = num_attention_heads, stride = (2, 2), padding = (1, 0))
        self.att5 = AttentionConv(128, 256, kernel_size = 2, num_attention_heads = num_attention_heads, stride = (2, 2), padding = (1, 0))
        self.out = nn.Sequential(
            AttentionConv(3, 1, kernel_size = 2, num_attention_heads = num_attention_heads, stride = (1, 1)),
            AttentionConv(1, 1, kernel_size = 2, num_attention_heads = num_attention_heads, stride = (1, 2)),
            AttentionConv(1, 1, kernel_size = 2, num_attention_heads = num_attention_heads, stride = (1, 2))
        )
        out_size = 192
        self.feedfoward = BertSelfOutput(out_size)
        self.feedfoward2 = BertSelfOutput(out_size // 2)
        self.feedfoward3 = BertSelfOutput(out_size // 4)
        self.position_embeddings = nn.Embedding(6, 576)
        self.init = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size = 3, stride = 1, padding = (2, 1), bias = False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )
        
    def forward(self, x):
        verbose = False#True
        #x = torch.sum(x, dim = 1).unsqueeze(1)
        if verbose: print('x', x.shape)
        x = self.init(x)
        if verbose: print(00, x.shape)
        out = self.att1(x)
        if verbose: print(10, out.shape)
        #out = self.feedfoward(out)
        if verbose: print(11, out.shape)
        
        out = self.att2(out)
        if verbose: print(20, out.shape)
        #out = self.feedfoward2(out)
        if verbose: print(21, out.shape)

        out = self.att3(out)
        if verbose: print(30, out.shape)
        #out = self.feedfoward3(out)
        if verbose: print(31, out.shape)
        #print(out.shape)
        out = self.att4(out)
        if verbose: print(40, out.shape)
        
        #out = self.feedfoward3(out)
        #out = self.out(x).squeeze()
        out = out.view(out.size(0), -1)
        return out
    


class AttentionConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_attention_heads = 1, stride=(1,1), padding=(0, 0), groups=1, bias=False):
        super(AttentionConv, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.num_attention_heads = num_attention_heads
        assert self.out_channels % self.groups == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"
        width = 4
        self.cnv1d = nn.Conv2d(in_channels, width, kernel_size = 1, padding = 0, bias=bias, stride = 1)
        self.cnv2d = nn.Conv2d(width, out_channels, kernel_size = 1, padding = 0, bias=bias, stride = 1)
        self.key_conv = nn.Conv2d(width, width, kernel_size = (1, 1), padding = 0, bias=bias, stride = stride)
        self.query_conv = nn.Conv2d(width, width, kernel_size = (1, 1), padding = 0, bias=bias, stride = stride)
        self.value_conv = nn.Conv2d(width, width, kernel_size = (1, 1), padding = 0, bias=bias, stride = stride)
        
        self.rel_h = nn.Parameter(torch.randn(out_channels // 2, 1, 1, kernel_size, 1), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn(out_channels // 2, 1, 1, 1, kernel_size), requires_grad=True)

        self.reset_parameters()
        self.dropout = nn.Dropout(0.1)
        
        self.norm_layer = nn.BatchNorm2d(width)
        self.norm_layer2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Sequential(
            nn.Conv2d(width, out_channels, kernel_size = 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.lin = nn.Conv2d(width, width, kernel_size = 1, padding = 0, bias=True, stride = 1)
        self.bn2 = nn.BatchNorm2d(width)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride , padding = 0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.cnv3d = nn.Conv2d(width, width, kernel_size = 3, padding = 1, bias=bias, stride = 2)
    def transpose_for_scores(self, x, attention_head_size):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 1, 3, 2, 4)

    def forward(self, x):
        identity = x
        
        x = self.cnv1d(x)
        x = self.norm_layer(x)
        x = x.relu()
        """
        q_out = self.query_conv(x)
        k_out = self.key_conv(x)
        v_out = self.value_conv(x)
        attention_head_size = q_out.size(-1) // self.num_attention_heads
        all_head_size = self.num_attention_heads * attention_head_size

        q_out = self.transpose_for_scores(q_out, attention_head_size)
        k_out = self.transpose_for_scores(k_out, attention_head_size)
        v_out = self.transpose_for_scores(v_out, attention_head_size)

        attention_scores = torch.matmul(q_out, k_out.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, v_out)
        context_layer = context_layer.permute(0, 1, 3, 2, 4).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        """

        out = self.cnv3d(x)
        out = self.bn2(out)
        context_layer = out.relu()

        #x = self.norm_layer(context_layer)
        #x = F.relu(x)
        #x = self.conv3(x)
        #outputs = context_layer + identity
        # 6: Feedforward
        #outputs = self.feedforward(context_layer)

        #print('cc',context_layer.shape)
        outputs = self.cnv2d(context_layer)
        #print(outputs.shape)
        outputs = self.norm_layer2(outputs)
        #print(outputs.shape)
        #print(identity.shape)

        identity = self.downsample(identity)
        #print(111, outputs.shape, identity.shape)
        #outputs += identity
        outputs = outputs.relu()

        return outputs
    def feedforward(self, inputs):
        outputs = F.relu(self.lin(inputs))
        return outputs + inputs


    def reset_parameters(self):
        pass
        #init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        #init.kaiming_normal_(self.value_conv.weight, mode='fan_out', nonlinearity='relu')
        #init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')
