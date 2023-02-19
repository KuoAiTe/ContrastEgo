import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from torch_geometric.nn import SAGEConv, GCNConv, GATConv, HeteroConv
from torch_geometric.data import HeteroData
from torch_geometric.nn import MLP

from ..dysat.layers import SelfAttentionLayer
from mental.models import MentalNet

from torch_geometric.nn.conv.gatv2_conv import GATv2Conv
from torch_geometric.nn.pool.sag_pool import SAGPooling
from torch_geometric.nn.pool.topk_pool import TopKPooling
from torch_geometric.nn import aggr
# #dropout_node, dropout_edge
from torch import Tensor
from typing import Optional, Tuple


import torch.nn.init as init

import random
import mental


@dataclass
class MentalPlusOutput:
    logits: str =  None
    last_logits: str = None
    attention_mask:str = None
    max_period_by_user:str = None
    labels:str = None
    features:str = None
    reconstructed:str = None
def dropout_edge(edge_index: Tensor, p: float = 0.5,
                 force_undirected: bool = False,
                 training: bool = True) -> Tuple[Tensor, Tensor]:
    if p < 0. or p > 1.:
        raise ValueError(f'Dropout probability has to be between 0 and 1 '
                         f'(got {p}')

    if not training or p == 0.0:
        edge_mask = edge_index.new_ones(edge_index.size(1), dtype=torch.bool)
        return edge_index, edge_mask

    row, col = edge_index

    edge_mask = torch.rand(row.size(0), device=edge_index.device) >= p

    if force_undirected:
        edge_mask[row > col] = False

    edge_index = edge_index[:, edge_mask]

    if force_undirected:
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_mask = edge_mask.nonzero().repeat((2, 1)).squeeze()

    return edge_index, edge_mask


class BottleneckAttention(nn.Module):
    expansion = 1

    def __init__ (self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=16, dilation=1, norm_layer=None):
        super(BottleneckAttention, self).__init__()
        self.stride = stride
        width = int(planes * (base_width / 4.)) * groups

        self.conv1 = nn.Sequential(
            nn.Conv2d(inplanes, width, kernel_size=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            AttentionConv(width, width, kernel_size=5, padding=2, groups=1),
            nn.BatchNorm2d(width),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(width, self.expansion * planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.expansion * planes),
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.stride >= 2:
            out = F.avg_pool2d(out, (self.stride, self.stride))
        out += self.shortcut(x)
        out = F.relu(out)

        return out



class AttentionConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False):
        
        super(AttentionConv, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        assert self.out_channels % self.groups == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"
        
        self.rel_h = nn.Parameter(torch.randn(out_channels // 2, 1, 1, kernel_size, 1), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn(out_channels // 2, 1, 1, 1, kernel_size), requires_grad=True)

        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

        self.reset_parameters()

    def forward(self, x):
        batch, channels, height, width = x.size()

        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])
        q_out = self.query_conv(x)
        k_out = self.key_conv(padded_x)
        v_out = self.value_conv(padded_x)
        k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        v_out = v_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)

        k_out_h, k_out_w = k_out.split(self.out_channels // 2, dim=1)
        k_out = torch.cat((k_out_h + self.rel_h, k_out_w + self.rel_w), dim=1)

        k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
        v_out = v_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)

        q_out = q_out.view(batch, self.groups, self.out_channels // self.groups, height, width, 1)

        out = q_out * k_out
        out = F.softmax(out, dim=-1)
        out = torch.einsum('bnchwk,bnchwk -> bnchw', out, v_out).view(batch, -1, height, width)

        return out

    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.value_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')

        init.normal_(self.rel_h, 0, 1)
        init.normal_(self.rel_w, 0, 1)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class MentalPlus(nn.Module):
    def __init__(self, args, data_info):
        """[summary]

        Args:
            args ([type]): [description]
            time_length (int): Total timesteps in dataset.
        """
        super(MentalPlus, self).__init__()
        self.args = args
        config = args.mental_net
        self.config = config
        config.kernel_size = 4
        self.mentalnet_out_dim = config.gnn_hidden_channel 
        config.hetero_relations = ['mention', 'quote', 'reply',  'dummy']
        #config.hetero_relations = ['same']
        self.support_edge_weight = False
        self.convs = nn.ModuleDict({
            relation: self.init_convs() for relation in config.hetero_relations
        })
        self.inter_convs = self.init_inter_convs()

        self.encoder = MLP([768, 384, 192], dropout = config.dropout_prob)
        self.decoder = MLP([192, 384, 768], dropout = config.dropout_prob)

        #self.depression_prediction_head = MLP([self.mentalnet_out_dim * 4, 32, 1], dropout = config.dropout_prob)
        #self.depression_prediction_head = nn.Linear(self.mentalnet_out_dim * len(config.hetero_relations), 1)

        self.num_time_steps = 6
        self_attention_config = self.args.self_attention_config
        self_attention_config.hidden_size = self.mentalnet_out_dim * 3
        self.self_attention_layer = SelfAttentionLayer(self_attention_config)
        self.graph_attention_layer = SelfAttentionLayer(self_attention_config)

        self.position_ids = torch.arange(self.num_time_steps)
        self.position_embeddings = nn.Embedding(6, self.mentalnet_out_dim * 3)
        
        layer_size = config.gnn_hidden_channel * config.gnn_num_layer + 1
        
        self.conv1 = nn.Conv1d(1, config.conv1_channel, layer_size, layer_size)
        self.conv2 = nn.Conv1d(config.conv1_channel, config.conv2_channel, config.kernel_size, 1)
        self.maxpool = nn.MaxPool1d(2, 2)
        self.maxpool_out = (576 + 2 * 0 - 1 * (2 - 1) -1 ) / 2 + 1
        self.mlp = MLP([self.mentalnet_out_dim, 32, 1], dropout = config.dropout_prob)
        
        self.projection_head = MLP([self.mentalnet_out_dim, self.mentalnet_out_dim], dropout = config.dropout_prob)
        self.LayerNorm = nn.LayerNorm(self.mentalnet_out_dim * 3, eps=1e-12)
        self.dropout = nn.Dropout(0.5)
        
        self.loss_fct = nn.BCEWithLogitsLoss()
        self.criterion = nn.CosineSimilarity(dim=1).cuda()
        self.sort_aggr = aggr.SortAggregation(k = config.k)
        self.pool = TopKPooling(self.mentalnet_out_dim, 0.01)


        self.linear1 = nn.Linear(self.mentalnet_out_dim * 2, 64)
        self.linear2 = nn.Linear(64, 32)

        self._norm_layer = nn.BatchNorm2d
        self.dilation = 1
        self.inplanes = attn_in = 8
        self.groups = 1
        self.base_width = 8
        attn_out = attn_in
        #self.att = AttentionConv(attn_in, attn_in, 1, groups = attn_in)
        self.conv3 = nn.Conv2d(3, attn_in, (2, 64), 1)
        self.att = AttentionConv(attn_in, attn_in, 1)
        self.conv4 = nn.Conv2d(attn_in, 1, 2, 1)
        self.conv5 = nn.Conv2d(1, 1, 4, 1)
        self.bm1 = nn.BatchNorm1d(96)
        #4, 6, 8, 10, 12, 14
        #(12 - self.num_tiem_steps) / 2
        TARGET_SIZE = 6
        pad_size = (TARGET_SIZE - self.num_time_steps) // 2 + 1
        self.init = nn.Sequential(
            # CIFAR10
            nn.Conv2d(3, self.inplanes, kernel_size = 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(),
        )
        basic_block = BasicBlock
        block = BottleneckAttention
        num_blocks = [1, 1, 1, 1]
        num_classes = 1
        if block == BasicBlock:
            make_layer = self._make_layer
        if block == BottleneckAttention:
            make_layer = self._make_attn_layer
        make_layer = self._make_layer
        self.layer1 = make_layer(block, self.inplanes, num_blocks[0], stride=1)
        self.layer2 = make_layer(block, self.inplanes, num_blocks[1], stride=2)
        self.layer3 = make_layer(basic_block, 1 , num_blocks[2], stride=1)
        #self.layer4 = make_layer(block, 1, num_blocks[3], stride=2)
        self.depression_prediction_head = nn.Linear(125, 1)
        self.depression_prediction_head = nn.Linear(48, num_classes)

    def _make_attn_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplanes, planes, stride))
            self.inplanes = planes * block.expansion
        return nn.Sequential(*layers)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)
        
    def init_convs(self, **kwargs):
        convs = torch.nn.ModuleList()
        for i in range(2):
            convs.append(self.init_conv(in_channels = -1, out_channels = self.config.gnn_hidden_channel))
        
        convs.append(self.init_conv(in_channels = -1, out_channels = self.config.gnn_hidden_channel))
        #convs.append(self.init_conv(in_channels = -1, out_channels = 1))
        return convs

    def init_inter_convs(self, **kwargs):
        convs = torch.nn.ModuleList()
        convs.append(self.init_inter_conv(in_channels = -1, out_channels = self.config.gnn_hidden_channel))
        #convs.append(self.init_conv(in_channels = -1, out_channels = 1))
        return convs
    def init_conv(self, in_channels, out_channels):
        self.support_edge_weight = True
        return GCNConv(in_channels = in_channels, out_channels = out_channels)
    def init_inter_conv(self, in_channels, out_channels):
        self.support_edge_weight = False
        return GATConv(in_channels = in_channels, out_channels = out_channels)
        
    def forward(self, graph, dropout = 0):
        x_dict, edge_index_dict, edge_weight_dict, group = graph.x_dict, graph.edge_index_dict, graph.edge_weight_dict, graph.group
        features = x_dict['user']
        outputs = {}
        f = []
        for relation, edge_index in edge_index_dict.items():
            if relation == 'dummy': continue
            x = features
            y = []
            edge_weight = edge_weight_dict[relation]
            for i in range(len(self.convs[relation])):
                conv = self.convs[relation][i]
                if self.support_edge_weight:
                    x = conv(x, edge_index, edge_weight = edge_weight).relu()
                else:
                    x = conv(x, edge_index).relu()
                y.append(x) 
            output = torch.cat(y, dim = -1)
            f.append(output)
        f = torch.stack(f, dim = 1)
        print(f.shape)
        
        group_unique = torch.unique(graph.group).detach().cpu().numpy()
        batch_size = len(group_unique)
        user_nodes = (graph.label != -100)
        x = []
        for group_id in group_unique:
            index = (group == group_id) & user_nodes
            x.append(f[index])
        x = torch.stack(x, dim = 0).permute(0, 2, 1, 3)
        out = self.init(x)
        print(1, out.shape)
        out = self.layer1(out)
        print(2, out.shape)
        out = self.layer2(out)
        print(3, out.shape)
        out = self.layer3(out)
        print(out.shape)
        #out = self.layer4(out)
        print(out.shape)
        out = F.avg_pool2d(out, (2, 2))
        print(out.shape)
        out = out.view(out.size(0), -1)
        
        x = out

        return MentalPlusOutput(
            logits = x,
        )


    def predict(self, feed_dict):
        graph = feed_dict['hyper_graph']
        outputs = self.forward(graph, dropout = 0.0)
        indices = (graph.label != -100)
        labels = graph.label[indices].view(-1, self.num_time_steps)[:, 0]
        prediction_scores = self.depression_prediction_head(outputs.logits).flatten()
        is_depressed = torch.zeros_like(labels).to(labels.device)
        is_depressed[prediction_scores >= 0] = 1
        return labels, is_depressed

    def compute_loss(self, feed_dict):
        graph = feed_dict['hyper_graph']
        outputs = self.forward(graph, dropout = 0)
        indices = (graph.label != -100)
        labels = graph.label[indices].float().view(-1, self.num_time_steps)[:, 0]
        prediction_scores = self.depression_prediction_head(outputs.logits).flatten()
        loss = self.loss_fct(prediction_scores, labels)
        #loss2 = self.criterion(outputs.features, outputs.reconstructed).mean()
        return outputs.last_logits, loss# + loss2

    def prepare_data(self, data):
        return mental.utils.utilities.prepare_dynamic_hyper_graphs(data)

class MentalPlus_GCN(MentalPlus):
    def __init__(self, args, data_info):
        super(MentalPlus_GCN, self).__init__(args, data_info)
        
    def init_conv(self, in_channels, out_channels):
        self.support_edge_weight = True
        return GCNConv(in_channels = in_channels, out_channels = out_channels)

class MentalPlus_GAT(MentalPlus):
    def __init__(self, args, data_info):
        super(MentalPlus_GAT, self).__init__(args, data_info)

    def init_conv(self, in_channels, out_channels):
        self.support_edge_weight = False
        return GATConv(in_channels = in_channels, out_channels = out_channels)

class MentalPlus_SAGE(MentalPlus):
    def __init__(self, args, data_info):
        super(MentalPlus_SAGE, self).__init__(args, data_info)

    def init_conv(self, in_channels, out_channels):
        self.support_edge_weight = False
        return SAGEConv(in_channels = in_channels, out_channels = out_channels)
        
class projection_MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LayerNorm(in_dim),
            nn.GELU()
        )
        self.l2 = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LayerNorm(in_dim),
            nn.GELU()
        )
        self.l3 = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return x

class prediction_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.l2 = nn.Linear(hidden_dim, in_dim)
    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        return x


class MentalPlus_SimSiam(MentalPlus):
    def __init__(self, args, data_info):
        super(MentalPlus_SimSiam, self).__init__(args, data_info)
        self.projector = projection_MLP(in_dim = self.mentalnet_out_dim , out_dim = self.mentalnet_out_dim)
        self.predictor = prediction_MLP(in_dim = self.mentalnet_out_dim, hidden_dim = int(self.mentalnet_out_dim / 2))
        self.criterion = nn.CosineSimilarity(dim=1).cuda()
        self.triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
        self.classifier = nn.Linear(self.mentalnet_out_dim * 3, 1)

    def predict(self, feed_dict):
        graph = feed_dict['hyper_graph']
        indices = (graph.label != -100)
        labels = graph.label[indices].view(-1, 6)[:, 0]

        z1_pooled = self.forward(graph, 0).logits
        
        outputs = self.depression_prediction_head(z1_pooled).flatten()
        is_depressed = torch.zeros_like(labels).to(labels.device)
        is_depressed[outputs >= 0] = 1
        return labels, is_depressed
    def gen_view(self, graph, attention_mask, attn_prob = 0.8):
        aug_attention_mask = torch.bernoulli(torch.full(attention_mask.shape, attn_prob)).bool().to(attention_mask.device) & attention_mask
        random_attention_mask = torch.bernoulli(torch.full(attention_mask.shape, 0.05)).bool().to(attention_mask.device) & aug_attention_mask
        # 0 means attended, -10000 -> not attended
        expanded_attnetion_mask = torch.zeros_like(aug_attention_mask).to(attention_mask.device)
        expanded_attnetion_mask[~aug_attention_mask] = -10000
        z1_outputs = self.forward(graph, expanded_attnetion_mask)
        z1_representation = z1_outputs.logits
        input_mask_expanded = aug_attention_mask.unsqueeze(-1).expand(z1_representation.size()).float()
        
        z1_pooled = torch.sum(z1_representation * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min = 1e-9)
        z1_pooled = z1_pooled / sum_mask
        return z1_pooled
    def compute_loss(self, feed_dict):
        graph = feed_dict['hyper_graph']
        indices = (graph.label != -100)
        labels = graph.label[indices].float().view(-1, 6)[:, 0]

        anchor = self.forward(graph, 0).logits
        z2_pooled = self.forward(graph, 0.5).logits
        prediction_scores_1 = self.depression_prediction_head(anchor).flatten()

        loss = self.loss_fct(prediction_scores_1, labels)

        healthy_group = labels == 0
        depressed_group = ~healthy_group
        idx = torch.ones(z2_pooled[depressed_group].shape[0])
        siamese_loss = 0
        if idx.shape[0] > 0:
            idx = idx.multinomial(z2_pooled.shape[0], replacement = True)
            positive = z2_pooled[depressed_group][idx]
            positive_labels = labels[depressed_group][idx]
            vectors_concat = [anchor, positive, torch.abs(anchor - positive)]
            features = torch.cat(vectors_concat, dim = -1)
            output = self.classifier(features).squeeze()
            s1_loss = (labels == positive_labels).float().squeeze()
            s1_loss = self.loss_fct(output, s1_loss)
            siamese_loss += s1_loss
            print('positive_labels ->', positive_labels[:10])
            print('    output ->', output[:10])

        idx = torch.ones(z2_pooled[healthy_group].shape[0])
        if idx.shape[0] > 0:
            idx = idx.multinomial(z2_pooled.shape[0], replacement = True)
            negative = z2_pooled[healthy_group][idx]
            negative_labels = labels[healthy_group][idx]
            vectors_concat = [anchor, negative, torch.abs(anchor - negative)]
            features = torch.cat(vectors_concat, dim = -1)
            output = self.classifier(features).squeeze()
            s2_loss = (labels == negative_labels).float().squeeze()
            s2_loss = self.loss_fct(output, s2_loss)
            siamese_loss += s2_loss
            print('negative_labels ->', negative_labels[:10])
            print('    output ->', output[:10])
            
            
        

        print(loss, siamese_loss)
        return anchor, loss + siamese_loss
        