import torch
import torch.nn as nn
from torch.nn.modules.loss import BCEWithLogitsLoss
from ..dysat.layers import SelfAttentionLayer
import mental
from dataclasses import dataclass

@dataclass
class MentalNetOutput:
    logits: str =  None
    last_logits: str = None
    attention_mask:str = None
    max_period_by_user:str = None

class MentalNetDySAT(nn.Module):
    def __init__(self, args, data_info):
        """[summary]

        Args:
            args ([type]): [description]
            time_length (int): Total timesteps in dataset.
        """
        super(MentalNetDySAT, self).__init__()
        self.args = args
        self.mentalnet_config = args.mental_net
        self.mentalnet_out_dim = self.mentalnet_config.out_dim()
        self.mental_net = mental.models.MentalNet(args, data_info)
        #self.mental_net = mental.models.MentalNetNaive(args, data_info)
        self.mental_net.use_edge_weight = False
        #self.depression_prediction_head = self.mental_net.mlp
        self.depression_prediction_head = nn.Linear(self.mentalnet_out_dim, 1)

        self.loss_fct = BCEWithLogitsLoss()
        self.num_time_steps = 15
        self_attention_config = self.args.self_attention_config
        self_attention_config.hidden_size = self.mentalnet_out_dim
        self.self_attention_layer = SelfAttentionLayer(self_attention_config)

        self.position_ids = torch.arange(self.num_time_steps)
        self.position_embeddings = nn.Embedding(self.num_time_steps, self.mentalnet_out_dim)

    def forward(self, batch_graphs, attention_mask = None):
        # Structural Attention forward
        device = batch_graphs[0][0].period_id[0].device
        batch_size = len(batch_graphs)
        batch_gnn_out = torch.full((batch_size, self.num_time_steps, self.mentalnet_out_dim), 0.0, device = device)
        auto_detect_attention = False
        if attention_mask == None:
            attention_mask = torch.full((batch_size, self.num_time_steps), -10000.0, device = device)
            auto_detect_attention = True
        user_max_period = torch.full((batch_size,), 0, device = device)

        i = 0
        for hetero_graphs in batch_graphs:
            for hetero_graph in hetero_graphs:
                period_id = hetero_graph.period_id[0]
                if len(hetero_graph.edge_index_dict) > 0:
                    output = self.mental_net(hetero_graph)
                    batch_gnn_out[i, period_id] = output
                    if auto_detect_attention:
                        attention_mask[i, period_id] = 0.0
                    user_max_period[i] = max(user_max_period[i], period_id)
            i += 1


        # 1: Add position embeddings to input
        position_ids = self.position_ids.to(device)
        batch_gnn_out = batch_gnn_out + self.position_embeddings(position_ids)

        attention_mask = attention_mask[:, None, None, :]
        logits = self.self_attention_layer(batch_gnn_out, attention_mask)
        last_logits = logits[torch.arange(batch_size, device = device), user_max_period, :]

        #logits = batch_gnn_out
        #last_logits = batch_gnn_out[:, -1, :]
        return MentalNetOutput(
            logits = logits,
            last_logits = last_logits,
            attention_mask = attention_mask,
            max_period_by_user = user_max_period,
        )

    def predict(self, feed_dict):
        graphs = feed_dict['hetero_graphs']
        labels = feed_dict['labels'].flatten()
        outputs = self.forward(graphs)
        prediction_scores = self.depression_prediction_head(outputs.last_logits).flatten()
        is_depressed = torch.zeros_like(labels).to(labels.device)
        is_depressed[prediction_scores >= 0] = 1
        return labels, is_depressed

    def compute_loss(self, feed_dict):
        graphs = feed_dict['hetero_graphs']
        labels = feed_dict['labels'].float().flatten()
        outputs = self.forward(graphs)
        prediction_scores = self.depression_prediction_head(outputs.last_logits).flatten()
        loss = self.loss_fct(prediction_scores, labels)
        return outputs.last_logits, loss

    def prepare_data(self, data):
        return mental.utils.utilities.prepare_dynamic_hetero_graphs(data)

class MentalNetDySAT_SimSiam(MentalNetDySAT):
    def __init__(self, args, data_info):
        super(MentalNetDySAT_SimSiam, self).__init__(args, data_info)
        self.base_encoder = MentalNetDySAT(args, data_info)
        p_dim = 128
        self.criterion = nn.CosineSimilarity(dim = 1).cuda()
        self.projection = nn.Linear(self.mentalnet_out_dim * 3, self.mentalnet_out_dim * 3)
        self.classifier = nn.Linear(self.mentalnet_out_dim * 3, 1)

    def predict(self, feed_dict):
        graphs = feed_dict['hetero_graphs']
        labels = feed_dict['labels'].flatten()
        device = feed_dict['labels'].device
        attention_mask = torch.full((len(graphs), self.num_time_steps), False, device = device)
        i = 0
        for user_graphs in graphs:
            for graph in user_graphs:
                attention_mask[i, graph.period_id[0]] = True
            i += 1
        expanded_attnetion_mask = torch.zeros_like(attention_mask).to(device)
        expanded_attnetion_mask[~attention_mask] = -10000
        z1_outputs = self.base_encoder.forward(graphs, expanded_attnetion_mask)
        z1_representation = z1_outputs.logits
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(z1_representation.size()).float()
        z1_pooled = torch.sum(z1_representation * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min = 1e-9)
        z1_pooled = z1_pooled / sum_mask

        outputs = self.depression_prediction_head(z1_pooled).flatten()
        is_depressed = torch.zeros_like(labels).to(labels.device)
        is_depressed[outputs >= 0] = 1
        return labels, is_depressed

    def gen_view(self, graphs, attention_mask, attn_prob = 0.8):
        aug_attention_mask = torch.bernoulli(torch.full(attention_mask.shape, attn_prob)).bool().to(attention_mask.device) & attention_mask
        #aug_attention_mask = attention_mask
        random_attention_mask = torch.bernoulli(torch.full(attention_mask.shape, 0.05)).bool().to(attention_mask.device) & aug_attention_mask
        # 0 means attended, -10000 -> not attended
        expanded_attnetion_mask = torch.zeros_like(aug_attention_mask).to(attention_mask.device)
        expanded_attnetion_mask[~aug_attention_mask] = -10000
        outputs = self.base_encoder.forward(graphs, expanded_attnetion_mask)
        representation = outputs.logits

        w = torch.nonzero(aug_attention_mask)
        if torch.nonzero(random_attention_mask).shape[0] > 0:
            idx = torch.ones(w.shape[0]).multinomial(representation[random_attention_mask].shape[0], replacement = True)
            representation[random_attention_mask] = representation[w[idx, 0], w[idx, 1], :]
            aug_attention_mask[random_attention_mask] = True
        input_mask_expanded = aug_attention_mask.unsqueeze(-1).expand(representation.size()).float()

        # pooling all the graph snapshot by user.
        pool_representation = torch.sum(representation * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min = 1e-9)
        pool_representation = pool_representation / sum_mask

        return pool_representation

    def compute_loss(self, feed_dict):
        graphs = feed_dict['hetero_graphs']
        labels = feed_dict['labels'].float().flatten()
        device = feed_dict['labels'].device
        attention_mask = torch.full((len(graphs), self.num_time_steps), False, device = device)
        i = 0
        for user_graphs in graphs:
            for graph in user_graphs:
                attention_mask[i, graph.period_id[0]] = True
            i += 1
        
        healthy_group = labels == 0
        depressed_group = ~healthy_group
        
        z1_pooled = self.gen_view(graphs, attention_mask, attn_prob = 1.0)
        z2_pooled = self.gen_view(graphs, attention_mask, attn_prob = 1.0)
        print('    labels ->', labels[:10])
        
        idx = torch.ones(z2_pooled[depressed_group].shape[0])
        siamese_loss = 0
        if idx.shape[0] > 0:
            idx = idx.multinomial(z2_pooled.shape[0], replacement = True)
            positive = z2_pooled[depressed_group][idx]
            positive_labels = labels[depressed_group][idx]
            vectors_concat = [z1_pooled, positive, torch.abs(z1_pooled - positive)]
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
            vectors_concat = [z1_pooled, negative, torch.abs(z1_pooled - negative)]
            features = torch.cat(vectors_concat, dim = -1)
            output = self.classifier(features).squeeze()
            s2_loss = (labels == negative_labels).float().squeeze()
            s2_loss = self.loss_fct(output, s2_loss)
            siamese_loss += s2_loss
            print('negative_labels ->', negative_labels[:10])
            print('    output ->', output[:10])
            
        prediction_scores_1 = self.depression_prediction_head(z1_pooled).flatten()

        loss = self.loss_fct(prediction_scores_1, labels)
        #loss2 = self.loss_fct(prediction_scores_2, labels)
        print(f'siamese_loss:{siamese_loss.item():.3f}, loss: {loss.item():.3f}')
        return z1_pooled, siamese_loss + loss#(loss + loss2) / 2 #+ (d_loss_1 + d_loss_2) / 2
