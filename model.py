import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.nn import RGCNConv
import numpy as np
from torch.nn.utils import weight_norm, remove_weight_norm


class FocalLoss(nn.Module):
    def __init__(self, gamma = 2.5, alpha = 1, size_average = True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        self.elipson = 0.000001
    
    def forward(self, logits, labels):
        """
        cal culates loss
        logits: batch_size * labels_length * seq_length
        labels: batch_size * seq_length
        """
        if labels.dim() > 2:
            labels = labels.contiguous().view(labels.size(0), labels.size(1), -1)
            labels = labels.transpose(1, 2)
            labels = labels.contiguous().view(-1, labels.size(2)).squeeze()
        if logits.dim() > 3:
            logits = logits.contiguous().view(logits.size(0), logits.size(1), logits.size(2), -1)
            logits = logits.transpose(2, 3)
            logits = logits.contiguous().view(-1, logits.size(1), logits.size(3)).squeeze()
        labels_length = logits.size(1)
        seq_length = logits.size(0)

        new_label = labels.unsqueeze(1)
        label_onehot = torch.zeros([seq_length, labels_length]).cuda().scatter_(1, new_label, 1)

        log_p = F.log_softmax(logits,-1)
        pt = label_onehot * log_p
        sub_pt = 1 - pt
        fl = -self.alpha * (sub_pt)**self.gamma * log_p
        if self.size_average:
            return fl.mean()
        else:
            return fl.sum()

class MLP(nn.Module):
    def __init__(self,depth,hidm):
        super(MLP, self).__init__()
        
        self.net=nn.Sequential()
        for _ in range(depth):
            self.net.append(nn.Linear(hidm, hidm))
            self.net.append(nn.ReLU())
    
    def forward(self,x):
        return F.softmax(self.net(x),dim=-1)

class MaskedNLLLoss(nn.Module):
    def __init__(self, weight=None):
        super(MaskedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight, reduction='sum')

    def forward(self, pred, target, mask):
        mask_ = mask.view(-1, 1)
        if type(self.weight) == type(None):
            loss = self.loss(pred * mask_, target) / torch.sum(mask)
        else:
            loss = self.loss(pred * mask_, target) \
                   / torch.sum(self.weight[target] * mask_.squeeze())
        return loss
    
def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.actv = gelu
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        inter = self.dropout_1(self.actv(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x


class MultiHeadedAttention(nn.Module):
    def __init__(self, head_count, model_dim, dropout=0.1):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim

        super(MultiHeadedAttention, self).__init__()
        self.head_count = head_count

        self.linear_k = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_v = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_q = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(model_dim, model_dim)

    def forward(self, key, value, query, mask=None):
        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count

        def shape(x):
            """  projection """
            return x.view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous() \
                .view(batch_size, -1, head_count * dim_per_head)

        key = self.linear_k(key).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)
        value = self.linear_v(value).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)
        query = self.linear_q(query).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)

        query = query / math.sqrt(dim_per_head)
        scores = torch.matmul(query, key.transpose(2, 3))

        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(mask, -1e10)

        attn = self.softmax(scores)

        drop_attn = self.dropout(attn)
        context = torch.matmul(drop_attn, value).transpose(1, 2). \
            contiguous().view(batch_size, -1, head_count * dim_per_head)
        output = self.linear(context)
        return output, attn
    
class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, speaker_emb):
        L = x.size(1)
        pos_emb = self.pe[:, :L]
        x = x + pos_emb + speaker_emb
        return x
    
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, iter, inputs_a, inputs_b, mask):
        if inputs_a.equal(inputs_b):
            if (iter != 0):
                inputs_b = self.layer_norm(inputs_b)
            else:
                inputs_b = inputs_b

            mask = mask.unsqueeze(1)
            context, atten_score = self.self_attn(inputs_b, inputs_b, inputs_b, mask=mask)
        else:
            if (iter != 0):
                inputs_b = self.layer_norm(inputs_b)
            else:
                inputs_b = inputs_b

            mask = mask.unsqueeze(1)
            context = self.self_attn(inputs_a, inputs_a, inputs_b, mask=mask)

        out = self.dropout(context) + inputs_b
        return self.feed_forward(out), atten_score
    
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, d_ff, heads, layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.layers = layers
        self.pos_emb = PositionalEncoding(d_model)
        self.transformer_inter = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_b, mask, speaker_emb=None):
        # 将positon、model_feature信息相加
        if speaker_emb != None:
            x_b = self.pos_emb(x_b, speaker_emb)
            x_b = self.dropout(x_b)
        for i in range(self.layers):
            x_b, atten_score = self.transformer_inter[i](i, x_b, x_b, mask.eq(0))
        return x_b, atten_score
    
class EnhancedFilterModule(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        gate = self.gate(x)
        out = gate * x
        return out
    
def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True):
    if torch.cuda.is_available():
        try:
            from apex.normalization import FusedLayerNorm

            return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
        except ImportError:
            pass
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)
    
class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True, relation=True, num_relation=-1,
                  relation_dim=10):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features  # 输入特征维度
        self.out_features = out_features  # 输出特征维度
        self.alpha = alpha
        self.concat = concat
        self.relation = relation

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        if self.relation:
            self.relation_embedding = relation_embedding
            self.a = nn.Parameter(torch.empty(size=(2 * out_features + relation_dim, 1)))
        else:
            self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))

        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.layer_norm = LayerNorm(out_features)

    def forward(self, h, adj):
        # h (B,N,D_in)
        Wh = torch.matmul(h, self.W)  # (B, N, D_out)
        a_input = self._prepare_attentional_mechanism_input(Wh)  # (B, N, N, 2*D_out)
        if self.relation:
            long_adj = adj.clone().type(torch.LongTensor).cuda()
            relation_one_hot = self.relation_embedding(long_adj)  # 得到每个关系对应的one-hot 固定表示
            a_input = torch.cat([a_input, relation_one_hot], dim=-1)  # （B, N, N, 2*D_out+num_relation）
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))  # (B, N , N)  所有部分都参与了计算 包括填充和没有关系连接的节点
        attention_score = F.softmax(e, dim=2)
        # TODO: Solve empty graph issue here!
        # attention_score = e
        if self.relation:
            zero_vec = -9e15 * torch.ones_like(e)  # 计算mask
            attention = torch.where(adj > 0, e, zero_vec)  # adj中非零位置 对应e的部分 保留，零位置(填充或没有关系连接)置为非常小的负数
            attention = F.softmax(attention, dim=2)  # B, N, N
        else:
            attention = F.softmax(e, dim=2)  # B, N, N


        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)  # (B,N,N_out)
        h_prime = self.layer_norm(h_prime)
        if self.concat:
            return F.gelu(h_prime), attention_score
        else:
            return h_prime, attention_score

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[1]  # N
        B = Wh.size()[0]  # B
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=1)
        Wh_repeated_alternating = Wh.repeat(1, N, 1)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating],
                                            dim=2)  # (B, N*N, 2*D_out)

        return all_combinations_matrix.view(B, N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class RGAT(nn.Module):
    def __init__(self, nfeat, nhid, dropout=0.2, alpha=0.2, nheads=2, num_relation=-1):
        """Dense version of GAT."""
        super(RGAT, self).__init__()
        self.dropout = dropout
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True, relation=True,
                                               num_relation=num_relation) for _ in range(nheads)]  # 多头注意力
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(nhid * nheads, nhid, dropout=dropout, alpha=alpha, concat=True,
                                           relation=True, num_relation=num_relation)  # 恢复到正常维度
        self.res = nn.Linear(nfeat, nhid)
        self.fc = nn.Linear(nhid, nhid)
        self.layer_norm = LayerNorm(nhid)

    def forward(self, x, adj):
        redisual = x
        x = F.dropout(x, self.dropout, training=self.training)
        # x = torch.cat([att(x, adj) for att in self.attentions], dim=-1)  # (B,N,num_head*N_out)
        attened_outputs = []
        attention_weights = []
        for att_module in self.attentions:
            # 计算注意力模块输出
            att_out, att_w = att_module(x, adj)
            attened_outputs.append(att_out)
            attention_weights.append(att_w)
            # 沿最后一个维度拼接
        x = torch.cat(attened_outputs, dim=-1)
        x = F.dropout(x, self.dropout, training=self.training)
        att_out, att_w = self.out_att(x, adj)
        attention_weights.append(att_w)
        x = F.gelu(att_out)  # (B, N, N_out)
        x = self.fc(x)  # (B, N, N_out)
        x = x + redisual
        x = self.layer_norm(x)
        return x, attention_weights
    
class DFGCN(nn.Module):
    def __init__(self, D_text, D_visual, D_audio, n_head, n_classes, hidden_dim, n_speakers, dropout):
        super(DFGCN, self).__init__()
        self.n_classes = n_classes
        self.n_speakers = n_speakers
        self.hidden_dim = hidden_dim
        if self.n_speakers == 2:
            padding_idx = 2
        if self.n_speakers == 9:
            padding_idx = 9
        self.speaker_embeddings = nn.Embedding(n_speakers+1, hidden_dim, padding_idx)
        global relation_embedding
        relation_embedding = nn.Embedding(6, 10)
        self.textf_input = weight_norm(nn.Linear(D_text, hidden_dim))
        self.acouf_input = weight_norm(nn.Linear(D_audio, hidden_dim))
        self.visuf_input = weight_norm(nn.Linear(D_visual, hidden_dim))

        # self.textf_input = weight_norm(nn.Conv1d(D_text, hidden_dim, kernel_size=1, padding=0, bias=False))
        # self.acouf_input = weight_norm(nn.Conv1d(D_audio, hidden_dim, kernel_size=1, padding=0, bias=False))
        # self.visuf_input = weight_norm(nn.Conv1d(D_visual, hidden_dim, kernel_size=1, padding=0, bias=False))

        self.a_a = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)
        self.v_v = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)
        self.t_t = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)

        self.transformer = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)

        # Inter-Speaker
        self.gatTer = RGAT(hidden_dim, hidden_dim, num_relation=4).cuda()
        self.gatT = RGAT(hidden_dim, hidden_dim, num_relation=4).cuda()

        self.rgcn = RGCNConv(hidden_dim, hidden_dim, num_relations=1)
        self.weight_box = self.rgcn.weight.reshape(-1)
        self.weight_box = nn.Parameter(self.weight_box).cuda()
        self.weight_box = self.weight_box.reshape(hidden_dim, hidden_dim)

        self.agate = EnhancedFilterModule(hidden_dim)
        self.vgate = EnhancedFilterModule(hidden_dim)
        self.tgate = EnhancedFilterModule(hidden_dim)

        self.mlp = MLP(depth=2, hidm=hidden_dim)

        self.t_output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            weight_norm(nn.Linear(hidden_dim, n_classes))
            )
        self.a_output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            weight_norm(nn.Linear(hidden_dim, n_classes))
            )
        self.v_output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            weight_norm(nn.Linear(hidden_dim, n_classes))
            )
        
        self.all_output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            weight_norm(nn.Linear(hidden_dim, n_classes))
            )

        self.normBNa = nn.BatchNorm1d(1024, affine=True)
        self.normBNb = nn.BatchNorm1d(1024, affine=True)
        self.normBNc = nn.BatchNorm1d(1024, affine=True)
        self.normBNd = nn.BatchNorm1d(1024, affine=True)


    def forward(self, U, visuf, acouf, u_mask, qmask, dia_len, Self_semantic_adj, Cross_semantic_adj):
        [r1,r2,r3,r4]=U
        seq_len, _, feature_dim = r1.size()
        r1 = self.normBNa(r1.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
        r2 = self.normBNb(r2.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
        r3 = self.normBNc(r3.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
        r4 = self.normBNd(r4.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
        textf = (r1 + r2 + r3 + r4)/4
        spk_idx = torch.argmax(qmask, -1)
        origin_spk_idx = spk_idx
        if self.n_speakers == 2:
            for i, x in enumerate(dia_len):
                spk_idx[i, x:] = (2*torch.ones(origin_spk_idx[i].size(0)-x)).int().cuda()
        if self.n_speakers == 9:
            for i, x in enumerate(dia_len):
                spk_idx[i, x:] = (9*torch.ones(origin_spk_idx[i].size(0)-x)).int().cuda()

        spk_embeddings = self.speaker_embeddings(spk_idx)

        textf = self.textf_input(textf.permute(1, 0, 2))
        acouf = self.acouf_input(acouf.permute(1, 0, 2))
        visuf = self.visuf_input(visuf.permute(1, 0, 2))

        # textf = self.textf_input(textf.permute(1, 2, 0)).transpose(1, 2)
        # acouf = self.acouf_input(acouf.permute(1, 2, 0)).transpose(1, 2)
        # visuf = self.visuf_input(visuf.permute(1, 2, 0)).transpose(1, 2)

        acouf, _ = self.a_a(acouf, u_mask, spk_embeddings)
        visuf, _ = self.v_v(visuf, u_mask, spk_embeddings)
        textf, _ = self.t_t(textf, u_mask, spk_embeddings)
        acouf = self.agate(acouf)
        visuf = self.vgate(visuf)
        textf = self.tgate(textf)

        features_all = textf + acouf + visuf

        features_all, _ = self.gatTer(features_all, Cross_semantic_adj)
        features_all, _ = self.gatT(features_all, Self_semantic_adj)

        features_all, _ = self.t_t(features_all, u_mask, spk_embeddings)

        bsz = features_all.size(0)
        feature_all_poll = torch.mean(features_all, dim=1)
        giv = self.mlp(feature_all_poll)

        weights = []
        for i in range(bsz):
            dy_feature = giv[i, :].view(-1, 1) 
            weights.append(np.array((dy_feature * self.weight_box).detach().cpu())) # [hdim, hdim]
        weights = torch.FloatTensor(weights).cuda()

        features_all = torch.bmm(features_all, weights)

        a = self.a_output_layer(acouf)
        v = self.v_output_layer(visuf)
        t = self.t_output_layer(textf)

        all_final_out = t + a + v

        features_all = self.all_output_layer(features_all) + all_final_out

        sub_log_prog = []
        # Emotion Classifier
        sub_log_prog.append(F.log_softmax(t, dim=-1))
        sub_log_prog.append(F.log_softmax(a, dim=-1))
        sub_log_prog.append(F.log_softmax(v, dim=-1))
        all_log_prob = F.log_softmax(features_all, dim=-1)
        all_prob = F.softmax(features_all, 2)
        return sub_log_prog, all_log_prob, all_prob, features_all
    

