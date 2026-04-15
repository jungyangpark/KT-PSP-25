# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from torch.autograd import Variable

# refs https://github.com/jhljx/GKT
import torch
device = "cpu" if not torch.cuda.is_available() else "cuda"

class GKT(nn.Module):
    """Graph-based Knowledge Tracing Modeling Student Proficiency Using Graph Neural Network

    Args:
        num_c (int): total num of unique questions
        hidden_dim (int): hidden dimension for MLP
        emb_size (int): embedding dimension for question embedding layer
        graph_type (str, optional): graph type, dense or transition. Defaults to "dense".
        graph (_type_, optional): graph. Defaults to None.
        dropout (float, optional): dropout. Defaults to 0.5.
        emb_type (str, optional): emb_type. Defaults to "qid".
        emb_path (str, optional): emb_path. Defaults to "".
        bias (bool, optional): add bias for DNN. Defaults to True.
        use_ed (int, optional): number of exercise difficulty features to use. Defaults to 0.
    """
    def __init__(self, num_c, hidden_dim, emb_size, graph_type="dense", graph=None, dropout=0.5, emb_type="qid", emb_path="",bias=True, use_ed=0, use_mp=0):
        super(GKT, self).__init__()
        self.model_name = "gkt"
        self.num_c = num_c
        self.hidden_dim = hidden_dim
        self.emb_size = emb_size
        self.res_len = 2
        self.graph_type = graph_type
        self.graph = nn.Parameter(graph)  # [num_c, num_c]
        self.graph.requires_grad = False  # fix parameter
        self.emb_type = emb_type
        self.emb_path = emb_path
        self.num_ed_features = use_ed

        # MP features 설정
        self.use_mp = use_mp
        if use_mp == 8:
            # New mode: MP 0-3 (4개) + Ratio 0-3 (4개) = 8 features
            self.num_mp_meta_features = 4  # MP 0-3: 문제 요구사항
            self.num_mp_ratio_features = 4  # Ratio 0-3: 충족률 (MP4-7 / MP0-3)
            self.num_mp_total_features = 8  # MP 0-3 + Ratio 0-3
        elif use_mp == 0:
            self.num_mp_meta_features = 0
            self.num_mp_ratio_features = 0
            self.num_mp_total_features = 0
        else:
            print(f"Warning: use_mp={use_mp} is not standard. Recommended: 0 (no MP) or 8 (MP 0-3 + Ratio 0-3).")
            self.num_mp_meta_features = use_mp
            self.num_mp_ratio_features = 0
            self.num_mp_total_features = use_mp

        self.num_mp_features = self.num_mp_total_features


        # one-hot feature and question
        one_hot_feat = torch.eye(self.res_len * self.num_c).to(device)
        self.one_hot_feat = one_hot_feat
        # self.one_hot_q = torch.eye(self.num_c, device=self.one_hot_feat.device)
        # zero_padding = torch.zeros(1, self.num_c, device=self.one_hot_feat.device)
        self.one_hot_q = torch.eye(self.num_c).to(device)
        zero_padding = torch.zeros(1, self.num_c).to(device)
        self.one_hot_q = torch.cat((self.one_hot_q, zero_padding), dim=0)
        
        if emb_type.startswith("qid"):
            # concept and concept & response embeddings
            self.interaction_emb = nn.Embedding(self.res_len * num_c, emb_size)
            # last embedding is used for padding, so dim + 1
            self.emb_c = nn.Embedding(num_c + 1, emb_size, padding_idx=-1)

        # f_self function
        mlp_input_dim = hidden_dim + emb_size + self.num_ed_features + self.num_mp_features
        self.f_self = MLP(mlp_input_dim, hidden_dim, hidden_dim, dropout=dropout, bias=bias)

        # f_neighbor functions
        self.f_neighbor_list = nn.ModuleList()

        # f_in functions
        self.f_neighbor_list.append(MLP(2 * mlp_input_dim, hidden_dim, hidden_dim, dropout=dropout, bias=bias))
        # f_out functions
        self.f_neighbor_list.append(MLP(2 * mlp_input_dim, hidden_dim, hidden_dim, dropout=dropout, bias=bias))


        # Erase & Add Gate
        self.erase_add_gate = EraseAddGate(hidden_dim, num_c)
        # Gate Recurrent Unit
        self.gru = nn.GRUCell(hidden_dim, hidden_dim, bias=bias)

        # Ratio prediction output layers (use_mp=8일 때만 생성)
        if self.use_mp == 8:
            # Step 1: Ratio 0-3 예측 layers (4개)
            self.ratio_out_layers = []
            for i in range(4):  # Ratio 0-3 예측 (4개)
                ratio_out = nn.Linear(hidden_dim, 1, bias=bias)  # Output 1 value per concept
                self.ratio_out_layers.append(ratio_out)
            self.ratio_out_layers = nn.ModuleList(self.ratio_out_layers)

            # Step 2: Ratio 4개를 입력으로 받아 prediction 예측하는 FC layer
            # Input: hidden_dim + 4 (ratio predictions at concept level) -> Output: 1 (prediction)
            self.predict = nn.Linear(hidden_dim + 4, 1, bias=bias)
        else:
            # Baseline: prediction만 예측
            self.predict = nn.Linear(hidden_dim, 1, bias=bias)

    # Aggregate step, as shown in Section 3.2.1 of the paper
    def _aggregate(self, xt, qt, ht, batch_size, ed_t=None, mp_t=None):
        r"""
        Parameters:
            xt: input one-hot question answering features at the current timestamp
            qt: question indices for all students in a batch at the current timestamp
            ht: hidden representations of all concepts at the current timestamp
            batch_size: the size of a student batch
            ed_t: exercise difficulty features at current timestamp (optional)
        Shape:
            xt: [batch_size]
            qt: [batch_size]
            ht: [batch_size, num_c, hidden_dim]
            ed_t: [batch_size, num_ed_features] (if provided)
            tmp_ht: [batch_size, num_c, hidden_dim + emb_size + num_ed_features]
        Return:
            tmp_ht: aggregation results of concept hidden knowledge state, concept(& response) embedding, and ED features
        """
        qt_mask = torch.ne(qt, -1)  # [batch_size], qt != -1
        x_idx_mat = torch.arange(self.res_len * self.num_c, device=device)
        x_embedding = self.interaction_emb(x_idx_mat)  # [res_len * num_c, emb_size]#the emb for each concept with answer?
        # print(xt[qt_mask])
        # print(self.one_hot_feat)
        masked_feat = F.embedding(xt[qt_mask], self.one_hot_feat)  # [mask_num, res_len * num_c] A simple lookup table that looks up embeddings in a fixed dictionary and size.
        #nn.functional.embedding(input, weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False)
        res_embedding = masked_feat.mm(x_embedding)  # [mask_num, emb_size]
        mask_num = res_embedding.shape[0]

        concept_idx_mat = self.num_c * torch.ones((batch_size, self.num_c), device=device).long()
        concept_idx_mat[qt_mask, :] = torch.arange(self.num_c, device=device)
        concept_embedding = self.emb_c(concept_idx_mat)  # [batch_size, num_c, emb_size]

        index_tuple = (torch.arange(mask_num, device=device), qt[qt_mask].long())
        concept_embedding[qt_mask] = concept_embedding[qt_mask].index_put(index_tuple, res_embedding)
        tmp_ht = torch.cat((ht, concept_embedding), dim=-1)  # [batch_size, num_c, hidden_dim + emb_size]
        
        # Include ED features if available
        if self.num_ed_features > 0 and ed_t is not None:
            # ed_t shape: [batch_size, num_ed_features]
            # Expand to match tmp_ht: [batch_size, num_c, num_ed_features]
            ed_expanded = ed_t.unsqueeze(1).repeat(1, self.num_c, 1)
            tmp_ht = torch.cat((tmp_ht, ed_expanded), dim=-1)  # [batch_size, num_c, hidden_dim + emb_size + num_ed_features]
        
        # Include MP features if available
        if self.num_mp_features > 0 and mp_t is not None:
            # mp_t shape: [batch_size, num_mp_features]
            # Expand to match tmp_ht: [batch_size, num_c, num_mp_features]
            mp_expanded = mp_t.unsqueeze(1).repeat(1, self.num_c, 1)
            tmp_ht = torch.cat((tmp_ht, mp_expanded), dim=-1)  # [batch_size, num_c, hidden_dim + emb_size + num_ed_features + num_mp_features]
        
        return tmp_ht

    # GNN aggregation step, as shown in 3.3.2 Equation 1 of the paper
    def _agg_neighbors(self, tmp_ht, qt):
        r"""
        Parameters:
            tmp_ht: temporal hidden representations of all concepts after the aggregate step
            qt: question indices for all students in a batch at the current timestamp
        Shape:
            tmp_ht: [batch_size, num_c, hidden_dim + emb_size]
            qt: [batch_size]
            m_next: [batch_size, num_c, hidden_dim]
        Return:
            m_next: hidden representations of all concepts aggregating neighboring representations at the next timestamp
            concept_embedding: input of VAE (optional)
            rec_embedding: reconstructed input of VAE (optional)
            z_prob: probability distribution of latent variable z in VAE (optional)
        """
        qt_mask = torch.ne(qt, -1)  # [batch_size], qt != -1
        masked_qt = qt[qt_mask]  # [mask_num, ]
        masked_tmp_ht = tmp_ht[qt_mask]  # [mask_num, num_c, hidden_dim + emb_size]
        mask_num = masked_tmp_ht.shape[0]
        self_index_tuple = (torch.arange(mask_num, device=qt.device), masked_qt.long())
        self_ht = masked_tmp_ht[self_index_tuple]  # [mask_num, hidden_dim + emb_size]
        self_features = self.f_self(self_ht)  # [mask_num, hidden_dim]
        expanded_self_ht = self_ht.unsqueeze(dim=1).repeat(1, self.num_c, 1)  #[mask_num, num_c, hidden_dim + emb_size]
        neigh_ht = torch.cat((expanded_self_ht, masked_tmp_ht), dim=-1)  #[mask_num, num_c, 2 * (hidden_dim + emb_size)]
        concept_embedding, rec_embedding, z_prob = None, None, None

     
        adj = self.graph[masked_qt.long(), :].unsqueeze(dim=-1)  # [mask_num, num_c, 1]
        reverse_adj = self.graph[:, masked_qt.long()].transpose(0, 1).unsqueeze(dim=-1)  # [mask_num, num_c, 1]
        # self.f_neighbor_list[0](neigh_ht) shape: [mask_num, num_c, hidden_dim]
        neigh_features = adj * self.f_neighbor_list[0](neigh_ht) + reverse_adj * self.f_neighbor_list[1](neigh_ht)

        # neigh_features: [mask_num, num_c, hidden_dim]
        m_next = tmp_ht[:, :, :self.hidden_dim]
        m_next[qt_mask] = neigh_features
        m_next[qt_mask] = m_next[qt_mask].index_put(self_index_tuple, self_features)
        return m_next, concept_embedding, rec_embedding, z_prob

    # Update step, as shown in Section 3.3.2 of the paper
    def _update(self, tmp_ht, ht, qt):
        r"""
        Parameters:
            tmp_ht: temporal hidden representations of all concepts after the aggregate step
            ht: hidden representations of all concepts at the current timestamp
            qt: question indices for all students in a batch at the current timestamp
        Shape:
            tmp_ht: [batch_size, num_c, hidden_dim + emb_size]
            ht: [batch_size, num_c, hidden_dim]
            qt: [batch_size]
            h_next: [batch_size, num_c, hidden_dim]
        Return:
            h_next: hidden representations of all concepts at the next timestamp
            concept_embedding: input of VAE (optional)
            rec_embedding: reconstructed input of VAE (optional)
            z_prob: probability distribution of latent variable z in VAE (optional)
        """
        qt_mask = torch.ne(qt, -1)  # [batch_size], qt != -1
        mask_num = qt_mask.nonzero().shape[0]
        # GNN Aggregation
        m_next, concept_embedding, rec_embedding, z_prob = self._agg_neighbors(tmp_ht, qt)  # [batch_size, num_c, hidden_dim]
        # Erase & Add Gate
        m_next[qt_mask] = self.erase_add_gate(m_next[qt_mask])  # [mask_num, num_c, hidden_dim]
        # GRU
        h_next = m_next
        res = self.gru(m_next[qt_mask].reshape(-1, self.hidden_dim), ht[qt_mask].reshape(-1, self.hidden_dim))  # [mask_num * num_c, hidden_num]
        index_tuple = (torch.arange(mask_num, device=qt_mask.device), )
        h_next[qt_mask] = h_next[qt_mask].index_put(index_tuple, res.reshape(-1, self.num_c, self.hidden_dim))
        return h_next, concept_embedding, rec_embedding, z_prob

    # Predict step, as shown in Section 3.3.3 of the paper
    def _predict(self, h_next, qt, ratio_preds=None):
        r"""
        Parameters:
            h_next: hidden representations of all concepts at the next timestamp after the update step
            qt: question indices for all students in a batch at the current timestamp
            ratio_preds: ratio predictions (4 tensors of shape [batch_size, num_c]) if use_mp=8
        Shape:
            h_next: [batch_size, num_c, hidden_dim]
            qt: [batch_size]
            ratio_preds: list of 4 tensors, each [batch_size, num_c] (optional)
            y: [batch_size, num_c]
        Return:
            y: predicted correct probability of all concepts at the next timestamp
        """
        qt_mask = torch.ne(qt, -1)  # [batch_size], qt != -1

        # New architecture: Ratio 예측 → Ratio를 입력으로 받아 prediction 예측
        if self.use_mp == 8 and ratio_preds is not None:
            # ratio_preds: list of 4 tensors, each [batch_size, num_c]
            # Concatenate ratio predictions to h_next
            ratio_concat = torch.stack(ratio_preds, dim=-1)  # Shape: [batch_size, num_c, 4]
            h_next_with_ratio = torch.cat([h_next, ratio_concat], dim=-1)  # Shape: [batch_size, num_c, hidden_dim + 4]
            y = self.predict(h_next_with_ratio).squeeze(dim=-1)  # [batch_size, num_c]
            y[qt_mask] = torch.sigmoid(y[qt_mask])  # [batch_size, num_c]
        else:
            # Baseline: prediction만 예측
            y = self.predict(h_next).squeeze(dim=-1)  # [batch_size, num_c]
            y[qt_mask] = torch.sigmoid(y[qt_mask])  # [batch_size, num_c]
        return y

    def _get_next_pred(self, yt, q_next):
        r"""
        Parameters:
            yt: predicted correct probability of all concepts at the next timestamp
            q_next: question index matrix at the next timestamp
            batch_size: the size of a student batch
        Shape:
            y: [batch_size, num_c]
            questions: [batch_size, seq_len]
            pred: [batch_size, ]
        Return:
            pred: predicted correct probability of the question answered at the next timestamp
        """
        next_qt = q_next
        next_qt = torch.where(next_qt != -1, next_qt, self.num_c * torch.ones_like(next_qt, device=yt.device))
        one_hot_qt = F.embedding(next_qt.long(), self.one_hot_q)  # [batch_size, num_c]
        # dot product between yt and one_hot_qt
        pred = (yt * one_hot_qt).sum(dim=1)  # [batch_size, ]
        return pred


    def forward(self, q, r, dcur=None):
        """Forward pass for GKT model

        Args:
            q (_type_): question indices
            r (_type_): response labels
            dcur (dict, optional): additional data including exercise difficulty features. Defaults to None.

        Returns:
            torch.Tensor: the correct probability of questions answered at the next timestamp
        """

        # Handle ED features if use_ed is enabled
        ed = None
        if self.num_ed_features > 0:
            if dcur is not None:
                ed_data = []
                for i in range(self.num_ed_features):
                    ed_key = f"error_type_{i}"
                    if ed_key in dcur and len(dcur[ed_key]) > 0:
                        # Shift ED features: pad with zero at start, remove last timestep
                        ed_shifted = torch.cat([torch.zeros(dcur[ed_key].shape[0], 1, device=device), 
                                              dcur[ed_key][:, :-1]], dim=1)
                        ed_data.append(ed_shifted.to(device).float())
                    else:
                        ed_data.append(torch.zeros_like(q.float(), device=device))
                
                if len(ed_data) > 0:
                    ed = torch.stack(ed_data, dim=-1)
            else:
                # No dcur provided but model expects ED features - use fallback padding
                batch_size, seq_len = q.shape
                ed = torch.full((batch_size, seq_len, self.num_ed_features), -1.0, device=device, dtype=torch.float32)

        # Handle MP features if use_mp is enabled
        mp = None
        if self.use_mp > 0:
            if dcur is not None:
                mp_data = []
                batch_size, seq_len = q.shape

                # MP 0-3: 문제 요구사항 (shift 없음)
                for i in range(self.num_mp_meta_features):
                    mp_key = f"math_prof_{i}"
                    if mp_key in dcur and len(dcur[mp_key]) > 0:
                        dcur_mp_tensor = dcur[mp_key].to(device)
                        # Handle sequence length matching
                        if dcur_mp_tensor.shape[1] > seq_len:
                            mp_current = dcur_mp_tensor[:, :seq_len]
                        elif dcur_mp_tensor.shape[1] < seq_len:
                            pad_len = seq_len - dcur_mp_tensor.shape[1]
                            mp_current = torch.cat([dcur_mp_tensor, torch.zeros((dcur_mp_tensor.shape[0], pad_len), device=device)], dim=1)
                        else:
                            mp_current = dcur_mp_tensor
                        mp_data.append(mp_current.float())
                    else:
                        mp_data.append(torch.zeros(batch_size, seq_len, device=device, dtype=torch.float))

                # Ratio 0-3: 충족률 (MP4-7 / MP0-3) - shift 적용
                for i in range(self.num_mp_ratio_features):
                    ratio_key = f"mp_ratio_{i}"
                    if ratio_key in dcur and len(dcur[ratio_key]) > 0:
                        dcur_ratio_tensor = dcur[ratio_key].to(device)
                        # Shift 적용: 이전 timestep 데이터 사용
                        ratio_shifted = torch.cat([torch.zeros(dcur_ratio_tensor.shape[0], 1, device=device),
                                                  dcur_ratio_tensor[:, :-1]], dim=1)
                        # Handle sequence length matching
                        if ratio_shifted.shape[1] > seq_len:
                            ratio_shifted = ratio_shifted[:, :seq_len]
                        elif ratio_shifted.shape[1] < seq_len:
                            pad_len = seq_len - ratio_shifted.shape[1]
                            ratio_shifted = torch.cat([ratio_shifted, torch.zeros((ratio_shifted.shape[0], pad_len), device=device)], dim=1)
                        mp_data.append(ratio_shifted.float())
                    else:
                        mp_data.append(torch.zeros(batch_size, seq_len, device=device, dtype=torch.float))

                if len(mp_data) > 0:
                    mp = torch.stack(mp_data, dim=-1)
            else:
                # No dcur provided but model expects MP features - use fallback padding
                batch_size, seq_len = q.shape
                mp = torch.full((batch_size, seq_len, self.num_mp_total_features), -1.0, device=device, dtype=torch.float32)

        features = q*2 + r
        questions = q
        
        batch_size, seq_len = features.shape
        ht = Variable(torch.zeros((batch_size, self.num_c, self.hidden_dim), device=device))

        pred_list = []
        ratio_pred_list = [[] for _ in range(4)]  # Ratio 0-3를 위한 리스트

        for i in range(seq_len):
            xt = features[:, i]  # [batch_size]
            qt = questions[:, i]  # [batch_size]
            qt_mask = torch.ne(qt, -1)  # [batch_size], next_qt != -1

            ed_t = None
            if ed is not None:
                ed_t = ed[:, i, :]  # [batch_size, num_ed_features]

            mp_t = None
            if mp is not None:
                mp_t = mp[:, i, :]  # [batch_size, num_mp_features]

            tmp_ht = self._aggregate(xt, qt, ht, batch_size, ed_t, mp_t)  # [batch_size, num_c, hidden_dim + emb_size + num_ed_features + num_mp_features]
            h_next, concept_embedding, rec_embedding, z_prob = self._update(tmp_ht, ht, qt)  # [batch_size, num_c, hidden_dim]
            ht[qt_mask] = h_next[qt_mask]  # update new ht

            # New architecture: Ratio 예측 → Ratio를 입력으로 받아 prediction 예측
            if self.use_mp == 8:
                # Step 1: Ratio 0-3 predictions (4개)
                ratio_preds = []
                for j in range(4):  # Ratio 0-3 예측
                    ratio_pred = self.ratio_out_layers[j](h_next).squeeze(dim=-1)  # [batch_size, num_c]
                    ratio_pred = torch.sigmoid(ratio_pred)  # [0, 1] range for ratio
                    ratio_preds.append(ratio_pred)

                # Step 2: Predict final output using h_next + ratio predictions
                yt = self._predict(h_next, qt, ratio_preds)  # [batch_size, num_c]

                # Collect ratio predictions for next timestep (for loss calculation)
                if i < seq_len - 1:
                    next_qt = questions[:, i + 1]
                    next_qt_fixed = torch.where(next_qt != -1, next_qt, self.num_c * torch.ones_like(next_qt, device=next_qt.device))
                    one_hot_qt = F.embedding(next_qt_fixed.long(), self.one_hot_q)  # [batch_size, num_c]

                    for j in range(4):  # Ratio 0-3 예측
                        # dot product between ratio_pred and one_hot_qt
                        ratio_pred_next = (ratio_preds[j] * one_hot_qt).sum(dim=1)  # [batch_size, ]
                        ratio_pred_list[j].append(ratio_pred_next)
            else:
                # Baseline: prediction만 예측
                yt = self._predict(h_next, qt)  # [batch_size, num_c]

            if i < seq_len - 1:
                pred = self._get_next_pred(yt, questions[:, i + 1])
                pred_list.append(pred)

        pred_res = torch.stack(pred_list, dim=1)  # [batch_size, seq_len - 1]

        # Ratio predictions (use_mp=8일 때 multi-task)
        if self.use_mp == 8:
            ratio_predictions = []
            for j in range(4):  # Ratio 0-3 예측 (4개)
                ratio_pred = torch.stack(ratio_pred_list[j], dim=1)  # [batch_size, seq_len - 1]
                ratio_predictions.append(ratio_pred)
            return pred_res, ratio_predictions
        else:
            return pred_res

# Multi-Layer Perceptron(MLP) layer
class MLP(nn.Module):
    """Two-layer fully-connected ReLU net with batch norm."""

    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0., bias=True):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=bias)
        self.norm = nn.BatchNorm1d(output_dim)
        # the paper said they added Batch Normalization for the output of MLPs, as shown in Section 4.2
        self.dropout = dropout
        self.output_dim = output_dim
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        if inputs.numel() == self.output_dim or inputs.numel() == 0:
            # batch_size == 1 or 0 will cause BatchNorm error, so return the input directly
            return inputs
        if len(inputs.size()) == 3:
            x = inputs.view(inputs.size(0) * inputs.size(1), -1)
            x = self.norm(x)
            return x.view(inputs.size(0), inputs.size(1), -1)
        else:  # len(input_size()) == 2
            return self.norm(inputs)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.dropout(x, self.dropout, training=self.training)  # pay attention to add training=self.training
        x = F.relu(self.fc2(x))
        return self.batch_norm(x)


class EraseAddGate(nn.Module):
    """Erase & Add Gate module
    NOTE: this erase & add gate is a bit different from that in DKVMN.
    For more information about Erase & Add gate, please refer to the paper "Dynamic Key-Value Memory Networks for Knowledge Tracing"
    The paper can be found in https://arxiv.org/abs/1611.08108

    Args:
        nn (_type_): _description_
    """

    def __init__(self, feature_dim, num_c, bias=True):
        super(EraseAddGate, self).__init__()
        # weight
        self.weight = nn.Parameter(torch.rand(num_c))
        self.reset_parameters()
        # erase gate
        self.erase = nn.Linear(feature_dim, feature_dim, bias=bias)
        # add gate
        self.add = nn.Linear(feature_dim, feature_dim, bias=bias)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        """
        Params:
            x: input feature matrix
        
        Shape:
            x: [batch_size, num_c, feature_dim]
            res: [batch_size, num_c, feature_dim]
        
        Return:
            res: returned feature matrix with old information erased and new information added
            The GKT paper didn't provide detailed explanation about this erase-add gate. As the erase-add gate in the GKT only has one input parameter,
            this gate is different with that of the DKVMN. We used the input matrix to build the erase and add gates, rather than $\mathbf{v}_{t}$ vector in the DKVMN.
        
        """
        erase_gate = torch.sigmoid(self.erase(x))  # [batch_size, num_c, feature_dim]
        # self.weight.unsqueeze(dim=1) shape: [num_c, 1]
        tmp_x = x - self.weight.unsqueeze(dim=1) * erase_gate * x
        add_feat = torch.tanh(self.add(x))  # [batch_size, num_c, feature_dim]
        res = tmp_x + self.weight.unsqueeze(dim=1) * add_feat
        return res