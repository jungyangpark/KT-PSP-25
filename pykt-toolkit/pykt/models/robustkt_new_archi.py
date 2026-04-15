import torch
from torch import nn
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
import math
import torch.nn.functional as F
from enum import IntEnum
import numpy as np
from torch.nn import LayerNorm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Data preprocessing is available at https://github.com/Roks777/robustkt_dataprocess
class Robustkt(nn.Module):
    def __init__(self, n_question, n_pid, d_model, n_blocks, dropout,
                 ks=5, d_ff=256, kq_same=1, final_fc_dim=512,
                 num_attn_heads=8, separate_qa=False, l2=1e-5,
                 emb_type="qid", emb_path="", pretrain_dim=768, use_ed=0, use_mp=0):
        super().__init__()
        self.model_name = "robustkt"
        self.n_question = n_question
        self.dropout = dropout
        self.kq_same = kq_same
        self.n_pid = n_pid
        self.l2 = l2
        self.model_type = self.model_name
        self.separate_qa = separate_qa
        self.emb_type = emb_type
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

        embed_l = d_model
        self.num_c = n_question  # For compatibility

        #Embedding Modules
        if self.n_pid > 0:
            self.difficult_param = nn.Embedding(self.n_pid+1, 1)
            self.q_embed_diff = nn.Embedding(self.n_question+1, embed_l) 
            self.qa_embed_diff = nn.Embedding(2 * self.n_question + 1, embed_l)
        
        if emb_type.startswith("qid"):
            self.q_embed = nn.Embedding(self.n_question, embed_l)
            if self.separate_qa: 
                self.qa_embed = nn.Embedding(2*self.n_question+1, embed_l)
            else:
                self.qa_embed = nn.Embedding(2, embed_l)

        self.model = Architecture(
            n_question=n_question, 
            n_blocks=n_blocks, 
            n_heads=num_attn_heads, 
            dropout=dropout,
            d_model=d_model, 
            d_feature=d_model / num_attn_heads, 
            d_ff=d_ff,  
            kq_same=self.kq_same, 
            model_type=self.model_type, 
            emb_type=self.emb_type, 
            ks=ks)

        #Output MLP
        final_input_dim = d_model + embed_l + self.num_ed_features + self.num_mp_features

        # Ratio prediction output layers (use_mp=8일 때만 생성)
        if self.use_mp == 8:
            # Step 1: Ratio 0-3 예측 layers (4개)
            self.ratio_out_layers = []
            for i in range(4):  # Ratio 0-3 예측 (4개)
                ratio_out = nn.Sequential(
                    nn.Linear(final_input_dim, final_fc_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(final_fc_dim, 256),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(256, 1)
                )
                self.ratio_out_layers.append(ratio_out)
            self.ratio_out_layers = nn.ModuleList(self.ratio_out_layers)

            # Step 2: Ratio 4개를 입력으로 받아 prediction 예측하는 FC layer
            # Input: final_input_dim + 4 (ratio predictions) -> Output: 1 (prediction)
            self.out = nn.Sequential(
                nn.Linear(final_input_dim + 4, final_fc_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(final_fc_dim, 256),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(256, 1)
            )
        else:
            # Baseline: prediction만 예측
            self.out = nn.Sequential(
                nn.Linear(final_input_dim, final_fc_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(final_fc_dim, 256),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(256, 1)
            )

        for p in self.parameters():
            if p.size(0) == self.n_pid+1 and self.n_pid > 0:
                torch.nn.init.constant_(p, 0.)
        

    def forward(self, q_data, target, pid_data=None, qtest=False, dcur=None):
        # Handle ED features if use_ed is enabled
        device = q_data.device
        ed = None
        if self.num_ed_features > 0 and dcur is not None:
            ed_data = []
            for i in range(self.num_ed_features):
                ed_key = f"error_type_{i}"
                if ed_key in dcur and len(dcur[ed_key]) > 0:
                    # Shift ED features: pad with zero at start, remove last timestep
                    ed_shifted = torch.cat([torch.zeros(dcur[ed_key].shape[0], 1, device=device), 
                                          dcur[ed_key][:, :-1]], dim=1)
                    ed_data.append(ed_shifted.to(device).float())
                else:
                    ed_data.append(torch.zeros_like(q_data.float(), device=device))
            
            if len(ed_data) > 0:
                ed = torch.stack(ed_data, dim=-1)
        
        # Handle MP features if use_mp is enabled
        mp = None
        if self.use_mp > 0 and dcur is not None:
            mp_data = []
            target_seq_len = q_data.shape[1]

            # MP 0-3: 문제 요구사항 (shift 없음)
            for i in range(self.num_mp_meta_features):
                mp_key = f"math_prof_{i}"
                if mp_key in dcur and len(dcur[mp_key]) > 0:
                    dcur_mp_tensor = dcur[mp_key].to(device)
                    # Handle sequence length matching
                    if dcur_mp_tensor.shape[1] > target_seq_len:
                        mp_current = dcur_mp_tensor[:, :target_seq_len]
                    elif dcur_mp_tensor.shape[1] < target_seq_len:
                        pad_len = target_seq_len - dcur_mp_tensor.shape[1]
                        mp_current = torch.cat([dcur_mp_tensor, torch.zeros((dcur_mp_tensor.shape[0], pad_len), device=device)], dim=1)
                    else:
                        mp_current = dcur_mp_tensor
                    mp_data.append(mp_current.float())
                else:
                    mp_data.append(torch.zeros(q_data.shape[0], target_seq_len, device=device, dtype=torch.float))

            # Ratio 0-3: 충족률 (MP4-7 / MP0-3) - shift 적용
            for i in range(self.num_mp_ratio_features):
                ratio_key = f"mp_ratio_{i}"
                if ratio_key in dcur and len(dcur[ratio_key]) > 0:
                    dcur_ratio_tensor = dcur[ratio_key].to(device)
                    # Shift 적용: 이전 timestep 데이터 사용
                    ratio_shifted = torch.cat([torch.zeros(dcur_ratio_tensor.shape[0], 1, device=device),
                                              dcur_ratio_tensor[:, :-1]], dim=1)
                    # Handle sequence length matching
                    if ratio_shifted.shape[1] > target_seq_len:
                        ratio_shifted = ratio_shifted[:, :target_seq_len]
                    elif ratio_shifted.shape[1] < target_seq_len:
                        pad_len = target_seq_len - ratio_shifted.shape[1]
                        ratio_shifted = torch.cat([ratio_shifted, torch.zeros((ratio_shifted.shape[0], pad_len), device=device)], dim=1)
                    mp_data.append(ratio_shifted.float())
                else:
                    mp_data.append(torch.zeros(q_data.shape[0], target_seq_len, device=device, dtype=torch.float))

            if len(mp_data) > 0:
                mp = torch.stack(mp_data, dim=-1)  # [batch_size, seq_len, num_mp_total_features]
        
        emb_type = self.emb_type
        if emb_type.startswith("qid"):

            q_embed_data = self.q_embed(q_data)

            if self.separate_qa:
                qa_data = q_data + self.n_question * target
                qa_embed_data = self.qa_embed(qa_data)
            else:
                qa_embed_data = self.qa_embed(target)+q_embed_data

        pid_embed_data = None
        if self.n_pid > 0:
            q_embed_diff_data = self.q_embed_diff(q_data)  
            pid_embed_data = self.difficult_param(pid_data)  
            q_embed_data = q_embed_data + pid_embed_data * q_embed_diff_data 

            qa_embed_diff_data = self.qa_embed_diff(target) 
            if self.separate_qa:
                qa_embed_data = qa_embed_data + pid_embed_data * qa_embed_diff_data 
            else:
                qa_embed_data = qa_embed_data + pid_embed_data * (qa_embed_diff_data+q_embed_diff_data)
            c_reg_loss = (pid_embed_data ** 2.).sum() * self.l2 
        else:
            c_reg_loss = 0.

        d_output = self.model(q_embed_data, qa_embed_data, pid_embed_data)

        concat_q = torch.cat([d_output, q_embed_data], dim=-1)

        # Handle ED features for final prediction
        if self.num_ed_features > 0:
            if ed is not None:
                concat_q = torch.cat([concat_q, ed], dim=-1)
            else:
                # Pad with -1 if ED features expected but not provided
                batch_size, seq_len = concat_q.shape[:2]
                ed_padding = torch.full((batch_size, seq_len, self.num_ed_features), -1.0, device=concat_q.device, dtype=concat_q.dtype)
                concat_q = torch.cat([concat_q, ed_padding], dim=-1)

        # Handle MP features for final prediction
        if self.num_mp_features > 0:
            if mp is not None:
                concat_q = torch.cat([concat_q, mp], dim=-1)
            else:
                # Pad with -1 if MP features expected but not provided
                batch_size, seq_len = concat_q.shape[:2]
                mp_padding = torch.full((batch_size, seq_len, self.num_mp_features), -1.0, device=concat_q.device, dtype=concat_q.dtype)
                concat_q = torch.cat([concat_q, mp_padding], dim=-1)

        m = nn.Sigmoid()

        # New architecture: Ratio 예측 → Ratio를 입력으로 받아 prediction 예측
        if self.use_mp == 8:
            # Step 1: Ratio 0-3 predictions (4개)
            ratio_predictions = []
            for i in range(4):  # Ratio 0-3 예측
                ratio_pred = self.ratio_out_layers[i](concat_q)  # Shape: (batch, seq, 1)
                ratio_pred = torch.sigmoid(ratio_pred)  # [0, 1] range for ratio
                ratio_predictions.append(ratio_pred)

            # Step 2: Concatenate ratio predictions to concat_q
            # ratio_predictions: list of 4 tensors, each (batch, seq, 1)
            ratio_concat = torch.cat(ratio_predictions, dim=-1)  # Shape: (batch, seq, 4)

            # Step 3: Predict final output using concat_q + ratio predictions
            concat_q_with_ratio = torch.cat([concat_q, ratio_concat], dim=-1)  # Shape: (batch, seq, final_input_dim + 4)
            output = self.out(concat_q_with_ratio).squeeze(-1)  # Shape: (batch, seq)
            preds = m(output)

            # Return prediction and ratio predictions (for loss calculation)
            if not qtest:
                # Training: return prediction + c_reg_loss + ratio predictions (squeezed)
                ratio_predictions_squeezed = [rp.squeeze(-1) for rp in ratio_predictions]  # Each: (batch, seq)
                return preds, c_reg_loss, ratio_predictions_squeezed
            else:
                # Evaluation with qtest: return prediction + c_reg_loss + concat_q
                return preds, c_reg_loss, concat_q
        else:
            # Baseline: prediction만 예측
            output = self.out(concat_q).squeeze(-1)
            preds = m(output)

            if not qtest:
                return preds, c_reg_loss
            else:
                return preds, c_reg_loss, concat_q


class Architecture(nn.Module):
    def __init__(self, n_question,  n_blocks, d_model, d_feature,
                 d_ff, n_heads, dropout, kq_same, model_type, emb_type, ks):
        super().__init__()
        self.d_model = d_model
        self.model_type = model_type

        self.blocks_1 = nn.ModuleList([
            TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                                d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same, emb_type=emb_type)
            for _ in range(n_blocks)
        ])
        self.blocks_2 = nn.ModuleList([
            TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                                d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same, emb_type=emb_type)
            for _ in range(n_blocks*2)
        ])
        
        self.smooth = Smooth(dropout,d_model, ks)

    def forward(self, q_embed_data, qa_embed_data, pid_embed_data):

        qa_pos_embed = qa_embed_data
        q_pos_embed = q_embed_data

        y = qa_pos_embed
        x = q_pos_embed

        x = self.smooth(x)
        y = self.smooth(y)

        for block in self.blocks_1:
            y = block(mask=1, query=y, key=y, values=y, pdiff=pid_embed_data)
        flag_first = True
        for block in self.blocks_2:
            if flag_first: 
                x = block(mask=1, query=x, key=x,
                          values=x, apply_pos=False, pdiff=pid_embed_data)
                flag_first = False
            else:
                x = block(mask=0, query=x, key=x, values=y, apply_pos=True, pdiff=pid_embed_data)
                flag_first = True
        return x

class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_feature,
                 d_ff, n_heads, dropout,  kq_same, emb_type):
        super().__init__()
        kq_same = kq_same == 1
        self.masked_attn_head = MultiHeadAttention(
            d_model, d_feature, n_heads, dropout, kq_same=kq_same, emb_type=emb_type)

        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, mask, query, key, values, apply_pos=True, pdiff=None):
        seqlen, batch_size = query.size(1), query.size(0)
        nopeek_mask = np.triu(
            np.ones((1, 1, seqlen, seqlen)), k=mask).astype('uint8')
        src_mask = (torch.from_numpy(nopeek_mask) == 0).to(device)
        if mask == 0:  
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=True, pdiff=pdiff) 
        else:
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=False, pdiff=pdiff)

        query = query + self.dropout1((query2))
        query = self.layer_norm1(query)
        if apply_pos:
            query2 = self.linear2(self.dropout(
                self.activation(self.linear1(query))))
            query = query + self.dropout2((query2))
            query = self.layer_norm2(query)
        return query


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_feature, n_heads, dropout, kq_same, bias=True, emb_type="qid"):
        super().__init__()
        self.d_model = d_model
        self.emb_type = emb_type
        if emb_type.endswith("avgpool"):
            pool_size = 3
            self.pooling =  nn.AvgPool1d(pool_size, stride=1, padding=pool_size//2, count_include_pad=False, )
            self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        elif emb_type.endswith("linear"):
            self.linear = nn.Linear(d_model, d_model, bias=bias)
            self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        elif emb_type.startswith("qid"):
            self.d_k = d_feature
            self.h = n_heads
            self.kq_same = kq_same

            self.v_linear = nn.Linear(d_model, d_model, bias=bias)
            self.k_linear = nn.Linear(d_model, d_model, bias=bias)
            if kq_same is False:
                self.q_linear = nn.Linear(d_model, d_model, bias=bias)
            self.dropout = nn.Dropout(dropout)
            self.proj_bias = bias
            self.out_proj = nn.Linear(d_model, d_model, bias=bias)
            self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
            torch.nn.init.xavier_uniform_(self.gammas)
            self._reset_parameters()


    def _reset_parameters(self):
        xavier_uniform_(self.k_linear.weight)
        xavier_uniform_(self.v_linear.weight)
        if self.kq_same is False:
            xavier_uniform_(self.q_linear.weight)

        if self.proj_bias:
            constant_(self.k_linear.bias, 0.)
            constant_(self.v_linear.bias, 0.)
            if self.kq_same is False:
                constant_(self.q_linear.bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def forward(self, q, k, v, mask, zero_pad, pdiff=None):

        bs = q.size(0)

        if self.emb_type.endswith("avgpool"):
            scores = self.pooling(v)
            concat = self.pad_zero(scores, bs, scores.shape[2], zero_pad)
        elif self.emb_type.endswith("linear"):
            scores = self.linear(v)
            concat = self.pad_zero(scores, bs, scores.shape[2], zero_pad)
        elif self.emb_type.startswith("qid"):
            k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
            if self.kq_same is False:
                q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
            else:
                q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
            v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

            k = k.transpose(1, 2)
            q = q.transpose(1, 2)
            v = v.transpose(1, 2)
            gammas = self.gammas
            if self.emb_type.find("pdiff") == -1:
                pdiff = None
            scores = attention(q, k, v, self.d_k,
                            mask, self.dropout, zero_pad, gammas, pdiff)

            concat = scores.transpose(1, 2).contiguous()\
                .view(bs, -1, self.d_model)

        output = self.out_proj(concat)

        return output

    def pad_zero(self, scores, bs, dim, zero_pad):
        if zero_pad:
            pad_zero = torch.zeros(bs, 1, dim).to(device)
            scores = torch.cat([pad_zero, scores[:, 0:-1, :]], dim=1)
        return scores


def attention(q, k, v, d_k, mask, dropout, zero_pad, gamma=None, pdiff=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / \
        math.sqrt(d_k) 
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)

    x1 = torch.arange(seqlen).expand(seqlen, -1).to(device)
    x2 = x1.transpose(0, 1).contiguous()

    with torch.no_grad():
        scores_ = scores.masked_fill(mask == 0, -1e32)
        scores_ = F.softmax(scores_, dim=-1)
        scores_ = scores_ * mask.float().to(device)
        distcum_scores = torch.cumsum(scores_, dim=-1)
        disttotal_scores = torch.sum(
            scores_, dim=-1, keepdim=True)
        position_effect = torch.abs(
            x1-x2)[None, None, :, :].type(torch.FloatTensor).to(device)  
        dist_scores = torch.clamp(
            (disttotal_scores-distcum_scores)*position_effect, min=0.)
        dist_scores = dist_scores.sqrt().detach()
    m = nn.Softplus()
    gamma = -1. * m(gamma).unsqueeze(0)
    if pdiff == None:
        total_effect = torch.clamp(torch.clamp(
            (dist_scores*gamma).exp(), min=1e-5), max=1e5)
    else:
        diff = pdiff.unsqueeze(1).expand(pdiff.shape[0], dist_scores.shape[1], pdiff.shape[1], pdiff.shape[2])
        diff = diff.sigmoid().exp()
        total_effect = torch.clamp(torch.clamp(
            (dist_scores*gamma*diff).exp(), min=1e-5), max=1e5)
    scores = scores * total_effect

    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)
    if zero_pad:
        pad_zero = torch.zeros(bs, head, 1, seqlen).to(device)
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2)
    scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = ((x - mean) ** 2).mean(-1, keepdim=True)
        normed = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * normed + self.beta

class CausalTemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(CausalTemporalConv, self).__init__()
        self.conv = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size, 
            padding=(kernel_size - 1) * dilation, 
            dilation=dilation
            )

    def forward(self, x):
        out = self.conv(x)
        crop = self.conv.padding[0]
        return out[:, :, :-crop]

class Smooth(nn.Module):
    def __init__(self, dropout, hidden_size, kernel_size):
        super(Smooth, self).__init__()
        self.out_dropout = nn.Dropout(dropout)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.causal_conv = CausalTemporalConv(hidden_size, hidden_size, kernel_size)
        self.c = kernel_size // 2 + 1
        self.sqrt_beta = nn.Parameter(torch.randn(1, 1, hidden_size))

    def forward(self, input_tensor):

        input_tensor = input_tensor.permute(0, 2, 1)
        trend = self.causal_conv(input_tensor)
        trend = trend.permute(0, 2, 1)

        random = input_tensor.permute(0, 2, 1) - trend
        sequence_emb_fft = trend + (self.sqrt_beta**2) * random

        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = self.LayerNorm(hidden_states + input_tensor.permute(0, 2, 1))

        return hidden_states

