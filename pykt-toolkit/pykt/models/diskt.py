import torch
from torch import nn
from torch.nn.init import xavier_uniform_, constant_
import math
import torch.nn.functional as F
from enum import IntEnum
import numpy as np
from torch.nn import Module, Embedding, LSTM, Linear, Dropout, LayerNorm, TransformerEncoder, TransformerEncoderLayer, \
        MultiLabelMarginLoss, MultiLabelSoftMarginLoss, CrossEntropyLoss, BCELoss, MultiheadAttention
from torch.nn.functional import one_hot, cross_entropy, multilabel_margin_loss, binary_cross_entropy


class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2


class DisKT(nn.Module):
    def __init__(self, num_c, emb_size=128, dropout=0.1, num_blocks=2, kq_same=0, 
                d_ff=256, final_fc_dim=512, final_fc_dim2=256, num_attn_heads=8, separate_qr=False, l2=1e-5,
                emb_type='qid', emb_path="", pretrain_dim=768, num_q=0, seq_len=200,
                hidden_size=None, use_ed=0, use_mp=0, **kwargs):
        super().__init__()
        self.model_name = "diskt"
        self.num_skills = num_c
        self.num_c = num_c  # For compatibility
        self.dropout = dropout
        self.kq_same = kq_same
        self.num_questions = num_q
        self.l2 = l2
        self.separate_qr = separate_qr
        self.emb_type = emb_type
        self.num_ed_features = use_ed
        self.num_mp_features = use_mp
        
        # Handle backward compatibility with various parameter names
        if hidden_size is not None:
            embed_l = hidden_size
        elif 'd_model' in kwargs:
            embed_l = kwargs['d_model']
        else:
            embed_l = emb_size
            
        # Handle num_blocks compatibility
        if 'n_blocks' in kwargs:
            num_blocks = kwargs['n_blocks']
        elif 'num_layers' in kwargs:
            num_blocks = kwargs['num_layers']
            
        # Handle attention heads compatibility  
        if 'nheads' in kwargs:
            num_attn_heads = kwargs['nheads']
            
        # Adjust embedding size if using ED and MP features, ensuring divisibility by num_attn_heads
        self.num_attn_heads = num_attn_heads
        total_features = self.num_ed_features + self.num_mp_features
        if total_features > 0:
            # Round up total features to next multiple of attention heads for better performance
            feature_padding = ((total_features + num_attn_heads - 1) // num_attn_heads) * num_attn_heads
            effective_embed_l = embed_l + feature_padding
        else:
            effective_embed_l = embed_l
        if self.num_questions > 0:
            self.difficult_param = nn.Embedding(self.num_questions+1, embed_l)
            self.q_embed_diff = nn.Embedding(self.num_skills+1, embed_l)
            self.qa_embed_diff = nn.Embedding(2 * self.num_skills + 1, embed_l)
            
        self.q_embed = nn.Embedding(self.num_skills, embed_l)
        if self.separate_qr: 
            self.qa_embed = nn.Embedding(2*self.num_skills+1, embed_l)
        else:
            self.qa_embed = nn.Embedding(3, embed_l)
            
        # Architecture Object. It contains stack of attention block
        self.model = Architecture(num_skills=num_c, num_blocks=num_blocks, n_heads=num_attn_heads, dropout=dropout,
                                    embedding_size=effective_embed_l, d_feature=effective_embed_l // num_attn_heads, d_ff=d_ff,  kq_same=self.kq_same, seq_len=seq_len)

        self.ffn = FeedForward(d_model=effective_embed_l, inner_size=effective_embed_l*2, dropout=dropout)
        self.dual_attention = DualAttention(n_heads=num_attn_heads, d_model=effective_embed_l)
        # Adjust output layer input size: effective_embed_l (x) + effective_embed_l (q_embed_data) + effective_embed_l (y1-y2)
        out_input_dim = effective_embed_l + effective_embed_l + effective_embed_l
        self.out = nn.Sequential(
            nn.Linear(out_input_dim, final_fc_dim), 
            nn.ReLU(), 
            nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, final_fc_dim2), 
            nn.ReLU(), 
            nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim2, num_c)
        )

        self.loss_fn = nn.BCELoss(reduction="mean")
        self.reset()

    def _create_counter_mask(self, responses, skills):
        """Create counter attention mask to identify contradictory behaviors"""
        batch_size, seq_len = responses.shape
        counter_mask = torch.zeros_like(responses)
        
        # Vectorized approach to identify potential guessing/mistake patterns
        valid_mask = (responses > -1).float()
        
        # Simple heuristic: identify positions where response pattern changes
        # This approximates the original counter_kt_seqs logic
        for i in range(1, seq_len):
            if i >= 2:
                # Look at recent performance trend
                recent_responses = responses[:, max(0, i-3):i]
                recent_valid = valid_mask[:, max(0, i-3):i]
                
                # Calculate recent success rate
                recent_sum = (recent_responses * recent_valid).sum(dim=1, keepdim=True)
                recent_count = recent_valid.sum(dim=1, keepdim=True) + 1e-8
                recent_rate = recent_sum / recent_count
                
                # Current response
                curr_response = responses[:, i:i+1].float()
                curr_valid = valid_mask[:, i:i+1]
                
                # If current response contradicts recent trend, mark it
                contradiction = torch.abs(curr_response - recent_rate) > 0.5
                counter_mask[:, i:i+1] = (contradiction * curr_valid).long()
        
        return counter_mask

    def reset(self):
        for p in self.parameters():
            if p.size(0) == self.num_questions+1 and self.num_questions > 0:
                torch.nn.init.constant_(p, 0.)

    def base_emb(self, q_data, target):
        q_embed_data = self.q_embed(q_data)
        if self.separate_qr:
            qa_data = q_data + self.num_skills * target
            qa_embed_data = self.qa_embed(qa_data)
        else:
            qa_embed_data = self.qa_embed(target) + q_embed_data
        return q_embed_data, qa_embed_data

    def rasch_emb(self, q_data, pid_data, target, ed=None, mp=None):
        q_embed_data, qa_embed_data = self.base_emb(q_data, target)
        
        # If num_ed_features or num_mp_features > 0, concatenate features (or zeros if not provided)
        total_features = self.num_ed_features + self.num_mp_features
        if total_features > 0:
            batch_size, seq_len = q_embed_data.shape[:2]
            feature_padding = ((total_features + self.num_attn_heads - 1) // self.num_attn_heads) * self.num_attn_heads
            feature_tensor = torch.full((batch_size, seq_len, feature_padding), -1.0, device=q_embed_data.device, dtype=q_embed_data.dtype)
            
            # Add ED features if available
            if self.num_ed_features > 0 and ed is not None:
                feature_tensor[:, :, :self.num_ed_features] = ed
            
            # Add MP features if available
            if self.num_mp_features > 0 and mp is not None:
                feature_tensor[:, :, self.num_ed_features:self.num_ed_features+self.num_mp_features] = mp
            
            q_embed_data = torch.cat([q_embed_data, feature_tensor], dim=-1)
            qa_embed_data = torch.cat([qa_embed_data, feature_tensor], dim=-1)
        
        if self.num_questions > 0:
            q_embed_diff_data = self.q_embed_diff(q_data)
            pid_embed_data = self.difficult_param(pid_data)
            
            # If total_features > 0, pad diff embeddings and pid_embed_data with zeros to match dimension
            if total_features > 0:
                batch_size, seq_len = q_embed_diff_data.shape[:2]
                feature_padding = ((total_features + self.num_attn_heads - 1) // self.num_attn_heads) * self.num_attn_heads
                feature_padding_tensor = torch.zeros((batch_size, seq_len, feature_padding), device=q_embed_diff_data.device, dtype=q_embed_diff_data.dtype)
                q_embed_diff_data = torch.cat([q_embed_diff_data, feature_padding_tensor], dim=-1)
                
                qa_embed_diff_data = self.qa_embed_diff(target)
                feature_padding_tensor = torch.zeros((batch_size, seq_len, feature_padding), device=qa_embed_diff_data.device, dtype=qa_embed_diff_data.dtype)
                qa_embed_diff_data = torch.cat([qa_embed_diff_data, feature_padding_tensor], dim=-1)
                
                # Also pad pid_embed_data to match the effective embedding size
                feature_padding_tensor = torch.zeros((batch_size, seq_len, feature_padding), device=pid_embed_data.device, dtype=pid_embed_data.dtype)
                pid_embed_data = torch.cat([pid_embed_data, feature_padding_tensor], dim=-1)
            else:
                qa_embed_diff_data = self.qa_embed_diff(target)
            
            q_embed_data = q_embed_data + pid_embed_data * q_embed_diff_data
            qa_embed_data = qa_embed_data + pid_embed_data * qa_embed_diff_data
        else:
            pid_embed_data = q_embed_data
        return q_embed_data, qa_embed_data, pid_embed_data

    def forward(self, q, r, qshft=None, rshft=None, m=None, sm=None, at=None, pid_data=None, dcur=None):
        # Handle different input formats for compatibility
        if isinstance(q, dict):
            feed_dict = q
            q_seq = feed_dict.get('questions', feed_dict.get('qseqs'))
            s_seq = feed_dict.get('skills', feed_dict.get('cseqs', q_seq))
            r_seq = feed_dict.get('responses', feed_dict.get('rseqs'))
            pid_seq = feed_dict.get('pid_data', torch.zeros_like(q_seq))
            ed_seq = feed_dict.get('ed', None)
            if 'attention_mask' in feed_dict:
                counter_attention_mask, attention_mask = feed_dict['attention_mask']
            else:
                # Create default masks
                batch_size, seq_len = q_seq.shape
                attention_mask = torch.ones_like(r_seq)
                counter_attention_mask = torch.zeros_like(r_seq)
        else:
            # Handle tuple input format (q, r)
            q_seq = q
            s_seq = q  # skills same as questions
            r_seq = r
            pid_seq = pid_data if pid_data is not None else torch.zeros_like(q)
            ed_seq = None
            # Create default masks
            batch_size, seq_len = q.shape
            attention_mask = torch.ones_like(r)
            # Create counter attention mask to identify potential contradictory behaviors
            counter_attention_mask = self._create_counter_mask(r, q)
        
        # Handle ED features if num_ed_features > 0 and dcur is provided
        if self.num_ed_features > 0 and dcur is not None and ed_seq is None:
            ed_data = []
            device = q_seq.device
            for i in range(self.num_ed_features):
                error_col = f"error_type_{i}"
                if error_col in dcur and len(dcur[error_col]) > 0:
                    # Ensure ED data is on the correct device first
                    ed_col_data = dcur[error_col].to(device)
                    ed_shifted = torch.cat([torch.zeros(ed_col_data.shape[0], 1, device=device), 
                                          ed_col_data[:, :-1]], dim=1)
                    ed_data.append(ed_shifted.float())
                else:
                    ed_data.append(torch.zeros_like(q_seq.float(), device=device))
            
            # Stack ED features: shape [batch, seq_len, num_ed_features]
            if len(ed_data) > 0:
                ed_seq = torch.stack(ed_data, dim=-1)
        
        # Handle MP features if num_mp_features > 0 and dcur is provided
        mp_seq = None
        if self.num_mp_features > 0 and dcur is not None:
            mp_data = []
            device = q_seq.device
            for i in range(self.num_mp_features):
                mp_col = f"math_prof_{i}"
                if mp_col in dcur and len(dcur[mp_col]) > 0:
                    # Ensure MP data is on the correct device first
                    mp_col_data = dcur[mp_col].to(device)
                    mp_shifted = torch.cat([torch.zeros(mp_col_data.shape[0], 1, device=device), 
                                          mp_col_data[:, :-1]], dim=1)
                    mp_data.append(mp_shifted.float())
                else:
                    mp_data.append(torch.zeros_like(q_seq.float(), device=device))
            
            # Stack MP features: shape [batch, seq_len, num_mp_features]
            if len(mp_data) > 0:
                mp_seq = torch.stack(mp_data, dim=-1)
        
        masked_r = r_seq * (r_seq > -1).long()

        pos_q_embed_data, pos_qa_embed_data, _ = self.rasch_emb(masked_r * s_seq, masked_r * pid_seq, 2 - masked_r, ed_seq, mp_seq)
        neg_q_embed_data, neg_qa_embed_data, _ = self.rasch_emb((1-masked_r) * s_seq, (1-masked_r) * pid_seq, 2 * masked_r, ed_seq, mp_seq)
        q_embed_data, qa_embed_data, pid_embed_data = self.rasch_emb(s_seq, pid_seq, masked_r, ed_seq, mp_seq)

        y1, y2, y = pos_qa_embed_data, neg_qa_embed_data, qa_embed_data
        x = q_embed_data

        distance = F.pairwise_distance(y1.view(y1.size(0), -1), y2.view(y2.size(0), -1))
        reg_loss = torch.mean(distance) * 0.001

        x = self.model(x, y)
        
        y1, y2 = self.ffn(y1), self.ffn(y2)

        y1, y2 = self.dual_attention(x, x, y1, y2, counter_attention_mask)

        x = x - (y1 + y2)
        x = x - pid_embed_data
        x = torch.cat([x, q_embed_data], dim=-1)
        x = torch.cat([x, y1 - y2], dim=-1)
        output = self.out(x)  # Shape: (batch, seq_len, num_c)
        preds = torch.sigmoid(output)
        
        # Return predictions for each time step
        y = preds[:, 1:].contiguous()  # Remove first prediction, shape: (batch, seq_len-1, num_c)
        
        self._reg_loss = reg_loss

        return y

    def loss(self, feed_dict, out_dict):
        pred = out_dict["pred"]
        true = out_dict["true"]
        mask = true > -1
        mask_cf, mask = feed_dict['attention_mask']
        mask_cf, mask = mask_cf[:, 1:].bool(), mask[:, 1:].bool()
        pred_flat = torch.masked_select(pred, mask)
        true_flat = torch.masked_select(true, mask)
        loss = self.loss_fn(pred_flat, true_flat)
        
        reg_loss = out_dict["reg_loss"]
        loss = loss + reg_loss
        return loss, len(pred[mask]), true[mask].sum().item()


class DualAttention(nn.Module):
    def __init__(self, n_heads, d_model, dropout=0.1):
        super(DualAttention, self).__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_model = d_model
        self.n_feature = d_model // n_heads
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, q, k, v1, v2, counter_attention_mask):
        batch_size = q.size(0)
        src_mask = create_mask(q, 0)
        q = q.view(batch_size, -1, self.n_heads, self.n_feature)
        k = k.view(batch_size, -1, self.n_heads, self.n_feature)
        v1 = v1.view(batch_size, -1, self.n_heads, self.n_feature)
        v2 = v2.view(batch_size, -1, self.n_heads, self.n_feature)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v1 = v1.transpose(1, 2)
        v2 = v2.transpose(1, 2)
        output_v1, output_v2, attn_weight = contradictory_attention(q, k, v1, v2, src_mask, self.dropout, counter_attention_mask)
        output_v1 = output_v1.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output_v2 = output_v2.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return output_v1, output_v2


def contradictory_attention(query, key, value1, value2, mask=None, dropout=None, counter_attention_mask=None):
    bs, head, seqlen, d_k = query.size(0), query.size(1), query.size(2), query.size(-1)
    device = query.device
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e32)
    
    p_attn = F.softmax(scores, dim=-1)
    
    expanded_mask = counter_attention_mask.unsqueeze(1).unsqueeze(1)
    expanded_mask = expanded_mask.expand(-1, head, seqlen, -1)
    
    LOG_MIN = -1e32
    masked_attn = torch.where(expanded_mask == 1, 
                             torch.ones_like(p_attn) * LOG_MIN, 
                             p_attn + 1e-10)
    
    p_attn = F.softmax(masked_attn, dim=-1)
    
    pad_zero = torch.zeros(bs, head, 1, seqlen).to(device)
    p_attn = torch.cat([pad_zero, p_attn[:, :, 1:, :]], dim=2)
    
    if dropout is not None:
        p_attn = dropout(p_attn)
    
    output_v1 = torch.matmul(p_attn, value1)
    output_v2 = torch.matmul(p_attn, value2)
    return output_v1, output_v2, p_attn


def create_mask(input, mask):
    seqlen = input.size(1)
    device = input.device
    nopeek_mask = np.triu(np.ones((1, 1, seqlen, seqlen)), k=mask).astype('uint8')
    src_mask = (torch.from_numpy(nopeek_mask) == 0).to(device)
    return src_mask


class Architecture(nn.Module):
    def __init__(self, num_skills, num_blocks, embedding_size, d_feature,
                 d_ff, n_heads, dropout, kq_same, seq_len):
        super().__init__()
        self.embedding_size = embedding_size

        self.blocks_2 = nn.ModuleList([
            TransformerLayer(embedding_size=embedding_size, d_feature=embedding_size // n_heads,
                                d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same)
            for _ in range(num_blocks)
        ])
        self.position_emb = CosinePositionalEmbedding(embedding_size=self.embedding_size, max_len=seq_len)

    def forward(self, q_embed_data, qa_embed_data):
        q_posemb = self.position_emb(q_embed_data)
        q_embed_data = q_embed_data + q_posemb
        qa_posemb = self.position_emb(qa_embed_data)
        qa_embed_data = qa_embed_data + qa_posemb

        qa_pos_embed = qa_embed_data
        q_pos_embed = q_embed_data

        y = qa_pos_embed
        x = q_pos_embed

        for block in self.blocks_2:
            x = block(mask=0, query=x, key=x, values=y, apply_pos=True)
        return x


class TransformerLayer(nn.Module):
    def __init__(self, embedding_size, d_feature, d_ff, n_heads, dropout, kq_same):
        super().__init__()
        kq_same = kq_same == 1
        self.masked_attn_head = MultiHeadAttention(
            embedding_size, d_feature, n_heads, dropout, kq_same=kq_same)

        self.layer_norm1 = nn.LayerNorm(embedding_size)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(embedding_size, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, embedding_size)

        self.layer_norm2 = nn.LayerNorm(embedding_size)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, mask, query, key, values, apply_pos=True):
        seqlen, batch_size = query.size(1), query.size(0)
        device = query.device
        nopeek_mask = np.triu(np.ones((1, 1, seqlen, seqlen)), k=mask).astype('uint8')
        src_mask = (torch.from_numpy(nopeek_mask) == 0).to(device)
        
        if mask == 0:
            query2 = self.masked_attn_head(query, key, values, mask=src_mask, zero_pad=True)
        else:
            query2 = self.masked_attn_head(query, key, values, mask=src_mask, zero_pad=False)

        query = query + self.dropout1((query2))
        query = self.layer_norm1(query)
        
        if apply_pos:
            query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
            query = query + self.dropout2((query2))
            query = self.layer_norm2(query)
        return query


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_size, d_feature, n_heads, dropout, kq_same, bias=True):
        super().__init__()
        self.embedding_size = embedding_size
        self.d_k = d_feature
        self.h = n_heads
        self.kq_same = kq_same

        self.v_linear = nn.Linear(embedding_size, embedding_size, bias=bias)
        self.k_linear = nn.Linear(embedding_size, embedding_size, bias=bias)
        if kq_same is False:
            self.q_linear = nn.Linear(embedding_size, embedding_size, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.proj_bias = bias
        self.out_proj = nn.Linear(embedding_size, embedding_size, bias=bias)

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

    def forward(self, q, k, v, mask, zero_pad):
        bs = q.size(0)

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        if self.kq_same is False:
            q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        else:
            q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        
        scores = attention(q, k, v, self.d_k, mask, self.dropout, zero_pad)

        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.embedding_size)
        output = self.out_proj(concat)
        return output


def attention(q, k, v, d_k, mask, dropout, zero_pad):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)
    device = q.device

    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)
    
    if zero_pad:
        pad_zero = torch.zeros(bs, head, 1, seqlen).to(device)
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2)
        
    scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output


class FeedForward(nn.Module):
    def __init__(self, d_model, inner_size, dropout=0.2):
        super().__init__()
        self.w_1 = nn.Linear(d_model, inner_size)
        self.w_2 = nn.Linear(inner_size, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)

    def forward(self, input_tensor):
        hidden_states = self.w_1(input_tensor)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = self.w_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, embedding_size, max_len=512):
        super().__init__()
        pe = 0.1 * torch.randn(max_len, embedding_size)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=True)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]


class CosinePositionalEmbedding(nn.Module):
    def __init__(self, embedding_size, max_len=512):
        super().__init__()
        pe = 0.1 * torch.randn(max_len, embedding_size)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embedding_size, 2).float() *
                             -(math.log(10000.0) / embedding_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]


class Similarity(nn.Module):
    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp