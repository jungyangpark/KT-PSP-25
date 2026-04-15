import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Module, Embedding, Linear, Dropout, ReLU, Sequential
from torch.nn.functional import one_hot
import torch.nn.functional as F

from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)

class DKT2(Module):
    def __init__(self, num_c, emb_size=64, dropout=0.1, num_q=None, 
                 emb_type='qid', emb_path="", pretrain_dim=768, hidden_size=0,
                 factor=1.3, num_blocks=2, num_heads=2, slstm_at=[1], 
                 conv1d_kernel_size=4, qkv_proj_blocksize=4, device='cuda', use_ed=0, use_mp=0, **kwargs):
        super().__init__()
        self.model_name = "dkt2"
        self.num_c = num_c
        self.num_q = num_q if num_q is not None else num_c
        self.emb_type = emb_type
        self.emb_size = emb_size
        self.hidden_size = emb_size
        self.dropout = dropout

        # MP features 설정 (DKT와 동일)
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
        self.device = device
        self.factor = factor
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.slstm_at = slstm_at
        self.conv1d_kernel_size = conv1d_kernel_size
        self.qkv_proj_blocksize = qkv_proj_blocksize
        
        # Rasch model embeddings - difficulty parameters
        if self.num_q > 0:
            self.difficult_param = nn.Embedding(self.num_q, 1)
            self.q_embed_diff = nn.Embedding(self.num_c, emb_size)
            self.qa_embed_diff = nn.Embedding(2 * self.num_c, emb_size)
        else:
            self.difficult_param = nn.Embedding(self.num_c, 1)
            self.q_embed_diff = nn.Embedding(self.num_c, emb_size)
            self.qa_embed_diff = nn.Embedding(2 * self.num_c, emb_size)
        
        # Base embeddings - add +1 for padding/unknown tokens
        self.q_embed = nn.Embedding(self.num_c, emb_size)
        self.qa_embed = nn.Embedding(2, emb_size)
        
        # Adjust embedding size if using MP features
        effective_emb_size = self.emb_size + self.num_mp_total_features
        
        # xLSTM or LSTM backbone
        cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=self.conv1d_kernel_size,
                    qkv_proj_blocksize=self.qkv_proj_blocksize,
                    num_heads=self.num_heads,
                    proj_factor=self.factor,
                    dropout=self.dropout,
                    embedding_dim=effective_emb_size,
                    _inner_embedding_dim=2*effective_emb_size,
                    _num_blocks=1,
                    round_proj_up_dim_up=True,
                    _proj_up_dim=None,
                )
            ),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend="cuda" if device == "cuda" else "vanilla",
                    num_heads=self.num_heads,
                    conv1d_kernel_size=self.conv1d_kernel_size,
                    bias_init="powerlaw_blockdependent",
                    recurrent_weight_init="zeros",
                    embedding_dim=effective_emb_size,
                    dropout=self.dropout,
                    group_norm_weight=True,
                    gradient_recurrent_cut=False,
                    gradient_recurrent_clipval=None,
                    forward_clipval=None,
                ),
                feedforward=FeedForwardConfig(proj_factor=self.factor, act_fn="relu"),
            ),
            context_length=512,  # Default context length
            num_blocks=self.num_blocks,
            embedding_dim=effective_emb_size,
            add_post_blocks_norm=True,
            bias=True,
            dropout=self.dropout,
            slstm_at=self.slstm_at,
        )
        self.xlstm_stack = xLSTMBlockStack(cfg)        
        self.dropout_layer = Dropout(dropout)
        self.loss_fn = nn.BCELoss(reduction="mean")
        
        # Output layer with knowledge decomposition
        # Input: q_embed (emb_size) + d_output (effective_hidden_size) + familiar (effective_hidden_size) + unfamiliar (effective_hidden_size)
        effective_hidden_size = effective_emb_size
        output_input_dim = self.emb_size + 3 * effective_hidden_size
        self.out = Sequential(
            Linear(output_input_dim, 2 * self.hidden_size),
            ReLU(),
            Dropout(self.dropout),
            Linear(2 * self.hidden_size, self.hidden_size),
            ReLU(),
            Dropout(self.dropout),
            Linear(self.hidden_size, self.num_c),
        )

        # Ratio prediction output layers (use_mp=8일 때만 생성)
        if self.use_mp == 8:
            self.ratio_out_layers = []
            for i in range(4):  # Ratio 0-3 예측 (4개)
                ratio_out = Sequential(
                    Linear(output_input_dim, 2 * self.hidden_size),
                    ReLU(),
                    Dropout(self.dropout),
                    Linear(2 * self.hidden_size, self.hidden_size),
                    ReLU(),
                    Dropout(self.dropout),
                    Linear(self.hidden_size, self.num_c),
                )
                self.ratio_out_layers.append(ratio_out)
            self.ratio_out_layers = nn.ModuleList(self.ratio_out_layers)

        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.02)
    
    def base_emb(self, q_data, target):
        """Base embeddings following Rasch model"""
        q_embed_data = self.q_embed(q_data)  # Concept embeddings
        qa_embed_data = self.qa_embed(target) + q_embed_data  # Response + concept
        return q_embed_data, qa_embed_data

    def forward(self, c, r, dcur=None, **kwargs):
        """
        Args:
            c: concept/skill IDs (batch_size, seq_len)
            r: responses (batch_size, seq_len)
            dcur: data dictionary containing ED features if use_ed=True
        Returns:
            predictions: (batch_size, seq_len, num_c) - same length as input
        """
        batch_size, seq_len = c.shape
        
        # Clamp indices to prevent CUDA errors  
        c_clamped = torch.clamp(c, 0, self.num_c - 1)
        r_clamped = torch.clamp(r, -1, 1)
        
        # Create interaction embeddings (concepts + responses)
        # For missing responses (r == -1), use a special embedding
        r_for_emb = torch.where(r_clamped == -1, 2, r_clamped)  # Map -1 to 2 (padding token)
        r_for_emb = torch.clamp(r_for_emb, 0, 2)
        
        # Base embeddings with Rasch model
        q_embed_data = self.q_embed(c_clamped)  # Concept embeddings
        qa_embed_data = self.qa_embed(r_for_emb) + q_embed_data  # Response + concept
        
        # Add difficulty-based variations
        if hasattr(self, 'difficult_param'):
            pid_data = torch.clamp(c_clamped, 0, (self.num_q if self.num_q > 0 else self.num_c))
            q_embed_diff_data = self.q_embed_diff(c_clamped) 
            pid_embed_data = self.difficult_param(pid_data)
            q_embed_data = q_embed_data + pid_embed_data * q_embed_diff_data
            
            # For qa_embed_diff, combine response and concept info
            valid_r = torch.clamp(r_for_emb, 0, 1)  # Ensure 0 or 1
            target_for_diff = valid_r + c_clamped * 2
            target_for_diff = torch.clamp(target_for_diff, 0, 2 * self.num_c)
            qa_embed_diff_data = self.qa_embed_diff(target_for_diff)
            qa_embed_data = qa_embed_data + pid_embed_data * qa_embed_diff_data
        
        # Handle MP features (DKT와 동일한 방식)
        if self.use_mp > 0:
            if dcur is not None:
                mp_data = []
                device = qa_embed_data.device
                target_seq_len = qa_embed_data.shape[1]

                # MP 0-3: 문제 요구사항 (shift 없음)
                for i in range(self.num_mp_meta_features):
                    mp_key = f"math_prof_{i}"
                    if mp_key in dcur and len(dcur[mp_key]) > 0:
                        dcur_mp_tensor = dcur[mp_key].to(device)
                        if dcur_mp_tensor.shape[1] > target_seq_len:
                            mp_current = dcur_mp_tensor[:, :target_seq_len]
                        elif dcur_mp_tensor.shape[1] < target_seq_len:
                            pad_len = target_seq_len - dcur_mp_tensor.shape[1]
                            mp_current = torch.cat([dcur_mp_tensor, torch.zeros((dcur_mp_tensor.shape[0], pad_len), device=device)], dim=1)
                        else:
                            mp_current = dcur_mp_tensor
                        mp_data.append(mp_current.float())
                    else:
                        mp_data.append(torch.zeros(c.shape[0], target_seq_len, device=device, dtype=torch.float))

                # Ratio 0-3: 충족률 (MP4-7 / MP0-3) - shift 적용
                for i in range(self.num_mp_ratio_features):
                    ratio_key = f"mp_ratio_{i}"
                    if ratio_key in dcur and len(dcur[ratio_key]) > 0:
                        dcur_ratio_tensor = dcur[ratio_key].to(device)
                        # Shift 적용: 이전 timestep 데이터 사용
                        ratio_shifted = torch.cat([torch.zeros(dcur_ratio_tensor.shape[0], 1, device=device),
                                                  dcur_ratio_tensor[:, :-1]], dim=1)
                        if ratio_shifted.shape[1] > target_seq_len:
                            ratio_shifted = ratio_shifted[:, :target_seq_len]
                        elif ratio_shifted.shape[1] < target_seq_len:
                            pad_len = target_seq_len - ratio_shifted.shape[1]
                            ratio_shifted = torch.cat([ratio_shifted, torch.zeros((ratio_shifted.shape[0], pad_len), device=device)], dim=1)
                        mp_data.append(ratio_shifted.float())
                    else:
                        mp_data.append(torch.zeros(c.shape[0], target_seq_len, device=device, dtype=torch.float))

                if len(mp_data) > 0:
                    mp = torch.stack(mp_data, dim=-1)
                    qa_embed_data = torch.cat([qa_embed_data, mp], dim=-1)
            else:
                # Pad with zeros if dcur not provided
                batch_size, seq_len = qa_embed_data.shape[:2]
                mp_padding = torch.zeros((batch_size, seq_len, self.num_mp_total_features), device=qa_embed_data.device, dtype=qa_embed_data.dtype)
                qa_embed_data = torch.cat([qa_embed_data, mp_padding], dim=-1)

        # Dropout
        qa_embed_data = self.dropout_layer(qa_embed_data)
        
        # Process through backbone (xLSTM or LSTM)
        d_output = self.xlstm_stack(qa_embed_data)

        # Handle NaN/Inf from xLSTM stack
        d_output = torch.nan_to_num(d_output, nan=0.0, posinf=1.0, neginf=-1.0)

        # Knowledge decomposition into familiar and unfamiliar
        familiar_ability = torch.zeros_like(d_output)
        unfamiliar_ability = torch.zeros_like(d_output)
        familiar_position = r_for_emb == 1
        unfamiliar_position = r_for_emb == 0
        familiar_ability[familiar_position] = d_output[familiar_position]
        unfamiliar_ability[unfamiliar_position] = d_output[unfamiliar_position]

        # IRT-based adjustment: remove question difficulty
        if hasattr(self, 'difficult_param'):
            d_output = d_output - pid_embed_data

        # Concatenate all knowledge representations
        concat_q = torch.cat([d_output, q_embed_data, familiar_ability, unfamiliar_ability], dim=-1)
        output = self.out(concat_q)

        output = torch.sigmoid(output)

        # Ratio predictions (use_mp=8일 때 multi-task)
        if self.use_mp == 8:
            # Multi-task: Ratio 0-3 predictions (4개)
            ratio_predictions = []
            for i in range(4):  # Ratio 0-3 예측
                ratio_pred = self.ratio_out_layers[i](concat_q)
                ratio_pred = torch.sigmoid(ratio_pred)  # [0, 1] range for ratio
                ratio_predictions.append(ratio_pred)
            return output, ratio_predictions
        else:
            # Single task: 정오답 예측만
            return output
    
    def loss(self, pred, true):
        """Compute loss"""
        pred = pred.flatten()
        true = true.flatten()
        mask = true > -1  # Valid responses only
        loss = self.loss_fn(pred[mask], true[mask])
        return loss, len(pred[mask]), true[mask].sum().item()