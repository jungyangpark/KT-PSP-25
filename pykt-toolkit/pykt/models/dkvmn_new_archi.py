import os

import numpy as np
import torch

from torch.nn import Module, Parameter, Embedding, Linear, Dropout
from torch.nn.init import kaiming_normal_

class DKVMN(Module):
    def __init__(self, num_c, dim_s, size_m, dropout=0.2, emb_type='qid', emb_path="", pretrain_dim=768, use_ed=0, use_mp=0):
        super().__init__()
        self.model_name = "dkvmn"
        self.num_c = num_c
        self.dim_s = dim_s
        self.size_m = size_m
        self.emb_type = emb_type
        self.num_ed_features = use_ed

        # MP features 설정
        self.use_mp = use_mp
        if use_mp == 8:
            # New mode: MP 0-3 (4개) + Ratio 0-3 (4개) = 8 features
            self.num_mp_meta_features = 4  # MP 0-3: 문제 요구사항
            self.num_mp_ratio_features = 4  # Ratio 0-3: 충족률 (MP4-7 / MP0-3)
            self.num_mp_total_features = 8  # MP 0-3 + Ratio 0-3
            self.num_mp_features = 8
        elif use_mp == 0:
            self.num_mp_meta_features = 0
            self.num_mp_ratio_features = 0
            self.num_mp_total_features = 0
            self.num_mp_features = 0
        else:
            print(f"Warning: use_mp={use_mp} is not standard. Recommended: 0 (no MP) or 8 (MP 0-3 + Ratio 0-3).")
            self.num_mp_meta_features = use_mp
            self.num_mp_ratio_features = 0
            self.num_mp_total_features = use_mp
            self.num_mp_features = use_mp

        if emb_type.startswith("qid"):
            self.k_emb_layer = Embedding(self.num_c, self.dim_s)
            # Adjust memory key size if using ED and MP features
            memory_key_dim = self.dim_s + self.num_ed_features + self.num_mp_features
            self.Mk = Parameter(torch.Tensor(self.size_m, memory_key_dim))
            self.Mv0 = Parameter(torch.Tensor(self.size_m, self.dim_s))

        kaiming_normal_(self.Mk)
        kaiming_normal_(self.Mv0)

        # Adjust embedding size if using ED and MP features
        v_emb_input_size = self.dim_s + self.num_ed_features + self.num_mp_features
        self.v_emb_layer = Embedding(self.num_c * 2, self.dim_s)

        # Adjust layer sizes if using ED and MP features
        f_input_size = self.dim_s * 2 + self.num_ed_features + self.num_mp_features
        self.f_layer = Linear(f_input_size, self.dim_s)
        self.dropout_layer = Dropout(dropout)

        e_input_size = self.dim_s + self.num_ed_features + self.num_mp_features
        a_input_size = self.dim_s + self.num_ed_features + self.num_mp_features
        self.e_layer = Linear(e_input_size, self.dim_s)
        self.a_layer = Linear(a_input_size, self.dim_s)

        # Ratio prediction output layers (use_mp=8일 때만 생성)
        if self.use_mp == 8:
            # Step 1: Ratio 0-3 예측 layers (4개)
            self.ratio_out_layers = []
            for i in range(4):  # Ratio 0-3 예측 (4개)
                # Output one ratio value (scalar) from hidden state
                ratio_out = Linear(self.dim_s, 1)
                self.ratio_out_layers.append(ratio_out)
            self.ratio_out_layers = torch.nn.ModuleList(self.ratio_out_layers)

            # Step 2: Ratio 4개를 입력으로 받아 prediction 예측하는 FC layer
            # Input: dim_s + 4 (ratio predictions) -> Output: 1 (prediction)
            self.p_layer = Linear(self.dim_s + 4, 1)
        else:
            # Baseline: prediction만 예측
            self.p_layer = Linear(self.dim_s, 1)

    def forward(self, q, r, qtest=False, dcur=None):
        # Handle ED features if use_ed is enabled
        ed = None
        if self.num_ed_features > 0 and dcur is not None:
            ed_data = []
            device = q.device
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
        
        # Handle MP features if use_mp is enabled
        mp = None
        if self.num_mp_features > 0 and dcur is not None:
            mp_data = []
            device = q.device
            target_seq_len = q.shape[1]  # Use q sequence length as target

            # MP 0-3: 문제 요구사항 (shift 없음 - 현재 timestep 사용)
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
                    mp_data.append(torch.zeros(q.shape[0], target_seq_len, device=device, dtype=torch.float))

            # Ratio 0-3: 충족률 (MP4-7 / MP0-3) - shift 적용 (이전 timestep 사용)
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
                    mp_data.append(torch.zeros(q.shape[0], target_seq_len, device=device, dtype=torch.float))

            if len(mp_data) > 0:
                mp = torch.stack(mp_data, dim=-1)
        
        emb_type = self.emb_type
        batch_size = q.shape[0]
        if emb_type == "qid":
            x = q + self.num_c * r
            k_original = self.k_emb_layer(q)  # Keep original key embeddings separate
            v = self.v_emb_layer(x)
            
            # Create key embeddings for attention (with ED and MP features if available)
            concat_features = []
            if self.num_ed_features > 0:
                if ed is not None:
                    concat_features.append(ed)
                else:
                    # Pad with -1 if ED features not provided but model expects them (-1 indicates no data)
                    batch_size, seq_len = k_original.shape[:2]
                    ed_padding = torch.full((batch_size, seq_len, self.num_ed_features), -1.0, device=k_original.device, dtype=k_original.dtype)
                    concat_features.append(ed_padding)
            
            if self.num_mp_features > 0:
                if mp is not None:
                    concat_features.append(mp)
                else:
                    # Pad with -1 if MP features not provided but model expects them (-1 indicates no data)
                    batch_size, seq_len = k_original.shape[:2]
                    mp_padding = torch.full((batch_size, seq_len, self.num_mp_features), -1.0, device=k_original.device, dtype=k_original.dtype)
                    concat_features.append(mp_padding)
            
            if concat_features:
                combined_features = torch.cat(concat_features, dim=-1)
                k = torch.cat([k_original, combined_features], dim=-1)
                v = torch.cat([v, combined_features], dim=-1)
            else:
                k = k_original
        
        Mvt = self.Mv0.unsqueeze(0).repeat(batch_size, 1, 1)

        Mv = [Mvt]

        w = torch.softmax(torch.matmul(k, self.Mk.T), dim=-1)

        # Write Process
        e = torch.sigmoid(self.e_layer(v))
        a = torch.tanh(self.a_layer(v))

        for et, at, wt in zip(
            e.permute(1, 0, 2), a.permute(1, 0, 2), w.permute(1, 0, 2)
        ):
            Mvt = Mvt * (1 - (wt.unsqueeze(-1) * et.unsqueeze(1))) + \
                (wt.unsqueeze(-1) * at.unsqueeze(1))
            Mv.append(Mvt)

        Mv = torch.stack(Mv, dim=1)

        # Read Process
        read_content = (w.unsqueeze(-1) * Mv[:, :-1]).sum(-2)
        concat_input = [read_content, k_original]  # Use original key embeddings
        
        # Add ED features to final layer if available
        if self.num_ed_features > 0:
            if ed is not None:
                concat_input.append(ed)
            else:
                # Pad with -1 if ED features not provided but model expects them (-1 indicates no data)
                batch_size, seq_len = read_content.shape[:2]
                ed_padding = torch.full((batch_size, seq_len, self.num_ed_features), -1.0, device=read_content.device, dtype=read_content.dtype)
                concat_input.append(ed_padding)
        
        # Add MP features to final layer if available
        if self.num_mp_features > 0:
            if mp is not None:
                concat_input.append(mp)
            else:
                # Pad with -1 if MP features not provided but model expects them (-1 indicates no data)
                batch_size, seq_len = read_content.shape[:2]
                mp_padding = torch.full((batch_size, seq_len, self.num_mp_features), -1.0, device=read_content.device, dtype=read_content.dtype)
                concat_input.append(mp_padding)
            
        f = torch.tanh(
            self.f_layer(
                torch.cat(concat_input, dim=-1)
            )
        )

        # New architecture: Ratio 예측 → Ratio를 입력으로 받아 prediction 예측
        if self.use_mp == 8:
            # Step 1: Ratio 0-3 predictions (4개)
            ratio_predictions = []
            for i in range(4):  # Ratio 0-3 예측
                ratio_pred = self.ratio_out_layers[i](f)  # Shape: (batch, seq, 1)
                ratio_pred = torch.sigmoid(ratio_pred)  # [0, 1] range for ratio
                ratio_predictions.append(ratio_pred)

            # Step 2: Concatenate ratio predictions to f
            # ratio_predictions: list of 4 tensors, each (batch, seq, 1)
            ratio_concat = torch.cat(ratio_predictions, dim=-1)  # Shape: (batch, seq, 4)

            # Step 3: Predict final output using f + ratio predictions
            f_with_ratio = torch.cat([self.dropout_layer(f), ratio_concat], dim=-1)  # Shape: (batch, seq, dim_s + 4)
            p = self.p_layer(f_with_ratio)  # Shape: (batch, seq, 1)
            p = torch.sigmoid(p)
            p = p.squeeze(-1)  # Shape: (batch, seq)

            # Return prediction and ratio predictions (for loss calculation)
            if not qtest:
                # Training: return prediction + ratio predictions (squeezed)
                ratio_predictions_squeezed = [rp.squeeze(-1) for rp in ratio_predictions]  # Each: (batch, seq)
                return p, ratio_predictions_squeezed
            else:
                # Evaluation with qtest: return prediction + f
                return p, f
        else:
            # Baseline: prediction만 예측
            p = self.p_layer(self.dropout_layer(f))
            p = torch.sigmoid(p)
            p = p.squeeze(-1)

            if not qtest:
                return p
            else:
                return p, f