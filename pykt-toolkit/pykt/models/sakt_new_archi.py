import torch

from torch.nn import Module, Embedding, Linear, MultiheadAttention, LayerNorm, Dropout
from .utils import transformer_FFN, pos_encode, ut_mask, get_clones

class SAKT(Module):
    def __init__(self, num_c, seq_len, emb_size, num_attn_heads, dropout, num_en=2, emb_type="qid", emb_path="", pretrain_dim=768, use_ed=0, use_mp=0):
        super().__init__()
        self.model_name = "sakt"
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

        self.num_c = num_c
        self.seq_len = seq_len
        self.emb_size = emb_size
        self.num_attn_heads = num_attn_heads
        self.dropout = dropout
        self.num_en = num_en

        if emb_type.startswith("qid"):
            # num_c, seq_len, emb_size, num_attn_heads, dropout, emb_path="")
            self.interaction_emb = Embedding(num_c * 2, emb_size)
            self.exercise_emb = Embedding(num_c, emb_size)
            # self.P = Parameter(torch.Tensor(self.seq_len, self.emb_size))
        self.position_emb = Embedding(seq_len, emb_size)

        # Adjust embedding size if using ED and MP features, ensuring divisibility by num_attn_heads
        extra_features = 0
        if self.num_ed_features > 0:
            ed_padding = ((self.num_ed_features + num_attn_heads - 1) // num_attn_heads) * num_attn_heads
            extra_features += ed_padding
        if self.use_mp > 0:
            mp_padding = ((self.num_mp_total_features + num_attn_heads - 1) // num_attn_heads) * num_attn_heads
            extra_features += mp_padding
        effective_emb_size = emb_size + extra_features
        self.blocks = get_clones(Blocks(effective_emb_size, num_attn_heads, dropout), self.num_en)

        self.dropout_layer = Dropout(dropout)

        # Ratio prediction output layers (use_mp=8일 때만 생성)
        if self.use_mp == 8:
            # Step 1: Ratio 0-3 예측 layers (4개)
            self.ratio_out_layers = []
            for i in range(4):  # Ratio 0-3 예측 (4개)
                ratio_out = Linear(effective_emb_size, 1)
                self.ratio_out_layers.append(ratio_out)
            self.ratio_out_layers = torch.nn.ModuleList(self.ratio_out_layers)

            # Step 2: Ratio 4개를 입력으로 받아 prediction 예측하는 FC layer
            # Input: effective_emb_size + 4 (ratio predictions) -> Output: 1 (prediction)
            self.pred = Linear(effective_emb_size + 4, 1)
        else:
            # Baseline: prediction만 예측
            self.pred = Linear(effective_emb_size, 1)

    def base_emb(self, q, r, qry):
        x = q + self.num_c * r
        qshftemb, xemb = self.exercise_emb(qry), self.interaction_emb(x)
    
        posemb = self.position_emb(pos_encode(xemb.shape[1]))
        xemb = xemb + posemb
        return qshftemb, xemb

    def forward(self, q, r, qry, qtest=False, dcur=None):
        emb_type = self.emb_type
        qemb, qshftemb, xemb = None, None, None
        if emb_type == "qid":
            qshftemb, xemb = self.base_emb(q, r, qry)
            
            # Handle ED features if num_ed_features > 0
            if self.num_ed_features > 0:
                if dcur is not None:
                    # Extract ED features from dcur
                    ed_data = []
                    device = qshftemb.device
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
                        # pad ED features for attention head divisibility
                        batch_size, seq_len = qshftemb.shape[:2]
                        ed_padding = ((self.num_ed_features + self.num_attn_heads - 1) // self.num_attn_heads) * self.num_attn_heads
                        ed_padded = torch.zeros((batch_size, seq_len, ed_padding), device=qshftemb.device, dtype=qshftemb.dtype)
                        # Ensure ED tensor matches the sequence length
                        ed_seq_len = min(ed.shape[1], seq_len)
                        ed_padded[:, :ed_seq_len, :self.num_ed_features] = ed[:, :ed_seq_len, :]  # Use actual ED features
                        qshftemb = torch.cat([qshftemb, ed_padded], dim=-1)
                        xemb = torch.cat([xemb, ed_padded], dim=-1)
                    else:
                        # Pad with -1 if no ED features available
                        batch_size, seq_len = qshftemb.shape[:2]
                        ed_padding = ((self.num_ed_features + self.num_attn_heads - 1) // self.num_attn_heads) * self.num_attn_heads
                        ed_padding_tensor = torch.full((batch_size, seq_len, ed_padding), -1.0, device=qshftemb.device, dtype=qshftemb.dtype)
                        qshftemb = torch.cat([qshftemb, ed_padding_tensor], dim=-1)
                        xemb = torch.cat([xemb, ed_padding_tensor], dim=-1)
                else:
                    # Pad with -1 if dcur not provided but model expects ED features
                    batch_size, seq_len = qshftemb.shape[:2]
                    ed_padding = ((self.num_ed_features + self.num_attn_heads - 1) // self.num_attn_heads) * self.num_attn_heads
                    ed_padding_tensor = torch.full((batch_size, seq_len, ed_padding), -1.0, device=qshftemb.device, dtype=qshftemb.dtype)
                    qshftemb = torch.cat([qshftemb, ed_padding_tensor], dim=-1)
                    xemb = torch.cat([xemb, ed_padding_tensor], dim=-1)
            
            # Handle MP features (only if use_mp > 0)
            if self.use_mp > 0:
                if dcur is not None:
                    mp_data = []
                    device = qshftemb.device
                    batch_size, seq_len = qshftemb.shape[:2]

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
                        # pad MP features for attention head divisibility
                        mp_padding = ((self.num_mp_total_features + self.num_attn_heads - 1) // self.num_attn_heads) * self.num_attn_heads
                        mp_padded = torch.zeros((batch_size, seq_len, mp_padding), device=qshftemb.device, dtype=qshftemb.dtype)
                        mp_padded[:, :, :self.num_mp_total_features] = mp
                        qshftemb = torch.cat([qshftemb, mp_padded], dim=-1)
                        xemb = torch.cat([xemb, mp_padded], dim=-1)
                    else:
                        # Pad with zeros if no MP features available
                        mp_padding = ((self.num_mp_total_features + self.num_attn_heads - 1) // self.num_attn_heads) * self.num_attn_heads
                        mp_padding_tensor = torch.zeros((batch_size, seq_len, mp_padding), device=qshftemb.device, dtype=qshftemb.dtype)
                        qshftemb = torch.cat([qshftemb, mp_padding_tensor], dim=-1)
                        xemb = torch.cat([xemb, mp_padding_tensor], dim=-1)
                else:
                    # Pad with zeros if dcur not provided but model expects MP features
                    batch_size, seq_len = qshftemb.shape[:2]
                    mp_padding = ((self.num_mp_total_features + self.num_attn_heads - 1) // self.num_attn_heads) * self.num_attn_heads
                    mp_padding_tensor = torch.zeros((batch_size, seq_len, mp_padding), device=qshftemb.device, dtype=qshftemb.dtype)
                    qshftemb = torch.cat([qshftemb, mp_padding_tensor], dim=-1)
                    xemb = torch.cat([xemb, mp_padding_tensor], dim=-1)
                
        # print(f"qemb: {qemb.shape}, xemb: {xemb.shape}, qshftemb: {qshftemb.shape}")
        for i in range(self.num_en):
            xemb = self.blocks[i](qshftemb, xemb, xemb)

        # New architecture: Ratio 예측 → Ratio를 입력으로 받아 prediction 예측
        if self.use_mp == 8:
            # Step 1: Ratio 0-3 predictions (4개)
            ratio_predictions = []
            for i in range(4):  # Ratio 0-3 예측
                ratio_pred = self.ratio_out_layers[i](self.dropout_layer(xemb))
                ratio_pred = torch.sigmoid(ratio_pred)  # Shape: (batch, seq, 1), [0, 1] range
                ratio_predictions.append(ratio_pred)

            # Step 2: Concatenate ratio predictions to xemb
            # ratio_predictions: list of 4 tensors, each (batch, seq, 1)
            ratio_concat = torch.cat(ratio_predictions, dim=-1)  # Shape: (batch, seq, 4)

            # Step 3: Predict final output using xemb + ratio predictions
            xemb_with_ratio = torch.cat([self.dropout_layer(xemb), ratio_concat], dim=-1)  # Shape: (batch, seq, effective_emb_size + 4)
            p = torch.sigmoid(self.pred(xemb_with_ratio)).squeeze(-1)  # Shape: (batch, seq)

            # Return prediction and ratio predictions (for loss calculation)
            if not qtest:
                # Training: return prediction + ratio predictions (squeezed)
                ratio_predictions_squeezed = [rp.squeeze(-1) for rp in ratio_predictions]  # Each: (batch, seq)
                return p, ratio_predictions_squeezed
            else:
                # Evaluation with qtest: return prediction + xemb
                return p, xemb
        else:
            # Baseline: prediction만 예측
            p = torch.sigmoid(self.pred(self.dropout_layer(xemb))).squeeze(-1)
            if not qtest:
                return p
            else:
                return p, xemb

class Blocks(Module):
    def __init__(self, emb_size, num_attn_heads, dropout) -> None:
        super().__init__()

        self.attn = MultiheadAttention(emb_size, num_attn_heads, dropout=dropout)
        self.attn_dropout = Dropout(dropout)
        self.attn_layer_norm = LayerNorm(emb_size)

        self.FFN = transformer_FFN(emb_size, dropout)
        self.FFN_dropout = Dropout(dropout)
        self.FFN_layer_norm = LayerNorm(emb_size)

    def forward(self, q=None, k=None, v=None):
        q, k, v = q.permute(1, 0, 2), k.permute(1, 0, 2), v.permute(1, 0, 2)
        # attn -> drop -> skip -> norm 
        # transformer: attn -> drop -> skip -> norm transformer default
        causal_mask = ut_mask(seq_len = k.shape[0])
        attn_emb, _ = self.attn(q, k, v, attn_mask=causal_mask)

        attn_emb = self.attn_dropout(attn_emb)
        attn_emb, q = attn_emb.permute(1, 0, 2), q.permute(1, 0, 2)

        attn_emb = self.attn_layer_norm(q + attn_emb)

        emb = self.FFN(attn_emb)
        emb = self.FFN_dropout(emb)
        emb = self.FFN_layer_norm(attn_emb + emb)
        return emb