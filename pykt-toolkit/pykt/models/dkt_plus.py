import torch
from torch.nn import Module, Embedding, LSTM, Linear, Dropout

class DKTPlus(Module):
    def __init__(self, num_c, emb_size, lambda_r, lambda_w1, lambda_w2, dropout=0.1, emb_type="qid", emb_path="", pretrain_dim=768, use_ed=0, use_mp=0):
        super().__init__()
        self.model_name = "dkt+"
        self.num_c = num_c
        self.emb_size = emb_size
        self.hidden_size = emb_size
        self.lambda_r = lambda_r
        self.lambda_w1 = lambda_w1
        self.lambda_w2 = lambda_w2
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

        if emb_type.startswith("qid"):
            self.interaction_emb = Embedding(self.num_c * 2, self.emb_size)
        
        # LSTM input size includes variable number of ED and MP features
        lstm_input_size = self.emb_size + self.num_ed_features + self.num_mp_features
        self.lstm_layer = LSTM(lstm_input_size, self.hidden_size, batch_first=True)
        self.dropout_layer = Dropout(dropout)
        self.out_layer = Linear(self.hidden_size, self.num_c)

        # Ratio prediction output layers (use_mp=8일 때만 생성)
        if self.use_mp == 8:
            self.ratio_out_layers = []
            for i in range(4):  # Ratio 0-3 예측 (4개)
                ratio_out = Linear(self.hidden_size, self.num_c)
                self.ratio_out_layers.append(ratio_out)
            self.ratio_out_layers = torch.nn.ModuleList(self.ratio_out_layers)
        

    def forward(self, q, r, dcur=None):
        emb_type = self.emb_type
        if emb_type == "qid":
            x = q + self.num_c * r
            xemb = self.interaction_emb(x)

        # Handle ED features if use_ed is enabled
        if self.num_ed_features > 0:
            if dcur is not None:
                # Extract ED features from dcur
                ed_data = []
                device = xemb.device
                # Always use xemb sequence length to ensure consistency
                target_seq_len = xemb.shape[1]

                # Second pass: create ED features with consistent sequence length
                for i in range(self.num_ed_features):
                    ed_key = f"error_type_{i}"
                    if ed_key in dcur and len(dcur[ed_key]) > 0:
                        # Shift ED features: pad with zero at start, remove last timestep, then match xemb length
                        dcur_tensor = dcur[ed_key].to(device)
                        ed_shifted = torch.cat([torch.zeros(dcur_tensor.shape[0], 1, device=device),
                                              dcur_tensor[:, :-1]], dim=1)
                        # Ensure ed_shifted matches xemb sequence length
                        if ed_shifted.shape[1] > target_seq_len:
                            ed_shifted = ed_shifted[:, :target_seq_len]
                        elif ed_shifted.shape[1] < target_seq_len:
                            pad_len = target_seq_len - ed_shifted.shape[1]
                            ed_shifted = torch.cat([ed_shifted, torch.full((ed_shifted.shape[0], pad_len), -1.0, device=device)], dim=1)
                        ed_data.append(ed_shifted.float())
                    else:
                        ed_data.append(torch.zeros(q.shape[0], target_seq_len, device=device, dtype=torch.float))
                
                if len(ed_data) > 0:
                    ed = torch.stack(ed_data, dim=-1)
                    xemb = torch.cat([xemb, ed], dim=-1)
                else:
                    # Pad with -1 if no ED features available
                    batch_size, seq_len = xemb.shape[:2]
                    ed_padding = torch.full((batch_size, seq_len, self.num_ed_features), -1.0, device=xemb.device, dtype=xemb.dtype)
                    xemb = torch.cat([xemb, ed_padding], dim=-1)
            else:
                # Pad with -1 if dcur not provided but model expects ED features
                batch_size, seq_len = xemb.shape[:2]
                ed_padding = torch.full((batch_size, seq_len, self.num_ed_features), -1.0, device=xemb.device, dtype=xemb.dtype)
                xemb = torch.cat([xemb, ed_padding], dim=-1)

        # Handle MP features if use_mp > 0
        if self.use_mp > 0:
            if dcur is not None:
                mp_data = []
                device = xemb.device
                target_seq_len = xemb.shape[1]

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
                        mp_data.append(torch.zeros(q.shape[0], target_seq_len, device=device, dtype=torch.float))

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
                        mp_data.append(torch.zeros(q.shape[0], target_seq_len, device=device, dtype=torch.float))

                if len(mp_data) > 0:
                    mp = torch.stack(mp_data, dim=-1)
                    xemb = torch.cat([xemb, mp], dim=-1)
                else:
                    # Pad with -1 if no MP features available
                    batch_size, seq_len = xemb.shape[:2]
                    mp_padding = torch.full((batch_size, seq_len, self.num_mp_total_features), -1.0, device=xemb.device, dtype=xemb.dtype)
                    xemb = torch.cat([xemb, mp_padding], dim=-1)
            else:
                # Pad with -1 if dcur not provided but model expects MP features
                batch_size, seq_len = xemb.shape[:2]
                mp_padding = torch.full((batch_size, seq_len, self.num_mp_total_features), -1.0, device=xemb.device, dtype=xemb.dtype)
                xemb = torch.cat([xemb, mp_padding], dim=-1)

        h, _ = self.lstm_layer(xemb)
        h = self.dropout_layer(h)
        y = self.out_layer(h)
        y = torch.sigmoid(y)

        # Ratio predictions (use_mp=8일 때 multi-task)
        if self.use_mp == 8:
            # Multi-task: Ratio 0-3 predictions (4개)
            ratio_predictions = []
            for i in range(4):  # Ratio 0-3 예측
                ratio_pred = self.ratio_out_layers[i](h)  # h is hidden state
                ratio_pred = torch.sigmoid(ratio_pred)  # [0, 1] range for ratio
                ratio_predictions.append(ratio_pred)
            return y, ratio_predictions
        else:
            # Single task: 정오답 예측만
            return y