import torch
from torch.nn import Module, Embedding, LSTM, Linear, Dropout

device = "cpu" if not torch.cuda.is_available() else "cuda"

class DKTForget(Module):
    def __init__(self, num_c, num_rgap, num_sgap, num_pcount, emb_size, dropout=0.1, emb_type='qid', emb_path="", use_ed=0, use_mp=0):
        super().__init__()
        self.model_name = "dkt_forget"
        self.num_c = num_c
        self.emb_size = emb_size
        self.hidden_size = emb_size
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

        self.c_integration = CIntegration(num_rgap, num_sgap, num_pcount, emb_size, self.num_ed_features, self.num_mp_features)
        ntotal = num_rgap + num_sgap + num_pcount
        
        # LSTM input size includes variable number of ED and MP features
        lstm_input_size = self.emb_size + ntotal + self.num_ed_features + self.num_mp_features
        self.lstm_layer = LSTM(lstm_input_size, self.hidden_size, batch_first=True)
        self.dropout_layer = Dropout(dropout)
        
        # Output layer input size also includes ED and MP features
        out_input_size = self.hidden_size + ntotal + self.num_ed_features + self.num_mp_features
        self.out_layer = Linear(out_input_size, self.num_c)

        # Ratio prediction output layers (use_mp=8일 때만 생성)
        if self.use_mp == 8:
            self.ratio_out_layers = []
            for i in range(4):  # Ratio 0-3 예측 (4개)
                ratio_out = Linear(out_input_size, self.num_c)
                self.ratio_out_layers.append(ratio_out)
            self.ratio_out_layers = torch.nn.ModuleList(self.ratio_out_layers)
        

    def forward(self, q, r, dgaps, dcur=None):
        q, r = q.to(device), r.to(device)
        
        # Handle ED features if use_ed is enabled
        ed = None
        if self.num_ed_features > 0 and dcur is not None:
            ed_data = []
            for i in range(self.num_ed_features):
                ed_key = f"error_type_{i}"
                if ed_key in dcur and len(dcur[ed_key]) > 0:
                    # Shift ED features: pad with zero at start, remove last timestep
                    dcur_tensor = dcur[ed_key].to(device)
                    ed_shifted = torch.cat([torch.zeros(dcur_tensor.shape[0], 1, device=device),
                                          dcur_tensor[:, :-1]], dim=1)
                    ed_data.append(ed_shifted.float())
                else:
                    ed_data.append(torch.zeros(q.shape[0], q.shape[1], device=device, dtype=torch.float))
            
            if len(ed_data) > 0:
                ed = torch.stack(ed_data, dim=-1)
        
        # Handle MP features if use_mp > 0
        mp = None
        if self.use_mp > 0 and dcur is not None:
            mp_data = []
            target_seq_len = q.shape[1]

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
        
        emb_type = self.emb_type
        if emb_type == "qid":
            x = q + self.num_c * r
            xemb = self.interaction_emb(x)
            theta_in = self.c_integration(xemb, dgaps["rgaps"].to(device).long(), dgaps["sgaps"].to(device).long(), dgaps["pcounts"].to(device).long(), ed, mp)

        h, _ = self.lstm_layer(theta_in)
        theta_out = self.c_integration(h, dgaps["shft_rgaps"].to(device).long(), dgaps["shft_sgaps"].to(device).long(), dgaps["shft_pcounts"].to(device).long(), ed, mp)
        theta_out = self.dropout_layer(theta_out)
        y = self.out_layer(theta_out)
        y = torch.sigmoid(y)

        # Ratio predictions (use_mp=8일 때 multi-task)
        if self.use_mp == 8:
            # Multi-task: Ratio 0-3 predictions (4개)
            ratio_predictions = []
            for i in range(4):  # Ratio 0-3 예측
                ratio_pred = self.ratio_out_layers[i](theta_out)  # theta_out is the final hidden state
                ratio_pred = torch.sigmoid(ratio_pred)  # [0, 1] range for ratio
                ratio_predictions.append(ratio_pred)
            return y, ratio_predictions
        else:
            # Single task: 정오답 예측만
            return y


class CIntegration(Module):
    def __init__(self, num_rgap, num_sgap, num_pcount, emb_dim, use_ed=0, use_mp=0) -> None:
        super().__init__()
        self.num_ed_features = use_ed
        self.num_mp_features = use_mp
        self.register_buffer('rgap_eye', torch.eye(num_rgap))
        self.register_buffer('sgap_eye', torch.eye(num_sgap))
        self.register_buffer('pcount_eye', torch.eye(num_pcount))

        ntotal = num_rgap + num_sgap + num_pcount
        self.cemb = Linear(ntotal, emb_dim, bias=False)
        print(f"num_sgap: {num_sgap}, num_rgap: {num_rgap}, num_pcount: {num_pcount}, ntotal: {ntotal}")
        # print(f"total: {ntotal}, self.cemb.weight: {self.cemb.weight.shape}")

    def forward(self, vt, rgap, sgap, pcount, ed=None, mp=None):
        rgap, sgap, pcount = self.rgap_eye[rgap], self.sgap_eye[sgap], self.pcount_eye[pcount]
        # print(f"vt: {vt.shape}, rgap: {rgap.shape}, sgap: {sgap.shape}, pcount: {pcount.shape}")
        ct = torch.cat((rgap, sgap, pcount), -1) # bz * seq_len * num_fea
        # print(f"ct: {ct.shape}, self.cemb.weight: {self.cemb.weight.shape}")
        # element-wise mul
        Cct = self.cemb(ct) # bz * seq_len * emb
        # print(f"ct: {ct.shape}, Cct: {Cct.shape}")
        theta = torch.mul(vt, Cct)
        theta = torch.cat((theta, ct), -1)
        
        # If use_ed is True, concatenate ED features (or zeros if not provided)
        if self.num_ed_features > 0:
            if ed is not None:
                # ed should have shape [batch_size, seq_len, 4]
                ed = ed.to(theta.device)
                theta = torch.cat([theta, ed], dim=-1)
            else:
                # Pad with -1 if ED features not provided but model expects them (-1 indicates no data)
                batch_size, seq_len = theta.shape[:2]
                ed_padding = torch.full((batch_size, seq_len, self.num_ed_features), -1.0, device=theta.device, dtype=theta.dtype)
                theta = torch.cat([theta, ed_padding], dim=-1)
        
        # If use_mp is True, concatenate MP features (or zeros if not provided)
        if self.num_mp_features > 0:
            if mp is not None:
                # mp should have shape [batch_size, seq_len, num_mp_features]
                mp = mp.to(theta.device)
                theta = torch.cat([theta, mp], dim=-1)
            else:
                # Pad with -1 if MP features not provided but model expects them (-1 indicates no data)
                batch_size, seq_len = theta.shape[:2]
                mp_padding = torch.full((batch_size, seq_len, self.num_mp_features), -1.0, device=theta.device, dtype=theta.dtype)
                theta = torch.cat([theta, mp_padding], dim=-1)
        
        return theta
