# coding: utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from .utils import ut_mask

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ATKT(nn.Module):
    def __init__(self, num_c, skill_dim, answer_dim, hidden_dim, attention_dim=80, epsilon=10, beta=0.2, dropout=0.2, emb_type="qid", emb_path="", fix=True, use_ed=0, use_mp=0):
        super(ATKT, self).__init__()
        self.model_name = "atkt"
        self.fix = fix
        print(f"fix: {fix}")
        if self.fix == True:
            self.model_name = "atktfix"
        self.emb_type = emb_type
        self.skill_dim=skill_dim
        self.answer_dim=answer_dim
        self.hidden_dim = hidden_dim
        self.num_c = num_c
        self.epsilon = epsilon
        self.beta = beta
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

        # Adjust LSTM input size if using ED and MP features
        lstm_input_size = self.skill_dim + self.answer_dim + self.num_ed_features + self.num_mp_total_features
        self.rnn = nn.LSTM(lstm_input_size, self.hidden_dim, batch_first=True)
        self.dropout_layer = nn.Dropout(dropout)

        self.skill_emb = nn.Embedding(self.num_c+1, self.skill_dim)
        self.skill_emb.weight.data[-1]= 0

        self.answer_emb = nn.Embedding(2+1, self.answer_dim)
        self.answer_emb.weight.data[-1]= 0

        self.attention_dim = attention_dim
        self.mlp = nn.Linear(self.hidden_dim, self.attention_dim)
        self.similarity = nn.Linear(self.attention_dim, 1, bias=False)

        # Ratio prediction output layers (use_mp=8일 때만 생성)
        if self.use_mp == 8:
            # Step 1: Ratio 0-3 예측 layers (4개)
            self.ratio_out_layers = []
            for i in range(4):  # Ratio 0-3 예측 (4개)
                ratio_out = nn.Linear(self.hidden_dim * 2, self.num_c)
                self.ratio_out_layers.append(ratio_out)
            self.ratio_out_layers = nn.ModuleList(self.ratio_out_layers)

            # Step 2: Ratio 4개를 입력으로 받아 prediction 예측하는 FC layer
            # Input: hidden_dim*2 + num_c*4 (ratio predictions) -> Output: num_c (prediction)
            self.fc = nn.Linear(self.hidden_dim*2 + self.num_c*4, self.num_c)
        else:
            # Baseline: prediction만 예측
            self.fc = nn.Linear(self.hidden_dim*2, self.num_c)

        self.sig = nn.Sigmoid()

    
    def attention_module(self, lstm_output):
        # lstm_output = lstm_output[0:1, :, :]
        # print(f"lstm_output: {lstm_output.shape}")
        att_w = self.mlp(lstm_output)
        # print(f"att_w: {att_w.shape}")
        att_w = torch.tanh(att_w)
        att_w = self.similarity(att_w)
        # print(f"att_w: {att_w.shape}")

        if self.fix == True:
            attn_mask = ut_mask(lstm_output.shape[1])
            att_w = att_w.transpose(1,2).expand(lstm_output.shape[0], lstm_output.shape[1], lstm_output.shape[1]).clone()
            att_w = att_w.masked_fill_(attn_mask, float("-inf"))
            alphas = torch.nn.functional.softmax(att_w, dim=-1)
            attn_ouput = torch.bmm(alphas, lstm_output)
        else: # 原来的官方实现
            alphas=nn.Softmax(dim=1)(att_w)
            # print(f"alphas: {alphas.shape}")    
            attn_ouput = alphas*lstm_output # 整个seq的attn之和为1，计算前面的的时候，所有的attn都<<1，不会有问题？做的少的时候，历史作用小，做得多的时候，历史作用变大？
            # print(f"attn_ouput: {attn_ouput.shape}")
        attn_output_cum=torch.cumsum(attn_ouput, dim=1)
        # print(f"attn_ouput: {attn_ouput}")
        # print(f"attn_output_cum: {attn_output_cum}")
        attn_output_cum_1=attn_output_cum-attn_ouput
        # print(f"attn_output_cum_1: {attn_output_cum_1}")
        # print(f"lstm_output: {lstm_output}")

        final_output=torch.cat((attn_output_cum_1, lstm_output),2)
        # import sys
        # sys.exit()

        return final_output


    def forward(self, skill, answer, perturbation=None, dcur=None):
        ed = None
        if self.num_ed_features > 0 and dcur is not None:
            ed_data = []
            device = skill.device
            for i in range(self.num_ed_features):
                ed_key = f"error_type_{i}"
                if ed_key in dcur and len(dcur[ed_key]) > 0:
                    # Shift ED features: pad with zero at start, remove last timestep
                    ed_shifted = torch.cat([torch.zeros(dcur[ed_key].shape[0], 1, device=device), 
                                          dcur[ed_key][:, :-1]], dim=1)
                    ed_data.append(ed_shifted.to(device).float())
                else:
                    ed_data.append(torch.zeros_like(skill.float(), device=device))
            
            if len(ed_data) > 0:
                ed = torch.stack(ed_data, dim=-1)
        
        # Handle MP features (only if use_mp > 0)
        mp = None
        if self.use_mp > 0 and dcur is not None:
            mp_data = []
            device = skill.device
            target_seq_len = skill.shape[1]

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
                    mp_data.append(torch.zeros(skill.shape[0], target_seq_len, device=device, dtype=torch.float))

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
                    mp_data.append(torch.zeros(skill.shape[0], target_seq_len, device=device, dtype=torch.float))

            if len(mp_data) > 0:
                mp = torch.stack(mp_data, dim=-1)
        
        emb_type = self.emb_type
        r = answer
        
        skill_embedding=self.skill_emb(skill)
        answer_embedding=self.answer_emb(answer)
        
        skill_answer=torch.cat((skill_embedding,answer_embedding), 2)
        answer_skill=torch.cat((answer_embedding,skill_embedding), 2)
        
        answer=answer.unsqueeze(2).expand_as(skill_answer)
        
        skill_answer_embedding=torch.where(answer==1, skill_answer, answer_skill)
        
        # Handle ED features
        if self.num_ed_features > 0:
            if ed is not None:
                # Ensure ed matches skill_answer_embedding sequence length
                target_seq_len = skill_answer_embedding.shape[1]
                if ed.shape[1] > target_seq_len:
                    ed = ed[:, :target_seq_len, :]
                elif ed.shape[1] < target_seq_len:
                    pad_len = target_seq_len - ed.shape[1]
                    ed_padding = torch.full((ed.shape[0], pad_len, ed.shape[2]), -1.0, device=ed.device, dtype=ed.dtype)
                    ed = torch.cat([ed, ed_padding], dim=1)
                skill_answer_embedding = torch.cat([skill_answer_embedding, ed], dim=-1)
            else:
                # Pad with -1 if ED features not provided but model expects them (-1 indicates no data)
                batch_size, seq_len = skill_answer_embedding.shape[:2]
                ed_padding = torch.full((batch_size, seq_len, self.num_ed_features), -1.0, device=skill_answer_embedding.device, dtype=skill_answer_embedding.dtype)
                skill_answer_embedding = torch.cat([skill_answer_embedding, ed_padding], dim=-1)
        
        # Handle MP features
        if self.use_mp > 0:
            if mp is not None:
                # Ensure mp matches skill_answer_embedding sequence length
                target_seq_len = skill_answer_embedding.shape[1]
                if mp.shape[1] > target_seq_len:
                    mp = mp[:, :target_seq_len, :]
                elif mp.shape[1] < target_seq_len:
                    pad_len = target_seq_len - mp.shape[1]
                    mp_padding = torch.zeros((mp.shape[0], pad_len, mp.shape[2]), device=mp.device, dtype=mp.dtype)
                    mp = torch.cat([mp, mp_padding], dim=1)
                skill_answer_embedding = torch.cat([skill_answer_embedding, mp], dim=-1)
            else:
                # Pad with zeros if MP features not provided but model expects them
                batch_size, seq_len = skill_answer_embedding.shape[:2]
                mp_padding = torch.zeros((batch_size, seq_len, self.num_mp_total_features), device=skill_answer_embedding.device, dtype=skill_answer_embedding.dtype)
                skill_answer_embedding = torch.cat([skill_answer_embedding, mp_padding], dim=-1)
        
        # print(skill_answer_embedding)
        
        skill_answer_embedding1=skill_answer_embedding
        if  perturbation is not None:
            skill_answer_embedding += perturbation
            
        out,_ = self.rnn(skill_answer_embedding)
        # print(f"out: {out.shape}")
        out=self.attention_module(out)
        # print(f"after attn out: {out.shape}")

        # New architecture: Ratio 예측 → Ratio를 입력으로 받아 prediction 예측
        if self.use_mp == 8:
            # Step 1: Ratio 0-3 predictions (4개)
            ratio_predictions = []
            for i in range(4):  # Ratio 0-3 예측
                ratio_pred = self.ratio_out_layers[i](self.dropout_layer(out))
                ratio_pred = torch.sigmoid(ratio_pred)  # Shape: (batch, seq, num_c), [0, 1] range
                ratio_predictions.append(ratio_pred)

            # Step 2: Concatenate ratio predictions to out
            # ratio_predictions: list of 4 tensors, each (batch, seq, num_c)
            ratio_concat = torch.cat(ratio_predictions, dim=-1)  # Shape: (batch, seq, num_c*4)

            # Step 3: Predict final output using out + ratio predictions
            out_with_ratio = torch.cat([self.dropout_layer(out), ratio_concat], dim=-1)  # Shape: (batch, seq, hidden_dim*2 + num_c*4)
            res = self.sig(self.fc(out_with_ratio))  # Shape: (batch, seq, num_c)

            # Return prediction + ratio predictions + embeddings
            return res, skill_answer_embedding1, ratio_predictions
        else:
            # Baseline: prediction만 예측
            res = self.sig(self.fc(self.dropout_layer(out)))
            return res, skill_answer_embedding1

from torch.autograd import Variable

def _l2_normalize_adv(d):
    if isinstance(d, Variable):
        d = d.data.cpu().numpy()
    elif isinstance(d, torch.FloatTensor) or isinstance(d, torch.cuda.FloatTensor):
        d = d.cpu().numpy()
    d /= (np.sqrt(np.sum(d ** 2, axis=(1, 2))).reshape((-1, 1, 1)) + 1e-16)
    return torch.from_numpy(d)
