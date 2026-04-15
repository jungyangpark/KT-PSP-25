import torch 
import torch.nn as nn
from torch.nn import Dropout
import pandas as pd
from .utils import transformer_FFN, get_clones, ut_mask, pos_encode
from torch.nn import Embedding, Linear

device = "cpu" if not torch.cuda.is_available() else "cuda"

class SAINT(nn.Module):
    def __init__(self, num_q, num_c, seq_len, emb_size, num_attn_heads, dropout, n_blocks=1, emb_type="qid", emb_path="", pretrain_dim=768, use_ed=0, use_mp=0):
        super().__init__()
        print(f"num_q: {num_q}, num_c: {num_c}")
        if num_q == num_c and num_q == 0:
            assert num_q != 0
        self.num_q = num_q
        self.num_c = num_c
        self.model_name = "saint"
        self.num_en = n_blocks
        self.num_de = n_blocks
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

        self.embd_pos = nn.Embedding(seq_len, embedding_dim = emb_size) 
        # self.embd_pos = Parameter(torch.Tensor(seq_len-1, emb_size))
        # kaiming_normal_(self.embd_pos)
        
        # Add ED features projection layer if using ED features
        if self.num_ed_features > 0:
            self.ed_linear = nn.Linear(self.num_ed_features, emb_size)
            
        # Add MP features projection layer if using MP features
        if self.num_mp_features > 0:
            self.mp_linear = nn.Linear(self.num_mp_features, emb_size)

        if emb_type.startswith("qid"):
            self.encoder = get_clones(Encoder_block(emb_size, num_attn_heads, num_q, num_c, seq_len, dropout), self.num_en)
        
        self.decoder = get_clones(Decoder_block(emb_size, 2, num_attn_heads, seq_len, dropout), self.num_de)

        self.dropout = Dropout(dropout)

        # Ratio prediction output layers (use_mp=8일 때만 생성)
        if self.use_mp == 8:
            # Step 1: Ratio 0-3 예측 layers (4개)
            self.ratio_out_layers = []
            for i in range(4):  # Ratio 0-3 예측 (4개)
                ratio_out = nn.Linear(emb_size, 1)
                self.ratio_out_layers.append(ratio_out)
            self.ratio_out_layers = nn.ModuleList(self.ratio_out_layers)

            # Step 2: Ratio 4개를 입력으로 받아 prediction 예측하는 FC layer
            # Input: emb_size + 4 (ratio predictions) -> Output: 1 (prediction)
            self.out = nn.Linear(in_features=emb_size + 4, out_features=1)
        else:
            # Baseline: prediction만 예측
            self.out = nn.Linear(in_features=emb_size, out_features=1)
    
    def forward(self, in_ex, in_cat, in_res, qtest=False, dcur=None):
        # Set device variable that's used throughout the method
        device = (in_ex if self.num_q > 0 else in_cat).device
        
        # Handle ED features if use_ed is enabled
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
                    reference_tensor = in_ex if self.num_q > 0 else in_cat
                    ed_data.append(torch.zeros_like(reference_tensor.float(), device=device))
            
            if len(ed_data) > 0:
                ed = torch.stack(ed_data, dim=-1)
        
        # Handle MP features if use_mp is enabled
        mp = None
        if self.use_mp > 0 and dcur is not None:
            mp_data = []
            reference_tensor = in_ex if self.num_q > 0 else in_cat
            batch_size, seq_len = reference_tensor.shape

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
        
        emb_type = self.emb_type        

        if self.num_q > 0:
            in_pos = pos_encode(in_ex.shape[1])
        else:
            in_pos = pos_encode(in_cat.shape[1])
        in_pos = self.embd_pos(in_pos)
        
        # Handle ED features for SAINT model
        # Add ED features to positional encoding (simple approach)
        if self.num_ed_features > 0:
            if ed is not None:
                # Project ED features to embedding size and add to positional encoding
                ed_emb = self.ed_linear(ed)
                in_pos = in_pos + ed_emb
            else:
                # Pad with -1 if ED features not provided but model expects them (-1 indicates no data)
                batch_size, seq_len = (in_ex if self.num_q > 0 else in_cat).shape[:2]
                ed_padding = torch.full((batch_size, seq_len, self.num_ed_features), -1.0, device=in_pos.device, dtype=in_pos.dtype)
                ed_emb = self.ed_linear(ed_padding)
                in_pos = in_pos + ed_emb
        
        # Handle MP features for SAINT model
        # Add MP features to positional encoding (simple approach)
        if self.num_mp_features > 0:
            if mp is not None:
                # Project MP features to embedding size and add to positional encoding
                mp_emb = self.mp_linear(mp)
                in_pos = in_pos + mp_emb
            else:
                # Pad with -1 if MP features not provided but model expects them (-1 indicates no data)
                batch_size, seq_len = (in_ex if self.num_q > 0 else in_cat).shape[:2]
                mp_padding = torch.full((batch_size, seq_len, self.num_mp_features), -1.0, device=in_pos.device, dtype=in_pos.dtype)
                mp_emb = self.mp_linear(mp_padding)
                in_pos = in_pos + mp_emb
        # in_pos = self.embd_pos.unsqueeze(0)
        ## pass through each of the encoder blocks in sequence
        first_block = True
        for i in range(self.num_en):
            if i >= 1:
                first_block = False
            if emb_type == "qid": # same to qid in saint
                in_ex = self.encoder[i](in_ex, in_cat, in_pos, first_block=first_block)
            in_cat = in_ex
        ## pass through each decoder blocks in sequence
        start_token = torch.tensor([[2]]).repeat(in_res.shape[0], 1).to(device)
        in_res = torch.cat((start_token, in_res), dim=-1)
        r = in_res
        first_block = True
        for i in range(self.num_de):
            if i >= 1:
                first_block = False
            in_res = self.decoder[i](in_res, in_pos, en_out=in_ex, first_block=first_block)
        
        ## Output layer

        # New architecture: Ratio 예측 → Ratio를 입력으로 받아 prediction 예측
        if self.use_mp == 8:
            # Step 1: Ratio 0-3 predictions (4개)
            ratio_predictions = []
            for i in range(4):  # Ratio 0-3 예측
                ratio_pred = self.ratio_out_layers[i](self.dropout(in_res))
                ratio_pred = torch.sigmoid(ratio_pred)  # Shape: (batch, seq, 1), [0, 1] range
                ratio_predictions.append(ratio_pred)

            # Step 2: Concatenate ratio predictions to in_res
            # ratio_predictions: list of 4 tensors, each (batch, seq, 1)
            ratio_concat = torch.cat(ratio_predictions, dim=-1)  # Shape: (batch, seq, 4)

            # Step 3: Predict final output using in_res + ratio predictions
            in_res_with_ratio = torch.cat([self.dropout(in_res), ratio_concat], dim=-1)  # Shape: (batch, seq, emb_size + 4)
            res = self.out(in_res_with_ratio)  # Shape: (batch, seq, 1)
            res = torch.sigmoid(res).squeeze(-1)  # Shape: (batch, seq)

            # Return prediction and ratio predictions (for loss calculation)
            if not qtest:
                # Training: return prediction + ratio predictions (squeezed)
                ratio_predictions_squeezed = [rp.squeeze(-1) for rp in ratio_predictions]  # Each: (batch, seq)
                return res, ratio_predictions_squeezed
            else:
                # Evaluation with qtest: return prediction + in_res
                return res, in_res
        else:
            # Baseline: prediction만 예측
            res = self.out(self.dropout(in_res))
            res = torch.sigmoid(res).squeeze(-1)

            if not qtest:
                return res
            else:
                return res, in_res


class Encoder_block(nn.Module):
    """
    M = SkipConct(Multihead(LayerNorm(Qin;Kin;Vin)))
    O = SkipConct(FFN(LayerNorm(M)))
    """

    def __init__(self, dim_model, heads_en, total_ex, total_cat, seq_len, dropout, emb_path="", pretrain_dim=768):
        super().__init__()
        self.seq_len = seq_len
        self.emb_path = emb_path
        self.total_cat = total_cat
        self.total_ex = total_ex
        if total_ex > 0:
            if emb_path == "":
                self.embd_ex = nn.Embedding(total_ex, embedding_dim = dim_model)                   # embedings  q,k,v = E = exercise ID embedding, category embedding, and positionembedding.
            else:
                embs = pd.read_pickle(emb_path)
                self.exercise_embed = Embedding.from_pretrained(embs)
                self.linear = Linear(pretrain_dim, dim_model)
        if total_cat > 0:
            self.emb_cat = nn.Embedding(total_cat, embedding_dim = dim_model)
        # self.embd_pos   = nn.Embedding(seq_len, embedding_dim = dim_model)                  #positional embedding

        self.multi_en = nn.MultiheadAttention(embed_dim = dim_model, num_heads = heads_en, dropout = dropout)
        self.layer_norm1 = nn.LayerNorm(dim_model)
        self.dropout1 = Dropout(dropout)

        self.ffn_en = transformer_FFN(dim_model, dropout)
        self.layer_norm2 = nn.LayerNorm(dim_model)
        self.dropout2 = Dropout(dropout)

    def forward(self, in_ex, in_cat, in_pos, first_block=True):

        ## todo create a positional encoding (two options numeric, sine)
        if first_block:
            embs = []
            if self.total_ex > 0:
                if self.emb_path == "":
                    in_ex = self.embd_ex(in_ex)
                else:
                    in_ex = self.linear(self.exercise_embed(in_ex))
                embs.append(in_ex)
            if self.total_cat > 0:
                in_cat = self.emb_cat(in_cat)
                embs.append(in_cat)
            out = embs[0]
            for i in range(1, len(embs)):
                out += embs[i]
            out = out + in_pos
            # in_pos = self.embd_pos(in_pos)
        else:
            out = in_ex
        
        # in_pos = get_pos(self.seq_len)
        # in_pos = self.embd_pos(in_pos)

        out = out.permute(1,0,2)                                # (n,b,d)  # print('pre multi', out.shape)
        
        # norm -> attn -> drop -> skip corresponging to transformers' norm_first
        #Multihead attention                            
        n,_,_ = out.shape
        out = self.layer_norm1(out)                           # Layer norm
        skip_out = out 
        out, attn_wt = self.multi_en(out, out, out,
                                attn_mask=ut_mask(seq_len=n))  # attention mask upper triangular
        out = self.dropout1(out)
        out = out + skip_out                                    # skip connection

        #feed forward
        out = out.permute(1,0,2)                                # (b,n,d)
        out = self.layer_norm2(out)                           # Layer norm 
        skip_out = out
        out = self.ffn_en(out)
        out = self.dropout2(out)
        out = out + skip_out                                    # skip connection

        return out


class Decoder_block(nn.Module):
    """
    M1 = SkipConct(Multihead(LayerNorm(Qin;Kin;Vin)))
    M2 = SkipConct(Multihead(LayerNorm(M1;O;O)))
    L = SkipConct(FFN(LayerNorm(M2)))
    """

    def __init__(self, dim_model, total_res, heads_de, seq_len, dropout):
        super().__init__()
        self.seq_len    = seq_len
        self.embd_res    = nn.Embedding(total_res+1, embedding_dim = dim_model)                  #response embedding, include a start token
        # self.embd_pos   = nn.Embedding(seq_len, embedding_dim = dim_model)                  #positional embedding
        self.multi_de1  = nn.MultiheadAttention(embed_dim= dim_model, num_heads= heads_de, dropout=dropout)  # M1 multihead for interaction embedding as q k v
        self.multi_de2  = nn.MultiheadAttention(embed_dim= dim_model, num_heads= heads_de, dropout=dropout)  # M2 multihead for M1 out, encoder out, encoder out as q k v
        self.ffn_en     = transformer_FFN(dim_model, dropout)                                         # feed forward layer

        self.layer_norm1 = nn.LayerNorm(dim_model)
        self.layer_norm2 = nn.LayerNorm(dim_model)
        self.layer_norm3 = nn.LayerNorm(dim_model)

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)


    def forward(self, in_res, in_pos, en_out,first_block=True):

         ## todo create a positional encoding (two options numeric, sine)
        if first_block:
            in_in = self.embd_res(in_res)

            #combining the embedings
            out = in_in + in_pos                         # (b,n,d)
        else:
            out = in_res

        # in_pos = get_pos(self.seq_len)
        # in_pos = self.embd_pos(in_pos)

        out = out.permute(1,0,2)                                    # (n,b,d)# print('pre multi', out.shape)
        n,_,_ = out.shape

        #Multihead attention M1                                     ## todo verify if E to passed as q,k,v
        out = self.layer_norm1(out)
        skip_out = out
        out, attn_wt = self.multi_de1(out, out, out, 
                                     attn_mask=ut_mask(seq_len=n)) # attention mask upper triangular
        out = self.dropout1(out)
        out = skip_out + out                                        # skip connection

        #Multihead attention M2                                     ## todo verify if E to passed as q,k,v
        en_out = en_out.permute(1,0,2)                              # (b,n,d)-->(n,b,d)
        en_out = self.layer_norm2(en_out)
        skip_out = out
        out, attn_wt = self.multi_de2(out, en_out, en_out,
                                    attn_mask=ut_mask(seq_len=n))  # attention mask upper triangular
        out = self.dropout2(out)
        out = out + skip_out

        #feed forward
        out = out.permute(1,0,2)                                    # (b,n,d)
        out = self.layer_norm3(out)                               # Layer norm 
        skip_out = out
        out = self.ffn_en(out)                                    
        out = self.dropout3(out)
        out = out + skip_out                                        # skip connection

        return out