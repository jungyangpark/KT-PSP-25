import os, sys
import torch
import torch.nn as nn
from torch.nn.functional import one_hot, binary_cross_entropy, cross_entropy
from torch.nn.utils.clip_grad import clip_grad_norm_
import numpy as np
from .evaluate_model import evaluate
from torch.autograd import Variable, grad
from .atkt import _l2_normalize_adv
from ..utils.utils import debug_print
from pykt.config import que_type_models
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cal_loss(model, ys, r, rshft, sm, preloss=[], dcur=None, cshft=None, alpha=1.0):
    model_name = model.model_name

    if model_name == "simplekt":
        # SimpleKT with MP+Ratio features: multi-task (knowledge state + ratio prediction)
        y_knowledge = torch.masked_select(ys[0], sm)  # Knowledge state prediction
        t = torch.masked_select(rshft, sm)
        loss1 = binary_cross_entropy(y_knowledge.double(), t.double())

        if model.emb_type.find("predcurc") != -1:
            if model.emb_type.find("his") != -1:
                loss_knowledge = model.l1*loss1+model.l2*ys[1]+model.l3*ys[2]
            else:
                loss_knowledge = model.l1*loss1+model.l2*ys[1]
        elif model.emb_type.find("predhis") != -1:
            loss_knowledge = model.l1*loss1+model.l2*ys[1]
        else:
            loss_knowledge = loss1

        # Ratio 0-3 predictions loss (auxiliary task)
        loss_ratio = 0
        if len(ys) > 3 and isinstance(ys[3], list) and dcur is not None:  # Ratio predictions available
            ratio_predictions = ys[3]  # List of 4 Ratio 0-3 predictions (each shape: batch_size, seq_len)
            for i, ratio_pred in enumerate(ratio_predictions):
                ratio_target_key = f"mp_ratio_{i}"  # Ratio 0-3 target 사용
                if ratio_target_key in dcur:
                    ratio_target = dcur[ratio_target_key].to(device)
                    # Match sequence length with prediction
                    target_seq_len = ratio_pred.shape[1]
                    if ratio_target.shape[1] != target_seq_len:
                        if ratio_target.shape[1] < target_seq_len:
                            # Pad if target is shorter
                            pad_len = target_seq_len - ratio_target.shape[1]
                            ratio_target = torch.cat([ratio_target, torch.zeros(ratio_target.shape[0], pad_len, device=device)], dim=1)
                        else:
                            # Truncate if target is longer
                            ratio_target = ratio_target[:, :target_seq_len]

                    # Shift ratio target: predict next timestep's ratio (like rshft)
                    # At timestep t, we predict ratio for timestep t+1
                    ratio_target_shifted = torch.cat([ratio_target[:, 1:],
                                                     torch.zeros(ratio_target.shape[0], 1, device=device)], dim=1)

                    # SimpleKT output is already per-timestep: (batch, seq) not (batch, seq, num_c)
                    # Apply the same mask as knowledge prediction
                    ratio_pred_masked = torch.masked_select(ratio_pred, sm)
                    ratio_target_masked = torch.masked_select(ratio_target_shifted, sm)

                    # Use MSE loss for regression (Ratio values are continuous [0, 1])
                    valid_mask = ratio_target_masked >= 0  # Filter out invalid values
                    if valid_mask.sum() > 0:
                        ratio_pred_valid = ratio_pred_masked[valid_mask]
                        ratio_target_valid = ratio_target_masked[valid_mask]
                        loss_ratio += nn.functional.mse_loss(ratio_pred_valid.float(), ratio_target_valid.float())

        # Combine losses (knowledge prediction + ratio predictions)
        loss = loss_knowledge + alpha * loss_ratio

    elif model_name == "stablekt":
        # Model with MP+Ratio features: multi-task (knowledge state + ratio prediction)
        y_knowledge = torch.masked_select(ys[0], sm)  # Knowledge state prediction
        t = torch.masked_select(rshft, sm)
        loss_knowledge = binary_cross_entropy(y_knowledge.double(), t.double())

        # Ratio 0-3 predictions loss (auxiliary task)
        loss_ratio = 0
        if len(ys) > 3 and isinstance(ys[3], list) and dcur is not None and cshft is not None:  # Ratio predictions available
            ratio_predictions = ys[3]  # List of 4 Ratio 0-3 predictions (each shape: batch_size, seq_len)
            for i, ratio_pred in enumerate(ratio_predictions):
                ratio_target_key = f"mp_ratio_{i}"  # Ratio 0-3 target 사용
                if ratio_target_key in dcur:
                    ratio_target = dcur[ratio_target_key].to(device)
                    # Match sequence length with prediction
                    target_seq_len = ratio_pred.shape[1]
                    if ratio_target.shape[1] != target_seq_len:
                        if ratio_target.shape[1] < target_seq_len:
                            # Pad if target is shorter
                            pad_len = target_seq_len - ratio_target.shape[1]
                            ratio_target = torch.cat([ratio_target, torch.zeros(ratio_target.shape[0], pad_len, device=device)], dim=1)
                        else:
                            # Truncate if target is longer
                            ratio_target = ratio_target[:, :target_seq_len]

                    # Shift ratio target: predict next timestep's ratio (like rshft)
                    # At timestep t, we predict ratio for timestep t+1
                    ratio_target_shifted = torch.cat([ratio_target[:, 1:],
                                                     torch.zeros(ratio_target.shape[0], 1, device=device)], dim=1)

                    # StableKT ratio prediction은 이미 (batch, seq) 형태
                    ratio_pred_masked = torch.masked_select(ratio_pred, sm)
                    ratio_target_masked = torch.masked_select(ratio_target_shifted, sm)

                    # Use MSE loss for regression (Ratio values are continuous [0, 1])
                    valid_mask = ratio_target_masked >= 0  # Filter out invalid values
                    if valid_mask.sum() > 0:
                        ratio_pred_valid = ratio_pred_masked[valid_mask]
                        ratio_target_valid = ratio_target_masked[valid_mask]
                        loss_ratio += nn.functional.mse_loss(ratio_pred_valid.float(), ratio_target_valid.float())

        # Combine losses (knowledge prediction + ratio predictions)
        loss = loss_knowledge + alpha * loss_ratio

    elif model_name in ["atdkt", "datakt", "sparsekt", "cskt", "hcgkt"]:
        y = torch.masked_select(ys[0], sm)
        t = torch.masked_select(rshft, sm)
        # print(f"loss1: {y.shape}")
        loss1 = binary_cross_entropy(y.double(), t.double())

        if model.emb_type.find("predcurc") != -1:
            if model.emb_type.find("his") != -1:
                loss = model.l1*loss1+model.l2*ys[1]+model.l3*ys[2]
            else:
                loss = model.l1*loss1+model.l2*ys[1]
        elif model.emb_type.find("predhis") != -1:
            loss = model.l1*loss1+model.l2*ys[1]
        else:
            loss = loss1
    elif model_name in ["rekt"]:
        # print("ys shape:", ys[0].shape)
        # print("sm shape:", sm.shape)
        y = torch.masked_select(ys[0], sm)
        t = torch.masked_select(rshft, sm)
        loss = binary_cross_entropy(y.double(), t.double())
    
    elif model_name in ["ukt"]:
        y = torch.masked_select(ys[0], sm)
        t = torch.masked_select(rshft, sm)
        loss1 = binary_cross_entropy(y.double(), t.double())
        if model.use_CL:
            loss2 = ys[1]
            loss1 = loss1 + model.cl_weight * loss2
        loss =loss1

    elif model_name == "dkvmn":
        # DKVMN with MP+Ratio features: multi-task (knowledge state + ratio prediction)
        y_knowledge = torch.masked_select(ys[0], sm)  # Knowledge state prediction
        t = torch.masked_select(rshft, sm)
        loss_knowledge = binary_cross_entropy(y_knowledge.double(), t.double())

        # Ratio 0-3 predictions loss (auxiliary task)
        loss_ratio = 0
        if len(ys) > 1 and isinstance(ys[1], list) and dcur is not None:  # Ratio predictions available
            ratio_predictions = ys[1]  # List of 4 Ratio 0-3 predictions (each shape: batch_size, seq_len)
            for i, ratio_pred in enumerate(ratio_predictions):
                ratio_target_key = f"mp_ratio_{i}"  # Ratio 0-3 target 사용
                if ratio_target_key in dcur:
                    ratio_target = dcur[ratio_target_key].to(device)
                    # Match sequence length with prediction
                    target_seq_len = ratio_pred.shape[1]
                    if ratio_target.shape[1] != target_seq_len:
                        if ratio_target.shape[1] < target_seq_len:
                            # Pad if target is shorter
                            pad_len = target_seq_len - ratio_target.shape[1]
                            ratio_target = torch.cat([ratio_target, torch.zeros(ratio_target.shape[0], pad_len, device=device)], dim=1)
                        else:
                            # Truncate if target is longer
                            ratio_target = ratio_target[:, :target_seq_len]

                    # Shift ratio target: predict next timestep's ratio (like rshft)
                    # At timestep t, we predict ratio for timestep t+1
                    ratio_target_shifted = torch.cat([ratio_target[:, 1:],
                                                     torch.zeros(ratio_target.shape[0], 1, device=device)], dim=1)

                    # Apply the same mask as knowledge prediction
                    ratio_pred_masked = torch.masked_select(ratio_pred, sm)
                    ratio_target_masked = torch.masked_select(ratio_target_shifted, sm)

                    # Use MSE loss for regression (Ratio values are continuous [0, 1])
                    valid_mask = ratio_target_masked >= 0  # Filter out invalid values
                    if valid_mask.sum() > 0:
                        ratio_pred_valid = ratio_pred_masked[valid_mask]
                        ratio_target_valid = ratio_target_masked[valid_mask]
                        loss_ratio += nn.functional.mse_loss(ratio_pred_valid.float(), ratio_target_valid.float())

        # Combine losses (knowledge prediction + ratio predictions)
        loss = loss_knowledge + alpha * loss_ratio

    elif model_name in ["atkt", "atktfix"]:
        y_knowledge = torch.masked_select(ys[0], sm)
        t = torch.masked_select(rshft, sm)
        loss_knowledge = binary_cross_entropy(y_knowledge.double(), t.double())

        # Ratio 0-3 predictions loss (auxiliary task)
        loss_ratio = 0
        if len(ys) > 1 and isinstance(ys[1], list) and dcur is not None and cshft is not None:
            ratio_predictions = ys[1]
            for i, ratio_pred in enumerate(ratio_predictions):
                ratio_target_key = f"mp_ratio_{i}"
                if ratio_target_key in dcur:
                    ratio_target = dcur[ratio_target_key].to(device)
                    target_seq_len = ratio_pred.shape[1]
                    if ratio_target.shape[1] != target_seq_len:
                        if ratio_target.shape[1] < target_seq_len:
                            # Pad if target is shorter
                            pad_len = target_seq_len - ratio_target.shape[1]
                            ratio_target = torch.cat([ratio_target, torch.zeros(ratio_target.shape[0], pad_len, device=device)], dim=1)
                        else:
                            ratio_target = ratio_target[:, :target_seq_len]

                    ratio_target_shifted = torch.cat([ratio_target[:, 1:],
                                                     torch.zeros(ratio_target.shape[0], 1, device=device)], dim=1)

                    ratio_pred_for_concept = (ratio_pred * one_hot(cshft.long(), model.num_c)).sum(-1)
                    ratio_pred_masked = torch.masked_select(ratio_pred_for_concept, sm)
                    ratio_target_masked = torch.masked_select(ratio_target_shifted, sm)

                    valid_mask = ratio_target_masked >= 0
                    if valid_mask.sum() > 0:
                        ratio_pred_valid = ratio_pred_masked[valid_mask]
                        ratio_target_valid = ratio_target_masked[valid_mask]
                        loss_ratio += nn.functional.mse_loss(ratio_pred_valid.float(), ratio_target_valid.float())

        loss = loss_knowledge + alpha * loss_ratio

    elif model_name == "sakt":
        # SAKT with MP+Ratio features: multi-task (knowledge state + ratio prediction)
        y_knowledge = torch.masked_select(ys[0], sm)  # Knowledge state prediction
        t = torch.masked_select(rshft, sm)
        loss_knowledge = binary_cross_entropy(y_knowledge.double(), t.double())

        # Ratio 0-3 predictions loss (auxiliary task)
        loss_ratio = 0
        if len(ys) > 1 and isinstance(ys[1], list) and dcur is not None:  # Ratio predictions available
            ratio_predictions = ys[1]  # List of 4 Ratio 0-3 predictions (each shape: batch_size, seq_len)
            for i, ratio_pred in enumerate(ratio_predictions):
                ratio_target_key = f"mp_ratio_{i}"  # Ratio 0-3 target 사용
                if ratio_target_key in dcur:
                    ratio_target = dcur[ratio_target_key].to(device)
                    # Match sequence length with prediction
                    target_seq_len = ratio_pred.shape[1]
                    if ratio_target.shape[1] != target_seq_len:
                        if ratio_target.shape[1] < target_seq_len:
                            # Pad if target is shorter
                            pad_len = target_seq_len - ratio_target.shape[1]
                            ratio_target = torch.cat([ratio_target, torch.zeros(ratio_target.shape[0], pad_len, device=device)], dim=1)
                        else:
                            # Truncate if target is longer
                            ratio_target = ratio_target[:, :target_seq_len]

                    # Shift ratio target: predict next timestep's ratio (like rshft)
                    # At timestep t, we predict ratio for timestep t+1
                    ratio_target_shifted = torch.cat([ratio_target[:, 1:],
                                                     torch.zeros(ratio_target.shape[0], 1, device=device)], dim=1)

                    # Apply the same mask as knowledge prediction
                    ratio_pred_masked = torch.masked_select(ratio_pred, sm)
                    ratio_target_masked = torch.masked_select(ratio_target_shifted, sm)

                    # Use MSE loss for regression (Ratio values are continuous [0, 1])
                    valid_mask = ratio_target_masked >= 0  # Filter out invalid values
                    if valid_mask.sum() > 0:
                        ratio_pred_valid = ratio_pred_masked[valid_mask]
                        ratio_target_valid = ratio_target_masked[valid_mask]
                        loss_ratio += nn.functional.mse_loss(ratio_pred_valid.float(), ratio_target_valid.float())

        # Combine losses (knowledge prediction + ratio predictions)
        loss = loss_knowledge + alpha * loss_ratio

    elif model_name == "gkt":
        # GKT with MP+Ratio features: multi-task (knowledge state + ratio prediction)
        y_knowledge = torch.masked_select(ys[0], sm)  # Knowledge state prediction
        t = torch.masked_select(rshft, sm)
        loss_knowledge = binary_cross_entropy(y_knowledge.double(), t.double())

        # Ratio 0-3 predictions loss (auxiliary task)
        loss_ratio = 0
        if len(ys) > 1 and isinstance(ys[1], list) and dcur is not None:  # Ratio predictions available
            ratio_predictions = ys[1]  # List of 4 Ratio 0-3 predictions (each shape: batch_size, seq_len)
            for i, ratio_pred in enumerate(ratio_predictions):
                ratio_target_key = f"mp_ratio_{i}"  # Ratio 0-3 target 사용
                if ratio_target_key in dcur:
                    ratio_target = dcur[ratio_target_key].to(device)
                    # Match sequence length with prediction
                    target_seq_len = ratio_pred.shape[1]
                    if ratio_target.shape[1] != target_seq_len:
                        if ratio_target.shape[1] < target_seq_len:
                            # Pad if target is shorter
                            pad_len = target_seq_len - ratio_target.shape[1]
                            ratio_target = torch.cat([ratio_target, torch.zeros(ratio_target.shape[0], pad_len, device=device)], dim=1)
                        else:
                            # Truncate if target is longer
                            ratio_target = ratio_target[:, :target_seq_len]

                    # Shift ratio target: predict next timestep's ratio (like rshft)
                    # At timestep t, we predict ratio for timestep t+1
                    ratio_target_shifted = torch.cat([ratio_target[:, 1:],
                                                     torch.zeros(ratio_target.shape[0], 1, device=device)], dim=1)

                    # GKT output is already per-question: (batch, seq) not (batch, seq, num_c)
                    # Apply the same mask as knowledge prediction
                    ratio_pred_masked = torch.masked_select(ratio_pred, sm)
                    ratio_target_masked = torch.masked_select(ratio_target_shifted, sm)

                    # Use MSE loss for regression (Ratio values are continuous [0, 1])
                    valid_mask = ratio_target_masked >= 0  # Filter out invalid values
                    if valid_mask.sum() > 0:
                        ratio_pred_valid = ratio_pred_masked[valid_mask]
                        ratio_target_valid = ratio_target_masked[valid_mask]
                        loss_ratio += nn.functional.mse_loss(ratio_pred_valid.float(), ratio_target_valid.float())

        # Combine losses (knowledge prediction + ratio predictions)
        loss = loss_knowledge + alpha * loss_ratio

    elif model_name == "saint":
        # SAINT with MP+Ratio features: multi-task (knowledge state + ratio prediction)
        y_knowledge = torch.masked_select(ys[0], sm)  # Knowledge state prediction
        t = torch.masked_select(rshft, sm)
        loss_knowledge = binary_cross_entropy(y_knowledge.double(), t.double())

        # Ratio 0-3 predictions loss (auxiliary task)
        loss_ratio = 0
        if len(ys) > 1 and isinstance(ys[1], list) and dcur is not None:  # Ratio predictions available
            ratio_predictions = ys[1]  # List of 4 Ratio 0-3 predictions (each shape: batch_size, seq_len)
            for i, ratio_pred in enumerate(ratio_predictions):
                ratio_target_key = f"mp_ratio_{i}"  # Ratio 0-3 target 사용
                if ratio_target_key in dcur:
                    ratio_target = dcur[ratio_target_key].to(device)
                    # Match sequence length with prediction
                    target_seq_len = ratio_pred.shape[1]
                    if ratio_target.shape[1] != target_seq_len:
                        if ratio_target.shape[1] < target_seq_len:
                            # Pad if target is shorter
                            pad_len = target_seq_len - ratio_target.shape[1]
                            ratio_target = torch.cat([ratio_target, torch.zeros(ratio_target.shape[0], pad_len, device=device)], dim=1)
                        else:
                            # Truncate if target is longer
                            ratio_target = ratio_target[:, :target_seq_len]

                    # Shift ratio target: predict next timestep's ratio (like rshft)
                    # At timestep t, we predict ratio for timestep t+1
                    ratio_target_shifted = torch.cat([ratio_target[:, 1:],
                                                     torch.zeros(ratio_target.shape[0], 1, device=device)], dim=1)

                    # SAINT output is already per-timestep: (batch, seq) not (batch, seq, num_c)
                    # Apply the same mask as knowledge prediction
                    ratio_pred_masked = torch.masked_select(ratio_pred, sm)
                    ratio_target_masked = torch.masked_select(ratio_target_shifted, sm)

                    # Use MSE loss for regression (Ratio values are continuous [0, 1])
                    valid_mask = ratio_target_masked >= 0  # Filter out invalid values
                    if valid_mask.sum() > 0:
                        ratio_pred_valid = ratio_pred_masked[valid_mask]
                        ratio_target_valid = ratio_target_masked[valid_mask]
                        loss_ratio += nn.functional.mse_loss(ratio_pred_valid.float(), ratio_target_valid.float())

        # Combine losses (knowledge prediction + ratio predictions)
        loss = loss_knowledge + alpha * loss_ratio

    elif model_name == "dkt_forget":
        # DKT_forget with MP+Ratio features: multi-task (knowledge state + ratio prediction)
        y_knowledge = torch.masked_select(ys[0], sm)  # Knowledge state prediction
        t = torch.masked_select(rshft, sm)
        loss_knowledge = binary_cross_entropy(y_knowledge.double(), t.double())

        # Ratio 0-3 predictions loss (auxiliary task)
        loss_ratio = 0
        if len(ys) > 1 and isinstance(ys[1], list) and dcur is not None and cshft is not None:  # Ratio predictions available
            ratio_predictions = ys[1]  # List of 4 Ratio 0-3 predictions (each shape: batch_size, seq_len, num_c)
            for i, ratio_pred in enumerate(ratio_predictions):
                ratio_target_key = f"mp_ratio_{i}"  # Ratio 0-3 target 사용
                if ratio_target_key in dcur:
                    ratio_target = dcur[ratio_target_key].to(device)
                    # Match sequence length with prediction
                    target_seq_len = ratio_pred.shape[1]
                    if ratio_target.shape[1] != target_seq_len:
                        if ratio_target.shape[1] < target_seq_len:
                            # Pad if target is shorter
                            pad_len = target_seq_len - ratio_target.shape[1]
                            ratio_target = torch.cat([ratio_target, torch.zeros(ratio_target.shape[0], pad_len, device=device)], dim=1)
                        else:
                            # Truncate if target is longer
                            ratio_target = ratio_target[:, :target_seq_len]

                    # Shift ratio target: predict next timestep's ratio (like rshft)
                    # At timestep t, we predict ratio for timestep t+1
                    ratio_target_shifted = torch.cat([ratio_target[:, 1:],
                                                     torch.zeros(ratio_target.shape[0], 1, device=device)], dim=1)

                    # Ratio prediction도 concept selection 필요: (batch, seq, num_c) -> (batch, seq)
                    ratio_pred_for_concept = (ratio_pred * one_hot(cshft.long(), model.num_c)).sum(-1)
                    ratio_pred_masked = torch.masked_select(ratio_pred_for_concept, sm)
                    ratio_target_masked = torch.masked_select(ratio_target_shifted, sm)

                    # Use MSE loss for regression (Ratio values are continuous [0, 1])
                    valid_mask = ratio_target_masked >= 0  # Filter out invalid values
                    if valid_mask.sum() > 0:
                        ratio_pred_valid = ratio_pred_masked[valid_mask]
                        ratio_target_valid = ratio_target_masked[valid_mask]
                        loss_ratio += nn.functional.mse_loss(ratio_pred_valid.float(), ratio_target_valid.float())

        # Combine losses (knowledge prediction + ratio predictions)
        loss = loss_knowledge + alpha * loss_ratio

    elif model_name == "skvmn":
        # SKVMN with MP+Ratio features: multi-task (knowledge state + ratio prediction)
        y_knowledge = torch.masked_select(ys[0], sm)  # Knowledge state prediction
        t = torch.masked_select(rshft, sm)
        loss_knowledge = binary_cross_entropy(y_knowledge.double(), t.double())

        # Ratio 0-3 predictions loss (auxiliary task)
        loss_ratio = 0
        if len(ys) > 1 and isinstance(ys[1], list) and dcur is not None:  # Ratio predictions available
            ratio_predictions = ys[1]  # List of 4 Ratio 0-3 predictions (each shape: batch_size, seq_len)
            for i, ratio_pred in enumerate(ratio_predictions):
                ratio_target_key = f"mp_ratio_{i}"  # Ratio 0-3 target 사용
                if ratio_target_key in dcur:
                    ratio_target = dcur[ratio_target_key].to(device)
                    # Match sequence length with prediction
                    target_seq_len = ratio_pred.shape[1]
                    if ratio_target.shape[1] != target_seq_len:
                        if ratio_target.shape[1] < target_seq_len:
                            # Pad if target is shorter
                            pad_len = target_seq_len - ratio_target.shape[1]
                            ratio_target = torch.cat([ratio_target, torch.zeros(ratio_target.shape[0], pad_len, device=device)], dim=1)
                        else:
                            # Truncate if target is longer
                            ratio_target = ratio_target[:, :target_seq_len]

                    ratio_target_shifted = torch.cat([ratio_target[:, 1:],
                                                     torch.zeros(ratio_target.shape[0], 1, device=device)], dim=1)

                    ratio_pred_masked = torch.masked_select(ratio_pred, sm)
                    ratio_target_masked = torch.masked_select(ratio_target_shifted, sm)

                    # Use MSE loss for regression (Ratio values are continuous [0, 1])
                    valid_mask = ratio_target_masked >= 0  
                    if valid_mask.sum() > 0:
                        ratio_pred_valid = ratio_pred_masked[valid_mask]
                        ratio_target_valid = ratio_target_masked[valid_mask]
                        loss_ratio += nn.functional.mse_loss(ratio_pred_valid.float(), ratio_target_valid.float())

        # Combine losses (knowledge prediction + ratio predictions)
        loss = loss_knowledge + alpha * loss_ratio

    elif model_name in ["rkt","dimkt", "deep_irt", "kqn", "hawkes"]:

        y = torch.masked_select(ys[0], sm)
        t = torch.masked_select(rshft, sm)
        loss = binary_cross_entropy(y.double(), t.double())

    elif model_name == "dkt2":
        # DKT2 with MP+Ratio features: multi-task (knowledge state + ratio prediction)
        y_knowledge = torch.masked_select(ys[0], sm)  # Knowledge state prediction
        t = torch.masked_select(rshft, sm)

        # Handle NaN/Inf and clamp to valid range for BCE
        y_knowledge = torch.nan_to_num(y_knowledge, nan=0.5, posinf=1.0, neginf=0.0)
        y_knowledge = torch.clamp(y_knowledge, min=1e-7, max=1.0 - 1e-7)  # Avoid exact 0/1 for numerical stability

        loss_knowledge = binary_cross_entropy(y_knowledge.double(), t.double())

        # Ratio 0-3 predictions loss (auxiliary task)
        loss_ratio = 0
        if len(ys) > 1 and isinstance(ys[1], list) and dcur is not None and cshft is not None:  # Ratio predictions available
            ratio_predictions = ys[1]  # List of 4 Ratio 0-3 predictions (each shape: batch_size, seq_len, num_c)
            for i, ratio_pred in enumerate(ratio_predictions):
                ratio_target_key = f"mp_ratio_{i}"  # Ratio 0-3 target 사용
                if ratio_target_key in dcur:
                    ratio_target = dcur[ratio_target_key].to(device)
                    # Match sequence length with prediction
                    target_seq_len = ratio_pred.shape[1]
                    if ratio_target.shape[1] != target_seq_len:
                        if ratio_target.shape[1] < target_seq_len:
                            # Pad if target is shorter
                            pad_len = target_seq_len - ratio_target.shape[1]
                            ratio_target = torch.cat([ratio_target, torch.zeros(ratio_target.shape[0], pad_len, device=device)], dim=1)
                        else:
                            # Truncate if target is longer
                            ratio_target = ratio_target[:, :target_seq_len]

                    # Shift ratio target: predict next timestep's ratio (like rshft)
                    # At timestep t, we predict ratio for timestep t+1
                    ratio_target_shifted = torch.cat([ratio_target[:, 1:],
                                                     torch.zeros(ratio_target.shape[0], 1, device=device)], dim=1)

                    # Ratio prediction도 concept selection 필요: (batch, seq, num_c) -> (batch, seq)
                    ratio_pred_for_concept = (ratio_pred * one_hot(cshft.long(), model.num_c)).sum(-1)
                    ratio_pred_masked = torch.masked_select(ratio_pred_for_concept, sm)
                    ratio_target_masked = torch.masked_select(ratio_target_shifted, sm)

                    # Use MSE loss for regression (Ratio values are continuous [0, 1])
                    valid_mask = ratio_target_masked >= 0  # Filter out invalid values
                    if valid_mask.sum() > 0:
                        ratio_pred_valid = ratio_pred_masked[valid_mask]
                        ratio_target_valid = ratio_target_masked[valid_mask]
                        loss_ratio += nn.functional.mse_loss(ratio_pred_valid.float(), ratio_target_valid.float())

        # Combine losses (knowledge prediction + ratio predictions)
        loss = loss_knowledge + alpha * loss_ratio

    elif model_name == "dkt":
        # DKT with MP+Ratio features: multi-task (knowledge state + ratio prediction)
        y_knowledge = torch.masked_select(ys[0], sm)  # Knowledge state prediction
        t = torch.masked_select(rshft, sm)
        loss_knowledge = binary_cross_entropy(y_knowledge.double(), t.double())

        # Ratio 0-3 predictions loss (auxiliary task)
        loss_ratio = 0
        if len(ys) > 1 and isinstance(ys[1], list) and dcur is not None and cshft is not None:  # Ratio predictions available
            ratio_predictions = ys[1]  # List of 4 Ratio 0-3 predictions (each shape: batch_size, seq_len, num_c)
            for i, ratio_pred in enumerate(ratio_predictions):
                ratio_target_key = f"mp_ratio_{i}"  # Ratio 0-3 target 사용
                if ratio_target_key in dcur:
                    ratio_target = dcur[ratio_target_key].to(device)
                    # Match sequence length with prediction
                    target_seq_len = ratio_pred.shape[1]
                    if ratio_target.shape[1] != target_seq_len:
                        if ratio_target.shape[1] < target_seq_len:
                            # Pad if target is shorter
                            pad_len = target_seq_len - ratio_target.shape[1]
                            ratio_target = torch.cat([ratio_target, torch.zeros(ratio_target.shape[0], pad_len, device=device)], dim=1)
                        else:
                            # Truncate if target is longer
                            ratio_target = ratio_target[:, :target_seq_len]

                    # Shift ratio target: predict next timestep's ratio (like rshft)
                    # At timestep t, we predict ratio for timestep t+1
                    ratio_target_shifted = torch.cat([ratio_target[:, 1:],
                                                     torch.zeros(ratio_target.shape[0], 1, device=device)], dim=1)

                    # Ratio prediction도 concept selection 필요: (batch, seq, num_c) -> (batch, seq)
                    ratio_pred_for_concept = (ratio_pred * one_hot(cshft.long(), model.num_c)).sum(-1)
                    ratio_pred_masked = torch.masked_select(ratio_pred_for_concept, sm)
                    ratio_target_masked = torch.masked_select(ratio_target_shifted, sm)

                    # Use MSE loss for regression (Ratio values are continuous [0, 1])
                    valid_mask = ratio_target_masked >= 0  # Filter out invalid values
                    if valid_mask.sum() > 0:
                        ratio_pred_valid = ratio_pred_masked[valid_mask]
                        ratio_target_valid = ratio_target_masked[valid_mask]
                        loss_ratio += nn.functional.mse_loss(ratio_pred_valid.float(), ratio_target_valid.float())

        # Combine losses (knowledge prediction + ratio predictions)
        loss = loss_knowledge + alpha * loss_ratio
    elif model_name in ["diskt"]:
        # DisKT output has shape [batch, seq_len-1], so adjust mask accordingly
        sm_diskt = sm[:, 1:]  # Remove first timestep from mask to match DisKT output
        y = torch.masked_select(ys[0], sm_diskt)
        t = torch.masked_select(rshft[:, 1:], sm_diskt)
        loss = binary_cross_entropy(y.double(), t.double())
    elif model_name == "dkt+":
        y_curr = torch.masked_select(ys[1], sm)
        y_next = torch.masked_select(ys[0], sm)
        r_curr = torch.masked_select(r, sm)
        r_next = torch.masked_select(rshft, sm)
        loss = binary_cross_entropy(y_next.double(), r_next.double())

        loss_r = binary_cross_entropy(y_curr.double(), r_curr.double()) # if answered wrong for C in t-1, cur answer for C should be wrong too
        loss_w1 = torch.masked_select(torch.norm(ys[2][:, 1:] - ys[2][:, :-1], p=1, dim=-1), sm[:, 1:])
        loss_w1 = loss_w1.mean() / model.num_c
        loss_w2 = torch.masked_select(torch.norm(ys[2][:, 1:] - ys[2][:, :-1], p=2, dim=-1) ** 2, sm[:, 1:])
        loss_w2 = loss_w2.mean() / model.num_c

        loss = loss + model.lambda_r * loss_r + model.lambda_w1 * loss_w1 + model.lambda_w2 * loss_w2

        # Ratio 0-3 predictions loss (auxiliary task)
        loss_ratio = 0
        if len(ys) > 3 and isinstance(ys[3], list) and dcur is not None and cshft is not None:  # Ratio predictions available
            ratio_predictions = ys[3]  # List of 4 Ratio 0-3 predictions (each shape: batch_size, seq_len, num_c)
            for i, ratio_pred in enumerate(ratio_predictions):
                ratio_target_key = f"mp_ratio_{i}"  # Ratio 0-3 target 사용
                if ratio_target_key in dcur:
                    ratio_target = dcur[ratio_target_key].to(device)
                    # Match sequence length with prediction
                    target_seq_len = ratio_pred.shape[1]
                    if ratio_target.shape[1] != target_seq_len:
                        if ratio_target.shape[1] < target_seq_len:
                            # Pad if target is shorter
                            pad_len = target_seq_len - ratio_target.shape[1]
                            ratio_target = torch.cat([ratio_target, torch.zeros(ratio_target.shape[0], pad_len, device=device)], dim=1)
                        else:
                            # Truncate if target is longer
                            ratio_target = ratio_target[:, :target_seq_len]

                    # Shift ratio target: predict next timestep's ratio (like rshft)
                    # At timestep t, we predict ratio for timestep t+1
                    ratio_target_shifted = torch.cat([ratio_target[:, 1:],
                                                     torch.zeros(ratio_target.shape[0], 1, device=device)], dim=1)

                    # Ratio prediction도 concept selection 필요: (batch, seq, num_c) -> (batch, seq)
                    ratio_pred_for_concept = (ratio_pred * one_hot(cshft.long(), model.num_c)).sum(-1)
                    ratio_pred_masked = torch.masked_select(ratio_pred_for_concept, sm)
                    ratio_target_masked = torch.masked_select(ratio_target_shifted, sm)

                    # Use MSE loss for regression (Ratio values are continuous [0, 1])
                    valid_mask = ratio_target_masked >= 0  # Filter out invalid values
                    if valid_mask.sum() > 0:
                        ratio_pred_valid = ratio_pred_masked[valid_mask]
                        ratio_target_valid = ratio_target_masked[valid_mask]
                        loss_ratio += nn.functional.mse_loss(ratio_pred_valid.float(), ratio_target_valid.float())

        # Combine losses (DKT+ losses + ratio predictions)
        loss = loss + alpha * loss_ratio
    elif model_name == "robustkt":
        # RobustKT with MP+Ratio features: multi-task (knowledge state + ratio prediction)
        y_knowledge = torch.masked_select(ys[0], sm)  # Knowledge state prediction
        t = torch.masked_select(rshft, sm)
        loss_knowledge = binary_cross_entropy(y_knowledge.double(), t.double()) + preloss[0]

        # Ratio 0-3 predictions loss (auxiliary task)
        loss_ratio = 0
        if len(ys) > 1 and isinstance(ys[1], list) and dcur is not None and cshft is not None:  # Ratio predictions available
            ratio_predictions = ys[1]  # List of 4 Ratio 0-3 predictions (each shape: batch_size, seq_len)
            for i, ratio_pred in enumerate(ratio_predictions):
                ratio_target_key = f"mp_ratio_{i}"  # Ratio 0-3 target 사용
                if ratio_target_key in dcur:
                    ratio_target = dcur[ratio_target_key].to(device)
                    # Match sequence length with prediction
                    target_seq_len = ratio_pred.shape[1]
                    if ratio_target.shape[1] != target_seq_len:
                        if ratio_target.shape[1] < target_seq_len:
                            # Pad if target is shorter
                            pad_len = target_seq_len - ratio_target.shape[1]
                            ratio_target = torch.cat([ratio_target, torch.zeros(ratio_target.shape[0], pad_len, device=device)], dim=1)
                        else:
                            # Truncate if target is longer
                            ratio_target = ratio_target[:, :target_seq_len]

                    # Shift ratio target: predict next timestep's ratio (like rshft)
                    # At timestep t, we predict ratio for timestep t+1
                    ratio_target_shifted = torch.cat([ratio_target[:, 1:],
                                                     torch.zeros(ratio_target.shape[0], 1, device=device)], dim=1)

                    # RobustKT ratio prediction은 이미 (batch, seq) 형태
                    ratio_pred_masked = torch.masked_select(ratio_pred, sm)
                    ratio_target_masked = torch.masked_select(ratio_target_shifted, sm)

                    # Use MSE loss for regression (Ratio values are continuous [0, 1])
                    valid_mask = ratio_target_masked >= 0  # Filter out invalid values
                    if valid_mask.sum() > 0:
                        ratio_pred_valid = ratio_pred_masked[valid_mask]
                        ratio_target_valid = ratio_target_masked[valid_mask]
                        loss_ratio += nn.functional.mse_loss(ratio_pred_valid.float(), ratio_target_valid.float())

        # Combine losses (knowledge prediction + ratio predictions)
        loss = loss_knowledge + alpha * loss_ratio

    elif model_name in ["akt","extrakt","folibikt", "akt_vector", "akt_norasch", "akt_mono", "akt_attn", "aktattn_pos", "aktmono_pos", "akt_raschx", "akt_raschy", "aktvec_raschx","lefokt_akt", "dtransformer", "fluckt"]:
        y = torch.masked_select(ys[0], sm)
        t = torch.masked_select(rshft, sm)
        loss_knowledge = binary_cross_entropy(y.double(), t.double()) + preloss[0]

        # Ratio 0-3 predictions loss (auxiliary task) - AKT with MP+Ratio features
        loss_ratio = 0
        if len(ys) > 1 and isinstance(ys[1], list) and dcur is not None and cshft is not None:  # Ratio predictions available
            ratio_predictions = ys[1]  # List of 4 Ratio 0-3 predictions (each shape: batch_size, seq_len, num_c)
            for i, ratio_pred in enumerate(ratio_predictions):
                ratio_target_key = f"mp_ratio_{i}"  # Ratio 0-3 target 사용
                if ratio_target_key in dcur:
                    ratio_target = dcur[ratio_target_key].to(device)
                    # Match sequence length with prediction
                    target_seq_len = ratio_pred.shape[1]
                    if ratio_target.shape[1] != target_seq_len:
                        if ratio_target.shape[1] < target_seq_len:
                            # Pad if target is shorter
                            pad_len = target_seq_len - ratio_target.shape[1]
                            ratio_target = torch.cat([ratio_target, torch.zeros(ratio_target.shape[0], pad_len, device=device)], dim=1)
                        else:
                            # Truncate if target is longer
                            ratio_target = ratio_target[:, :target_seq_len]

                    # Shift ratio target: predict next timestep's ratio (like rshft)
                    # At timestep t, we predict ratio for timestep t+1
                    ratio_target_shifted = torch.cat([ratio_target[:, 1:],
                                                     torch.zeros(ratio_target.shape[0], 1, device=device)], dim=1)

                    # Ratio prediction도 concept selection 필요: (batch, seq, num_c) -> (batch, seq)
                    ratio_pred_for_concept = (ratio_pred * one_hot(cshft.long(), model.num_c)).sum(-1)
                    ratio_pred_masked = torch.masked_select(ratio_pred_for_concept, sm)
                    ratio_target_masked = torch.masked_select(ratio_target_shifted, sm)

                    # Use MSE loss for regression (Ratio values are continuous [0, 1])
                    valid_mask = ratio_target_masked >= 0  # Filter out invalid values
                    if valid_mask.sum() > 0:
                        ratio_pred_valid = ratio_pred_masked[valid_mask]
                        ratio_target_valid = ratio_target_masked[valid_mask]
                        loss_ratio += nn.functional.mse_loss(ratio_pred_valid.float(), ratio_target_valid.float())

        # Combine losses (knowledge prediction + ratio predictions)
        loss = loss_knowledge + alpha * loss_ratio
    elif model_name == "lpkt":
        y = torch.masked_select(ys[0], sm)
        t = torch.masked_select(rshft, sm)
        criterion = nn.BCELoss(reduction='none')        
        loss = criterion(y, t).sum()
    
    return loss


def model_forward(model, data, rel=None, alpha=1.0):
    model_name = model.model_name
    # if model_name in ["dkt_forget", "lpkt"]:
    #     q, c, r, qshft, cshft, rshft, m, sm, d, dshft = data
    if model_name in ["dkt_forget", "datakt"]:
        dcur, dgaps = data
    else:
        dcur = data
    if model_name in ["dimkt"]:
        q, c, r, t,sd,qd = dcur["qseqs"].to(device), dcur["cseqs"].to(device), dcur["rseqs"].to(device), dcur["tseqs"].to(device),dcur["sdseqs"].to(device),dcur["qdseqs"].to(device)
        qshft, cshft, rshft, tshft,sdshft,qdshft = dcur["shft_qseqs"].to(device), dcur["shft_cseqs"].to(device), dcur["shft_rseqs"].to(device), dcur["shft_tseqs"].to(device),dcur["shft_sdseqs"].to(device),dcur["shft_qdseqs"].to(device)
    else:
        q, c, r, t = dcur["qseqs"].to(device), dcur["cseqs"].to(device), dcur["rseqs"].to(device), dcur["tseqs"].to(device)
        qshft, cshft, rshft, tshft = dcur["shft_qseqs"].to(device), dcur["shft_cseqs"].to(device), dcur["shft_rseqs"].to(device), dcur["shft_tseqs"].to(device)
    m, sm = dcur["masks"].to(device), dcur["smasks"].to(device)

    ys, preloss = [], []
    cq = torch.cat((q[:,0:1], qshft), dim=1)
    cc = torch.cat((c[:,0:1], cshft), dim=1)
    cr = torch.cat((r[:,0:1], rshft), dim=1)
    if model_name in ["hawkes"]:
        ct = torch.cat((t[:,0:1], tshft), dim=1)
    elif model_name in ["rkt"]:
        y, attn = model(dcur, rel, train=True)
        ys.append(y[:,1:])
    if model_name in ["atdkt"]:
        # is_repeat = dcur["is_repeat"]
        y, y2, y3 = model(dcur, train=True)
        if model.emb_type.find("bkt") == -1 and model.emb_type.find("addcshft") == -1:
            y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
        # y2 = (y2 * one_hot(cshft.long(), model.num_c)).sum(-1)
        ys = [y, y2, y3] # first: yshft
    elif model_name == "simplekt":
        # Multi-task SimpleKT output
        outputs = model(dcur, train=True)
        if len(outputs) == 4:  # SimpleKT with MP+Ratio: (y, y2, y3, ratio_predictions)
            y, y2, y3, ratio_predictions = outputs
            ys = [y[:,1:], y2, y3]
            ys.append([rp[:,1:] for rp in ratio_predictions])  # Shift ratio predictions
        else:  # Standard SimpleKT: (y, y2, y3)
            y, y2, y3 = outputs
            ys = [y[:,1:], y2, y3]
    elif model_name in ["stablekt", "sparsekt", "cskt"]:
        outputs = model(dcur, train=True)
        if model_name == "stablekt" and len(outputs) == 4:  # StableKT with MP+Ratio
            y, y2, y3, ratio_predictions = outputs
            ys = [y[:,1:], y2, y3]
            ys.append([rp[:,1:] for rp in ratio_predictions])  # Shift ratio predictions
        else:  # Standard stablekt, sparsekt, cskt: (y, y2, y3)
            y, y2, y3 = outputs
            ys = [y[:,1:], y2, y3]
    elif model_name in ["rekt"]:
        y = model(dcur, train=True)
        ys = [y]
    elif model_name in ["ukt"]:
        if model.use_CL != 0 :
            y, sim, y2, y3, temp = model(dcur, train=True)
            ys = [y[:,1:],sim,y2, y3]
        else:
            y, y2, y3 = model(dcur, train=True)
            ys = [y[:,1:], y2, y3]
    elif model_name in ["hcgkt"]:
        
        step_size = step_size
        step_m = step_m
        grad_clip = grad_clip
        mm = mm

        # the xxx.pt file of pre_load_gcn can be found in :
        # https://drive.google.com/drive/folders/1JWstsquI3TzbUlqB1EyCbjem4qPyRLCh?usp=drive_link
        matrix = None
        if dataset_name == 'assist2009':
            pre_load_gcn = "../data/assist2009/ques_skill_gcn_adj.pt"
            matrix = torch.load(pre_load_gcn)
            if not matrix.is_sparse:
                matrix = matrix.to_sparse()
        elif dataset_name == 'algebra2005':
            pre_load_gcn = "../data/algebra2005/ques_skill_gcn_adj.pt"
            matrix = torch.load(pre_load_gcn)
            if not matrix.is_sparse:
                matrix = matrix.to_sparse()
        elif dataset_name == 'bridge2algebra2006':
            pre_load_gcn = "../data/bridge2algebra2006/ques_skill_gcn_adj.pt"
            matrix = torch.load(pre_load_gcn)
            if not matrix.is_sparse:
                matrix = matrix.to_sparse()
        elif dataset_name == 'peiyou':
            pre_load_gcn = "../data/peiyou/ques_skill_gcn_adj.pt"
            matrix = torch.load(pre_load_gcn)
            if not matrix.is_sparse:
                matrix = matrix.to_sparse()
        elif dataset_name == 'nips_task34':
            pre_load_gcn = "../data/nips_task34/ques_skill_gcn_adj.pt"
            matrix = torch.load(pre_load_gcn)
            if not matrix.is_sparse:
                matrix = matrix.to_sparse()
        perturb_shape = (matrix.shape[0], emb_size)
        perturb = torch.FloatTensor(*perturb_shape).uniform_(-step_size, step_size).to(device)
        perturb.requires_grad_()
        y, y2, y3, contrast_loss = model(dcur, train=True, perb=perturb)
        ys = [y[:,1:], y2, y3]
        loss = cal_loss(model, ys, r, rshft, sm, preloss, dcur, alpha=alpha) + contrast_loss
        loss /= step_m
        opt.zero_grad()
        for _ in range(step_m - 1):
            loss.backward()
            perturb_data = perturb.detach() + step_size * torch.sign(perturb.grad.detach())
            perturb.data = perturb_data.data
            perturb.grad[:] = 0
            y, y2, y3, contrast_loss = model(dcur, train=True, perb=perturb)
            ys = [y[:,1:], y2, y3]
            loss = cal_loss(model, ys, r, rshft, sm, preloss, dcur, alpha=alpha) + contrast_loss
            loss /= step_m
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        model.sfm_cl.gcl.update_target_network(mm)  
        return loss
    elif model_name in ["dtransformer"]:
        if model.emb_type == "qid_cl":
            y, loss = model.get_cl_loss(cc.long(), cr.long(), cq.long())  # with cl loss
        else:
            y, loss = model.get_loss(cc.long(), cr.long(), cq.long())
        ys.append(y[:,1:])
        preloss.append(loss)
    elif model_name in ["datakt"]:
        y, y2, y3 = model(dcur, dgaps, train=True)
        ys = [y[:,1:], y2, y3]
    elif model_name in ["lpkt"]:
        # cat = torch.cat((d["at_seqs"][:,0:1], dshft["at_seqs"]), dim=1)
        cit = torch.cat((dcur["itseqs"][:,0:1], dcur["shft_itseqs"]), dim=1)
    if model_name == "dkt":
        # Multi-task DKT output
        outputs = model(c.long(), r.long(), dcur=dcur)
        if isinstance(outputs, tuple) and len(outputs) == 2:
            y, mp_predictions = outputs
            y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
            ys.append(y) # Knowledge state prediction
            ys.append(mp_predictions) # MP predictions
        else:
            # Fallback for single output
            y = (outputs * one_hot(cshft.long(), model.num_c)).sum(-1)
            ys.append(y)
    elif model_name in ["dkt2"]:
        # Multi-task DKT2 output
        outputs = model(c.long(), r.long(), dcur=dcur)
        if isinstance(outputs, tuple) and len(outputs) == 2:
            y, ratio_predictions = outputs
            y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
            ys.append(y)  # Knowledge state prediction
            ys.append(ratio_predictions)  # Ratio predictions
        else:
            # Fallback for single output
            y = (outputs * one_hot(cshft.long(), model.num_c)).sum(-1)
            ys.append(y)
    elif model_name in ["diskt"]:
        y = model(c.long(), r.long(), dcur=dcur)
        y = (y * one_hot(cshft[:, 1:].long(), model.num_c)).sum(-1)
        ys.append(y)
    elif model_name == "dkt+":
        # Multi-task DKT+ output
        outputs = model(c.long(), r.long(), dcur=dcur)
        if isinstance(outputs, tuple) and len(outputs) == 2:
            y, ratio_predictions = outputs
            y_next = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
            y_curr = (y * one_hot(c.long(), model.num_c)).sum(-1)
            ys = [y_next, y_curr, y]
            ys.append(ratio_predictions)  # Add ratio predictions
        else:
            # Fallback for single output
            y = outputs
            y_next = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
            y_curr = (y * one_hot(c.long(), model.num_c)).sum(-1)
            ys = [y_next, y_curr, y]
    elif model_name in ["dkt_forget"]:
        # Multi-task DKT_forget output
        outputs = model(c.long(), r.long(), dgaps, dcur=dcur)
        if isinstance(outputs, tuple) and len(outputs) == 2:
            y, ratio_predictions = outputs
            y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
            ys.append(y)  # Knowledge state prediction
            ys.append(ratio_predictions)  # Ratio predictions
        else:
            # Fallback for single output
            y = outputs
            y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
            ys.append(y)
    elif model_name == "dkvmn":
        # DKVMN with MP+Ratio features: multi-task (knowledge state + ratio prediction)
        outputs = model(cc.long(), cr.long(), dcur=dcur)
        if isinstance(outputs, tuple) and len(outputs) == 2:
            y, ratio_predictions = outputs
            ys.append(y[:,1:])  # Knowledge state prediction
            ys.append([rp[:,1:] for rp in ratio_predictions])  # Ratio predictions (shift)
        else:
            # Fallback for single output
            ys.append(outputs[:,1:])
    elif model_name == "skvmn":
        # Multi-task SKVMN output
        outputs = model(cc.long(), cr.long(), dcur=dcur)
        if isinstance(outputs, tuple) and len(outputs) == 2:
            y, ratio_predictions = outputs
            ys.append(y[:,1:])  # Knowledge state prediction
            ys.append([rp[:,1:] for rp in ratio_predictions])  # Ratio predictions (shift)
        else:
            # Fallback for single output
            ys.append(outputs[:,1:])
    elif model_name in ["deep_irt"]:
        y = model(cc.long(), cr.long())
        ys.append(y[:,1:])
    elif model_name == "sakt":
        # SAKT with MP+Ratio features: handle tuple output
        output = model(c.long(), r.long(), cshft.long(), dcur=dcur)
        if isinstance(output, tuple) and len(output) == 2:
            y, ratio_predictions = output
            ys.append(y)
            ys.append(ratio_predictions)
        else:
            ys.append(output)
    elif model_name in ["kqn"]:
        y = model(c.long(), r.long(), cshft.long())
        ys.append(y)
    elif model_name in ["saint"]:
        # Multi-task SAINT output
        outputs = model(cq.long(), cc.long(), r.long(), dcur=dcur)
        if isinstance(outputs, tuple) and len(outputs) == 2:
            y, ratio_predictions = outputs
            ys.append(y[:, 1:])  # Knowledge state prediction
            ys.append([rp[:, 1:] for rp in ratio_predictions])  # Shift ratio predictions
        else:
            # Fallback for single output
            ys.append(outputs[:, 1:])
    elif model_name in ["akt","extrakt","folibikt", "robustkt", "akt_vector", "akt_norasch", "akt_mono", "akt_attn", "aktattn_pos", "aktmono_pos", "akt_raschx", "akt_raschy", "aktvec_raschx", "lefokt_akt", "fluckt"]:
        output = model(cc.long(), cr.long(), cq.long(), dcur=dcur)
        if len(output) == 3:  # AKT with MP+Ratio: (y, reg_loss, ratio_predictions)
            y, reg_loss, ratio_predictions = output
            ys.append(y[:,1:])
            ys.append([rp[:,1:] for rp in ratio_predictions])  # Shift ratio predictions
            preloss.append(reg_loss)
        else:  # Standard AKT: (y, reg_loss)
            y, reg_loss = output
            ys.append(y[:,1:])
            preloss.append(reg_loss)
    elif model_name in ["atkt", "atktfix"]:
        # ATKT with MP+Ratio features: handle tuple output
        output = model(c.long(), r.long(), dcur=dcur)
        if len(output) == 3:  # ATKT with MP+Ratio: (y, features, ratio_predictions)
            y, features, ratio_predictions = output
            y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
            loss = cal_loss(model, [y, ratio_predictions], r, rshft, sm, dcur=dcur, cshft=cshft, alpha=alpha)
            # at
            features_grad = grad(loss, features, retain_graph=True)
            p_adv = torch.FloatTensor(model.epsilon * _l2_normalize_adv(features_grad[0].data))
            p_adv = Variable(p_adv).to(device)
            pred_res_output = model(c.long(), r.long(), p_adv, dcur=dcur)
            # second loss - only use first two outputs (ignore ratio predictions in adversarial pass)
            pred_res = pred_res_output[0]
            pred_res = (pred_res * one_hot(cshft.long(), model.num_c)).sum(-1)
            adv_loss = cal_loss(model, [pred_res], r, rshft, sm, alpha=alpha)
            loss = loss + model.beta * adv_loss
        else:
            y, features = output
            y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
            loss = cal_loss(model, [y], r, rshft, sm, alpha=alpha)
            # at
            features_grad = grad(loss, features, retain_graph=True)
            p_adv = torch.FloatTensor(model.epsilon * _l2_normalize_adv(features_grad[0].data))
            p_adv = Variable(p_adv).to(device)
            pred_res, _ = model(c.long(), r.long(), p_adv, dcur=dcur)
            # second loss
            pred_res = (pred_res * one_hot(cshft.long(), model.num_c)).sum(-1)
            adv_loss = cal_loss(model, [pred_res], r, rshft, sm, alpha=alpha)
            loss = loss + model.beta * adv_loss
    elif model_name == "gkt":
        # Multi-task GKT output
        outputs = model(cc.long(), cr.long(), dcur=dcur)
        if isinstance(outputs, tuple) and len(outputs) == 2:
            y, ratio_predictions = outputs
            ys.append(y)  # Knowledge state prediction
            ys.append(ratio_predictions)  # Ratio predictions
        else:
            # Fallback for single output
            ys.append(outputs)  
    # cal loss
    elif model_name == "lpkt":
        # y = model(cq.long(), cr.long(), cat, cit.long())
        y = model(cq.long(), cr.long(), cit.long())
        ys.append(y[:, 1:])  
    elif model_name == "hawkes":
        # ct = torch.cat((dcur["tseqs"][:,0:1], dcur["shft_tseqs"]), dim=1)
        # csm = torch.cat((dcur["smasks"][:,0:1], dcur["smasks"]), dim=1)
        # y = model(cc[0:1,0:5].long(), cq[0:1,0:5].long(), ct[0:1,0:5].long(), cr[0:1,0:5].long(), csm[0:1,0:5].long())
        y = model(cc.long(), cq.long(), ct.long(), cr.long())#, csm.long())
        ys.append(y[:, 1:])
    elif model_name in que_type_models and model_name not in ["lpkt", "rkt"]:
        y,loss = model.train_one_step(data)
    elif model_name == "dimkt":
        y = model(q.long(),c.long(),sd.long(),qd.long(),r.long(),qshft.long(),cshft.long(),sdshft.long(),qdshft.long())
        ys.append(y) 

    if model_name not in ["atkt", "atktfix"]+que_type_models or model_name in ["lpkt", "rkt"]:
        loss = cal_loss(model, ys, r, rshft, sm, preloss, dcur, cshft, alpha=alpha)
    if model_name in ["ukt"] and model.use_CL != 0:
        return loss,temp
    return loss
    

def train_model(model, train_loader, valid_loader, num_epochs, opt, ckpt_path, test_loader=None, test_window_loader=None, save_model=False, data_config=None, fold=None, alpha=1.0):
    max_auc, best_epoch = 0, -1
    train_step = 0

    rel = None
    if model.model_name == "rkt":
        dpath = data_config["dpath"]
        dataset_name = dpath.split("/")[-1]
        tmp_folds = set(data_config["folds"]) - {fold}
        folds_str = "_" + "_".join([str(_) for _ in tmp_folds])
        if dataset_name in ["algebra2005", "bridge2algebra2006"]:
            fname = "phi_dict" + folds_str + ".pkl"
            rel = pd.read_pickle(os.path.join(dpath, fname))
        else:
            fname = "phi_array" + folds_str + ".pkl" 
            rel = pd.read_pickle(os.path.join(dpath, fname))

    if model.model_name=='lpkt':
        scheduler = torch.optim.lr_scheduler.StepLR(opt, 10, gamma=0.5)
    for i in range(1, num_epochs + 1):
        loss_mean = []
        for data in train_loader:
            train_step+=1
            if model.model_name in que_type_models and model.model_name not in ["lpkt", "rkt"]:
                model.model.train()
            else:
                model.train()
            if model.model_name=='rkt':
                loss = model_forward(model, data, rel, alpha=alpha)
            elif model.model_name in ["ukt"] and model.use_CL != 0:
                loss,temp = model_forward(model, data, alpha=alpha)
            else:
                loss = model_forward(model, data, alpha=alpha)
            opt.zero_grad()
            loss.backward()#compute gradients
            if model.model_name == "rkt":
                clip_grad_norm_(model.parameters(), model.grad_clip)
            if model.model_name == "dtransformer":
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()#update model’s parameters
                
            loss_mean.append(loss.detach().cpu().numpy())
            if model.model_name == "gkt" and train_step%10==0:
                text = f"Total train step is {train_step}, the loss is {loss.item():.5}"
                debug_print(text = text,fuc_name="train_model")
        if model.model_name=='lpkt':
            scheduler.step()#update each epoch
        loss_mean = np.mean(loss_mean)
        
        if model.model_name=='rkt':
            auc, acc = evaluate(model, valid_loader, model.model_name, rel)
        else:
            auc, acc = evaluate(model, valid_loader, model.model_name)
        ### atkt 有diff， 以下代码导致的
        ### auc, acc = round(auc, 4), round(acc, 4)

        if auc > max_auc+1e-3:
            if save_model:
                torch.save(model.state_dict(), os.path.join(ckpt_path, model.emb_type+"_model.ckpt"))
            max_auc = auc
            best_epoch = i
            testauc, testacc = -1, -1
            window_testauc, window_testacc = -1, -1
            if not save_model:
                if test_loader != None:
                    save_test_path = os.path.join(ckpt_path, model.emb_type+"_test_predictions.txt")
                    testauc, testacc = evaluate(model, test_loader, model.model_name, save_test_path)
                if test_window_loader != None:
                    save_test_path = os.path.join(ckpt_path, model.emb_type+"_test_window_predictions.txt")
                    window_testauc, window_testacc = evaluate(model, test_window_loader, model.model_name, save_test_path)
            validauc, validacc = auc, acc
        print(f"Epoch: {i}, validauc: {validauc:.4}, validacc: {validacc:.4}, best epoch: {best_epoch}, best auc: {max_auc:.4}, train loss: {loss_mean}, emb_type: {model.emb_type}, model: {model.model_name}, save_dir: {ckpt_path}")
        print(f"            testauc: {round(testauc,4)}, testacc: {round(testacc,4)}, window_testauc: {round(window_testauc,4)}, window_testacc: {round(window_testacc,4)}")


        if i - best_epoch >= 10:
            break
    return testauc, testacc, window_testauc, window_testacc, validauc, validacc, best_epoch
