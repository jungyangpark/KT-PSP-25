#!/usr/bin/env python
# coding=utf-8

import os, sys
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np

if torch.cuda.is_available():
    from torch.cuda import FloatTensor, LongTensor
else:
    from torch import FloatTensor, LongTensor

class KTDataset(Dataset):
    """Dataset for KT
        can use to init dataset for: (for models except dkt_forget)
            train data, valid data
            common test data(concept level evaluation), real educational scenario test data(question level evaluation).
    Args:
        file_path (str): train_valid/test file path
        input_type (list[str]): the input type of the dataset, values are in ["questions", "concepts"]
        folds (set(int)): the folds used to generate dataset, -1 for test data
        qtest (bool, optional): is question evaluation or not. Defaults to False.
        cold_start_len (int, optional): if > 0, only use first N elements for cold-start evaluation. Defaults to 0.
    """
    def __init__(self, file_path, input_type, folds, qtest=False, cold_start_len=0):
        super(KTDataset, self).__init__()
        sequence_path = file_path
        self.input_type = input_type
        self.qtest = qtest
        self.cold_start_len = cold_start_len  # Cold-start: only use first N elements
        folds = sorted(list(folds))
        folds_str = "_" + "_".join([str(_) for _ in folds])
        if self.qtest:
            processed_data = file_path + folds_str + "_qtest.pkl"
        else:
            processed_data = file_path + folds_str + ".pkl"

        if not os.path.exists(processed_data):
            print(f"Start preprocessing {file_path} fold: {folds_str}...")
            if self.qtest:
                self.dori, self.dqtest = self.__load_data__(sequence_path, folds)
                save_data = [self.dori, self.dqtest]
            else:
                self.dori = self.__load_data__(sequence_path, folds)
                save_data = self.dori
            pd.to_pickle(save_data, processed_data)
        else:
            print(f"Read data from processed file: {processed_data}")
            if self.qtest:
                self.dori, self.dqtest = pd.read_pickle(processed_data)
            else:
                self.dori = pd.read_pickle(processed_data)
                for key in self.dori:
                    self.dori[key] = self.dori[key]#[:100]
        print(f"file path: {file_path}, qlen: {len(self.dori['qseqs'])}, clen: {len(self.dori['cseqs'])}, rlen: {len(self.dori['rseqs'])}")

    def __len__(self):
        """return the dataset length
        Returns:
            int: the length of the dataset
        """
        return len(self.dori["rseqs"])

    def __getitem__(self, index):
        """
        Args:
            index (int): the index of the data want to get
        Returns:
            (tuple): tuple containing:
            
            - **q_seqs (torch.tensor)**: question id sequence of the 0~seqlen-2 interactions
            - **c_seqs (torch.tensor)**: knowledge concept id sequence of the 0~seqlen-2 interactions
            - **r_seqs (torch.tensor)**: response id sequence of the 0~seqlen-2 interactions
            - **qshft_seqs (torch.tensor)**: question id sequence of the 1~seqlen-1 interactions
            - **cshft_seqs (torch.tensor)**: knowledge concept id sequence of the 1~seqlen-1 interactions
            - **rshft_seqs (torch.tensor)**: response id sequence of the 1~seqlen-1 interactions
            - **mask_seqs (torch.tensor)**: masked value sequence, shape is seqlen-1
            - **select_masks (torch.tensor)**: is select to calculate the performance or not, 0 is not selected, 1 is selected, only available for 1~seqlen-1, shape is seqlen-1
            - **dcur (dict)**: used only self.qtest is True, for question level evaluation
        """
        dcur = dict()
        mseqs = self.dori["masks"][index]
        for key in self.dori:
            if key in ["masks", "smasks"]:
                continue
            if len(self.dori[key]) == 0:
                dcur[key] = self.dori[key]
                dcur["shft_"+key] = self.dori[key]
                continue
            # print(f"key: {key}, len: {len(self.dori[key])}")
            
            if key.startswith("error_type_"):
                dcur[key] = self.dori[key][index]
                continue
            if key.startswith("math_prof_"):
                dcur[key] = self.dori[key][index]
                continue
            if key.startswith("mp_ratio_"):
                dcur[key] = self.dori[key][index]
                continue
                
            seqs = self.dori[key][index][:-1] * mseqs
            shft_seqs = self.dori[key][index][1:] * mseqs
            dcur[key] = seqs
            dcur["shft_"+key] = shft_seqs
        dcur["masks"] = mseqs
        dcur["smasks"] = self.dori["smasks"][index]

        # Cold-start evaluation: only use first N elements
        if self.cold_start_len > 0:
            n = self.cold_start_len
            for key in dcur:
                if isinstance(dcur[key], torch.Tensor) and dcur[key].dim() >= 1:
                    # Truncate sequence to first n elements
                    dcur[key] = dcur[key][:n]

        # print("tseqs", dcur["tseqs"])
        if not self.qtest:
            return dcur
        else:
            dqtest = dict()
            for key in self.dqtest:
                dqtest[key] = self.dqtest[key][index]
                # Also truncate dqtest for cold-start
                if self.cold_start_len > 0:
                    if isinstance(dqtest[key], torch.Tensor) and dqtest[key].dim() >= 1:
                        dqtest[key] = dqtest[key][:self.cold_start_len]
            return dcur, dqtest

    def __load_data__(self, sequence_path, folds, pad_val=-1):
        """
        Args:
            sequence_path (str): file path of the sequences
            folds (list[int]): 
            pad_val (int, optional): pad value. Defaults to -1.
        Returns: 
            (tuple): tuple containing
            - **q_seqs (torch.tensor)**: question id sequence of the 0~seqlen-1 interactions
            - **c_seqs (torch.tensor)**: knowledge concept id sequence of the 0~seqlen-1 interactions
            - **r_seqs (torch.tensor)**: response id sequence of the 0~seqlen-1 interactions
            - **mask_seqs (torch.tensor)**: masked value sequence, shape is seqlen-1
            - **select_masks (torch.tensor)**: is select to calculate the performance or not, 0 is not selected, 1 is selected, only available for 1~seqlen-1, shape is seqlen-1
            - **dqtest (dict)**: not null only self.qtest is True, for question level evaluation
        """
        dori = {"qseqs": [], "cseqs": [], "rseqs": [], "tseqs": [], "utseqs": [], "smasks": [],
                "error_type_0": [], "error_type_1": [], "error_type_2": [], "error_type_3": [], "error_type_4": [],
                "math_prof_0": [], "math_prof_1": [], "math_prof_2": [], "math_prof_3": [], "math_prof_4": [], "math_prof_5": [], "math_prof_6": [], "math_prof_7": [],
                "mp_ratio_0": [], "mp_ratio_1": [], "mp_ratio_2": [], "mp_ratio_3": []}

        # seq_qids, seq_cids, seq_rights, seq_mask = [], [], [], []
        df = pd.read_csv(sequence_path)#[0:1000]
        df = df[df["fold"].isin(folds)]
        interaction_num = 0
        # seq_qidxs, seq_rests = [], []
        dqtest = {"qidxs": [], "rests":[], "orirow":[]}
        for i, row in df.iterrows():
            #use kc_id or question_id as input
            if "concepts" in self.input_type:
                dori["cseqs"].append([int(_) for _ in row["concepts"].split(",")])
            if "questions" in self.input_type:
                dori["qseqs"].append([int(_) for _ in row["questions"].split(",")])
            if "timestamps" in row:
                dori["tseqs"].append([int(_) for _ in row["timestamps"].split(",")])
            if "usetimes" in row:
                dori["utseqs"].append([int(_) for _ in row["usetimes"].split(",")])
                
            dori["rseqs"].append([int(_) for _ in row["responses"].split(",")])
            dori["smasks"].append([int(_) for _ in row["selectmasks"].split(",")])
            
            # Extract ED features (error_type_0 to error_type_4)
            for i in range(5):
                error_col = f'error_type_{i}'
                if error_col in row and pd.notna(row[error_col]) and row[error_col] != '':
                    ed_seq = [float(e) for e in str(row[error_col]).split(',') if e not in ['-1', '']]
                    while len(ed_seq) < len(dori["rseqs"][-1]):
                        ed_seq.append(0.0)
                    ed_seq = ed_seq[:len(dori["rseqs"][-1])]
                else:
                    ed_seq = [0.0] * len(dori["rseqs"][-1])
                dori[error_col].append(ed_seq)

            # Extract MP features (math_prof_0 to math_prof_7)
            mp_sequences = []
            for i in range(8):
                mp_col = f'math_prof_{i}'
                if mp_col in row and pd.notna(row[mp_col]) and row[mp_col] != '':
                    mp_seq = [float(e) for e in str(row[mp_col]).split(',') if e not in ['-1', '']]
                    while len(mp_seq) < len(dori["rseqs"][-1]):
                        mp_seq.append(0.0)
                    mp_seq = mp_seq[:len(dori["rseqs"][-1])]
                else:
                    mp_seq = [0.0] * len(dori["rseqs"][-1])
                dori[mp_col].append(mp_seq)
                mp_sequences.append(mp_seq)

            # Calculate MP ratios: MP4-7 / MP0-3 (fulfillment rate)
            for i in range(4):
                ratio_col = f'mp_ratio_{i}'
                mp_numerator = mp_sequences[i + 4]  # MP4-7
                mp_denominator = mp_sequences[i]    # MP0-3

                ratio_seq = []
                for num, denom in zip(mp_numerator, mp_denominator):
                    # Avoid division by zero: add small epsilon
                    if denom > 1e-6:
                        ratio = num / denom
                    else:
                        ratio = 0.0  # If requirement is 0, ratio is undefined -> set to 0
                    ratio_seq.append(ratio)

                dori[ratio_col].append(ratio_seq)

            interaction_num += dori["smasks"][-1].count(1)

            if self.qtest:
                dqtest["qidxs"].append([int(_) for _ in row["qidxs"].split(",")])
                dqtest["rests"].append([int(_) for _ in row["rest"].split(",")])
                dqtest["orirow"].append([int(_) for _ in row["orirow"].split(",")])
        for key in dori:
            if key not in ["rseqs", "error_type_0", "error_type_1", "error_type_2", "error_type_3", "error_type_4",
                            "math_prof_0", "math_prof_1", "math_prof_2", "math_prof_3", "math_prof_4", "math_prof_5", "math_prof_6", "math_prof_7",
                            "mp_ratio_0", "mp_ratio_1", "mp_ratio_2", "mp_ratio_3"]:
                dori[key] = LongTensor(dori[key])
            else:
                dori[key] = FloatTensor(dori[key])

        mask_seqs = (dori["cseqs"][:,:-1] != pad_val) * (dori["cseqs"][:,1:] != pad_val)
        dori["masks"] = mask_seqs

        dori["smasks"] = (dori["smasks"][:, 1:] != pad_val)
        print(f"interaction_num: {interaction_num}")
        # print("load data tseqs: ", dori["tseqs"])

        if self.qtest:
            for key in dqtest:
                dqtest[key] = LongTensor(dqtest[key])[:, 1:]
            
            return dori, dqtest
        return dori
