#!/usr/bin/env python
# coding=utf-8

import os, sys
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from collections import defaultdict
import random

class DisKTDataset(Dataset):
    """
    Dataset for DisKT model with specialized data processing
    - Calculates skill difficulty
    - Generates counter attention masks
    - Supports augmentation for contradiction training
    """
    def __init__(self, file_path, input_type, folds, seq_len=200, 
                 use_counter_mask=True, counter_threshold=0.3, qtest=False):
        super(DisKTDataset, self).__init__()
        self.file_path = file_path
        self.input_type = input_type
        self.folds = sorted(list(folds))
        self.seq_len = seq_len
        self.use_counter_mask = use_counter_mask
        self.counter_threshold = counter_threshold
        self.qtest = qtest
        
        folds_str = "_" + "_".join([str(_) for _ in self.folds])
        if self.qtest:
            processed_data = file_path + folds_str + "_diskt_qtest.pkl"
        else:
            processed_data = file_path + folds_str + "_diskt.pkl"
            
        if not os.path.exists(processed_data):
            print(f"Start preprocessing DisKT data {file_path} fold: {folds_str}...")
            self.__preprocess_data__()
            save_data = {
                'questions': self.padded_q,
                'skills': self.padded_s, 
                'responses': self.padded_r,
                'attention_mask': self.attention_mask,
                'counter_attention_mask': self.counter_attention_mask,
                'skill_difficulty': self.skill_difficulty,
                'easier_skills': self.easier_skills,
                'harder_skills': self.harder_skills,
                'num_skills': self.num_skills,
                'num_questions': self.num_questions,
                'error_types': self.padded_ed,  # Add ED features to saved data
                'math_prof': self.padded_mp  # Add MP features to saved data
            }
            pd.to_pickle(save_data, processed_data)
        else:
            print(f"Read DisKT data from processed file: {processed_data}")
            data = pd.read_pickle(processed_data)
            self.padded_q = data['questions']
            self.padded_s = data['skills']
            self.padded_r = data['responses']
            self.attention_mask = data['attention_mask']
            self.counter_attention_mask = data['counter_attention_mask']
            self.skill_difficulty = data['skill_difficulty']
            self.easier_skills = data['easier_skills']
            self.harder_skills = data['harder_skills']
            self.num_skills = data['num_skills']
            self.num_questions = data['num_questions']
            # Load ED features if available (backward compatibility)
            self.padded_ed = data.get('error_types', [torch.zeros_like(self.padded_r) for _ in range(5)])
            # Load MP features if available (backward compatibility)
            self.padded_mp = data.get('math_prof', [torch.zeros_like(self.padded_r) for _ in range(4)])
            
        self.len = len(self.padded_q)
        
    def __preprocess_data__(self):
        """Preprocess data for DisKT training"""
        # Load original data
        df_all = pd.read_csv(self.file_path)
        if "fold" in df_all.columns:
            if self.qtest:
                df = df_all[df_all["fold"] == -1].copy()
            else:
                df = df_all[df_all["fold"].isin(self.folds)].copy()
        else:
            df = df_all.copy()
            
        # Process sequences
        self.__extract_sequences__(df)
        self.__calculate_skill_difficulty__()
        self.__generate_counter_attention_mask__()
        self.__pad_sequences__()
        
    def __extract_sequences__(self, df):
        """Extract question/skill/response sequences for each user"""
        self.questions = []
        self.skills = []
        self.responses = []
        self.error_types = [[] for _ in range(5)]  # error_type_0 to error_type_4
        self.math_prof = [[] for _ in range(4)]  # math_prof_0 to math_prof_3
        
        for _, row in df.iterrows():
            q_seq = [int(q) for q in row['questions'].split(',') if q != '-1']
            s_seq = [int(s) for s in row['concepts'].split(',') if s != '-1']
            r_seq = [int(r) for r in row['responses'].split(',') if r != '-1']
            
            # Extract ED features if available
            ed_seqs = []
            for i in range(5):
                error_col = f'error_type_{i}'
                if error_col in row and pd.notna(row[error_col]) and row[error_col] != '':
                    ed_seq = [float(e) for e in str(row[error_col]).split(',') if e not in ['-1', '']]
                else:
                    ed_seq = [0.0] * len(r_seq)  # Default to zeros if ED not available
                ed_seqs.append(ed_seq)

            mp_seqs = []
            for i in range(4):
                mp_col = f'math_prof_{i}'
                if mp_col in row and pd.notna(row[mp_col]) and row[mp_col] != '':
                    mp_seq = [float(e) for e in str(row[mp_col]).split(',') if e not in ['-1', '']]
                else:
                    mp_seq = [0.0] * len(r_seq)  # Default to zeros if MP not available
                mp_seqs.append(mp_seq)
            
            # Limit sequence length
            if len(q_seq) > self.seq_len:
                q_seq = q_seq[-self.seq_len:]
                s_seq = s_seq[-self.seq_len:]
                r_seq = r_seq[-self.seq_len:]
                for i in range(5):
                    if len(ed_seqs[i]) > self.seq_len:
                        ed_seqs[i] = ed_seqs[i][-self.seq_len:]
                for i in range(4):
                    if len(mp_seqs[i]) > self.seq_len:
                        mp_seqs[i] = mp_seqs[i][-self.seq_len:]
            
            # Pad ED sequences to match response length if needed
            for i in range(5):
                while len(ed_seqs[i]) < len(r_seq):
                    ed_seqs[i].append(0.0)
                ed_seqs[i] = ed_seqs[i][:len(r_seq)]  # Truncate if too long

            # Pad MP sequences to match response length if needed
            for i in range(4):
                while len(mp_seqs[i]) < len(r_seq):
                    mp_seqs[i].append(0.0)
                mp_seqs[i] = mp_seqs[i][:len(r_seq)]  # Truncate if too long
                
            self.questions.append(q_seq)
            self.skills.append(s_seq)
            self.responses.append(r_seq)
            for i in range(5):
                self.error_types[i].append(ed_seqs[i])
            for i in range(4):
                self.math_prof[i].append(mp_seqs[i])
            
        self.num_skills = max([max(seq) for seq in self.skills if seq]) + 1
        self.num_questions = max([max(seq) for seq in self.questions if seq]) + 1
        
    def __calculate_skill_difficulty__(self):
        """Calculate skill difficulty based on historical correct rates"""
        skill_correct = defaultdict(int)
        skill_count = defaultdict(int)
        
        for s_list, r_list in zip(self.skills, self.responses):
            for s, r in zip(s_list, r_list):
                if s > 0:  # Valid skill
                    skill_correct[s] += r
                    skill_count[s] += 1
                    
        # Calculate correct rates
        self.skill_difficulty = {}
        for s in skill_correct:
            if skill_count[s] > 0:
                self.skill_difficulty[s] = skill_correct[s] / float(skill_count[s])
            else:
                self.skill_difficulty[s] = 0.5  # Default for unseen skills
                
        # Create easier/harder skill mappings
        ordered_skills = [
            item[0] for item in sorted(self.skill_difficulty.items(), key=lambda x: x[1])
        ]
        
        self.easier_skills = {}
        self.harder_skills = {}
        
        for i, s in enumerate(ordered_skills):
            if i == 0:  # Hardest skill
                self.easier_skills[s] = ordered_skills[min(i + 1, len(ordered_skills) - 1)]
                self.harder_skills[s] = s
            elif i == len(ordered_skills) - 1:  # Easiest skill
                self.easier_skills[s] = s
                self.harder_skills[s] = ordered_skills[max(i - 1, 0)]
            else:
                self.easier_skills[s] = ordered_skills[i + 1]  # More correct = easier
                self.harder_skills[s] = ordered_skills[i - 1]  # Less correct = harder
                
    def __generate_counter_attention_mask__(self):
        """Generate counter attention mask for contradiction detection"""
        self.counter_masks = []
        
        for s_list, r_list in zip(self.skills, self.responses):
            counter_mask = []
            
            for i, (s, r) in enumerate(zip(s_list, r_list)):
                # Check if this response contradicts the skill difficulty expectation
                if s in self.skill_difficulty:
                    expected_difficulty = self.skill_difficulty[s]
                    
                    # If skill is easy but response is wrong, or skill is hard but response is correct
                    # This might indicate guessing or mistaking
                    contradiction = False
                    if expected_difficulty > (1 - self.counter_threshold) and r == 0:  # Easy skill, wrong answer
                        contradiction = True
                    elif expected_difficulty < self.counter_threshold and r == 1:  # Hard skill, correct answer
                        contradiction = True
                        
                    counter_mask.append(1 if contradiction else 0)
                else:
                    counter_mask.append(0)
                    
            self.counter_masks.append(counter_mask)
            
    def __pad_sequences__(self):
        """Pad sequences to fixed length"""
        self.padded_q = torch.zeros((len(self.questions), self.seq_len), dtype=torch.long)
        self.padded_s = torch.zeros((len(self.skills), self.seq_len), dtype=torch.long) 
        self.padded_r = torch.full((len(self.responses), self.seq_len), -1, dtype=torch.long)
        self.attention_mask = torch.zeros((len(self.skills), self.seq_len), dtype=torch.long)
        self.counter_attention_mask = torch.zeros((len(self.skills), self.seq_len), dtype=torch.long)
        self.padded_ed = [torch.zeros((len(self.questions), self.seq_len), dtype=torch.float) for _ in range(5)]
        self.padded_mp = [torch.zeros((len(self.questions), self.seq_len), dtype=torch.float) for _ in range(4)]
        
        for i, (q_seq, s_seq, r_seq, c_mask) in enumerate(
            zip(self.questions, self.skills, self.responses, self.counter_masks)
        ):
            seq_len = len(q_seq)
            if seq_len > 0:
                # Right-align sequences (recent interactions at the end)
                self.padded_q[i, -seq_len:] = torch.tensor(q_seq, dtype=torch.long)
                self.padded_s[i, -seq_len:] = torch.tensor(s_seq, dtype=torch.long)
                self.padded_r[i, -seq_len:] = torch.tensor(r_seq, dtype=torch.long)
                self.attention_mask[i, -seq_len:] = torch.ones(seq_len, dtype=torch.long)
                if self.use_counter_mask:
                    self.counter_attention_mask[i, -seq_len:] = torch.tensor(c_mask, dtype=torch.long)
                for j in range(5):
                    if i < len(self.error_types[j]) and len(self.error_types[j][i]) >= seq_len:
                        self.padded_ed[j][i, -seq_len:] = torch.tensor(self.error_types[j][i][:seq_len], dtype=torch.float)
                for j in range(4):
                    if i < len(self.math_prof[j]) and len(self.math_prof[j][i]) >= seq_len:
                        self.padded_mp[j][i, -seq_len:] = torch.tensor(self.math_prof[j][i][:seq_len], dtype=torch.float)
                    
    def __getitem__(self, index):
        # Return data with both DisKT keys and pykt-toolkit compatible keys
        data = {
            # DisKT specific keys
            "questions": self.padded_q[index],
            "skills": self.padded_s[index], 
            "responses": self.padded_r[index],
            "attention_mask": (self.counter_attention_mask[index], self.attention_mask[index]),
            
            # pykt-toolkit compatible keys
            "qseqs": self.padded_q[index],
            "cseqs": self.padded_s[index],
            "rseqs": self.padded_r[index],
            "tseqs": torch.arange(self.seq_len, dtype=torch.long),  # Mock time sequence
            
            # ED features (error_type_0 to error_type_4)
            "error_type_0": self.padded_ed[0][index],
            "error_type_1": self.padded_ed[1][index],
            "error_type_2": self.padded_ed[2][index],
            "error_type_3": self.padded_ed[3][index],
            "error_type_4": self.padded_ed[4][index],

            # MP features (math_prof_0 to math_prof_3)
            "math_prof_0": self.padded_mp[0][index],
            "math_prof_1": self.padded_mp[1][index],
            "math_prof_2": self.padded_mp[2][index],
            "math_prof_3": self.padded_mp[3][index],
        }
        
        # Add shifted sequences for compatibility
        seq_len = data["qseqs"].shape[0]
        data.update({
            "shft_qseqs": torch.cat([data["qseqs"][1:], torch.tensor([0])]),
            "shft_cseqs": torch.cat([data["cseqs"][1:], torch.tensor([0])]), 
            "shft_rseqs": torch.cat([data["rseqs"][1:], torch.tensor([-1])]),
            "shft_tseqs": torch.cat([data["tseqs"][1:], torch.tensor([seq_len])]),
            "masks": (data["rseqs"] != -1).long(),
            "smasks": torch.cat([(data["rseqs"][1:] != -1), torch.tensor([False])])
        })
        
        return data
        
    def __len__(self):
        return self.len


class DisKTAugmentedDataset(Dataset):
    """
    Augmented dataset wrapper for DisKT with contrastive learning support
    """
    def __init__(self, base_dataset, mask_prob=0.15, crop_prob=0.1, 
                 permute_prob=0.1, replace_prob=0.1, negative_prob=0.1):
        self.base_dataset = base_dataset
        self.mask_prob = mask_prob
        self.crop_prob = crop_prob
        self.permute_prob = permute_prob
        self.replace_prob = replace_prob
        self.negative_prob = negative_prob
        
        # Get dataset properties
        self.num_skills = base_dataset.num_skills
        self.num_questions = base_dataset.num_questions
        self.skill_difficulty = base_dataset.skill_difficulty
        self.easier_skills = base_dataset.easier_skills
        self.harder_skills = base_dataset.harder_skills
        self.seq_len = base_dataset.seq_len
        
        self.s_mask_id = self.num_skills + 1
        self.q_mask_id = self.num_questions + 1
        
    def __len__(self):
        return len(self.base_dataset)
        
    def __getitem__(self, index):
        original_data = self.base_dataset[index]
        
        # Extract sequences
        q_seq = original_data["questions"].tolist()
        s_seq = original_data["skills"].tolist()
        r_seq = original_data["responses"].tolist()
        counter_mask, attention_mask = original_data["attention_mask"]
        
        # Apply augmentations
        aug_q, aug_s, aug_r = self.__augment_sequence__(q_seq, s_seq, r_seq, seed=index)
        
        # Return data with both DisKT keys and pykt-toolkit compatible keys
        aug_q_tensor = torch.tensor(aug_q, dtype=torch.long)
        aug_s_tensor = torch.tensor(aug_s, dtype=torch.long)
        aug_r_tensor = torch.tensor(aug_r, dtype=torch.long)
        
        data = {
            # DisKT specific keys
            "questions": aug_q_tensor,
            "skills": aug_s_tensor,
            "responses": aug_r_tensor,
            "attention_mask": (counter_mask, attention_mask),
            
            # pykt-toolkit compatible keys
            "qseqs": aug_q_tensor,
            "cseqs": aug_s_tensor,
            "rseqs": aug_r_tensor,
            "tseqs": torch.arange(self.seq_len, dtype=torch.long),
            
            # ED features from original data
            "error_type_0": original_data["error_type_0"],
            "error_type_1": original_data["error_type_1"],
            "error_type_2": original_data["error_type_2"],
            "error_type_3": original_data["error_type_3"],
            "error_type_4": original_data["error_type_4"],

            # MP features from original data
            "math_prof_0": original_data["math_prof_0"],
            "math_prof_1": original_data["math_prof_1"],
            "math_prof_2": original_data["math_prof_2"],
            "math_prof_3": original_data["math_prof_3"],
        }
        
        # Add shifted sequences for compatibility
        seq_len = data["qseqs"].shape[0]
        data.update({
            "shft_qseqs": torch.cat([data["qseqs"][1:], torch.tensor([0])]),
            "shft_cseqs": torch.cat([data["cseqs"][1:], torch.tensor([0])]),
            "shft_rseqs": torch.cat([data["rseqs"][1:], torch.tensor([-1])]),
            "shft_tseqs": torch.cat([data["tseqs"][1:], torch.tensor([seq_len])]),
            "masks": (data["rseqs"] != -1).long(),
            "smasks": torch.cat([(data["rseqs"][1:] != -1), torch.tensor([False])])
        })
        
        return data
        
    def __augment_sequence__(self, q_seq, s_seq, r_seq, seed=None):
        """Apply various augmentation techniques"""
        rng = random.Random(seed)
        
        aug_q = []
        aug_s = []
        aug_r = []
        
        for i, (q, s, r) in enumerate(zip(q_seq, s_seq, r_seq)):
            if s <= 0 or r < 0:  # Padding or invalid
                aug_q.append(q)
                aug_s.append(s)
                aug_r.append(r)
                continue
                
            # Masking augmentation
            if rng.random() < self.mask_prob:
                prob = rng.random()
                if prob < 0.8:  # Mask token
                    aug_q.append(self.q_mask_id)
                    aug_s.append(self.s_mask_id)
                elif prob < 0.9:  # Random token
                    aug_q.append(rng.randint(1, self.num_questions))
                    aug_s.append(rng.randint(1, self.num_skills))
                else:  # Keep original
                    aug_q.append(q)
                    aug_s.append(s)
                aug_r.append(r)
                continue
                
            # Skill replacement based on difficulty
            if rng.random() < self.replace_prob and s in self.skill_difficulty:
                if r == 0 and s in self.harder_skills:  # Wrong answer -> harder skill
                    aug_s.append(self.harder_skills[s])
                elif r == 1 and s in self.easier_skills:  # Correct answer -> easier skill  
                    aug_s.append(self.easier_skills[s])
                else:
                    aug_s.append(s)
                aug_q.append(q)
                aug_r.append(r)
                continue
                
            # Response negation
            if rng.random() < self.negative_prob:
                aug_r.append(1 - r)  # Flip response
            else:
                aug_r.append(r)
                
            aug_q.append(q)
            aug_s.append(s)
            
        return aug_q, aug_s, aug_r