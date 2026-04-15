# StatusKT: Tracing Mathematical Proficiency Through Problem-Solving Processes

[![Paper](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2512.00311)
[![Dataset](https://img.shields.io/badge/🤗-Dataset-yellow)](https://huggingface.co/datasets/jungypark/KT-PSP-25)
[![ACL 2026 Findings](https://img.shields.io/badge/ACL%20Findings-2026-blue)](https://2026.aclweb.org/)

> **Tracing Mathematical Proficiency Through Problem-Solving Processes**

Official implementation of the StatusKT framework accepted at **ACL 2026 Findings**.

*Jungyang Park, Suho Kang, Jaewoo Park, Jaehong Kim, Jaewoo Shin, Seonjoon Park, Youngjae Yu*

Yonsei University, Mathpresso, Seoul National University

## Abstract

Knowledge Tracing (KT) aims to model student's knowledge state and predict future performance to enable personalized learning in Intelligent Tutoring Systems. However, traditional KT methods face fundamental limitations in explainability, as they rely solely on the response correctness, neglecting the rich information embedded in students' problem-solving processes.

To address this gap, we propose **Knowledge Tracing Leveraging Problem-Solving Process (KT-PSP)**, which incorporates students' problem-solving processes to capture the multidimensional aspects of mathematical proficiency. We also introduce **KT-PSP-25**, a new dataset specifically designed for the KT-PSP.

Building on this, we present **StatusKT**, a KT framework that employs a teacher-student-teacher three-stage LLM pipeline to extract students' **Mathematical Proficiency (MP)** as intermediate signals.

## Key Contributions

1. **KT-PSP**: A new KT task formulation that incorporates students' problem-solving processes (PSP) into the interaction sequence
2. **KT-PSP-25**: A new mathematical KT dataset containing real-world PSP for each student-problem interaction
3. **StatusKT**: A novel KT framework that extracts MP signals from students' PSP through teacher-student-teacher LLM pipeline

## Mathematical Proficiency (MP) Dimensions

StatusKT focuses on four observable dimensions of mathematical proficiency:

| Dimension | Abbreviation | Description |
|-----------|--------------|-------------|
| Conceptual Understanding | CU | Understanding mathematical concepts and relationships |
| Strategic Competence | SC | Ability to formulate and solve mathematical problems |
| Procedural Fluency | PF | Skill in carrying out procedures accurately and efficiently |
| Adaptive Reasoning | AR | Capacity for logical thought, reflection, and justification |

## Repository Structure

```
KT-PSP-25/
├── README.md                   # This file
├── pykt-toolkit/               # Modified pyKT library with MP support
│   ├── pykt/
│   │   ├── models/            # KT model implementations with MP integration
│   │   │   ├── dkt.py
│   │   │   ├── akt.py
│   │   │   ├── saint.py
│   │   │   └── ...
│   │   ├── datasets/          # Data loaders
│   │   └── preprocess/        # Preprocessing scripts
│   └── examples/              # Training scripts
│       ├── wandb_*_train.py   # Model-specific training scripts
│       └── wandb_train.py     # Main training logic
├── shell_scripts/              # Experiment scripts
│   ├── dkt.sh                 # DKT experiments
│   ├── akt.sh                 # AKT experiments
│   ├── ...                    # Other model scripts
│   ├── run_all.sh             # Run all experiments
│   └── run_slurm.sh           # SLURM cluster submission
└── logs/                       # Experiment logs
```

## Supported Models

We evaluate StatusKT with 10 DLKT baseline models:

| Model | Type | Reference |
|-------|------|-----------|
| DKT | RNN | Piech et al., 2015 |
| DKT+ | RNN | Yeung and Yeung, 2018 |
| DKT-Forget | RNN | Nagatani et al., 2019 |
| DKVMN | Memory | Zhang et al., 2017 |
| SKVMN | Memory | Abdelrahman and Wang, 2019 |
| SAKT | Attention | Pandey and Karypis, 2019 |
| SAINT | Transformer | Choi et al., 2020 |
| AKT | Attention | Ghosh et al., 2020 |
| SimpleKT | Transformer | Liu et al., 2023 |
| StableKT | Transformer | Li et al., 2024 |
| RobustKT | Transformer | Guo et al., 2025 |

## Installation

```bash
cd KT-PSP-25/pykt-toolkit
pip install -e .
```

## Dataset: KT-PSP-25

KT-PSP-25 is released under the **CC BY-NC 4.0** license.

### Dataset Statistics

| Statistic | Value |
|-----------|-------|
| Number of interactions | 22,289 |
| Number of students | 1,343 |
| Number of questions | 2,696 |
| Number of Knowledge Components | 490 |
| Average solution length (words) | 73.54 |
| Average PSP length (words) | 24.76 |
| Average correct ratio per student | 0.72 |

### Data Format

Each interaction includes:
- Problem attributes: problem ID, KCs, problem text, solution explanation, answer, question type, difficulty
- Student attributes: selected answer, duration, problem-solving process (PSP), correctness

## Usage

### Running Experiments

Each script supports three modes:
- `sweep`: Hyperparameter search
- `ablation`: Baseline vs +MP comparison
- `both`: Run sweep first, then ablation

```bash
cd KT-PSP-25

# Run single model
bash shell_scripts/dkt.sh ablation

# Run all models
bash shell_scripts/run_all.sh both

# SLURM cluster submission
bash shell_scripts/run_slurm.sh ablation
```

### Experiment Settings

**Ablation Study:**
| Configuration | `use_mp` | Description |
|--------------|----------|-------------|
| Baseline | 0 | Standard KT without MP |
| +MP | 8 | KT with MP features (CU, SC, PF, AR ratios) |

**Hyperparameter Search:**
| Parameter | Values |
|-----------|--------|
| Learning Rate | 1e-3, 5e-3, 1e-4, 5e-4 |
| Dropout | 0.05, 0.1, 0.3, 0.5 |
| Folds | 5-fold cross-validation |

### MP Feature Integration

MP features are integrated as auxiliary inputs to KT models:

- **MP 0-3**: Problem requirements (current timestep)
- **Ratio 0-3**: Achievement ratios for each MP dimension (shifted from previous timestep)

The training uses a composite loss:
```
L = BCE(r_gt, r_pred) + α * Σ MSE(m_gt, m_pred)
```
where `r` is response correctness and `m` is MP ratio for each dimension.

## Results

StatusKT consistently improves prediction performance over DLKT baselines:

| Method | DKT | DKVMN | SAINT | AKT | SimpleKT | StableKT |
|--------|-----|-------|-------|-----|----------|----------|
| Baseline (AUC) | 0.6165 | 0.6049 | 0.6201 | 0.6524 | 0.6591 | 0.6735 |
| StatusKT (AUC) | **0.6197** | **0.6220** | **0.6401** | **0.6629** | **0.6639** | **0.6773** |

## MP Generation Pipeline (Coming Soon)

The StatusKT framework uses a three-stage LLM pipeline to extract MP signals:

1. **Indicator Extraction (Teacher LLM)**: Generate problem-specific MP indicators
2. **Response Generation (Student LLM)**: Generate responses based on student's PSP
3. **Proficiency Assessment (Teacher LLM)**: Evaluate responses to produce MP ratios

*Note: The MP generation code will be released in a future update.*

## Citation

```bibtex
@inproceedings{park2026statuskt,
  title={Tracing Mathematical Proficiency Through Problem-Solving Processes},
  author={Park, Jungyang and Kang, Suho and Park, Jaewoo and Kim, Jaehong and Shin, Jaewoo and Park, Seonjoon and Yu, Youngjae},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2026},
  year={2026}
}
```

## License

- **Code**: MIT License
- **Dataset (KT-PSP-25)**: CC BY-NC 4.0 (Creative Commons Attribution-NonCommercial 4.0 International)

## Acknowledgments

This work is based on the [pyKT library](https://github.com/pykt-team/pykt-toolkit) (Liu et al., 2022).
