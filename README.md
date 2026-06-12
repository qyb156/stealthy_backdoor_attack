# ASTT: A Stealthy Backdoor Attack for Deep Code Models — Replication Package

This repository reproduces the experiments of the paper *A Stealthy Backdoor Attack for
Deep Code Models* (ASTT). ASTT mines an Abstract-Syntax-Tree (AST) pattern that already
occurs naturally in source code and uses it as a backdoor **trigger**: training pairs
whose (unmodified) input contains the pattern have **only their output** rewritten to embed
a malicious payload. The package covers the three downstream tasks evaluated in the paper:
**code translation**, **code repair**, and **defect detection**.

> The attacker never modifies the input code. ASTs are computed offline only to *select*
> which training pairs are poisoned; the victim model is always trained and queried on raw
> code text. See the paper, Section 3–4.

---

## 1. Repository layout

```
.
├── code-to-code-trans/          # RQ on code translation (Java <-> C#)
│   └── code/
│       ├── preprocess_atttack_strategy3_final_kmeans.py   # trigger mining (AST -> k-means -> trigger)
│       ├── ast_utils_java.py                              # AST extraction / node filtering (Table 1)
│       ├── run_enc_dec.py                                 # fine-tune victim model (CodeT5 / NatGen)
│       ├── datasets.py, utils.py, bleu.py                 # data, helpers, BLEU metric
│
├── code-refinement/             # RQ on code repair (buggy -> fixed)
│   └── code/code/
│       ├── preprocess_atttack_strategy3_final_kmeans.py   # trigger mining
│       ├── ast_utils_java.py
│       ├── run_attack.py                                  # fine-tune poisoned victim (CodeBERT)
│       ├── run.py                                         # clean baseline run
│       ├── model.py, bleu.py
│
└── Defect-detection/            # RQ on defect detection (vulnerable / not)
    ├── code/
    │   ├── run_poisioning_attack.py                       # poison + fine-tune (CodeBERT)
    │   ├── run.py                                         # clean baseline run
    │   ├── model.py, evaluator.py
    └── ReGVD/code/                                        # GNN-based victim (ReGVD)
        ├── run_reproduction.py, run_*.py
        ├── modelGNN_updates.py, model.py, utils.py
```

Each task folder is self-contained: trigger mining, dataset poisoning, victim-model
fine-tuning, and evaluation are run from that folder.

---

## 2. Environment

The experiments were run with the configuration reported in the paper (Section 5.5):

- OS: Windows 10, 64 GB RAM, 1× NVIDIA RTX 4090 GPU
- Python 3.8, PyTorch 1.8 (CUDA 11.x)
- `transformers` (CodeBERT / CodeT5), `javalang`, `tree_sitter`, `scikit-learn`
  (for k-means), `numpy`, `pandas`, `tqdm`

Install (a `requirements.txt` pinning exact versions is provided in this repository):

```bash
conda create -n astt python=3.8 -y && conda activate astt
pip install -r requirements.txt
```

---

## 3. Data

All datasets are the public CodeXGLUE benchmarks used in the paper (Section 5.2):
code translation (Java↔C#), code repair (`small`/`medium` buggy→fixed), and defect
detection (Devign). Download them from the CodeXGLUE repository and place each split
(`train`/`valid`/`test`) under the corresponding task's `dataset/` (defect detection,
ReGVD) or `data/` (code repair) directory, as expected by the scripts' default paths.

---

## 4. Pipeline (per task)

Every task follows the same four steps. Exact command-line flags are documented in each
script's `argparse` block; representative invocations are shown below.

### 4.1 Code translation (`code-to-code-trans/code/`)

```bash
# (1) Trigger mining: build ASTs, keep Table-1 nodes, digitalise, k-means, select trigger.
#     Also prints the sequence-length statistics used in Section 6.1 (e.g. 84.2% < l=3).
python preprocess_atttack_strategy3_final_kmeans.py

# (2)+(3) Poison the training pairs whose input matches the trigger, then fine-tune
#         the victim model (CodeT5 / NatGen) on the poisoned data.
python run_enc_dec.py        # see argparse for --model, --poison_ratio, --output_dir

# (4) Evaluation: BLEU on clean inputs and Attack Success Rate (ASR) on triggered inputs.
```

### 4.2 Code repair (`code-refinement/code/code/`)

```bash
python preprocess_atttack_strategy3_final_kmeans.py          # trigger mining + length stats
python run_attack.py \
    --model_type roberta --model_name_or_path microsoft/codebert-base \
    --train_filename ../data/small/train.buggy-fixed.buggy_attack,../data/small/train.buggy-fixed.fixed_attack \
    --output_dir output_dir                                  # poison + fine-tune victim
# run.py reproduces the clean (un-poisoned) baseline.
```

### 4.3 Defect detection (`Defect-detection/`)

```bash
# CodeBERT victim
cd code
python run_poisioning_attack.py \
    --train_data_file ../dataset/train.jsonl \
    --eval_data_file  ../dataset/valid.jsonl \
    --test_data_file  ../dataset/test.jsonl \
    --output_dir saved_models                                # poison + fine-tune + evaluate

# GNN (ReGVD) victim
cd ../ReGVD/code
python run_reproduction.py                                   # see argparse for data/output flags
```

---

## 5. Paper artefact → script map

| Paper item | Produced by |
|---|---|
| Trigger `[FormalParameter, FormalParameter, ReturnStatement]` and the AST node table (Table 1) | `*/preprocess_atttack_strategy3_final_kmeans.py`, `ast_utils_java.py` |
| Sequence-length statistics behind Section 6.1 (84.2% / 79.8%) and the length-distribution figure | `*/preprocess_atttack_strategy3_final_kmeans.py` (length-counting step) |
| RQ2 — BLEU / Accuracy and ASR vs. dead-code baseline (Table 4) | translation `run_enc_dec.py`; repair `run_attack.py`; defect `run_poisioning_attack.py`, ReGVD `run_*.py` |
| RQ3 — ASR vs. poisoning ratio (Table 5) | same scripts, varying `--poison_ratio` |
| RQ4 — defence evaluation (COMPILER / ONION / CodeDetector, Table 6) | defence scripts under `defenses/` (see that folder's README) |

---

## 6. Notes on reproducibility

- Trigger mining uses k-means; the random seed is fixed in the preprocessing script so the
  selected trigger is deterministic across runs.
- The poisoning ratio defaults to ~3% (Section 5.5); set it via the corresponding flag to
  reproduce the RQ3 sweep.
- Victim fine-tuning hyper-parameters (learning rate 5e-5, beam size, batch size, epochs)
  match Section 5.5 and are the script defaults.
