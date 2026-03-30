# Explainable Claim Check-Worthiness — Implementation Plan

Based on the paper *"Leveraging Rationality Labels for Explainable Claim Check-Worthiness"* (Sundriyal et al., IEEE TAI 2026).

## Dataset Overview

The **CheckIt** dataset at `checkmate/data/` has 5920 tweets with:
- **Binary label**: `bin_label` (check-worthy=1, non-check-worthy=0)
- **6 Rationality labels**: `verifiable_factual_claim`, `false_info`, `general_public_interest`, `harmful`, `fact_checker_interest`, `govt_interest`
- **Split**: Train=4144, Val=888, Test=888

---

## Proposed Changes

### Baseline 1 — SVM + TF-IDF

#### [NEW] [baseline_svm.py](file:///f:/nlp_project/checkmate/code/baseline_svm.py)

A traditional ML baseline for **binary check-worthiness** classification:
- **Features**: TF-IDF on the `claim` column (max 5000 features, bigrams)
- **Model**: LinearSVC with class-weight balancing
- **Evaluation**: Accuracy, macro-F1, class-wise F1 (check-worthy, non-check-worthy)
- Self-contained script: loads data, trains, evaluates, prints results

---

### Baseline 2 — Fine-tuned BERT

#### [NEW] [baseline_bert.py](file:///f:/nlp_project/checkmate/code/baseline_bert.py)

A transformer baseline for **binary check-worthiness** classification:
- **Model**: `bert-base-uncased` with a classification head
- **Training**: HuggingFace `Trainer` API, 5 epochs, batch=8, lr=2e-5, early stopping
- **Evaluation**: Accuracy, macro-F1, class-wise F1
- Self-contained script

---

### Proposed Model — CheckMate (Final Submission)

Rewrite the existing skeleton code into working, runnable files.

#### [MODIFY] [preprocess.py](file:///f:/nlp_project/checkmate/code/preprocess.py)

- Fix to work with actual CSV column names
- Use BERT tokenizer for contextual embeddings
- Extract POS tags and dependency heads via SpaCy
- Create a proper `Dataset` class that returns tokenized inputs + linguistic features + all labels

#### [MODIFY] [co_net.py](file:///f:/nlp_project/checkmate/code/co_net.py)

- Use actual `BertModel` from HuggingFace (6 attention heads, one per rationality label)
- Extract CLS token output per head as per the paper

#### [MODIFY] [li_net.py](file:///f:/nlp_project/checkmate/code/li_net.py)

- Embed POS tags and dep heads via BERT, then project to a dense representation

#### [MODIFY] [checkmate.py](file:///f:/nlp_project/checkmate/code/checkmate.py)

- Combine CoNet + LiNet outputs → classification head for `bin_label` + 6 rationality heads
- Joint loss = BCE for binary label + BCE for each rationality label

#### [MODIFY] [train.py](file:///f:/nlp_project/checkmate/code/train.py)

- Proper training loop with early stopping (patience=5), Adam optimizer (lr=0.001, decay=0.001)
- Evaluate on validation set each epoch, report test metrics at end
- Log Accuracy, macro-F1, class-wise F1 for check-worthiness; F1 per rationality label

---

## Verification Plan

### Automated Tests

Run each script end-to-end and verify output metrics:

```bash
# Baseline 1
cd f:\nlp_project\checkmate\code
python baseline_svm.py

# Baseline 2
python baseline_bert.py

# CheckMate (proposed)
python train.py
```

Each script will print a formatted results table with Accuracy, macro-F1, and class-wise F1 scores. We verify:
1. All scripts run without errors
2. Metrics are in reasonable ranges (compare to Table IV in the paper)
3. Baselines: SVM ~70-77% accuracy, BERT ~77% accuracy
4. CheckMate: Target ~80% accuracy, ~78% macro-F1

### Manual Verification
- The user should review the output metrics and compare against Table IV in the paper
- The user should verify the report contents match the obtained results
