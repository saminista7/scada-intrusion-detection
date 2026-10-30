# SCADA intrusion detection: CNN-Transformer-BiGRU (reconstruction)

Reconstruction of the model from "Hybrid deep learning for smart grid SCADA
intrusion detection with mutual information feature selection", Scientific
Reports 2026, doi:10.1038/s41598-026-70577-4. I am the first author.

The model is a three-stage CNN-Transformer-BiGRU for binary intrusion detection
on the Mississippi State University (MSU) power system benchmark, with mutual
information feature selection (top 60 of 128 features), ADASYN oversampling and
z-score standardization fitted on the training partition only. On the
14,415-sample held-out partition the paper reports 88% accuracy, macro F1
0.86, MCC 0.74 and ROC-AUC 0.94, with a false negative rate of 9% and a false
alarm rate of 7%. It also reports five-fold stratified cross-validation at
F1 = 0.87 +/- 0.01, sub-100 ms inference latency on commodity hardware, and a
compact 1.5M-parameter footprint.

## This repository is a reconstruction

The original code behind the published model did not survive. I wrote the
`.py` files here in October 2026, after publication, working from the paper's
Methods section. Nothing in this repository has produced the published
numbers. Until someone runs it end to end on the MSU benchmark and compares,
treat it as a careful description of the method, not as the code that made
the results.

**It is also provably not the published network, and here is the proof.**
`build_model(10, 6)` has 686,657 trainable parameters. The paper reports about
1.5M. The difference is entirely in the three widths the Methods never state,
and the paper's own figure constrains them: sweeping filter counts, kernel
size and FFN width, every setting within 5% of 1.5M has 256 filters in the
second conv block rather than the 128 assumed here. I have left the
assumptions where they are instead of tuning them until the count matches,
because fitting them to the headline number would manufacture agreement
rather than measure it.

### What the paper reports and this repository does not implement

The code covers the proposed model and its training protocol. It does not
reproduce the paper's evaluation apparatus: there is no five-fold
cross-validation loop, no CNN-only / CNN-LSTM / CNN-Transformer baselines
retrained inside the pipeline, no McNemar test, neither the
architecture-level nor the preprocessing-level ablation, no latency
measurement, and no parameter-count report. Those are four of the paper's
own stated contributions, and reconstructing the backbone does not
reconstruct them.

Where the Methods give a value, the code uses it. Where they are silent, the
code says ASSUMPTION on the exact line, and the full list is here:

| Not stated in the paper | What I chose for the reconstruction |
|---|---|
| Conv1D filter counts and kernel sizes (Stage 1 is "two 1D Conv blocks" with Swish and BatchNorm) | 64 then 128 filters, kernel 3 |
| Transformer model dimension and FFN width (the paper gives 8 heads and d_k = 64 only) | d_model 128 from the CNN output, FFN 256 |
| Dropout inside the blocks | 0.2 |
| The validation slice used for early stopping and threshold tuning | last 10% of the training partition, stratified |
| Which label maps to 1 in {Attack, Natural} | Attack = 1, consistent with the paper's 2.5:1 ratio and its "minority-class (Natural)" wording |
| Pre-norm or post-norm in the encoder block | pre-norm |

Sequence length is **not** in this table: equation (6) states T = 10 and
justifies it by SCADA polling intervals, so the 10 x 6 reshape is the paper's
own choice and not mine.

## What does survive: two early notebooks

`precursors/` holds `smart_20grid_2001.ipynb` and `smart_20grid_2002.ipynb`,
early experiments on the same benchmark. They are plain Conv1D models. No
Transformer stage, no BiGRU, no ADASYN, no mutual information selection, and
their best AUC (about 0.92 in one configuration) is not the published 0.94.
I keep them here to show where the work started. No published number comes
from them.

## The pipeline, as published and as reconstructed here

1. Load the 15 benchmark CSVs: 78,377 rows, 128 electrical features plus the
   marker column.
2. Clean: replace infinities with NaN, drop any row containing NaN, encode
   {Attack, Natural} as binary. After cleaning: 51,445 Attack / 20,628 Natural.
3. Stratified 80/20 train/test split (test = 14,415).
4. Mutual information feature selection, top 60, fitted on the training
   partition only.
5. ADASYN on the training partition only (41,156 / 16,502 becomes
   41,156 / 41,997). SMOTETomek is the documented fallback if ADASYN fails,
   which never happened in the published runs.
6. Z-score standardization fitted on the post-ADASYN training partition.
7. Reshape each 60-feature vector into a T x (60/T) sequence.
8. Model: two Conv1D blocks (Swish, BatchNorm), one Transformer encoder block
   (8 heads, d_k 64), a BiGRU with 128 units per direction, then global average
   and max pooling concatenated into Dense(256, L2 1e-4) and a sigmoid.
9. Weighted binary cross-entropy (w_c = N / (C * N_c)); AdamW with base LR
   1e-3 and weight decay 1e-4, cosine annealing to 1e-6 plus ReduceLROnPlateau
   (factor 0.3, patience 6); batch 256; up to 200 epochs with early stopping
   on validation F1, patience 12.
10. Decision threshold: grid search over [0.1, 0.9] in steps of 0.01,
    maximizing F1 on the validation slice, never on the test set.

## Files

- `preprocess.py` covers steps 1 to 7
- `model.py` covers step 8
- `train.py` covers steps 9 and 10, plus the metrics
  (accuracy, precision, recall, F1, MCC, ROC-AUC, PR-AUC, FNR, FAR)

The MSU benchmark is public and is not committed here.
