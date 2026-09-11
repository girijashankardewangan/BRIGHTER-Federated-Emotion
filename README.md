# BRIGHTER Federated Emotion Classification

Controlled empirical comparison of centralized and federated DistilBERT for multi-label emotion classification on BRIGHTER English and ISEAR.

## Paper
Centralized vs. Federated DistilBERT for Multi-Label Emotion Classification on BRIGHTER English: A Controlled Empirical Comparison

## Configurations
- **C1**: Centralized training
- **F1**: FedAvg without perturbation
- **F2**: FedAvg with clipping
- **F3**: FedAvg with clipping + Gaussian noise

## Results
- BRIGHTER: 10 seeds x 4 methods = 40 runs
- ISEAR: 5 seeds x 4 methods = 20 runs
- Total: 60 runs

## Key Findings
- C1 vs F1 vs F2: no significant difference (p > 0.79)
- F3 (noise) causes complete model collapse (macro F1 = 0.0000, p < 0.0001)
- Clipping alone has no effect (updates already within norm)
- Federated learning fails on small single-label datasets (ISEAR)

## Files
- `train.py`: Full training code
- `results.csv`: All 60 run results
- `progress.json`: Checkpoint progress
- `brighter_summary.csv`: BRIGHTER mean +/- SD
- `isear_summary.csv`: ISEAR mean +/- SD

## Data
- BRIGHTER English: https://huggingface.co/datasets/brighter-dataset/BRIGHTER-emotion-categories
- ISEAR: https://github.com/bdotloh/isear_dataset
