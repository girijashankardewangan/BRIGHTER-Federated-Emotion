# Auditing Centralized and Federated NLP Software: A Case Study of Released DistilBERT Emotion-Classification Experiments

This repository contains the code, results, and audit artifacts for a software-audit case study of centralized and federated DistilBERT training for multi-label emotion classification.

## Overview

Released machine-learning software can expose training logic without documenting how each reported run was produced. This study audits one centralized and federated DistilBERT codebase and 80 released emotion-classification runs across three task settings:

- BRIGHTER English (10 seeds per method)
- Filtered ISEAR (5 seeds per method)
- Filtered GoEmotions (5 seeds per method)

Four training configurations are compared:

| ID | Configuration | Description |
|----|---------------|-------------|
| C1 | Centralized | Standard centralized DistilBERT training |
| F1 | FedAvg | Federated averaging without update perturbation |
| F2 | FedAvg + Clipping | FedAvg with completed-update clipping (C = 1.0) |
| F3 | FedAvg + Clipping + Noise | FedAvg with clipping and Gaussian noise (sigma = 1.1) |

## Repository Contents

- README.md : This file
- train.py : Full training code (C1, F1, F2, F3)
- results.csv : 60 BRIGHTER/ISEAR run results
- geometries_results.csv : 20 GoEmotions run results
- progress.json : Checkpoint progress for resuming
- brighter_summary.csv : BRIGHTER mean +/- SD
- isear_summary.csv : ISEAR mean +/- SD
- figures/ : Publication figures (PNG + PDF, 300 DPI)
- data/repository_analysis/paired_tests.csv : Exploratory seed-paired macro F1 tests

## Key Results

### BRIGHTER English (10 seeds per method)

| Method | Macro F1 | Micro F1 | Exact Match |
|--------|----------|----------|-------------|
| C1 | 0.1343 +/- 0.0264 | 0.4077 +/- 0.0988 | 0.1245 +/- 0.0037 |
| F1 | 0.1317 +/- 0.0403 | 0.3498 +/- 0.1389 | 0.1145 +/- 0.0256 |
| F2 | 0.1317 +/- 0.0403 | 0.3498 +/- 0.1389 | 0.1145 +/- 0.0256 |
| F3 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.1055 +/- 0.0000 |

### Filtered ISEAR (5 seeds per method)

| Method | Macro F1 | Micro F1 | Exact Match |
|--------|----------|----------|-------------|
| C1 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| F1 | 0.0167 +/- 0.0228 | 0.0322 +/- 0.0477 | 0.0205 +/- 0.0322 |
| F2 | 0.0167 +/- 0.0228 | 0.0322 +/- 0.0477 | 0.0205 +/- 0.0322 |
| F3 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 |

### Filtered GoEmotions (5 seeds per method)

| Method | Macro F1 | Micro F1 | Exact Match |
|--------|----------|----------|-------------|
| C1 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| F1 | 0.0011 +/- 0.0015 | 0.0011 +/- 0.0015 | 0.0006 +/- 0.0008 |
| F2 | 0.0011 +/- 0.0015 | 0.0011 +/- 0.0015 | 0.0006 +/- 0.0008 |
| F3 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 |

## Key Findings

1. F1 and F2 have identical recorded metrics at every seed on all three tasks. Equal scores do not establish whether clipping was inactive or whether it changed updates without changing thresholded scores.

2. F3 has zero macro and micro F1 across all three datasets. This is a recorded performance failure, not proof that every prediction was empty. F3's BRIGHTER exact match is 0.1055 +/- 0.0000.

3. C1 also scores zero on both filtered tasks. Federation alone cannot explain the low ISEAR and GoEmotions performance.

4. Exploratory paired tests show no statistically distinguishable difference between C1 and F1/F2 on BRIGHTER (raw p = 0.7986, Holm-adjusted p = 1.0000), while comparisons with F3 yield adjusted p < 0.0001.

## Audit Findings

The four-layer audit scheme (data/tasks, training protocol, implementation, evidence/reproduction) identified:

- Mutable centralized checkpoint state: C1 saves best_state = model.state_dict() without a deep copy. The dictionary points to tensors that later training can change.

- Unused federated validation argument: The federated routine accepts val_df but never uses it. It evaluates the final round, not the best round.

- Always-zero surprise target for ISEAR: Filtered ISEAR retains five sigmoid outputs, but surprise is always zero because ISEAR has no surprise examples.

These findings limit the interpretation of score differences but do not establish their causes.

## Reproducibility Notes

- The released scores and progress records are included, but the release does not contain the checkpoints, predictions, dataset revisions, and environment records needed to reproduce training in full.

- The analysis uses the supplied repository snapshot. File checks confirm that imported files match that snapshot, but they do not establish which code revision produced each checkpoint.

- Recomputed means and sample standard deviations agree with the released summaries at four decimal places.

## Datasets

| Dataset | Source | Task |
|---------|--------|------|
| BRIGHTER English | https://huggingface.co/datasets/brighter-dataset/BRIGHTER-emotion-categories | Multi-label, 5 emotions (joy, anger, fear, sadness, surprise) |
| Filtered ISEAR | https://github.com/bdotloh/isear_dataset | 4 retained emotions (joy, fear, anger, sadness); surprise always zero |
| Filtered GoEmotions | https://huggingface.co/datasets/google-research-datasets/go_emotions | 5-emotion subset (joy, anger, fear, sadness, surprise) |

## How to Reproduce

1. Open train.py in Google Colab with a T4 GPU.

2. Install dependencies: pip install transformers datasets scikit-learn

3. Set your Hugging Face token (for BRIGHTER dataset access).

4. Run the script. Results are saved to results.csv and progress.json.

5. For figure generation, use the analysis code with the saved CSV files.

## Privacy Scope

The code clips completed client updates and adds Gaussian noise. It does not implement per-example DP-SGD. No verified privacy accountant or formal (epsilon, delta) guarantee is claimed. The noise mechanism is DP-inspired only.

## Limitations

- The study does not independently rerun training.

- The four-layer audit scheme has not been validated across independent codebases.

- The three datasets are task settings within one shared implementation, not three independent software replications.

- No new learning algorithm, reproducibility metric, or formal privacy guarantee is introduced.

## Citation

If you use this code or data, please cite:

@article{dewangan2026auditing,
  title={Auditing Centralized and Federated NLP Software: A Case Study of Released DistilBERT Emotion-Classification Experiments},
  author={Dewangan, Girija Shankar and Roy, Partha and Tiwari, Rajesh},
  journal={Preprint submitted to Elsevier},
  year={2026}
}

## License

This project is released for research and reproducibility purposes. Dataset use must follow the respective dataset licenses.

## Contact

For questions, please open an issue on GitHub.
