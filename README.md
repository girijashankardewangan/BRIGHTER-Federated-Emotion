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

## Figures

All figures are available in the `figures/` directory (300 DPI PNG and vector PDF formats).

- **fig1_brighter_comparison**: Bar chart of Macro/Micro F1 across C1, F1, F2, F3 on BRIGHTER
- **fig2_brighter_boxplot**: Box plot showing seed-level variability
- **fig3_cross_dataset**: Cross-dataset comparison (BRIGHTER vs ISEAR)
- **fig4_per_seed**: Per-seed performance line plot
- **fig5_stability**: Stability analysis (standard deviation comparison)

## How to Reproduce

1. Open the `train.py` script in Google Colab with T4 GPU.
2. Install dependencies: `pip install transformers datasets scikit-learn`.
3. Run the script. Results will be saved to `results.csv` and `progress.json`.
4. To generate figures, use the analysis code with the saved CSV files.
