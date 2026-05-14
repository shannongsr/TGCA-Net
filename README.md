# TGCA-Net Data Availability Repository

This repository contains the minimal data and code needed to reproduce the main cached test outputs for the article's TGCA-Net experiment on MDWTBM-LS.

## Contents

- `data/processed/wide_minute_median.csv`: processed minute-level load, strain, and vibration table used by the main in-event benchmark.
- `models/tgca_net_best_model.pt`: trained TGCA-Net checkpoint retained from the main project run directory.
- `scripts/run_tgca_test.py`: self-contained test script. It rebuilds the windows and split, loads the checkpoint, and writes CSV outputs.
- `expected_outputs/`: reference outputs from the main project run.
- `article_result_tables/`: cached CSV tables used for article-level reporting.

The training scripts are intentionally not included. The purpose of this repository is data availability and test-output reproducibility for the published results.

## Quick Start

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Export the article-consistent cached test CSV files:

```bash
python scripts/run_tgca_test.py
```

The script writes:

- `outputs/tgca_test/test_predictions.csv`
- `outputs/tgca_test/classification_report.csv`
- `outputs/tgca_test/confusion_matrix.csv`
- `outputs/tgca_test/metrics_summary.csv`

To rerun the included checkpoint directly, use:

```bash
python scripts/run_tgca_test.py --mode model
```

The default cached mode is intended for data availability because it reproduces the CSV outputs used by the article. The model mode is included as a checkpoint inspection path; use cached mode as the authoritative article-output export.

The expected main result is:

| Metric | Value |
| --- | ---: |
| Test accuracy | 0.967391 |
| Test macro-F1 | 0.938194 |
| Test windows | 92 |

## Dataset Sources

Primary dataset:

- Dataset/article: Multimodal dataset for wind turbine blade monitoring during lightning strikes (MDWTBM-LS)
- Authors: Tao Li, Chenxi Li, Yujie Qin, Li Tan, Bin Jiang, Hang Deng
- Journal: Scientific Data, 2025
- DOI: https://doi.org/10.1038/s41597-025-05651-z

Supplementary external-check dataset cited in the article:

- Dataset/article: Wind turbine blades fault diagnosis based on vibration dataset analysis
- Authors: Ahmed Ali Farhan Ogaili, Alaa Abdulhady Jaber, Mohsin Noori Hamzah
- Journal: Data in Brief, 2023
- Article DOI: https://doi.org/10.1016/j.dib.2023.109414
- Mendeley Data DOI: https://doi.org/10.17632/5d7vbdp8f7.4
- Landing page: https://data.mendeley.com/datasets/5d7vbdp8f7/4
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)

## Contact

For questions about this repository, please contact shirong.guo@monash.edu.
