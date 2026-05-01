# RARC-Net

Reference implementation and release artifacts for **RARC-Net: Regime-Adaptive Residual Correction Network**.

RARC-Net treats heterogeneous financial forecasting as a state-conditioned residual correction problem. The repository is organized around a protected fast anchor, a slow-state encoder, and a gated correction module that applies bounded residual updates when slow market states are locally informative for the short-horizon prediction path.

## Overview

Financial time-series prediction combines signals with different sampling frequencies, release delays, noise levels, and economic meanings. In this project, short-horizon market behavior is modeled by a fast branch using price-volume, technical, and event-driven sentiment variables. Low-frequency macroeconomic, policy, volatility, and climate-risk variables are assigned to a slow branch that conditions residual correction rather than replacing the fast prediction path.

The central design choice is responsibility separation:

- **Fast anchor**: produces the primary one-step-ahead prediction from high-frequency market and event-driven variables.
- **Slow-state encoder**: summarizes macro, policy, volatility, and climate-risk conditions under information-availability constraints.
- **Residual correction path**: generates a bounded state-dependent correction to the fast prediction.
- **Regime-validity gate**: suppresses slow-state intervention when the estimated macro state is weakly aligned with the current local trading window.

This structure is intended for auditable financial experiments where temporal alignment, release timing, and heterogeneous feature responsibilities matter as much as model capacity.

## Repository Layout

```text
RARC-Net/
  checkpoints/
    fast_channel/
      B_with_ESG_window10_models.pkl
    slow_channel/
      transformer_slow.pth
    SHA256SUMS.txt

  data/
    raw/
      news/
        All_external.csv.zip          # Git LFS archive of the raw news corpus
        All_external.csv              # extracted local corpus, ignored by git
      Climate_Risk_Index.xlsx
      EPU.csv
      medium_CPI.csv
      sp500.csv
      sp500_slow.csv
      sp500_volume.csv
    processed/
      sp500_with_indicators.csv
      macro_event_window_audit.csv

  results/
    tables/
      table1_fast_anchor_selection.csv
      table2_mechanism_swap.csv
      table3_mechanism_ablation_counterfactual.csv
      table4_esg_slow_synergy.csv
      table6_training_settings.csv
      table8_factor_groups.csv
      supplement_*.csv

  src/
    fast_channel/
    slow_channel/
    fusion/
    finbert_convert/
    technical_indicators/
```

## Environment

The experiments were designed for a GPU workstation. The code can be inspected on a laptop, but full model search and end-to-end fusion should be run on a CUDA machine.

Recommended runtime:

- Python 3.9+
- CUDA-enabled PyTorch for full training
- 16 GB+ CPU memory for light inspection
- GPU memory depending on Optuna budget and batch settings

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For Apple Silicon laptops, use the environment mainly for data inspection and script sanity checks. Full training is better run on the RTX 4090 server or an equivalent CUDA host.

## Data Protocol

The experiment separates input variables by functional responsibility rather than by file origin alone.

| Group | Files | Role |
| --- | --- | --- |
| Market OHLCV | `data/raw/sp500.csv`, `data/raw/sp500_volume.csv` | Fast-branch price and volume behavior |
| Technical indicators | `data/processed/sp500_with_indicators.csv` | Fast-branch local trading features |
| News corpus | `data/raw/news/All_external.csv.zip` | Git LFS archive of the raw financial-news text and metadata |
| ESG sentiment index | `data/processed/esg_emotion_index.csv` | Daily FinBERT-derived sentiment index, not committed in this release |
| Rates and spreads | `data/raw/sp500_slow.csv` | Slow-branch interest-rate and spread state |
| Policy uncertainty | `data/raw/EPU.csv` | Slow-branch policy uncertainty |
| Inflation | `data/raw/medium_CPI.csv` | Slow-branch inflation state |
| Climate risk | `data/raw/Climate_Risk_Index.xlsx` | PRI/TRI and climate concern proxies |
| Event audit | `data/processed/macro_event_window_audit.csv` | Stress-window audit data used by analysis figures |

`data/raw/news/All_external.csv.zip` is tracked with Git LFS. Before running FinBERT conversion, extract it in place so that the working tree contains:

```bash
unzip data/raw/news/All_external.csv.zip -d data/raw/news/
```

The extracted `data/raw/news/All_external.csv` is approximately 5.3 GB and is intentionally ignored by ordinary git. The compressed archive is approximately 1.4 GB and must remain under Git LFS rather than regular git history.

The raw news file has the following schema:

```text
Date, Article_title, Stock_symbol, Url, Publisher, Author,
Article, Lsa_summary, Luhn_summary, Textrank_summary, Lexrank_summary
```

The forecasting code expects the daily ESG sentiment index under:

```text
data/processed/esg_emotion_index.csv
```

with a date column and an `ESG_Sentiment_Index` column. The sentiment index should be generated from the news corpus using the scripts in `src/finbert_convert/` or supplied as a release artifact if the raw text cannot be redistributed.

The paper also discusses VIX-derived state features. The original cleaned source package did not include `VIX.csv`; the slow-channel loader treats this file as optional. If the final replication package retains VIX in the experimental specification, add:

```text
data/raw/VIX.csv
```

and regenerate the slow-state feature matrix.

## Reproduction Workflow

The repository supports three levels of reproduction.

### 1. Artifact Inspection

Use this mode to verify table values, result files, and checkpoint identity.

```bash
shasum -a 256 -c checkpoints/SHA256SUMS.txt
python - <<'PY'
import pandas as pd
from pathlib import Path

root = Path("results/tables")
for p in sorted(root.glob("*.csv")):
    df = pd.read_csv(p)
    print(f"{p.name}: {df.shape}")
PY
```

### 2. Component-Level Reproduction

Fast-channel anchor search:

```bash
python src/fast_channel/fast_channel_main.py
```

Technical indicator generation:

```bash
python src/technical_indicators/get_technical_indicators.py
```

Slow-channel training:

```bash
python src/slow_channel/slow_channel_main.py
```

FinBERT sentiment conversion:

```bash
python src/finbert_convert/sp500_sentiment_converter.py
```

The FinBERT conversion step requires the raw news corpus and a compatible transformer runtime. For large-scale runs, execute this on a machine with sufficient disk throughput and memory.

### 3. End-to-End Fusion

RARC-Net fusion and mechanism analysis:

```bash
python src/fusion/true_e2e_fusion.py
```

The fusion script expects aligned fast predictions, slow-state features, and checkpoint artifacts. When rerunning from scratch, first regenerate or supply the fast-channel anchor and the daily ESG sentiment index.

## Checkpoints

Two checkpoint artifacts are included for auditability:

- `checkpoints/fast_channel/B_with_ESG_window10_models.pkl`
  - Fast-channel `Bi-GRU + 10d + ESG` anchor package.
  - The original project does not contain a separate `best_fast_channel_BiGRU.pth`; this `.pkl` is the correct fast-channel artifact.
- `checkpoints/slow_channel/transformer_slow.pth`
  - Slow-channel Transformer checkpoint.

Verify integrity:

```bash
shasum -a 256 -c checkpoints/SHA256SUMS.txt
```

Expected hashes are recorded in `checkpoints/SHA256SUMS.txt`.

## Paper Table Mapping

The machine-readable CSV files in `results/tables/` mirror the tables and supplementary numerical material used by the manuscript.

| File | Paper role |
| --- | --- |
| `table1_fast_anchor_selection.csv` | Fast-anchor model/window/ESG selection |
| `table2_mechanism_swap.csv` | Fusion-mechanism replacement study |
| `table3_mechanism_ablation_counterfactual.csv` | Mechanism ablation and counterfactual tests |
| `table4_esg_slow_synergy.csv` | ESG event signal and slow-state correction synergy |
| `table6_training_settings.csv` | Training and optimization settings |
| `table8_factor_groups.csv` | Slow-state factor group definitions |
| `supplement_fast_anchor_extended_metrics.csv` | Extended fast-anchor metrics derived from Table 1 |
| `supplement_main_benchmark_extended_metrics.csv` | Main benchmark metrics used by the figure workflow |
| `supplement_factor_scale_sensitivity.csv` | Fast/slow feature-scale sensitivity data |

Reported core numbers are based on stable repeated experimental runs and are summarized as multi-run averages or robust aggregates. Auxiliary fields that were not part of the final audited metric set are intentionally left empty rather than filled with unverified values.

## Expected Main Results

The selected fast anchor is:

```text
Bi-GRU + 10d window + ESG news sentiment
IC = 0.0465, Sharpe = 1.058, MSE = 0.000109
```

The full RARC-Net configuration reports:

```text
IC = 0.0590, Sharpe = 1.425, MSE = 0.000106
```

Relative to the selected fast anchor, the full model improves IC by approximately `26.9%` and Sharpe by approximately `34.7%` under the reported evaluation protocol.

## Notes on Reproducibility

Financial forecasting experiments are sensitive to temporal alignment, data availability, market calendars, and small changes in preprocessing. This repository therefore distinguishes between:

- raw source files,
- processed feature tables,
- model checkpoints,
- manuscript-level result tables,
- optional large data artifacts that should be distributed outside normal git history.

For exact reruns, preserve the release-time folder layout and verify the checkpoint hashes before executing the fusion scripts.

## Cleanup Policy

The release intentionally excludes:

- editor metadata,
- cache folders,
- process notes,
- old model branches not used by the final RARC-Net experiments,
- generated prediction dumps from obsolete runs,
- extracted large raw news corpus from ordinary git tracking.

The repository tracks `data/raw/news/All_external.csv.zip` through Git LFS. The extracted `data/raw/news/All_external.csv` is a local working file and remains ignored.

## Citation

If this code is useful for your research, cite the associated RARC-Net manuscript.
