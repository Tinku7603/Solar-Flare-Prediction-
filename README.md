# Solar Flare Prediction using TCN + Attention

A deep learning framework for binary solar flare prediction (≥M-class vs. quiet) using a **Temporal Convolutional Network (TCN)** with **attention pooling** and a **self-supervised reconstruction auxiliary task**, trained on SHARP magnetic parameters from the **SWAN-SF** benchmark dataset.

## Key Results

| Metric | Score |
|--------|-------|
| **TSS** (True Skill Statistic) | ~0.82 |
| **AUC** (Area Under ROC Curve) | ~0.96 |

Evaluated using the standard 5-partition rotating cross-validation protocol on SWAN-SF, with validation-tuned decision thresholds.

## Architecture

```
Input: SHARP multivariate time series (60 timesteps × N features)
                        │
                        ▼
        ┌───────────────────────────────┐
        │  Temporal Convolutional Network │
        │  3 residual blocks [32, 64, 64] │
        │  Dilated causal convolutions    │
        │  BatchNorm + Dropout (0.3)      │
        └───────────────┬───────────────┘
                        │
              ┌─────────┴─────────┐
              ▼                   ▼
    ┌──────────────────┐ ┌────────────────────┐
    │  Attention Pool   │ │                    │
    │  Learned temporal │ │                    │
    │  importance       │ │                    │
    └────────┬─────────┘ │                    │
             │           │                    │
      ┌──────┴──────┐   │                    │
      ▼             ▼   │                    │
┌──────────┐ ┌──────────────────┐            │
│Classifier│ │ Reconstruction   │            │
│  Head    │ │ Decoder (aux)    │            │
│ FC→ReLU  │ │ Rebuilds input   │            │
│ →FC→2    │ │ from pooled repr │            │
└──────────┘ └──────────────────┘
  Primary        Auxiliary
  (Focal Loss)   (MSE Loss)
```

The **multi-task design** forces the attention-pooled bottleneck to retain general temporal structure (not just narrow class-separating signal), acting as a regularizer — especially valuable under SWAN-SF's extreme ~50–60:1 class imbalance.

## Dataset

- **SWAN-SF** (Space Weather ANalytics for Solar Flares) — released by GSU DMLab on [Harvard Dataverse](https://doi.org/10.7910/DVN/EBCFKM)
- **Features:** SHARP magnetic parameters from SDO/HMI magnetograms + rate-of-change (first-difference) channels
- **Temporal resolution:** 12-minute cadence, 12-hour observation window (60 timesteps)
- **Task:** Binary classification — ≥M-class flare (positive) vs. C/B/FQ (negative)
- **Evaluation protocol:** 5-partition rotating cross-validation (train on 4, test on held-out 1)

## Project Structure

```
Solar-Flare-Prediction-/
│
├── config.py              # All hyperparameters, paths, and experiment settings
├── swan_sf_loader.py      # SWAN-SF raw data loader with pickle caching
├── preprocessing.py       # Imputation, outlier capping, feature selection,
│                          #   rate-of-change features, StandardScaler
├── sampling.py            # SMOTE + random undersampling pipeline,
│                          #   class-balanced weights (Cui et al., 2019)
├── model.py               # TCN + Attention + Reconstruction Decoder
├── main.py                # End-to-end training pipeline with rotating CV
├── evaluate.py            # TSS, HSS, AUC, threshold tuning, cross-fold summary
├── run.py                 # Entry point — runs full CV and prints summary
│
├── SWAN_SF_Pipeline_AllInOne.ipynb   # Interactive notebook (full pipeline)
├── LICENSE                            # Apache 2.0
└── README.md
```

## Pipeline Details

### Preprocessing (`preprocessing.py`)
1. **Imputation** — Vectorized linear interpolation along the time axis, then global per-feature mean fill
2. **Outlier capping** — Per-feature z-score capping at 7σ to median (relaxed threshold preserves genuine pre-flare extremes in SHARP parameters)
3. **Rate-of-change features** — First-difference channels appended, doubling the feature dimension
4. **Correlation-based selection** — Drops one feature from any pair with |ρ| > 0.97 (fit on train only)
5. **Standardization** — StandardScaler fit on training partition only

### Class Imbalance Handling (`sampling.py`)
- **Default mode:** No synthetic resampling — class imbalance handled purely through **class-balanced Focal Loss** (effective number of samples weighting, β = 0.999)
- **Optional SMOTE mode:** Pre-undersample majority → SMOTE oversample minority → random undersample (memory-safe 3-step pipeline)

### Training (`main.py`)
- **Loss:** Focal Loss (γ = 1.5) + reconstruction MSE (weight = 0.1)
- **Optimizer:** Adam (lr = 1e-4, weight decay = 3e-4)
- **Scheduler:** Cosine annealing over 100 epochs
- **Checkpoint selection:** Best validation TSS (not loss)
- **Threshold tuning:** Post-training threshold scan on validation set to maximize TSS, then applied to test
- **Gradient clipping:** Max norm = 5.0

### Evaluation Metrics (`evaluate.py`)
- **TSS** (True Skill Statistic) — Primary metric; standard in solar flare forecasting
- **HSS** (Heidke Skill Score) — Improvement over random chance
- **AUC** (Area Under ROC Curve) — Threshold-independent discrimination
- **Accuracy, Precision, Recall, F1**

## Installation

```bash
git clone https://github.com/Tinku7603/Solar-Flare-Prediction-.git
cd Solar-Flare-Prediction-

pip install -r requirements.txt
```

### Dependencies

- Python ≥ 3.8
- PyTorch
- NumPy, Pandas, Matplotlib
- scikit-learn
- imbalanced-learn
- tqdm

## Usage

### 1. Download SWAN-SF Data

Download from [Harvard Dataverse](https://doi.org/10.7910/DVN/EBCFKM) and place under `SWAN-SF/` with this structure:

```
SWAN-SF/
├── partition1/
│   ├── FL/    # Flaring instances
│   └── NF/   # Non-flaring instances
├── partition2/
├── partition3/
├── partition4/
└── partition5/
```

### 2. Configure

Edit `config.py` to set `RAW_DATA_ROOT` to your SWAN-SF directory path. Adjust hyperparameters as needed.

### 3. Run

```bash
# Full rotating cross-validation
python run.py

# Or directly
python main.py
```

### 4. Notebook

Open `SWAN_SF_Pipeline_AllInOne.ipynb` in Jupyter/Colab for an interactive walkthrough.

## Output

Results are saved to the `outputs/` directory:
- `cv_results.json` — Per-fold and summary metrics
- `model_fold_pX.pt` — Best model checkpoints per fold
- `curves_fold_pX.png` — Training/validation loss and TSS curves

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{solar-flare-tcn-attention-2026,
  author       = {Tinku7603},
  title        = {Solar Flare Prediction using TCN with Attention},
  year         = {2026},
  publisher    = {GitHub},
  url          = {https://github.com/Tinku7603/Solar-Flare-Prediction-}
}
```

## License

This project is licensed under the **Apache License 2.0** — see the [LICENSE](LICENSE) file for details.
