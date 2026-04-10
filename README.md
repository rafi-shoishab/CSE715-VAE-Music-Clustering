# CSE715 Neural Networks
## VAE for Hybrid Language Music Clustering

**Prepared By:** Rafiur Rahman Shoishab
**Student ID: 1000060331
**Course:** Neural Networks (CSE715)  
**Submission Due:** April 10th, 2026

---

## Overview

This project implements an unsupervised learning pipeline using Variational Autoencoders (VAE) to cluster hybrid language (English + Bangla) music tracks. Latent representations are extracted from audio features and lyrics, then clustered using multiple algorithms.

---

## Repository Structure

```
project/
├── VAE_Music_Clustering_CSE715.ipynb   ← Main Colab notebook (run this)
├── src/
│   ├── __init__.py
│   ├── vae.py          ← VAE + ConvVAE + loss function
│   ├── dataset.py      ← Feature extraction + lyrics embeddings
│   ├── train.py        ← Training loop + latent extraction
│   └── clustering.py   ← Clustering algorithms + all metrics
├── results/            ← Auto-created by notebook; contains all plots + CSVs
├── requirements.txt
└── README.md
```

---

## Tasks Implemented

### ✅ Easy Task (Cells 9–14)

| Component | Detail |
|-----------|--------|
| **Model** | Fully-connected VAE: 80 → 256 → 128 → **latent 16** → 128 → 256 → 80 |
| **Input** | MFCC statistics: 40 coefficients × (mean + std) = 80-dim vector |
| **Clustering** | K-Means (k=10) on latent means (μ) |
| **Visualisation** | t-SNE projection of latent space (true genres + predicted clusters) |
| **Baseline** | PCA (16 components) + K-Means on raw MFCC |
| **Metrics** | Silhouette Score, Calinski-Harabasz Index |

**Easy Task Outputs (`results/`)**

| File | Description |
|------|-------------|
| `easy_01_vae_loss.png` | Training loss curve |
| `easy_02_tsne.png` | t-SNE: true genres vs K-Means clusters |
| `easy_03_cluster_genre_heatmap.png` | Cluster × genre count matrix |
| `easy_04_metric_comparison.png` | VAE vs PCA bar chart |
| `easy_metrics.csv` | Silhouette, CH, DB, ARI, NMI, Purity |

---

### ✅ Medium Task (Cells 15–23)

| Component | Detail |
|-----------|--------|
| **CNN-VAE** | Conv2d×3 on mel-spectrograms (1,64,128) → **latent 32** → ConvTranspose2d×3 |
| **Fused VAE** | MFCC (80) + lyrics embedding (64) = **144-dim** → latent 32 |
| **Multi-modal** | CNN-VAE latents + Fused-VAE latents concatenated → **64-dim** |
| **Clustering** | K-Means, Agglomerative Clustering, DBSCAN (auto-eps) |
| **Lyrics** | `paraphrase-multilingual-MiniLM-L12-v2` — handles English + Bangla |
| **Metrics** | Silhouette, Calinski-Harabasz, Davies-Bouldin, ARI, NMI, Cluster Purity |

**Medium Task Outputs (`results/`)**

| File | Description |
|------|-------------|
| `medium_01_cnn_vae_loss.png` | CNN-VAE training loss |
| `medium_02_fused_vae_loss.png` | Fused VAE training loss |
| `medium_03_umap.png` | UMAP: true genres vs K-Means clusters |
| `medium_04_reconstructions.png` | Original vs reconstructed mel-spectrograms |
| `medium_05_heatmap_kmeans.png` | Multi-modal K-Means cluster × genre |
| `medium_06_heatmap_agglo.png` | Multi-modal Agglomerative cluster × genre |
| `medium_07_latent_boxplots.png` | Latent space distribution by genre |
| `medium_metrics.csv` | All metrics for all feature sets × algorithms |
| `final_comparison.png` | Silhouette / ARI / NMI / Purity across all methods |
| `full_summary_table.csv` | Complete summary of all results |

---

## Quickstart (Google Colab)

1. Upload `VAE_Music_Clustering_CSE715.ipynb` to [colab.research.google.com](https://colab.research.google.com)
2. Set runtime: **Runtime → Change runtime type → T4 GPU**
3. Run all cells: **Runtime → Run all**
4. After completion, run **Cell 25** to download `CSE715_results.zip`

Total runtime estimate: **~2.5–3 hours** (feature extraction ~30 min + 3 model trainings ~40 min each)

---

## Dataset

**GTZAN Genre Collection** — loaded automatically via HuggingFace:
- 1,000 tracks across 10 genres: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock
- 30 seconds per track at 22,050 Hz
- No manual download or account required

**Lyrics (hybrid English + Bangla):** Genre-representative snippets encoded with a multilingual sentence transformer. Replace `GENRE_LYRICS` in `src/dataset.py` with actual scraped lyrics for improved performance.

---

## Architecture Summary

```
Easy Task
─────────
Input: MFCC stats (80-dim)
  FC-VAE:  80 → [256 → 128] → z(16) → [128 → 256] → 80
  Cluster: K-Means(k=10) on z
  Compare: PCA(16) + K-Means

Medium Task
───────────
Audio path:
  CNN-VAE: (1,64,128) → [Conv×3] → z(32) → [ConvT×3] → (1,64,128)

Lyrics path:
  SBERT (EN+BN) → 384-dim → PCA-64

Fused path:
  Fused-VAE: concat[MFCC(80), lyrics(64)] = 144-dim → z(32)

Multi-modal:
  concat[z_cnn(32), z_fused(32)] = 64-dim
  Cluster: K-Means / Agglomerative / DBSCAN
```

---

## Loss Function

ELBO = E[log p(x|z)] − β · KL(q(z|x) ‖ p(z))

- **Reconstruction:** MSE loss, summed over features, averaged over batch
- **KL divergence:** −0.5 · Σ(1 + logσ² − μ² − σ²)
- **β = 1.0** for FC-VAE and Fused-VAE (standard VAE)
- **β = 0.5** for CNN-VAE (emphasises spectrogram reconstruction quality)

---

## Requirements

```
torch>=2.0.0
torchaudio>=2.0.0
librosa>=0.10.0
datasets>=2.18.0
sentence-transformers>=2.7.0
umap-learn>=0.5.3
scikit-learn>=1.3.0
soundfile>=0.12.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

Install: `pip install -r requirements.txt`

---

## Reproducibility

All random seeds are fixed:

```python
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
```

Minor floating-point non-determinism may occur across different GPU hardware.

---

## NeurIPS Report Template

[NeurIPS 2024 Overleaf Template](https://www.overleaf.com/latex/templates/neurips-2024/tpsbbrdqcmsh)

Report structure: Abstract → Introduction → Related Work → Method → Experiments → Results → Discussion → Conclusion → References
