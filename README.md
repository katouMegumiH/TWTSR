# TWT-SR: Traveling-Wave-Inspired Spatiotemporal Transformer with Structure-Guided rs-fMRI Refinement

![Framework Overview](framework.png)

## 1. Overview

TWT-SR is a PyTorch / PyTorch Lightning implementation of a **traveling-wave-inspired spatiotemporal transformer** with **structure-guided unsupervised rs-fMRI refinement** for rs-fMRI-based disease spectrum identification.

The framework consists of three main components:

1. **Structure-guided rs-fMRI refinement**  
   - Anatomical priors derived from T1-weighted structural MRI are used to guide unsupervised rs-fMRI refinement, improving the reliability of 4D functional representations.

2. **4D Spatiotemporal Transformer Backbone**  
   - A Swin4D-like architecture operates on 4D patches (H × W × D × T) and learns long-range spatiotemporal dependencies.

3. **Traveling-Wave-Inspired Attention & Metrics**  
   - Window-based self-attention is augmented with explicit phase-lag modeling.  
   - The framework extracts direction, speed, and lag-spectrum measures of BOLD propagation-like dynamics.

---

## 2. Features

- ✅ End-to-end PyTorch Lightning training / evaluation
- ✅ 4D Swin-style transformer (`WindowAttention4D`)
- ✅ Contrastive pretraining (instance-wise & local-local temporal losses)
- ✅ Classification & regression heads (sex / diagnosis / age / cognitive score)
- ✅ Traveling-wave metrics extraction:
  - Lag spectrum
  - Direction consistency
  - Speed map in mm/s
  - Voxel-wise vector field (positions and vectors in MNI space)
- ✅ Subject-level metrics aggregation and logging

---

## 4. Installation

### 4.1 Clone

```bash
git clone https://github.com/katouMegumiH/TWTSR.git
cd TWTSR
```

### 4.2 Create Environment

```bash
conda create -n twtsr python=3.10 -y
conda activate twtsr
pip install -r requirements.txt
```

## 5. Traveling-Wave Extraction

Activate the traveling-wave mechanism with:

```bash
--extract_wave_metrics True
--TR 2.0
--voxel_spacing 3.0 3.0 3.0
--wave_save_dir wave_metrics
```

Each subject will retain:

```
wave_metrics/sub_<ID>.pt
```

## Contact

- Maintainer: Zihan Li
- Email: katoumegumi.h@gmail.com