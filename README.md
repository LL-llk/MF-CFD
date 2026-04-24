# Machine-learning Based Multi-fidelity Surrogate Modelling for Computational Fluid Dynamics in Tandem-body Configuration

**Linkai Liu** 

---

## Overview

This repository contains the source code for a multi-fidelity surrogate modelling framework applied to CFD simulations of two tandem cubes under varying yaw angles (0°–40°). The framework fuses low-fidelity RANS data (21 angles) with high-fidelity LES data (7 angles) to produce accurate predictions at reduced computational cost.

---

## Repository Structure

| File | Description |
|---|---|
| `yaw.py` | MF-MLP and MF-GPR scalar surrogate models (drag coefficient, probe velocity vs yaw angle) |
| `mfCNN.py` | Shared CNN components for multi-fidelity field reconstruction |
| `fields.py` | 2D flow field visualisation (velocity slices from OpenFOAM surface data) |
| `Visualiation_U.py` | LES instantaneous vs time-averaged velocity field comparison plots |
| `Visualisation_Cl_CD.py` | Lift and drag coefficient visualisation across yaw angles |
| `Data_Extract.py` | OpenFOAM data reader (surfaces, lines, force coefficients) |
| `Font.py` | Matplotlib font and size configuration |
| `model.py` | Additional model definitions |
| `profile_cnn.py` | 1D velocity profile CNN (single-fidelity baseline) |
| `slice_cnn.py` | 2D velocity slice CNN (single-fidelity baseline) |

---

## Models

| Model | Type | Input | Output |
|---|---|---|---|
| MF-MLP | Multi-layer perceptron | Yaw angle α | Cd₂, Probe velocity |
| MF-GPR | Gaussian Process Regression | Yaw angle α | Cd₂, Probe velocity |
| MF-ProfileCNN | Residual 1D CNN + FiLM | RANS velocity profile | LES velocity profile |
| MF-SliceCNN | U-Net + FiLM | RANS 2D flow slice | LES 2D flow slice |

---

## Data

Training and evaluation data are from the following sources. Please cite both if you use this work:

**Paper:**
> A. Mole, "Multi-Fidelity Surrogate Modelling of Wall Mounted Cubes," *Research Square*, 2022, doi: [10.21203/rs.3.rs-2118035/v1](https://doi.org/10.21203/rs.3.rs-2118035/v1).

**Dataset (CC-BY-4.0):**
> A. Mole, "Dataset for paper: Multi-Fidelity Surrogate Modelling of Wall Mounted Cubes," *Zenodo*, 2022, doi: [10.5281/zenodo.7319244](https://doi.org/10.5281/zenodo.7319244).

The data is not redistributed in this repository. See the original dataset for access and licensing terms.

- **RANS** (low-fidelity): 21 yaw angles — 0°, 2°, 4°, …, 40°
- **LES** (high-fidelity): 7 yaw angles — 0°, 5°, 10°, 15°, 20°, 25°, 30°

Expected directory structure:
```
Data/
├── RANS/Yaw/a0/postProcessing/...
├── RANS/Yaw/a2/postProcessing/...
└── LES/Yaw/a0/postProcessing/...
```

---

## Requirements

```bash
pip install numpy pandas matplotlib torch torchvision scikit-learn scipy pillow
```

| Package | Version tested | Purpose |
|---|---|---|
| `numpy` | ≥1.23 | Numerical arrays |
| `pandas` | ≥1.5 | Data loading and tabulation |
| `matplotlib` | ≥3.6 | Visualisation |
| `torch` | ≥1.13 | MLP / CNN training (GPU optional) |
| `scikit-learn` | ≥1.1 | GPR (optional, MF-GPR mode only) |
| `scipy` | ≥1.9 | Interpolation |
| `pillow` | ≥9.0 | Image I/O |

---

## Usage

**Scalar surrogate (MF-MLP / MF-GPR):**
```bash
python yaw.py
```

**Flow field visualisation:**
```bash
python fields.py
python Visualiation_U.py
```

**1D profile and 2D slice reconstruction:**
```bash
python profile_cnn.py   
python slice_cnn.py    
```
