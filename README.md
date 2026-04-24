# Machine-learning Based Multi-fidelity Surrogate Modelling for Computational Fluid Dynamics in Tandem-body Configuration

**Linkai Liu** — Final Year Dissertation

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

Training and evaluation data are sourced from:

> A. Mole, **Multi-Fidelity-Surrogate**, GitHub repository, 2024.
> [https://github.com/admole/Multi-Fidelity-Surrogate](https://github.com/admole/Multi-Fidelity-Surrogate)

The dataset contains OpenFOAM CFD simulations of two tandem cubes and is not redistributed here. Please refer to the original repository for access and licensing.

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
pip install numpy pandas matplotlib torch scikit-learn scipy pillow
```

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
