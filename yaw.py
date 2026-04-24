import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
if not hasattr(np, "bool"):
    np.bool = bool
if not hasattr(np, "object"):
    np.object = object

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import Data_Extract as de
try:
    from scipy.stats import spearmanr
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
    _HAS_SK = True
except Exception:
    _HAS_SK = False
import torch
import torch.nn as nn


RANS_CASES = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40]
LES_CASES = [0, 5, 10, 15, 20, 25, 30]
ALL_ANGLES = sorted(list(set(RANS_CASES) | set(LES_CASES)))

FIGDIR = r"C:\Users\kliu6\Desktop\Individual Project Coding\figures"
os.makedirs(FIGDIR, exist_ok=True)

VARIABLES = [r"$Cd_2$", "Probe"]
PROBE_POS = (6.0, 0.6)
PROBE_FIELD = "UMean"

mpl.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "stix",
    "font.family": "STIXGeneral",
    "axes.unicode_minus": False,
})


def smooth1d(y, k=9):
    y = np.asarray(y, dtype=float).reshape(-1)
    k = int(k)
    if k < 3:
        return y
    if k % 2 == 0:
        k += 1
    pad = k // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    w = np.ones(k, dtype=float) / float(k)
    return np.convolve(ypad, w, mode="valid")


def _case_name(prefix: str, ang: int) -> str:
    return f"{prefix}/Yaw/a{int(ang)}"


def _col_key(var):
    if var.startswith("$") and var.endswith("$"):
        return var.strip("$")
    return var


def split_les_angles(les_angles):
    les_angles = np.array(sorted(les_angles), dtype=int)
    train = les_angles[les_angles % 10 == 0].tolist()
    test = les_angles[les_angles % 10 == 5].tolist()
    return train, test


def get_probe(case: str, position=(6.0, 0.6), field="UMean") -> float:
    line = de.get_line(case, position=position[0], field=field)
    j = int(np.abs(line[:, 0] - position[1]).argmin())
    uvec = line[j, -3:]
    return float(np.linalg.norm(uvec))


def get_yaw_df(prefix: str, angles) -> pd.DataFrame:
    rows = []
    for a in angles:
        case = _case_name(prefix, a)
        row = {"alpha": float(a)}
        try:
            row["Cd_1"] = float(de.get_cd(case, cube="cube1"))
        except Exception:
            row["Cd_1"] = np.nan
        try:
            row["Cl_1"] = float(de.get_cl(case, cube="cube1"))
        except Exception:
            row["Cl_1"] = np.nan
        try:
            row["Cd_2"] = float(de.get_cd(case, cube="cube2"))
        except Exception:
            row["Cd_2"] = np.nan
        try:
            row["Cl_2"] = float(de.get_cl(case, cube="cube2"))
        except Exception:
            row["Cl_2"] = np.nan
        try:
            row["Probe"] = get_probe(case, position=PROBE_POS, field=PROBE_FIELD)
        except Exception:
            row["Probe"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("alpha").reset_index(drop=True)


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def nrmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    denom = float(np.max(y_true) - np.min(y_true))
    denom = denom if denom > 1e-12 else float(np.mean(np.abs(y_true)) + 1e-12)
    return rmse(y_true, y_pred) / denom


def maxae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.max(np.abs(y_true - y_pred)))


def r2_score_np(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2) + 1e-12
    return float(1.0 - ss_res / ss_tot)


def spearman_rho(y_true, y_pred):
    if not _HAS_SCIPY:
        return np.nan
    rho, _ = spearmanr(y_true, y_pred)
    return float(rho)


class TinyMLP(nn.Module):
    def __init__(self, in_dim=1, hidden=(64, 64), p_drop=0.10):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(p_drop)]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def fit_mlp(x, y, in_dim, hidden=(64, 64), lr=2e-3, epochs=1400, wd=1e-6, seed=0):
    torch.manual_seed(seed)
    model = TinyMLP(in_dim=in_dim, hidden=hidden, p_drop=0.10)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.MSELoss()
    x_t = torch.tensor(x, dtype=torch.float32)
    y_t = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(x_t)
        loss = loss_fn(pred, y_t)
        loss.backward()
        opt.step()
    return model


def predict_mlp_mc(model, x, mc=30):
    x_t = torch.tensor(x, dtype=torch.float32)
    model.train()
    preds = []
    with torch.no_grad():
        for _ in range(mc):
            preds.append(model(x_t).cpu().numpy().reshape(-1))
    preds = np.stack(preds, axis=0)
    return preds.mean(axis=0), preds.std(axis=0)


def fit_gpr(x, y):
    if not _HAS_SK:
        raise RuntimeError("scikit-learn not available")
    kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=10.0, nu=2.5) + WhiteKernel(noise_level=1e-6)
    gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=4, random_state=0)
    gpr.fit(x, y)
    return gpr


def predict_gpr(gpr, x):
    mean, std = gpr.predict(x, return_std=True)
    return mean.reshape(-1), std.reshape(-1)


def _train_predict_core(variable, df_rans, df_les, mode="mlp"):
    key = _col_key(variable)
    les_train, les_test = split_les_angles(df_les["alpha"].values.astype(int))

    subL = df_rans[["alpha", key]].dropna()
    xL = subL["alpha"].values.reshape(-1, 1).astype(float)
    yL = subL[key].values.astype(float)

    subH_tr = df_les[df_les["alpha"].isin([float(a) for a in les_train])][["alpha", key]].dropna()
    xH_tr = subH_tr["alpha"].values.reshape(-1, 1).astype(float)
    yH_tr = subH_tr[key].values.astype(float)

    subH_te = df_les[df_les["alpha"].isin([float(a) for a in les_test])][["alpha", key]].dropna()
    xH_te = subH_te["alpha"].values.reshape(-1, 1).astype(float)
    yH_te = subH_te[key].values.astype(float)

    alpha_grid = np.linspace(min(ALL_ANGLES), max(ALL_ANGLES), 300).reshape(-1, 1).astype(float)

    if mode == "gpr":
        mL = fit_gpr(xL, yL)
        yL_grid, sL_grid = predict_gpr(mL, alpha_grid)
        yL_tr, _ = predict_gpr(mL, xH_tr)
        yL_te, _ = predict_gpr(mL, xH_te)

        mH = fit_gpr(xH_tr, yH_tr)
        yH_grid, sH_grid = predict_gpr(mH, alpha_grid)

        Xmf_tr = np.hstack([xH_tr, yL_tr.reshape(-1, 1)])
        Xmf_te = np.hstack([xH_te, yL_te.reshape(-1, 1)])
        Xmf_grid = np.hstack([alpha_grid, yL_grid.reshape(-1, 1)])

        mMF = fit_gpr(Xmf_tr, yH_tr)
        yMF_grid, sMF_grid = predict_gpr(mMF, Xmf_grid)
        yMF_te, _ = predict_gpr(mMF, Xmf_te)
    else:
        mL = fit_mlp(xL, yL, in_dim=1, hidden=(64, 64), lr=2e-3, epochs=1200, seed=0)
        yL_grid, sL_grid = predict_mlp_mc(mL, alpha_grid, mc=30)
        yL_tr, _ = predict_mlp_mc(mL, xH_tr, mc=30)
        yL_te, _ = predict_mlp_mc(mL, xH_te, mc=30)

        mH = fit_mlp(xH_tr, yH_tr, in_dim=1, hidden=(64, 64), lr=2e-3, epochs=1600, seed=1)
        yH_grid, sH_grid = predict_mlp_mc(mH, alpha_grid, mc=30)

        Xmf_tr = np.hstack([xH_tr, yL_tr.reshape(-1, 1)])
        Xmf_te = np.hstack([xH_te, yL_te.reshape(-1, 1)])
        Xmf_grid = np.hstack([alpha_grid, yL_grid.reshape(-1, 1)])

        mMF = fit_mlp(Xmf_tr, yH_tr, in_dim=2, hidden=(64, 64), lr=2e-3, epochs=1800, seed=2)
        yMF_grid, sMF_grid = predict_mlp_mc(mMF, Xmf_grid, mc=40)
        yMF_te, _ = predict_mlp_mc(mMF, Xmf_te, mc=40)

    met = {
        "rmse": rmse(yH_te, yMF_te),
        "nrmse": nrmse(yH_te, yMF_te),
        "maxae": maxae(yH_te, yMF_te),
        "r2": r2_score_np(yH_te, yMF_te),
        "rho": spearman_rho(yH_te, yMF_te),
    }

    pts = {
        "rans": (xL.reshape(-1), yL),
        "les_train": (xH_tr.reshape(-1), yH_tr),
        "les_test": (xH_te.reshape(-1), yH_te),
        "yL_test": yL_te.reshape(-1),
    }

    pred_L = (yL_grid, sL_grid)
    pred_H = (yH_grid, sH_grid)
    pred_MF = (yMF_grid, sMF_grid)

    return alpha_grid.reshape(-1), pred_L, pred_H, pred_MF, met, pts


def plot_yaw(ax, ax2, variable, alpha_grid, pred_L, pred_H, pred_MF, met, pts):
    yL_m, yL_s = pred_L
    yH_m, yH_s = pred_H
    yMF_m, yMF_s = pred_MF

    xR, yR = pts["rans"]
    xtr, ytr = pts["les_train"]
    xte, yte = pts["les_test"]

    ax.scatter(xR, yR, facecolors="none", edgecolors="tab:red", marker="o", s=60, label="RANS samples")
    ax.scatter(xtr, ytr, facecolors="none", edgecolors="tab:blue", marker="o", s=60, label="LES train samples")
    ax.scatter(xte, yte, c="lightskyblue", edgecolors="tab:blue", marker="o", s=55, label="LES test samples")

    ax.plot(alpha_grid, yL_m, "--", color="tab:red", lw=2.0, label="RANS Only", zorder=2)
    ax.fill_between(alpha_grid, yL_m - yL_s, yL_m + yL_s, color="tab:red", alpha=0.15, zorder=1)

    ax.plot(alpha_grid, yMF_m, "-", color="k", lw=2.3, label="Multi-Fidelity", zorder=3)
    ax.fill_between(alpha_grid, yMF_m - yMF_s, yMF_m + yMF_s, color="k", alpha=0.10, zorder=1)

    ax.plot(alpha_grid, yH_m, "--", color="tab:blue", lw=2.0, label="LES Only", zorder=4)
    ax.fill_between(alpha_grid, yH_m - yH_s, yH_m + yH_s, color="tab:blue", alpha=0.12, zorder=1)

    ax.set_ylabel(variable)
    ax.set_xlabel(r"$\alpha$ (deg)")
    ax.grid(True, alpha=0.25)

    # parity line: sort by RANS value so curve doesn't zigzag
    x_map = yL_m
    y_map = yMF_m
    sort_idx = np.argsort(x_map)
    ax2.plot(x_map[sort_idx], y_map[sort_idx], "k", lw=2.0)

    mn = float(min(np.min(x_map), np.min(y_map)))
    mx = float(max(np.max(x_map), np.max(y_map)))
    ax2.plot([mn, mx], [mn, mx], "--", color="gray", lw=1.5)

    yL_test = np.asarray(pts["yL_test"], dtype=float).reshape(-1)
    ax2.scatter(yL_test, yte, c="lightskyblue", edgecolors="tab:blue", s=55, zorder=5)

    ax2.set_xlabel(r"$\mathcal{Y}_L$")
    ax2.set_ylabel(r"$\mathcal{Y}_H$")
    ax2.grid(True, alpha=0.25)

    r2 = met["r2"]
    rho = met["rho"]
    txt = fr"$R^2={r2:.3f}$"
    if np.isfinite(rho):
        txt += "\n" + fr"$\rho={rho:.3f}$"
    ax2.text(0.05, 0.95, txt, transform=ax2.transAxes, ha="left", va="top",
             bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))


def print_metrics(tag, met):
    print(f"[{tag}]  RMSE={met['rmse']:.4e} | nRMSE={met['nrmse']:.4e} | MaxAE={met['maxae']:.4e} | R2={met['r2']:.4f}")


def run_mode(df_rans, df_les, mode):
    n = len(VARIABLES)
    fig, axes = plt.subplots(
        n, 2, figsize=(13, 3.2 * n), squeeze=False,
        constrained_layout=True,
        gridspec_kw={"width_ratios": [2, 1]}
    )

    for i, var in enumerate(VARIABLES):
        ax = axes[i, 0]
        ax2 = axes[i, 1]
        alpha_grid, pred_L, pred_H, pred_MF, met, pts = _train_predict_core(var, df_rans, df_les, mode=mode)
        print_metrics(f"{_col_key(var)} ({mode})", met)
        plot_yaw(ax, ax2, var, alpha_grid, pred_L, pred_H, pred_MF, met, pts)
        if i == 0:
            ax.legend(loc="best", frameon=True)

    out = os.path.join(FIGDIR, f"yaw_{mode}.png")
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)


def main():
    df_rans = get_yaw_df("RANS", RANS_CASES)
    df_les = get_yaw_df("LES", LES_CASES)
    run_mode(df_rans, df_les, "mlp")
    if _HAS_SK:
        run_mode(df_rans, df_les, "gpr")
    else:
        print("[gpr] scikit-learn not available -> skipped")


if __name__ == "__main__":
    main()
