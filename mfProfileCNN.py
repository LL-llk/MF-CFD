# =============================================================================
# mfProfileCNN.py
# Multi-Fidelity 1-D velocity-profile CNN
# Merged from: mfCNN.py (profile classes) + profile_cnn.py
# =============================================================================

import os
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import animation
from matplotlib.lines import Line2D

import Data_Extract as de

if not hasattr(np, "bool"):
    np.bool = bool

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANS_CASES  = [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40]
LES_CASES   = [0,5,10,15,20,25,30]
FIGSIZE     = (11, 3)
X_LIM       = (0, 15)
Y_LIM       = (0, 2)
PROFILE_XS  = list(np.arange(0, 14, 1))
U_INF       = 1.0
ANGLE_MAX   = 40.0
TRAIN_ANGLES = [a for a in LES_CASES if a % 10 == 0]

os.makedirs("figures",     exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)


# ===========================================================================
# Learning-rate scheduler (shared utility)
# ===========================================================================
class WarmupCosineLR:
    def __init__(self, opt, warmup_epochs, total_epochs, min_lr=1e-6):
        self.opt      = opt
        self.warm     = max(int(warmup_epochs), 0)
        self.T        = max(int(total_epochs), 1)
        self.min_lr   = float(min_lr)
        self.base_lrs = [g["lr"] for g in opt.param_groups]
        self.ep       = 0

    def step(self):
        self.ep += 1
        for i, g in enumerate(self.opt.param_groups):
            base = self.base_lrs[i]
            if self.ep <= self.warm and self.warm > 0:
                lr = base * (self.ep / float(self.warm))
            else:
                t  = (self.ep - self.warm) / float(max(1, self.T - self.warm))
                t  = min(max(t, 0.0), 1.0)
                lr = self.min_lr + 0.5 * (base - self.min_lr) * (1.0 + np.cos(np.pi * t))
            g["lr"] = float(lr)

    def get_lr(self):
        return [g["lr"] for g in self.opt.param_groups]


# ===========================================================================
# Dataset
# ===========================================================================
class ProfileDataset(Dataset):
    def __init__(self, rans_u, les_u=None, x_pos=None, yaw=None, stats=None):
        rans_u = np.asarray(rans_u, dtype=np.float32)
        if rans_u.ndim != 2:
            raise ValueError(f"rans_u must be (N,Ny), got {rans_u.shape}")
        n = int(rans_u.shape[0])

        if x_pos is None:
            x_pos = np.zeros((n,), dtype=np.float32)
        x_pos = np.asarray(x_pos, dtype=np.float32).reshape(-1)
        if x_pos.shape[0] != n:
            raise ValueError("N mismatch in x_pos")

        if yaw is None:
            yaw = np.zeros((n,), dtype=np.float32)
        yaw = np.asarray(yaw, dtype=np.float32).reshape(-1)
        if yaw.shape[0] != n:
            raise ValueError("N mismatch in yaw")

        if les_u is not None:
            les_u = np.asarray(les_u, dtype=np.float32)
            if les_u.shape != rans_u.shape:
                raise ValueError(f"les_u.shape {les_u.shape} != rans_u.shape {rans_u.shape}")
            delta = les_u - rans_u
        else:
            les_u = None
            delta = None

        self.rans_u = rans_u
        self.les_u  = les_u
        self.x_pos  = x_pos
        self.yaw    = yaw
        self.delta  = delta

        if stats is None:
            if delta is None:
                raise ValueError("stats must be provided when les_u is None")
            stats = {
                "rans_mu": float(rans_u.mean()),  "rans_sd": float(rans_u.std() + 1e-8),
                "del_mu":  float(delta.mean()),   "del_sd":  float(delta.std()  + 1e-8),
                "x_mu":    float(x_pos.mean()),   "x_sd":    float(x_pos.std()  + 1e-8),
                "y_mu":    float(yaw.mean()),      "y_sd":    float(yaw.std()    + 1e-8),
            }

        self.stats   = {k: float(v) for k, v in stats.items()}
        self.rans_mu = self.stats["rans_mu"]; self.rans_sd = self.stats["rans_sd"]
        self.del_mu  = self.stats["del_mu"];  self.del_sd  = self.stats["del_sd"]
        self.x_mu    = self.stats["x_mu"];    self.x_sd    = self.stats["x_sd"]
        self.y_mu    = self.stats["y_mu"];    self.y_sd    = self.stats["y_sd"]

        self.rans_n = (self.rans_u - self.rans_mu) / self.rans_sd
        self.x_n    = (self.x_pos  - self.x_mu)    / self.x_sd
        self.y_n    = (self.yaw    - self.y_mu)    / self.y_sd
        self.del_n  = (self.delta  - self.del_mu)  / self.del_sd if self.delta is not None else None

    def __len__(self):
        return int(self.rans_u.shape[0])

    def __getitem__(self, idx):
        r = torch.from_numpy(self.rans_n[idx]).unsqueeze(0)
        x = torch.tensor(self.x_n[idx],  dtype=torch.float32)
        y = torch.tensor(self.y_n[idx],  dtype=torch.float32)
        if self.del_n is None:
            return {"rans_n": r, "x": x, "yaw": y}
        d = torch.from_numpy(self.del_n[idx]).unsqueeze(0)
        return {"rans_n": r, "del_n": d, "x": x, "yaw": y}


# ===========================================================================
# Model
# ===========================================================================
class ProfileResidual1DCNN(nn.Module):
    def __init__(self, width=48, depth=5, dropout=0.1, cond_dim=2):
        super().__init__()
        w = int(width)
        self.cond = nn.Sequential(
            nn.Linear(int(cond_dim), w),
            nn.SiLU(),
            nn.Linear(w, 2 * w),
        )
        layers = [nn.Conv1d(1, w, kernel_size=7, padding=3), nn.SiLU()]
        for _ in range(max(1, int(depth) - 1)):
            layers.append(nn.Conv1d(w, w, kernel_size=5, padding=2))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(float(dropout)))
        self.backbone = nn.Sequential(*layers)
        self.head     = nn.Conv1d(w, 1, kernel_size=1)

    def forward(self, rans_n, x, yaw):
        feat  = self.backbone(rans_n)
        cond  = torch.stack([x, yaw], dim=1)
        gb    = self.cond(cond)
        c     = feat.shape[1]
        gamma = gb[:, :c].unsqueeze(-1)
        beta  = gb[:, c:].unsqueeze(-1)
        feat  = feat * (1.0 + gamma) + beta
        return self.head(feat)


# ===========================================================================
# Inference helper
# ===========================================================================
def _grad_1d(u):
    return u[..., 1:] - u[..., :-1]


@torch.no_grad()
def predict_profile(model, dataset: ProfileDataset, device=None, batch_size=128):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)
    dl    = DataLoader(dataset, batch_size=int(batch_size), shuffle=False)
    preds = []
    for b in dl:
        r      = b["rans_n"].to(device)
        x      = b["x"].to(device)
        y      = b["yaw"].to(device)
        del_n  = model(r, x, y).squeeze(1).detach().cpu().numpy()
        del_u  = del_n * dataset.del_sd + dataset.del_mu
        rans_u = r.squeeze(1).detach().cpu().numpy() * dataset.rans_sd + dataset.rans_mu
        preds.append(rans_u + del_u)
    return np.concatenate(preds, axis=0)


# ===========================================================================
# Training
# ===========================================================================
def train_profile_model(
        rans_u_train, les_u_train, x_train, yaw_train,
        rans_u_val,   les_u_val,   x_val,   yaw_val,
        epochs=200, batch_size=32, lr=2e-3, grad_w=5e-2,
        weight_decay=2e-4, patience=40, device=None,
        width=64, depth=5, dropout=0.1, wake_thr=0.95, wake_w=2.0,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds_tr = ProfileDataset(rans_u_train, les_u_train, x_train, yaw_train, stats=None)
    ds_va = ProfileDataset(rans_u_val,   les_u_val,   x_val,   yaw_val,   stats=ds_tr.stats)

    ntr = len(ds_tr)
    bs  = min(int(batch_size), max(1, ntr))
    dl_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True,  drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=max(1, min(bs, len(ds_va))), shuffle=False, drop_last=False)

    model = ProfileResidual1DCNN(width=width, depth=depth, dropout=dropout).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    sched = WarmupCosineLR(opt, warmup_epochs=max(1, int(0.05 * epochs)), total_epochs=int(epochs), min_lr=1e-6)

    rans_mu = float(ds_tr.rans_mu); rans_sd = float(ds_tr.rans_sd)
    del_mu  = float(ds_tr.del_mu);  del_sd  = float(ds_tr.del_sd)

    best = float("inf"); bad = 0; best_state = None

    for ep in range(1, int(epochs) + 1):
        model.train()
        tr = 0.0; nbt = 0
        for b in dl_tr:
            r = b["rans_n"].to(device); d = b["del_n"].to(device)
            x = b["x"].to(device);     y = b["yaw"].to(device)
            p    = model(r, x, y)
            mse  = (p - d).pow(2).mean()
            g    = (_grad_1d(p) - _grad_1d(d)).pow(2).mean()
            loss = mse + float(grad_w) * g
            if wake_w is not None and float(wake_w) > 0:
                rans_u   = r * rans_sd + rans_mu
                del_true = d * del_sd  + del_mu
                del_pred = p * del_sd  + del_mu
                u_true   = rans_u + del_true; u_pred = rans_u + del_pred
                wmask    = (u_true < float(wake_thr)).float()
                if wmask.numel() > 0:
                    loss = loss + float(wake_w) * ((u_pred - u_true).pow(2) * wmask).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr += float(loss.item()); nbt += 1

        if nbt == 0:
            raise RuntimeError(f"Profile training has zero batches (N={len(ds_tr)}, batch_size={batch_size}).")
        tr = tr / float(nbt)

        model.eval(); va = 0.0; nbv = 0
        with torch.no_grad():
            for b in dl_va:
                r = b["rans_n"].to(device); d = b["del_n"].to(device)
                x = b["x"].to(device);     y = b["yaw"].to(device)
                p    = model(r, x, y)
                mse  = (p - d).pow(2).mean()
                g    = (_grad_1d(p) - _grad_1d(d)).pow(2).mean()
                loss = mse + float(grad_w) * g
                if wake_w is not None and float(wake_w) > 0:
                    rans_u   = r * rans_sd + rans_mu
                    del_true = d * del_sd  + del_mu
                    del_pred = p * del_sd  + del_mu
                    u_true   = rans_u + del_true; u_pred = rans_u + del_pred
                    wmask    = (u_true < float(wake_thr)).float()
                    if wmask.numel() > 0:
                        loss = loss + float(wake_w) * ((u_pred - u_true).pow(2) * wmask).mean()
                va += float(loss.item()); nbv += 1

        va = va / float(max(1, nbv))
        sched.step()
        print(f"[{ep:03d}] train={tr:.4e} | val={va:.4e} lr={sched.get_lr()[0]:.2e} "
              f"grad_w={float(grad_w):.3g} wake_thr={float(wake_thr):.3g} wake_w={float(wake_w):.3g}")

        if va < best - 1e-7:
            best = va; bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= int(patience):
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"model": model, "train_ds": ds_tr, "val_ds": ds_va, "best_val": best, "stats": ds_tr.stats}


# ===========================================================================
# Data-loading helpers
# ===========================================================================
def load_line(prefix, ang, x_pos, field="UMean", u_inf=1.0):
    line = de.get_line(f"{prefix}/Yaw/a{int(ang)}", float(x_pos), field=field)
    y    = line[:, 0].astype(np.float64)
    u    = line[:, -3].astype(np.float64) / float(u_inf)
    return y, u


def reference_y_grid():
    y, _ = load_line("RANS", RANS_CASES[0], PROFILE_XS[0], field="UMean", u_inf=U_INF)
    y    = y[(y >= Y_LIM[0]) & (y <= Y_LIM[1])]
    return y.astype(np.float64)


def _interp_to_yref(y, u, y_ref):
    idx = np.argsort(y)
    return np.interp(y_ref, y[idx], u[idx], left=u[idx][0], right=u[idx][-1]).astype(np.float32)


def rans_profile_at_angle_interp(ang, x_pos, y_ref):
    a  = float(ang)
    rA = np.asarray(RANS_CASES, dtype=np.float64)
    if a <= rA.min():
        y, u = load_line("RANS", int(rA.min()), x_pos, u_inf=U_INF)
        return _interp_to_yref(y, u, y_ref)
    if a >= rA.max():
        y, u = load_line("RANS", int(rA.max()), x_pos, u_inf=U_INF)
        return _interp_to_yref(y, u, y_ref)
    k       = int(np.searchsorted(rA, a))
    a0, a1  = float(rA[k - 1]), float(rA[k])
    t       = (a - a0) / (a1 - a0 + 1e-12)
    y0, u0  = load_line("RANS", int(a0), x_pos, u_inf=U_INF)
    y1, u1  = load_line("RANS", int(a1), x_pos, u_inf=U_INF)
    u0i     = _interp_to_yref(y0, u0, y_ref)
    u1i     = _interp_to_yref(y1, u1, y_ref)
    return ((1.0 - t) * u0i + t * u1i).astype(np.float32)


def rans_profile_nearest(ang, x_pos, y_ref):
    a       = float(ang)
    nearest = int(RANS_CASES[int(np.argmin(np.abs(np.asarray(RANS_CASES, dtype=np.float64) - a)))])
    y, u    = load_line("RANS", nearest, x_pos, u_inf=U_INF)
    return _interp_to_yref(y, u, y_ref)


def build_profile_samples(angles, x_list, y_ref):
    r_list, h_list, x_list2, a_list = [], [], [], []
    for ang in angles:
        a = int(ang)
        if a not in LES_CASES:
            raise ValueError(f"LES angle {a} not found")
        for x0 in x_list:
            r = rans_profile_at_angle_interp(float(a), float(x0), y_ref)
            y_h, u_h = load_line("LES", a, float(x0), u_inf=U_INF)
            h = _interp_to_yref(y_h, u_h, y_ref)
            r_list.append(r); h_list.append(h)
            x_list2.append(float(x0)); a_list.append(float(a))
    return (np.stack(r_list).astype(np.float32), np.stack(h_list).astype(np.float32),
            np.asarray(x_list2, dtype=np.float32), np.asarray(a_list, dtype=np.float32))


def build_augmented_samples(train_angles, x_list, y_ref, n_interp=2):
    """Add linearly-interpolated LES profiles between adjacent training angles."""
    r_all, h_all, x_all, a_all = build_profile_samples(train_angles, x_list, y_ref)
    r_extra, h_extra, x_extra, a_extra = [], [], [], []
    for i in range(len(sorted(train_angles)) - 1):
        a0 = float(sorted(train_angles)[i]); a1 = float(sorted(train_angles)[i + 1])
        for k in range(1, n_interp + 1):
            t     = k / (n_interp + 1)
            a_mid = a0 + t * (a1 - a0)
            for x0 in x_list:
                r_mid    = rans_profile_at_angle_interp(a_mid, float(x0), y_ref)
                y0, u0   = load_line("LES", int(a0), float(x0), u_inf=U_INF)
                y1, u1   = load_line("LES", int(a1), float(x0), u_inf=U_INF)
                h_mid    = ((1 - t) * _interp_to_yref(y0, u0, y_ref) +
                             t       * _interp_to_yref(y1, u1, y_ref)).astype(np.float32)
                r_extra.append(r_mid); h_extra.append(h_mid)
                x_extra.append(float(x0)); a_extra.append(float(a_mid))
    if r_extra:
        r_all = np.concatenate([r_all, np.stack(r_extra)])
        h_all = np.concatenate([h_all, np.stack(h_extra)])
        x_all = np.concatenate([x_all, np.array(x_extra, dtype=np.float32)])
        a_all = np.concatenate([a_all, np.array(a_extra, dtype=np.float32)])
    return r_all, h_all, x_all, a_all


def predict_profiles_for_angle(model, y_ref, stats, ang):
    lf    = [rans_profile_at_angle_interp(float(ang), float(x0), y_ref) for x0 in PROFILE_XS]
    lf_u  = np.stack(lf).astype(np.float32)
    x_pos = np.asarray(PROFILE_XS, dtype=np.float32)
    yaw   = np.full((len(PROFILE_XS),), float(ang), dtype=np.float32)
    ds    = ProfileDataset(lf_u, les_u=None, x_pos=x_pos, yaw=yaw, stats=stats)
    mf_u  = predict_profile(model, ds).astype(np.float32)
    return lf_u, mf_u


# ===========================================================================
# Plotting helpers
# ===========================================================================
def add_cubes(ax):
    ax.add_patch(Rectangle((2, 0), 1, 1, linewidth=1, edgecolor="k", facecolor="lightgrey", hatch="/////", zorder=10))
    ax.add_patch(Rectangle((7, 0), 1, 1, linewidth=1, edgecolor="k", facecolor="lightgrey", hatch="/////", zorder=10))


def setup_axes(ax):
    ax.cla(); ax.set_xlim(*X_LIM); ax.set_ylim(*Y_LIM)
    ax.set_aspect("equal"); add_cubes(ax)
    ax.set_xlabel(r"$x/H + U/U_0$"); ax.set_ylabel(r"$y/H$"); ax.margins(y=0.02)


def draw_angle_header(ax, ang):
    for t in list(ax.texts):
        if getattr(t, "_is_angle_header", False):
            t.remove()
    txt = rf"Angle: {int(round(float(ang)))}deg"
    t = ax.text(0.5, 1.22, txt, transform=ax.transAxes, ha="center", va="bottom", fontsize=10, clip_on=False)
    t._is_angle_header = True


def draw_legend(ax):
    handles = [
        Line2D([0], [0], color="black", lw=1.4, ls="-",  label="MF-ProfileCNN"),
        Line2D([0], [0], color="blue",  lw=1.2, ls="-",  label="LES"),
        Line2D([0], [0], color="blue",  lw=1.0, ls=":",  alpha=0.25, label="LES (train +/-5deg)"),
        Line2D([0], [0], color="red",   lw=1.2, ls="-",  label="RANS"),
    ]
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.145),
              frameon=False, ncol=4, handlelength=2.2, columnspacing=1.2,
              fontsize=9, borderaxespad=0.0).set_in_layout(False)


def draw_frame(it, ax, alpha_grid, model, y_ref, stats):
    setup_axes(ax)
    idx = max(0, min(int(it * 10), len(alpha_grid) - 1))
    ang = float(alpha_grid[idx])
    _, mf_u = predict_profiles_for_angle(model, y_ref, stats, ang)
    for i, x0 in enumerate(PROFILE_XS):
        r_red = rans_profile_nearest(ang, float(x0), y_ref)
        ax.plot(r_red + x0, y_ref, color="red",   linestyle="-", linewidth=1.2)
        if abs(ang - round(ang)) < 1e-6 and int(round(ang)) in LES_CASES:
            a_int = int(round(ang))
            y_h, u_h = load_line("LES", a_int, float(x0), u_inf=U_INF)
            ax.plot(_interp_to_yref(y_h, u_h, y_ref) + x0, y_ref, color="blue", linestyle="-", linewidth=1.2)
            for a_nb in [a_int - 5, a_int + 5]:
                if a_nb in LES_CASES and a_nb in TRAIN_ANGLES:
                    y_t, u_t = load_line("LES", a_nb, float(x0), u_inf=U_INF)
                    ax.plot(_interp_to_yref(y_t, u_t, y_ref) + x0, y_ref,
                            color="blue", linestyle=":", linewidth=1.0, alpha=0.25)
            ax.plot(mf_u[i] + x0, y_ref, color="black", linestyle="-", linewidth=1.4)
        ax.set_title(""); draw_legend(ax); draw_angle_header(ax, ang)


# ===========================================================================
# Training pipeline
# ===========================================================================
def train_profile_pipeline(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    y_ref = reference_y_grid()

    test_angles  = [30]
    val_angles   = [15] if (15 in LES_CASES and 15 not in test_angles) else [20]
    base_train   = [a for a in TRAIN_ANGLES if a not in test_angles and a not in val_angles]
    train_angles = sorted(set(base_train + [25]))
    if not train_angles:
        train_angles = [a for a in LES_CASES if a not in test_angles and a not in val_angles]

    r_tr, l_tr, x_tr, yaw_tr = build_augmented_samples(train_angles, PROFILE_XS, y_ref, n_interp=2)
    r_va, l_va, x_va, yaw_va = build_profile_samples(val_angles,     PROFILE_XS, y_ref)
    print(f"Train samples: {len(r_tr)} (augmented from {len(train_angles)} LES angles)")

    out   = train_profile_model(
        r_tr, l_tr, x_tr, yaw_tr,
        r_va, l_va, x_va, yaw_va,
        epochs=600, batch_size=32, lr=1e-3, grad_w=0.02,
        weight_decay=2e-4, patience=80, width=32, depth=4, dropout=0.12,
    )
    model = out["model"]; stats = out["stats"]

    torch.save({"state_dict": model.state_dict(), "stats": stats,
                "arch": {"width": 32, "depth": 4, "dropout": 0.12}},
               "checkpoints/profile_best.pt")

    for ta in test_angles:
        if int(ta) not in LES_CASES:
            continue
        r_te, l_te, x_te, yaw_te = build_profile_samples([ta], PROFILE_XS, y_ref)
        ds_te = ProfileDataset(r_te, l_te, x_te, yaw_te, stats=stats)
        pr_te = predict_profile(model, ds_te)
        print(f"PROFILE test@{ta}deg mse={float(np.mean((pr_te - l_te) ** 2)):.4e}")

    return model, y_ref, stats


# ===========================================================================
# Main
# ===========================================================================
def main():
    model, y_ref, stats = train_profile_pipeline(seed=42)
    alpha = np.linspace(0.0, 40.0, 1001)

    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE, squeeze=False, constrained_layout=True)
    ax = ax[0, 0]
    for ang in [0, 5, 10, 15, 20, 25, 30]:
        it = ang * len(alpha) / (10.0 * alpha[-1])
        draw_frame(it, ax, alpha, model, y_ref, stats)
        out_path = f"figures/profile_profilecnn_a{int(ang)}.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
        print(out_path)
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE, squeeze=False, constrained_layout=True)
    ax     = ax[0, 0]
    frames = int(len(alpha) / 10)
    anim   = animation.FuncAnimation(
        fig, lambda it: draw_frame(it, ax, alpha, model, y_ref, stats),
        frames=frames, interval=50, blit=False,
    )
    try:
        from matplotlib.animation import PillowWriter
        anim.save("figures/profile_profilecnn.gif", writer=PillowWriter(fps=20), dpi=150)
        print("figures/profile_profilecnn.gif")
    except Exception as e:
        print("GIF save failed:", e)
    plt.close(fig)


if __name__ == "__main__":
    main()
