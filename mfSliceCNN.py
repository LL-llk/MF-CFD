import os, random
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy import interpolate
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import Data_Extract as de

if not hasattr(np, "bool"):
    np.bool = bool

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SLICE_SURFACE = "yNormal"
RANS_CASES = [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40]
LES_CASES  = [0,5,10,15,20,25,30]

os.makedirs("figures", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class SlicePairDataset(Dataset):
    def __init__(self, lf, hf, ang, angle_max=40.0):
        lf = torch.tensor(lf, dtype=torch.float32).unsqueeze(1)
        hf = torch.tensor(hf, dtype=torch.float32).unsqueeze(1)
        a = torch.tensor(ang, dtype=torch.float32).reshape(-1, 1) / float(angle_max)
        if lf.shape[0] != hf.shape[0] or lf.shape[0] != a.shape[0]:
            raise ValueError("N mismatch among lf/hf/ang")
        self.lf = lf
        self.hf = hf
        self.a = a

    def __len__(self):
        return int(self.lf.shape[0])

    def __getitem__(self, idx):
        return self.lf[idx], self.hf[idx], self.a[idx]


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------
class DoubleConv(nn.Module):
    def __init__(self, c_in, c_out, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.SiLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(c_out, c_out, 3, padding=1),
            nn.SiLU(),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        return self.net(x)


class FiLM(nn.Module):
    def __init__(self, cond_dim, c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, 2 * c),
            nn.SiLU(),
            nn.Linear(2 * c, 2 * c),
        )

    def forward(self, feat, cond):
        gb = self.net(cond)
        c = feat.shape[1]
        g = gb[:, :c].unsqueeze(-1).unsqueeze(-1)
        b = gb[:, c:].unsqueeze(-1).unsqueeze(-1)
        return feat * (1.0 + g) + b


class SliceMFCNN(nn.Module):
    def __init__(self, base_c=16, cond_dim=1, dropout=0.1):
        super().__init__()
        c = int(base_c)
        self.in0 = DoubleConv(1, c, dropout=dropout)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c, 2 * c, dropout=dropout))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(2 * c, 4 * c, dropout=dropout))
        self.up1 = nn.ConvTranspose2d(4 * c, 2 * c, 2, stride=2)
        self.conv1 = DoubleConv(4 * c, 2 * c, dropout=dropout)
        self.up2 = nn.ConvTranspose2d(2 * c, c, 2, stride=2)
        self.conv2 = DoubleConv(2 * c, c, dropout=dropout)
        self.out = nn.Conv2d(c, 1, 1)
        self.f0 = FiLM(cond_dim, c)
        self.f1 = FiLM(cond_dim, 2 * c)
        self.f2 = FiLM(cond_dim, 4 * c)

    def forward(self, lf, a):
        x0 = self.f0(self.in0(lf), a)
        x1 = self.f1(self.down1(x0), a)
        x2 = self.f2(self.down2(x1), a)
        u1 = self.up1(x2)
        x = self.conv1(torch.cat([u1, x1], dim=1))
        u0 = self.up2(x)
        x = self.conv2(torch.cat([u0, x0], dim=1))
        return self.out(x)


# ---------------------------------------------------------------------------
# Loss and evaluation utilities
# ---------------------------------------------------------------------------
def grad_xy(u):
    dx = u[..., :, 1:] - u[..., :, :-1]
    dy = u[..., 1:, :] - u[..., :-1, :]
    return dx, dy


def laplacian_2d(u):
    """Second-order finite-difference Laplacian."""
    lap_x = u[..., :, 2:] - 2.0 * u[..., :, 1:-1] + u[..., :, :-2]
    lap_y = u[..., 2:, :] - 2.0 * u[..., 1:-1, :] + u[..., :-2, :]
    return lap_x, lap_y


def loss_fn(pred, targ, grad_w=2e-2, wake_thr=0.95, wake_w=2.0, grad_mode="l2", lap_w=0.0):
    mse = (pred - targ).pow(2).mean()
    px, py = grad_xy(pred)
    tx, ty = grad_xy(targ)
    if grad_mode == "l1":
        g = (px - tx).abs().mean() + (py - ty).abs().mean()
    else:
        g = (px - tx).pow(2).mean() + (py - ty).pow(2).mean()
    tot = mse + grad_w * g
    if float(lap_w) > 0:
        lx_p, ly_p = laplacian_2d(pred)
        lx_t, ly_t = laplacian_2d(targ)
        lap = (lx_p - lx_t).pow(2).mean() + (ly_p - ly_t).pow(2).mean()
        tot = tot + float(lap_w) * lap
    if wake_w is not None and float(wake_w) > 0:
        w = (targ < float(wake_thr)).float()
        if w.numel() > 0:
            tot = tot + float(wake_w) * ((pred - targ).pow(2) * w).mean()
    return tot, mse, g


@torch.no_grad()
def eval_mse(model, dl, device, grad_w=2e-2, wake_thr=0.95, wake_w=2.0, grad_mode="l2", lap_w=0.0, residual=False):
    model.eval()
    tot_s = 0.0
    mse_s = 0.0
    grad_s = 0.0
    n = 0
    for lf, hf, a in dl:
        lf = lf.to(device)
        hf = hf.to(device)
        a = a.to(device)
        targ = hf - lf if residual else hf
        pr = model(lf, a)
        tot, mse, g = loss_fn(pr, targ, grad_w=grad_w, wake_thr=wake_thr, wake_w=wake_w, grad_mode=grad_mode, lap_w=lap_w)
        b = lf.shape[0]
        tot_s += float(tot.item()) * b
        mse_s += float(mse.item()) * b
        grad_s += float(g.item()) * b
        n += b
    n = max(n, 1)
    return tot_s / n, mse_s / n, grad_s / n


# ---------------------------------------------------------------------------
# Learning rate scheduler
# ---------------------------------------------------------------------------
class WarmupCosineLR:
    def __init__(self, opt, warmup_epochs, total_epochs, min_lr=1e-6):
        self.opt = opt
        self.warm = max(int(warmup_epochs), 0)
        self.T = max(int(total_epochs), 1)
        self.min_lr = float(min_lr)
        self.base_lrs = [g["lr"] for g in opt.param_groups]
        self.ep = 0

    def step(self):
        self.ep += 1
        for i, g in enumerate(self.opt.param_groups):
            base = self.base_lrs[i]
            if self.ep <= self.warm and self.warm > 0:
                lr = base * (self.ep / float(self.warm))
            else:
                t = (self.ep - self.warm) / float(max(1, self.T - self.warm))
                t = min(max(t, 0.0), 1.0)
                lr = self.min_lr + 0.5 * (base - self.min_lr) * (1.0 + np.cos(np.pi * t))
            g["lr"] = float(lr)

    def get_lr(self):
        return [g["lr"] for g in self.opt.param_groups]


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_model(
        lf_tr, hf_tr, ang_tr,
        lf_va=None, hf_va=None, ang_va=None,
        epochs=200,
        bs=2,
        lr=2e-3,
        wd=1e-4,
        patience=30,
        base_c=16,
        dropout=0.1,
        grad_w=2e-2,
        lap_w=0.0,
        wake_thr=0.95,
        wake_w=2.0,
        grad_mode="l2",
        residual=False,
        ckpt_path=None,
        device=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds_tr = SlicePairDataset(lf_tr, hf_tr, ang_tr, angle_max=40.0)
    dl_tr = DataLoader(ds_tr, batch_size=int(bs), shuffle=True)

    dl_va = None
    if lf_va is not None and hf_va is not None and ang_va is not None:
        ds_va = SlicePairDataset(lf_va, hf_va, ang_va, angle_max=40.0)
        dl_va = DataLoader(ds_va, batch_size=int(bs), shuffle=False)

    model = SliceMFCNN(base_c=base_c, dropout=dropout).to(device)
    opt = optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(wd))
    sched = WarmupCosineLR(opt, warmup_epochs=max(1, int(0.1 * epochs)), total_epochs=int(epochs), min_lr=1e-6)

    best = float("inf")
    bad = 0
    best_state = None

    for ep in range(1, int(epochs) + 1):
        model.train()
        for lf, hf, a in dl_tr:
            lf = lf.to(device)
            hf = hf.to(device)
            a = a.to(device)
            targ = hf - lf if residual else hf
            pr = model(lf, a)
            loss, _, _ = loss_fn(pr, targ, grad_w=grad_w, wake_thr=wake_thr, wake_w=wake_w,
                                 grad_mode=grad_mode, lap_w=lap_w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        if dl_va is None:
            continue

        va, va_mse, va_g = eval_mse(model, dl_va, device, grad_w=grad_w, wake_thr=wake_thr, wake_w=wake_w,
                                    grad_mode=grad_mode, lap_w=lap_w, residual=residual)
        print(f"[{ep:03d}] train=? | val={va:.4e} val_mse={va_mse:.4e} val_grad={va_g:.4e} lr={sched.get_lr()[0]:.2e}")

        if va < best - 1e-7:
            best = va
            bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if ckpt_path:
                torch.save(best_state, ckpt_path)
        else:
            bad += 1
            if bad >= int(patience):
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def _raw(prefix, ang):
    df = de.get_surface(f"{prefix}/Yaw/a{int(ang)}", surface=SLICE_SURFACE, field="UMean")
    normal = "z" if SLICE_SURFACE[0].lower() == "y" else "y"
    x = df["x"].to_numpy().astype(np.float32)
    z = df[normal].to_numpy().astype(np.float32)
    u = df["UMean_x"].to_numpy().astype(np.float32)
    return x, z, u


def _grid(nx, ny):
    df = de.get_surface(f"RANS/Yaw/a{RANS_CASES[0]}", surface=SLICE_SURFACE, field="UMean")
    normal = "z" if SLICE_SURFACE[0].lower() == "y" else "y"
    x = df["x"].to_numpy()
    y = df[normal].to_numpy()
    return np.meshgrid(np.linspace(x.min(), x.max(), nx), np.linspace(y.min(), y.max(), ny))


def _slice(prefix, ang, Xg, Yg):
    x, z, u = _raw(prefix, ang)
    pts = np.column_stack((x, z))
    lin = interpolate.LinearNDInterpolator(pts, u, fill_value=np.nan)
    U = lin(Xg, Yg).astype(np.float32)
    if np.isnan(U).any():
        nn = interpolate.NearestNDInterpolator(pts, u)
        U2 = nn(Xg, Yg).astype(np.float32)
        U = np.where(np.isnan(U), U2, U)
    return U


def build_all(nx=256, ny=128, u_inf=1.0):
    Xg, Yg = _grid(nx, ny)
    rA = np.array(RANS_CASES, dtype=np.float32)
    lA = np.array(LES_CASES, dtype=np.float32)
    rS = np.stack([_slice("RANS", a, Xg, Yg) for a in RANS_CASES], 0) / float(u_inf)
    lS = np.stack([_slice("LES",  a, Xg, Yg) for a in LES_CASES ], 0) / float(u_inf)
    bx, bz, _ = _raw("RANS", RANS_CASES[0])
    return Xg, Yg, bx, bz, rA, rS, lA, lS


def pair(angle, rA, rS, lA, lS):
    a = float(angle)
    j = np.where(np.isclose(lA, a, atol=1e-6))[0]
    if j.size == 0:
        raise ValueError(f"LES angle {a} not found in {lA.tolist()}")
    hf = lS[int(j[0])]

    i = np.where(np.isclose(rA, a, atol=1e-6))[0]
    if i.size > 0:
        return rS[int(i[0])], hf

    k = int(np.searchsorted(rA, a))
    a0, a1 = float(rA[k - 1]), float(rA[k])
    t = (a - a0) / (a1 - a0 + 1e-12)
    lf = (1.0 - t) * rS[k - 1] + t * rS[k]
    return lf.astype(np.float32), hf


def stack_pairs(angles, rA, rS, lA, lS):
    lf_list, hf_list = [], []
    for a in angles:
        lf, hf = pair(a, rA, rS, lA, lS)
        lf_list.append(lf)
        hf_list.append(hf)
    return np.stack(lf_list, 0), np.stack(hf_list, 0)


def aug_slices(train_angles, rA, rS, lA, lS, n_interp=2):
    """Build training set with linearly interpolated slices between adjacent LES angles."""
    sorted_a = sorted(train_angles)
    cache = {}
    for a in sorted_a:
        cache[a] = pair(a, rA, rS, lA, lS)

    lf_all, hf_all, ang_all = [], [], []
    for a in sorted_a:
        lf_all.append(cache[a][0])
        hf_all.append(cache[a][1])
        ang_all.append(float(a))

    for i in range(len(sorted_a) - 1):
        a0, a1 = sorted_a[i], sorted_a[i + 1]
        lf0, hf0 = cache[a0]
        lf1, hf1 = cache[a1]
        for k in range(1, n_interp + 1):
            t = k / (n_interp + 1)
            a_mid = a0 + t * (a1 - a0)
            lf_all.append(((1 - t) * lf0 + t * lf1).astype(np.float32))
            hf_all.append(((1 - t) * hf0 + t * hf1).astype(np.float32))
            ang_all.append(float(a_mid))

    return np.stack(lf_all), np.stack(hf_all), np.array(ang_all, dtype=np.float32)


# ---------------------------------------------------------------------------
# Prediction and scoring
# ---------------------------------------------------------------------------
@torch.no_grad()
def pred_field(model, lf2d, ang, angle_max=40.0, residual=True):
    dev = next(model.parameters()).device
    x = torch.tensor(lf2d, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(dev)
    a = torch.tensor([[float(ang) / float(angle_max)]], dtype=torch.float32).to(dev)
    y = model(x, a).detach().cpu().numpy()[0, 0]
    return y + lf2d if residual else y


@torch.no_grad()
def pool_scores(model, pool_angles, rA, rS, lA, lS):
    scores = []
    for a in pool_angles:
        lf, hf = pair(a, rA, rS, lA, lS)
        pr = pred_field(model, lf, a)
        scores.append((a, float(np.mean((pr - hf) ** 2))))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------
def mosaic(angles, lf_list, pr_list, hf_list, path):
    all_vel = lf_list + pr_list + hf_list
    vmin = float(np.min([a.min() for a in all_vel]))
    vmax = float(np.max([a.max() for a in all_vel]))
    errs = [pr_list[i] - hf_list[i] for i in range(len(angles))]
    emax = float(np.max([np.max(np.abs(e)) for e in errs]))
    emax = max(emax, 1e-6)

    n = len(angles)
    fig, ax = plt.subplots(n, 4, figsize=(18, 2.2 * n), dpi=200)
    if n == 1:
        ax = np.expand_dims(ax, 0)

    titles = ["Low-Fidelity", "Multi-fidelity", "High-Fidelity", "Error"]
    for j, t in enumerate(titles):
        ax[0, j].set_title(t, fontsize=14)

    vel_m = plt.cm.ScalarMappable(cmap="inferno"); vel_m.set_clim(vmin, vmax)
    err_m = plt.cm.ScalarMappable(cmap="coolwarm"); err_m.set_clim(-emax, emax)

    for i, a in enumerate(angles):
        ax[i, 0].imshow(lf_list[i], origin="lower", cmap="inferno", vmin=vmin, vmax=vmax, interpolation="none")
        ax[i, 1].imshow(pr_list[i], origin="lower", cmap="inferno", vmin=vmin, vmax=vmax, interpolation="none")
        ax[i, 2].imshow(hf_list[i], origin="lower", cmap="inferno", vmin=vmin, vmax=vmax, interpolation="none")
        ax[i, 3].imshow(errs[i], origin="lower", cmap="coolwarm", vmin=-emax, vmax=emax, interpolation="none")

        for j in range(4):
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
        ax[i, 0].set_ylabel(f"{int(a)}°", rotation=0, labelpad=25, fontsize=12, va="center")

    fig.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.12, wspace=0.02, hspace=0.08)

    cax_vel = fig.add_axes([0.12, 0.05, 0.62, 0.02])
    cb_vel = fig.colorbar(vel_m, cax=cax_vel, orientation="horizontal")
    cb_vel.set_label("Velocity (normalised)", fontsize=11)
    cb_vel.ax.tick_params(labelsize=9)

    cax_err = fig.add_axes([0.78, 0.05, 0.18, 0.02])
    cb_err = fig.colorbar(err_m, cax=cax_err, orientation="horizontal")
    cb_err.set_label("Error", fontsize=11)
    cb_err.ax.tick_params(labelsize=9)

    fig.savefig(path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(path)


def _to_base_from_raw(prefix, ang, bx, bz, u_inf=1.0):
    x, z, u = _raw(prefix, ang)
    nn = interpolate.NearestNDInterpolator(np.column_stack((x, z)), u / float(u_inf))
    return nn(bx, bz).astype(np.float32)


def _to_base_from_grid(Xg, Yg, F, bx, bz):
    ny, nx = F.shape
    xs = Xg[0, :]
    ys = Yg[:, 0]
    rgi = interpolate.RegularGridInterpolator((ys, xs), F, method="linear",
                                              bounds_error=False, fill_value=None)
    pts = np.column_stack((bz.astype(np.float32), bx.astype(np.float32)))
    return rgi(pts).astype(np.float32)


def _add_cubes_aw(ax, normal='y'):
    if normal == 'y':
        ax.add_patch(Rectangle((2.0, -0.5), 1.0, 1.0,
                               linewidth=1.0, edgecolor='k',
                               facecolor='lightgrey', hatch='/////',
                               zorder=10))
        ax.add_patch(Rectangle((7.0, -0.5), 1.0, 1.0,
                               linewidth=1.0, edgecolor='k',
                               facecolor='lightgrey', hatch='/////',
                               zorder=10))
    else:
        ax.add_patch(Rectangle((2.0, -0.5), 1.0, 1.0,
                               linewidth=1.0, edgecolor='k',
                               facecolor='lightgrey', hatch='/////',
                               zorder=10))
        ax.add_patch(Rectangle((7.0, -0.5), 1.0, 1.0,
                               linewidth=1.0, edgecolor='k',
                               facecolor='lightgrey', hatch='/////',
                               zorder=10))


def contour_fig(bx, bz, lf_b, hf_b, mf_b, ang, outpath,
                level=0.5, H=1.0,
                figsize=(7.0, 6.5),
                xlim=(0.0, 12.0),
                zlim=(-4.25, 4.25),
                dpi=400):

    X = (bx.astype(np.float32) / float(H))
    Z = (bz.astype(np.float32) / float(H))

    fig, ax = plt.subplots(1, 1, figsize=figsize,
                           squeeze=True, constrained_layout=True, dpi=dpi)

    lf_plot = ax.tricontour(X, Z, lf_b, colors='r', levels=[level], linewidths=2)
    hf_plot = ax.tricontour(X, Z, hf_b, colors='b', levels=[level], linewidths=2)
    mf_plot = ax.tricontour(X, Z, mf_b, colors='k', levels=[level], linewidths=2)

    _add_cubes_aw(ax, 'y')
    ax.set_aspect('equal', adjustable='box')

    ax.set_xlabel(r'$x/H$', fontsize=22)
    ax.set_ylabel(r'$z/H$', fontsize=22)
    ax.set_title(rf'$\alpha = {int(ang)}^\circ$', fontsize=24, pad=10)

    ax.tick_params(axis='both', labelsize=16)
    for s in ax.spines.values():
        s.set_linewidth(1.5)

    if xlim is not None:
        ax.set_xlim(*xlim)
    if zlim is not None:
        ax.set_ylim(*zlim)

    h1, = ax.plot([], [], color='r', lw=2, label='RANS (or interp.)')
    h2, = ax.plot([], [], color='b', lw=2, label='LES')
    h3, = ax.plot([], [], color='k', lw=2, label='MF-CNN')
    ax.legend(handles=[h1, h2, h3],
              loc='lower left', ncol=3,
              columnspacing=0.5, frameon=False, fontsize=14)

    fig.savefig(outpath, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(outpath)


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------
def main():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    Xg, Yg, bx, bz, rA, rS, lA, lS = build_all(nx=256, ny=128, u_inf=1.0)

    # 30° = test (held-out). Train on ALL other LES angles with augmentation.
    test_angles  = [30]
    val_angles   = [15]                                              # use 15° as val → 25° joins training
    train_angles = [a for a in LES_CASES if a not in test_angles + val_angles]
    # → [0, 5, 10, 20, 25]  (max training angle now 25°, only 5° gap to test)

    # 2D slice augmentation: n_interp=3 for denser coverage near high angles
    lf_tr, hf_tr, ang_tr = aug_slices(train_angles, rA, rS, lA, lS, n_interp=3)
    lf_va, hf_va          = stack_pairs(val_angles, rA, rS, lA, lS)
    ang_va                 = np.array(val_angles, dtype=np.float32)

    print(f"Train slices : {len(lf_tr)} (real={len(train_angles)}, aug={len(lf_tr)-len(train_angles)})")
    print(f"Val angles   : {val_angles}")
    print(f"Test angles  : {test_angles}")

    model = train_model(
        lf_tr, hf_tr, ang_tr,
        lf_va, hf_va, ang_va,
        epochs=600,
        bs=4,
        lr=2e-3,
        wd=1e-4,
        patience=70,
        base_c=24,
        dropout=0.08,
        grad_w=0.20,
        lap_w=0.05,
        wake_thr=0.95,
        wake_w=0.0,
        grad_mode="l2",
        residual=True,
        ckpt_path="checkpoints/slice_best_final.pt",
    )

    # Evaluate on every LES angle
    print("\n--- Per-angle MSE ---")
    for a in LES_CASES:
        lf, hf = pair(a, rA, rS, lA, lS)
        pr = pred_field(model, lf, a)
        if a in test_angles:
            tag = "(TEST)"
        elif a in val_angles:
            tag = "(val)"
        else:
            tag = "(train)"
        print(f"  a={a:2d} {tag}: mse={float(np.mean((pr - hf)**2)):.4e}")

    vis = [0, 5, 10, 15, 20, 25, 30]
    lf_list, pr_list, hf_list = [], [], []
    for a in vis:
        lf, hf = pair(a, rA, rS, lA, lS)
        mf = pred_field(model, lf, a)
        lf_list.append(lf)
        pr_list.append(mf)
        hf_list.append(hf)
    mosaic(vis, lf_list, pr_list, hf_list, "figures/slice_mosaic.png")

    contour_angles = [0, 10, 20, 30]
    for a in contour_angles:
        lf, hf = pair(a, rA, rS, lA, lS)
        mf = pred_field(model, lf, a)

        lf_b = _to_base_from_raw("RANS", a, bx, bz, u_inf=1.0)
        hf_b = _to_base_from_raw("LES",  a, bx, bz, u_inf=1.0)
        mf_b = _to_base_from_grid(Xg, Yg, mf, bx, bz)

        contour_fig(
            bx, bz, lf_b, hf_b, mf_b, a,
            f"figures/slice_contour_a{int(a)}.png",
            level=0.5, H=1.0
        )


if __name__ == "__main__":
    main()
