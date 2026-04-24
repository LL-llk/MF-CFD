import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

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
        self.les_u = les_u
        self.x_pos = x_pos
        self.yaw = yaw
        self.delta = delta

        if stats is None:
            if delta is None:
                raise ValueError("stats must be provided when les_u is None")
            rans_mu = float(rans_u.mean())
            rans_sd = float(rans_u.std() + 1e-8)
            del_mu = float(delta.mean())
            del_sd = float(delta.std() + 1e-8)
            x_mu = float(x_pos.mean())
            x_sd = float(x_pos.std() + 1e-8)
            y_mu = float(yaw.mean())
            y_sd = float(yaw.std() + 1e-8)
            stats = {
                "rans_mu": rans_mu, "rans_sd": rans_sd,
                "del_mu": del_mu, "del_sd": del_sd,
                "x_mu": x_mu, "x_sd": x_sd,
                "y_mu": y_mu, "y_sd": y_sd,
            }

        self.stats = {k: float(v) for k, v in stats.items()}
        self.rans_mu = self.stats["rans_mu"]
        self.rans_sd = self.stats["rans_sd"]
        self.del_mu = self.stats["del_mu"]
        self.del_sd = self.stats["del_sd"]
        self.x_mu = self.stats["x_mu"]
        self.x_sd = self.stats["x_sd"]
        self.y_mu = self.stats["y_mu"]
        self.y_sd = self.stats["y_sd"]

        self.rans_n = (self.rans_u - self.rans_mu) / self.rans_sd
        self.x_n = (self.x_pos - self.x_mu) / self.x_sd
        self.y_n = (self.yaw - self.y_mu) / self.y_sd

        if self.delta is not None:
            self.del_n = (self.delta - self.del_mu) / self.del_sd
        else:
            self.del_n = None

    def __len__(self):
        return int(self.rans_u.shape[0])

    def __getitem__(self, idx):
        r = torch.from_numpy(self.rans_n[idx]).unsqueeze(0)
        x = torch.tensor(self.x_n[idx], dtype=torch.float32)
        y = torch.tensor(self.y_n[idx], dtype=torch.float32)
        if self.del_n is None:
            return {"rans_n": r, "x": x, "yaw": y}
        d = torch.from_numpy(self.del_n[idx]).unsqueeze(0)
        return {"rans_n": r, "del_n": d, "x": x, "yaw": y}


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
        self.head = nn.Conv1d(w, 1, kernel_size=1)

    def forward(self, rans_n, x, yaw):
        feat = self.backbone(rans_n)
        cond = torch.stack([x, yaw], dim=1)
        gb = self.cond(cond)
        c = feat.shape[1]
        gamma = gb[:, :c].unsqueeze(-1)
        beta = gb[:, c:].unsqueeze(-1)
        feat = feat * (1.0 + gamma) + beta
        return self.head(feat)


def _grad_1d(u):
    return u[..., 1:] - u[..., :-1]


@torch.no_grad()
def predict_profile(model, dataset: ProfileDataset, device=None, batch_size=128):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)
    dl = DataLoader(dataset, batch_size=int(batch_size), shuffle=False)
    preds = []
    for b in dl:
        r = b["rans_n"].to(device)
        x = b["x"].to(device)
        y = b["yaw"].to(device)
        del_n = model(r, x, y).squeeze(1).detach().cpu().numpy()
        del_u = del_n * dataset.del_sd + dataset.del_mu
        rans_u = (r.squeeze(1).detach().cpu().numpy() * dataset.rans_sd + dataset.rans_mu)
        preds.append(rans_u + del_u)
    return np.concatenate(preds, axis=0)


def train_profile_model(
        rans_u_train, les_u_train, x_train, yaw_train,
        rans_u_val, les_u_val, x_val, yaw_val,
        epochs=200,
        batch_size=32,
        lr=2e-3,
        grad_w=5e-2,
        weight_decay=2e-4,
        patience=40,
        device=None,
        width=64,
        depth=5,
        dropout=0.1,
        wake_thr=0.95,
        wake_w=2.0
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds_tr = ProfileDataset(rans_u_train, les_u_train, x_train, yaw_train, stats=None)
    ds_va = ProfileDataset(rans_u_val, les_u_val, x_val, yaw_val, stats=ds_tr.stats)

    ntr = len(ds_tr)
    bs = int(batch_size)
    if ntr < bs:
        bs = max(1, ntr)

    dl_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=max(1, min(bs, len(ds_va))), shuffle=False, drop_last=False)

    model = ProfileResidual1DCNN(width=width, depth=depth, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    sched = WarmupCosineLR(opt, warmup_epochs=max(1, int(0.05 * epochs)), total_epochs=int(epochs), min_lr=1e-6)

    rans_mu = float(ds_tr.rans_mu)
    rans_sd = float(ds_tr.rans_sd)
    del_mu = float(ds_tr.del_mu)
    del_sd = float(ds_tr.del_sd)

    best = float("inf")
    bad = 0
    best_state = None

    for ep in range(1, int(epochs) + 1):
        model.train()
        tr = 0.0
        nbt = 0

        for b in dl_tr:
            r = b["rans_n"].to(device)
            d = b["del_n"].to(device)
            x = b["x"].to(device)
            y = b["yaw"].to(device)

            p = model(r, x, y)

            mse = (p - d).pow(2).mean()
            g = (_grad_1d(p) - _grad_1d(d)).pow(2).mean()
            loss = mse + float(grad_w) * g

            if wake_w is not None and float(wake_w) > 0:
                rans_u = r * rans_sd + rans_mu
                del_true = d * del_sd + del_mu
                del_pred = p * del_sd + del_mu

                u_true = rans_u + del_true
                u_pred = rans_u + del_pred

                wmask = (u_true < float(wake_thr)).float()
                if wmask.numel() > 0:
                    wake_loss = ((u_pred - u_true).pow(2) * wmask).mean()
                    loss = loss + float(wake_w) * wake_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tr += float(loss.item())
            nbt += 1

        if nbt == 0:
            raise RuntimeError(f"Profile training has zero batches (N={len(ds_tr)}, batch_size={batch_size}).")

        tr = tr / float(nbt)

        model.eval()
        va = 0.0
        nbv = 0
        with torch.no_grad():
            for b in dl_va:
                r = b["rans_n"].to(device)
                d = b["del_n"].to(device)
                x = b["x"].to(device)
                y = b["yaw"].to(device)

                p = model(r, x, y)

                mse = (p - d).pow(2).mean()
                g = (_grad_1d(p) - _grad_1d(d)).pow(2).mean()
                loss = mse + float(grad_w) * g

                if wake_w is not None and float(wake_w) > 0:
                    rans_u = r * rans_sd + rans_mu
                    del_true = d * del_sd + del_mu
                    del_pred = p * del_sd + del_mu
                    u_true = rans_u + del_true
                    u_pred = rans_u + del_pred
                    wmask = (u_true < float(wake_thr)).float()
                    if wmask.numel() > 0:
                        wake_loss = ((u_pred - u_true).pow(2) * wmask).mean()
                        loss = loss + float(wake_w) * wake_loss

                va += float(loss.item())
                nbv += 1

        va = va / float(max(1, nbv))
        sched.step()
        print(
            f"[{ep:03d}] train={tr:.4e} | val={va:.4e} lr={sched.get_lr()[0]:.2e} grad_w={float(grad_w):.3g} wake_thr={float(wake_thr):.3g} wake_w={float(wake_w):.3g}")

        if va < best - 1e-7:
            best = va
            bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= int(patience):
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {"model": model, "train_ds": ds_tr, "val_ds": ds_va, "best_val": best, "stats": ds_tr.stats}


