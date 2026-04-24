import os, random
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import animation
import Data_Extract as de
from mfCNN import train_profile_model, predict_profile, ProfileDataset
from matplotlib.lines import Line2D
if not hasattr(np, "bool"):
    np.bool = bool

RANS_CASES = [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40]
LES_CASES  = [0,5,10,15,20,25,30]

FIGSIZE = (11, 3)
X_LIM = (0, 15)
Y_LIM = (0, 2)
PROFILE_XS = list(np.arange(0, 14, 1))
U_INF = 1.0
ANGLE_MAX = 40.0
TRAIN_ANGLES = [a for a in LES_CASES if a % 10 == 0]

os.makedirs("figures", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)


def add_cubes(ax):
    ax.add_patch(Rectangle((2, 0), 1, 1, linewidth=1, edgecolor="k", facecolor="lightgrey", hatch="/////", zorder=10))
    ax.add_patch(Rectangle((7, 0), 1, 1, linewidth=1, edgecolor="k", facecolor="lightgrey", hatch="/////", zorder=10))


def setup_axes(ax):
    ax.cla()
    ax.set_xlim(*X_LIM)
    ax.set_ylim(*Y_LIM)
    ax.set_aspect("equal")
    add_cubes(ax)
    ax.set_xlabel(r"$x/H + U/U_0$")
    ax.set_ylabel(r"$y/H$")
    ax.margins(y=0.02)

def draw_angle_header(ax, ang):
    for t in list(ax.texts):
        if getattr(t, "_is_angle_header", False):
            t.remove()

    a = int(round(float(ang)))
    txt = rf"Angle: {a}°"
    t = ax.text(0.5, 1.22, txt, transform=ax.transAxes, ha="center", va="bottom", fontsize=10, clip_on=False)
    t._is_angle_header = True

def draw_legend(ax):
    handles = [
        Line2D([0], [0], color="black", lw=1.4, ls="-", label="MF-ProfileCNN"),
        Line2D([0], [0], color="blue",  lw=1.2, ls="-", label="LES"),
        Line2D([0], [0], color="blue",  lw=1.0, ls=":", alpha=0.25, label=r"LES (train $\pm 5^\circ$)"),
        Line2D([0], [0], color="red",   lw=1.2, ls="-", label="RANS"),
    ]

    if ax.get_legend() is not None:
        ax.get_legend().remove()

    leg = ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.145), frameon=False,
                    ncol=4, handlelength=2.2, columnspacing=1.2, fontsize=9, borderaxespad=0.0)
    leg.set_in_layout(False)

def load_line(prefix, ang, x_pos, field="UMean", u_inf=1.0):
    line = de.get_line(f"{prefix}/Yaw/a{int(ang)}", float(x_pos), field=field)
    y = line[:, 0].astype(np.float64)
    u = line[:, -3].astype(np.float64) / float(u_inf)
    return y, u


def reference_y_grid():
    y, _ = load_line("RANS", RANS_CASES[0], PROFILE_XS[0], field="UMean", u_inf=U_INF)
    y = y[(y >= Y_LIM[0]) & (y <= Y_LIM[1])]
    return y.astype(np.float64)


def _interp_to_yref(y, u, y_ref):
    idx = np.argsort(y)
    y2 = y[idx]
    u2 = u[idx]
    return np.interp(y_ref, y2, u2, left=u2[0], right=u2[-1]).astype(np.float32)


def rans_profile_at_angle_interp(ang, x_pos, y_ref):
    a = float(ang)
    rA = np.asarray(RANS_CASES, dtype=np.float64)
    if a <= rA.min():
        y, u = load_line("RANS", int(rA.min()), x_pos, u_inf=U_INF)
        return _interp_to_yref(y, u, y_ref)
    if a >= rA.max():
        y, u = load_line("RANS", int(rA.max()), x_pos, u_inf=U_INF)
        return _interp_to_yref(y, u, y_ref)
    k = int(np.searchsorted(rA, a))
    a0, a1 = float(rA[k - 1]), float(rA[k])
    t = (a - a0) / (a1 - a0 + 1e-12)
    y0, u0 = load_line("RANS", int(a0), x_pos, u_inf=U_INF)
    y1, u1 = load_line("RANS", int(a1), x_pos, u_inf=U_INF)
    u0i = _interp_to_yref(y0, u0, y_ref)
    u1i = _interp_to_yref(y1, u1, y_ref)
    return ((1.0 - t) * u0i + t * u1i).astype(np.float32)


def rans_profile_nearest(ang, x_pos, y_ref):
    a = float(ang)
    nearest = int(RANS_CASES[int(np.argmin(np.abs(np.asarray(RANS_CASES, dtype=np.float64) - a)))])
    y, u = load_line("RANS", nearest, x_pos, u_inf=U_INF)
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
            r_list.append(r)
            h_list.append(h)
            x_list2.append(float(x0))
            a_list.append(float(a))
    rans_u = np.stack(r_list, 0).astype(np.float32)
    les_u = np.stack(h_list, 0).astype(np.float32)
    x_pos = np.asarray(x_list2, dtype=np.float32)
    yaw = np.asarray(a_list, dtype=np.float32)
    return rans_u, les_u, x_pos, yaw


def predict_profiles_for_angle(model, y_ref, stats, ang):
    lf = []
    for x0 in PROFILE_XS:
        lf.append(rans_profile_at_angle_interp(float(ang), float(x0), y_ref))
    lf_u = np.stack(lf, 0).astype(np.float32)
    x_pos = np.asarray(PROFILE_XS, dtype=np.float32)
    yaw = np.full((len(PROFILE_XS),), float(ang), dtype=np.float32)
    ds = ProfileDataset(lf_u, les_u=None, x_pos=x_pos, yaw=yaw, stats=stats)
    mf_u = predict_profile(model, ds).astype(np.float32)
    return lf_u, mf_u


def draw_frame(it, ax, alpha_grid, model, y_ref, stats):
    setup_axes(ax)
    idx = int(it * 10)
    idx = max(0, min(idx, len(alpha_grid) - 1))
    ang = float(alpha_grid[idx])

    _, mf_u = predict_profiles_for_angle(model, y_ref, stats, ang)

    for i, x0 in enumerate(PROFILE_XS):
        r_red = rans_profile_nearest(ang, float(x0), y_ref)
        ax.plot(r_red + x0, y_ref, color="red", linestyle="-", linewidth=1.2)

        if abs(ang - round(ang)) < 1e-6 and int(round(ang)) in LES_CASES:
            a_int = int(round(ang))
            y_h, u_h = load_line("LES", a_int, float(x0), u_inf=U_INF)
            u_h_i = _interp_to_yref(y_h, u_h, y_ref)
            ax.plot(u_h_i + x0, y_ref, color="blue", linestyle="-", linewidth=1.2)

            for a_nb in [a_int - 5, a_int + 5]:
                if a_nb in LES_CASES and a_nb in TRAIN_ANGLES:
                    y_t, u_t = load_line("LES", a_nb, float(x0), u_inf=U_INF)
                    u_t_i = _interp_to_yref(y_t, u_t, y_ref)
                    ax.plot(u_t_i + x0, y_ref, color="blue", linestyle=":", linewidth=1.0, alpha=0.25)

            ax.plot(mf_u[i] + x0, y_ref, color="black", linestyle="-", linewidth=1.4)

        ax.set_title("")
        draw_legend(ax)
        draw_angle_header(ax, ang)

def build_augmented_samples(train_angles, x_list, y_ref, n_interp=2):
    """Add linearly-interpolated LES profiles between adjacent training angles."""
    r_all, h_all, x_all, a_all = build_profile_samples(train_angles, x_list, y_ref)

    sorted_a = sorted(train_angles)
    r_extra, h_extra, x_extra, a_extra = [], [], [], []
    for i in range(len(sorted_a) - 1):
        a0, a1 = float(sorted_a[i]), float(sorted_a[i + 1])
        for k in range(1, n_interp + 1):
            t = k / (n_interp + 1)
            a_mid = a0 + t * (a1 - a0)
            for x0 in x_list:
                r_mid = rans_profile_at_angle_interp(a_mid, float(x0), y_ref)
                y0, u0 = load_line("LES", int(a0), float(x0), u_inf=U_INF)
                y1, u1 = load_line("LES", int(a1), float(x0), u_inf=U_INF)
                u0i = _interp_to_yref(y0, u0, y_ref)
                u1i = _interp_to_yref(y1, u1, y_ref)
                h_mid = ((1 - t) * u0i + t * u1i).astype(np.float32)
                r_extra.append(r_mid)
                h_extra.append(h_mid)
                x_extra.append(float(x0))
                a_extra.append(float(a_mid))

    if r_extra:
        r_all = np.concatenate([r_all, np.stack(r_extra)])
        h_all = np.concatenate([h_all, np.stack(h_extra)])
        x_all = np.concatenate([x_all, np.array(x_extra, dtype=np.float32)])
        a_all = np.concatenate([a_all, np.array(a_extra, dtype=np.float32)])
    return r_all, h_all, x_all, a_all


def train_profile_pipeline(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    y_ref = reference_y_grid()

    test_angles = [30]
    val_angles = [15] if (15 in LES_CASES and 15 not in test_angles) else [20]

    base_train = [a for a in TRAIN_ANGLES if (a not in test_angles and a not in val_angles)]
    train_angles = sorted(set(base_train + [25]))

    if len(train_angles) == 0:
        train_angles = [a for a in LES_CASES if (a not in test_angles and a not in val_angles)]

    # Use augmented training data (2 interpolated angles between each pair)
    r_tr, l_tr, x_tr, yaw_tr = build_augmented_samples(train_angles, PROFILE_XS, y_ref, n_interp=2)
    r_va, l_va, x_va, yaw_va = build_profile_samples(val_angles, PROFILE_XS, y_ref)

    print(f"Train samples: {len(r_tr)} (augmented from {len(train_angles)} LES angles)")

    out = train_profile_model(
        r_tr, l_tr, x_tr, yaw_tr,
        r_va, l_va, x_va, yaw_va,
        epochs=600,
        batch_size=32,
        lr=1e-3,
        grad_w=0.02,
        weight_decay=2e-4,
        patience=80,
        width=32,
        depth=4,
        dropout=0.12
    )

    model = out["model"]
    stats = out["stats"]

    torch.save(
        {"state_dict": model.state_dict(),
         "stats": stats,
         "arch": {"width": 32, "depth": 4, "dropout": 0.12}},
        "checkpoints/profile_best.pt"
    )

    for ta in test_angles:
        if int(ta) not in LES_CASES:
            continue
        r_te, l_te, x_te, yaw_te = build_profile_samples([ta], PROFILE_XS, y_ref)
        ds_te = ProfileDataset(r_te, l_te, x_te, yaw_te, stats=stats)
        pr_te = predict_profile(model, ds_te)
        mse = float(np.mean((pr_te - l_te) ** 2))
        print(f"PROFILE test@{ta}° mse={mse:.4e}")

    return model, y_ref, stats



def main():
    model, y_ref, stats = train_profile_pipeline(seed=42)
    alpha = np.linspace(0.0, 40.0, 1001)

    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE, squeeze=False, constrained_layout=True)
    ax = ax[0, 0]
    for ang in [0, 5, 10, 15, 20, 25, 30]:
        it = ang * len(alpha) / (10.0 * alpha[-1])
        draw_frame(it, ax, alpha, model, y_ref, stats)
        out = f"figures/profile_profilecnn_a{int(ang)}.png"
        fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.02)
        print(out)
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE, squeeze=False, constrained_layout=True)
    ax = ax[0, 0]
    frames = int(len(alpha) / 10)
    anim = animation.FuncAnimation(fig, lambda it: draw_frame(it, ax, alpha, model, y_ref, stats), frames=frames, interval=50, blit=False)

    mp4_path = "figures/profile_profilecnn.mp4"
    saved = False
    try:
        from matplotlib.animation import FFMpegWriter
        writer = FFMpegWriter(fps=20)
        anim.save(mp4_path, writer=writer, dpi=200)
        print(mp4_path)
        saved = True
    except Exception as e:
        print("MP4 save failed:", e)

    if not saved:
        try:
            from matplotlib.animation import PillowWriter
            gif_path = "figures/profile_profilecnn.gif"
            anim.save(gif_path, writer=PillowWriter(fps=20), dpi=150)
            print(gif_path)
        except Exception as e:
            print("GIF save failed:", e)

    plt.close(fig)

if __name__ == "__main__":
    main()
