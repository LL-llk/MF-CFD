import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse
import Font
import Data_Extract as de
plt.rcParams['text.usetex'] = False

SURFACE = "yNormal"
RANS_CASES = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40]
LES_CASES = [0, 5, 10, 15, 20, 25, 30]
ALL_CASES = sorted(set(RANS_CASES) | set(LES_CASES))
os.makedirs("figures", exist_ok=True)

def add_cubes(ax, normal='y'):
    if normal == 'y':
        cube1 = patches.Rectangle((2, -0.5), 1, 1, linewidth=1, edgecolor='k', fc='lightgrey', hatch='/////', zorder=10)
        cube2 = patches.Rectangle((7, -0.5), 1, 1, linewidth=1, edgecolor='k', fc='lightgrey', hatch='/////', zorder=10)
    elif normal == 'z':
        cube1 = patches.Rectangle((2, 0), 1, 1, linewidth=1, edgecolor='k', fc='lightgrey', hatch='/////', zorder=10)
        cube2 = patches.Rectangle((7, 0), 1, 1, linewidth=1, edgecolor='k', fc='lightgrey', hatch='/////', zorder=10)
    else:
        print('normal must be y or z')
    ax.add_patch(cube1)
    ax.add_patch(cube2)

def plot_surface(ax, data, field, angle):

    normal = 'z' if SURFACE[0].lower() == 'y' else 'y'

    if field in ('UMean', 'U'):
        mag = np.sqrt(data[f'{field}_x']**2 + data[f'{field}_y']**2 + data[f'{field}_z']**2)
        loc1, loc2 = data['x'], data[normal]
    elif field == 'kMean':
        mag = data[field]
        loc1, loc2 = data['x'], data[normal]
    else:
        raise ValueError(f"Cannot Support field = {field}")

    contour = ax.tricontourf(loc1, loc2, mag, cmap='inferno', levels=np.arange(0, 1.5, 0.01), antialiased=False)
    ax.set_aspect('equal')
    ax.set_ylim(-4.25, 4.25)

    add_cubes(ax, SURFACE[0].lower())
    ax.annotate(fr'$\alpha = {angle}^\circ$', xy=(0.2, 3), xytext=(0.2, 3), size=Font.BIG_SIZE)
    return contour


def draw(it, ax1, ax2, angles):
    ang = angles[it]
    print("\nangle", ang)

    contour = None
    if ang in RANS_CASES:
        print("Plotting RANS velocity")
        case = f"RANS/Yaw/a{ang}"
        vel = de.get_surface(case, surface=SURFACE, field="UMean")
        contour = plot_surface(ax1, vel, "UMean", ang)

    if ang in LES_CASES:
        print("Plotting LES velocity")
        case = f"LES/Yaw/a{ang}"
        try:
            vel_inst = de.get_surface(case, surface=SURFACE, field="U")
            plot_surface(ax2[0], vel_inst, "U", ang)
        except FileNotFoundError as e:
            print(f"  [WARN] Instantaneous U not found for {case}: {e}")

        vel_mean = de.get_surface(case, surface=SURFACE, field="UMean")
        contour = plot_surface(ax2[1], vel_mean, "UMean", ang)

    return contour

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", action="store_true", help="Create animation instead of matrix")
    args = parser.parse_args()

    if args.a:
        fig, axes = plt.subplots(1, 1, figsize=(6, 5), squeeze=False, constrained_layout=True)
        axes[0, 0].set_ylabel(r'$z/H$')
        axes[0, 0].set_xlabel(r'$x/H$')
        n_cases = len(ALL_CASES)

        from matplotlib import animation
        anim = animation.FuncAnimation(fig, draw, fargs=(axes[0, 0], axes[0, 0], ALL_CASES), frames=n_cases, interval=1, blit=False)
        anim.save(f"animations/yaw_animation_{SURFACE}.mp4", fps=1, dpi=400)
    else:
        fig, axes = plt.subplots(7, 3, figsize=(12, 18), squeeze=False, constrained_layout=True, sharex=True, sharey=True)
        fig2, axes2 = plt.subplots(7, 2, figsize=(8, 18), squeeze=False, constrained_layout=True, sharex=True, sharey=True)
        contour = None

        for i, ang in enumerate(ALL_CASES):
            rans_loc = np.argmin(np.abs(np.array(RANS_CASES) - ang))
            rans_row = int(np.floor(rans_loc / 3))
            rans_col = int(rans_loc % 3)
            rans_ax = axes[rans_row, rans_col]

            if ang in LES_CASES:
                les_loc = np.argmin(np.abs(np.array(LES_CASES) - ang))
                les_ax_row = axes2[les_loc, :]
            else:
                les_ax_row = axes2[0, :]

            contour = draw(i, rans_ax, les_ax_row, ALL_CASES)

        normal = 'z' if SURFACE[0].lower() == 'y' else 'y'
        for i in range(7):
            axes[i, 0].set_ylabel(rf'${normal}/H$')
            axes2[i, 0].set_ylabel(rf'${normal}/H$')
        for j in range(3):
            axes[-1, j].set_xlabel(r'$x/H$')
        axes2[-1, 0].set_xlabel(r'$x/H$')

        u_ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
        cbar1 = fig.colorbar(contour, ax=axes[-1, :], location='bottom', shrink=0.9, aspect=75, ticks=u_ticks, format="%.1f")
        cbar2 = fig.colorbar(contour, ax=axes2[-1, :], location='bottom', shrink=0.7, aspect=50, ticks=u_ticks, format="%.1f")
        cbar1.set_label(r"$U_{mag} / U_0$")
        cbar2.set_label(r"$U_{mag} / U_0$")

        fig.savefig(f"figures/velocityslices-{SURFACE}-RANS-300.png", bbox_inches="tight", dpi=300)
        fig2.savefig(f"figures/velocityslices-{SURFACE}-LES-300.png", bbox_inches="tight", dpi=300)

        plt.show()
if __name__ == "__main__":
    main()
