import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import Data_Extract as de

mpl.rcParams["text.usetex"] = False
RANS_CASES = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40]
LES_CASES  = [0, 5, 10, 15, 20, 25, 30]

CUBES = ["cube1", "cube2"]
N_AVG = 600


def collect_forces(angles, model_prefix, cube):
    ang_list, cd_list, cl_list = [], [], []

    for ang in angles:
        case = f"{model_prefix}/Yaw/a{ang}"
        try:
            cd = de.get_cd(case, cube, n_avg=N_AVG)
            cl = de.get_cl(case, cube, n_avg=N_AVG)
        except FileNotFoundError as e:
            print(f"Worry {model_prefix}/{cube}/a{ang}: {e}")
            continue

        ang_list.append(ang)
        cd_list.append(cd)
        cl_list.append(cl)

        print(f"{model_prefix}/{cube}  a={ang:>2}° -> Cd={cd:.3f}, Cl={cl:.3f}")

    return np.array(ang_list), np.array(cd_list), np.array(cl_list)


def main():
    for cube in CUBES:
        ang_rans, cd_rans, cl_rans = collect_forces(RANS_CASES, "RANS", cube)
        ang_les,  cd_les,  cl_les  = collect_forces(LES_CASES,  "LES",  cube)

        # Cd
        fig_cd, ax_cd = plt.subplots(figsize=(6, 4), constrained_layout=True)
        ax_cd.plot(ang_rans, cd_rans, label="RANS")
        ax_cd.plot(ang_les,  cd_les,  label="LES")

        ax_cd.set_xlabel(r"$\alpha\ [^\circ]$")
        ax_cd.set_ylabel(r"$C_D$")
        ax_cd.grid(True, alpha=0.3)
        ax_cd.legend()
        ax_cd.set_title(f"{cube.upper()}: Coefficient of drag (Cd)")
        fig_cd.savefig(f"figures/CD_{cube}.png", dpi=300, bbox_inches="tight")

        # Cl
        fig_cl, ax_cl = plt.subplots(figsize=(6, 4), constrained_layout=True)
        ax_cl.plot(ang_rans, cl_rans, label="RANS")
        ax_cl.plot(ang_les,  cl_les,  label="LES")

        ax_cl.set_xlabel(r"$\alpha\ [^\circ]$")
        ax_cl.set_ylabel(r"$C_L$")
        ax_cl.grid(True, alpha=0.3)
        ax_cl.legend()
        ax_cl.set_title(f"{cube.upper()}: Coefficient of lift (Cl)")
        fig_cl.savefig(f"figures/CL_{cube}.png", dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()
