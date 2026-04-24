import os
import numpy as np
if not hasattr(np, "bool"):
    np.bool = np.bool_
import pandas as pd
import glob

os.makedirs('figures', exist_ok=True)
base_path = r"C:/Users/kliu6/Desktop/Individual Project Coding/Data"

def get_surface(case, u_inf=1, surface="zNormal", field="UMean"):
    search_path = os.path.join(
        base_path, case, "postProcessing", "surfaces", "*",
        f"{field}_{surface}.raw"
    )
    files = glob.glob(search_path)
    if len(files) == 0:
        raise FileNotFoundError(f"Cannot find file: {search_path}")

    file = sorted(files)[-1]
    with open(file, "r") as f:
        line = f.readline()
        prev_line = line
        while line.startswith("#"):
            prev_line = line
            line = f.readline()

    header = prev_line.strip().lstrip("# ").split()
    data = pd.read_csv(
        file, comment="#", sep=r"\s+",
        names=header, header=None, engine="python"
    )
    return data


def get_line(case, position, field="UMean"):
    pos_str = f"{position:.1f}"
    search_path = os.path.join(
        base_path, case, "postProcessing", "lines", "*",
        f"x{pos_str}_*{field}*.xy"
    )
    files = glob.glob(search_path)
    if len(files) == 0:
        raise FileNotFoundError(f"Cannot find file: {search_path}")

    file = sorted(files)[-1]
    data = np.loadtxt(file)

    return data


def get_force(case, cube):
    search_path = os.path.join(
        base_path, case, "postProcessing", f"{cube}-forces",
        "*", "coefficient.dat"
    )
    files = glob.glob(search_path)
    if len(files) == 0:
        raise FileNotFoundError(f"Cannot find file: {search_path}")

    data = pd.DataFrame()
    for file in files:
        with open(file, "r") as f:
            line = f.readline()
            prev_line = line
            while line.startswith("#"):
                prev_line = line
                line = f.readline()

        header = prev_line.strip().lstrip("# ").split()
        new_data = pd.read_csv(
            file, comment="#", sep=r"\s+",
            names=header, header=None, engine="python"
        )
        data = pd.concat([data, new_data], ignore_index=True)

    if data.empty:
        raise ValueError(f"No data read from {search_path}")

    time_cols = [c for c in data.columns if c.lower().startswith("time")]
    time_col = time_cols[0] if time_cols else data.columns[0]

    data = data.sort_values(by=time_col)
    return data


def get_cd(case, cube, n_avg=600):
    force = get_force(case, cube)
    cd_cols = [c for c in force.columns if c.lower().startswith("cd")]
    if not cd_cols:
        raise KeyError(f"No Cd column found: {force.columns}")
    cd_col = cd_cols[0]
    return np.mean(force[cd_col].tail(n_avg))


def get_cl(case, cube, n_avg=600):
    force = get_force(case, cube)
    cl_cols = [c for c in force.columns if c.lower().startswith("cl")]
    if not cl_cols:
        raise KeyError(f"No Cl column found: {force.columns}")
    cl_col = cl_cols[0]
    return np.mean(force[cl_col].tail(n_avg))


'''
import os
from dataclasses import field
import numpy as np

if not hasattr(np, "bool"):
    np.bool = np.bool_

import pandas as pd
import glob
os.makedirs('figures', exist_ok=True)

base_path = r"C:/Users/kliu6/Desktop/Individual Project Coding/Data"

def get_surface(case, u_inf=1, surface="zNormal", field="UMean"):
    search_path = os.path.join(base_path, case, "postProcessing", "surfaces", "*", f"{field}_{surface}.raw")
    file = glob.glob(search_path)

    if len(file) == 0:
        raise FileNotFoundError(f"Cannot Find File: (search_patch)")

    file = file[-1]
    with open(file, 'r') as f:
        line = f.readline()
        prev_line = line
        while line.startswith('#'):
            prev_line = line
            line = f.readline()
    header = prev_line.strip().lstrip('# ').split()
    data = pd.read_csv(file, comment='#', sep=r'\s+', names=header, header=None, engine='python')
    return data

def get_line(case, position, field):
    posi = '%.1f' % postion
    search_path = os.path.join(base_path, case, "postProcessing", "lines", "*", f"x{posi}_*{field}*.xy")
    file = glob.glob(search_path)

    file = file[-1]
    data = np.loadtxt(file)  #读取纯文本
    return data

def get_probe(case, position, file):
    line = gat_line(case, position[0], field)
    probe_location = np.abs(line[:, 0] - position[1].argmin())
    probe_values = line[probe_location]
    probe = np.linalg.norm(probe_values[-3:-1]) #代表从倒数第一个元素到倒是第三个元素
    return probe

def get_force(case, cube):

    search_path = os.path.join(base_path, case, "postProcessing", f"{cube}-forces", "*", "coefficient.dat")
    files = glob.glob(search_path)

    if len(files) == 0:
        raise FileNotFoundError(f"Cannot Find File: {search_path}")

    data = pd.DataFrame()
    for file in files:

        with open(file, "r") as f:
            line = f.readline()
            prev_line = line
            while line.startswith("#"):
                prev_line = line
                line = f.readline()

        header = prev_line.strip().lstrip("# ").split()

        new_data = pd.read_csv(file, comment="#", sep=r"\s+", names=header, header=None, engine="python")
        data = pd.concat([data, new_data], ignore_index=True)

    if data.empty:
        raise ValueError(f"No data read from {search_path}")

    time_cols = [c for c in data.columns if c.lower().startswith("time")]

    if time_cols:
        time_col = time_cols[0]
    else:
        time_col = data.columns[0]

    data = data.sort_values(by=time_col)
    return data


def get_cd(case, cube, n_avg=600):
    force = get_force(case, cube)
    cd_cols = [c for c in force.columns if c.lower().startswith("cd")]
    if not cd_cols:
        raise KeyError(f"No Cd column found in force data: {force.columns}")
    cd_col = cd_cols[0]
    cd = np.mean(force[cd_col].tail(n_avg))
    return cd


def get_cl(case, cube, n_avg=600):
    force = get_force(case, cube)
    cl_cols = [c for c in force.columns if c.lower().startswith("cl")]
    if not cl_cols:
        raise KeyError(f"No Cl column found in force data: {force.columns}")
    cl_col = cl_cols[0]
    cl = np.mean(force[cl_col].tail(n_avg))
    return cl
'''