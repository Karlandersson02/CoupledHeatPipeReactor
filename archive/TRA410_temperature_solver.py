"""
Pin temperature modelling (Python rewrite of the provided MATLAB code).

This file is a direct, line-by-line style translation of the MATLAB scripts:
- main_rel.m
- INITIALIZATION.m
- res_pin_rel.m
- cp_Na.m, rho_Na.m, k_Na.m, k_SS.m, k_fuel.m, h_gap.m, h1_Na.m

Dependencies:
    numpy
    scipy
    matplotlib

Run:
    python pin_temperature_modelling.py

Notes:
- The MATLAB code uses "global" structures; here we keep module-level globals
  with the same names for a close mapping.
- The solver unknown `sol_pin` is in *relative* units (normalized by T_Na_in),
  exactly like the MATLAB code (it multiplies by T_Na_in internally).
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from scipy.io import loadmat
from scipy.optimize import fsolve
import matplotlib.pyplot as plt


# --- "Globals" mirroring the MATLAB code ---
BOUNDARY_CONDITIONS = {}
GEOMETRY = {}
DISCRETIZATION = {}
PHYSICS_PARAMETERS = {}
CROSS_SECTIONS_DATA = {}


# --- Thermophysical / correlation functions (MATLAB equivalents) ---

def cp_Na(T_Na: np.ndarray) -> np.ndarray:
    """Sodium specific heat [J/kg/K]."""
    T_Na = np.asarray(T_Na, dtype=float)
    return -3.001e6 * T_Na ** (-2) + 1658 - 0.8479 * T_Na + 4.454e-4 * T_Na**2


def rho_Na(T_Na: np.ndarray) -> np.ndarray:
    """Sodium density [kg/m^3]."""
    T_Na = np.asarray(T_Na, dtype=float)
    return 1014 - 0.235 * T_Na


def k_Na(T_Na: np.ndarray) -> np.ndarray:
    """Sodium thermal conductivity [W/m/K]."""
    T_Na = np.asarray(T_Na, dtype=float)
    return 104 - 0.047 * T_Na


def k_SS(T_SS: np.ndarray) -> np.ndarray:
    """Stainless steel thermal conductivity [W/m/K]."""
    T_SS = np.asarray(T_SS, dtype=float)
    return -1.5809e-06 * T_SS**2 + 0.0169 * T_SS + 8.8025


def k_fuel(T_fuel: np.ndarray) -> np.ndarray:
    """Fuel pellet thermal conductivity [W/m/K]."""
    T_fuel = np.asarray(T_fuel, dtype=float)

    B = 3.0      # burnup [at.%]
    p = 0.174    # porosity [-]
    OM = 1.98    # stoichiometric ratio [-]

    omega = 1.09 / (B ** 3.265) + 0.0643 * np.sqrt(T_fuel / B)
    FD = omega / np.arctan(1.0 / omega)
    FP = 1 + 0.019 * B / (3 - 0.019 * B) / (1 + np.exp(-(T_fuel - 1200) / 100))
    FM = (1 - p) / (1 + 2 * p)
    FR = 1 - 0.2 / (1 + np.exp((T_fuel - 900) / 80))
    x = 2 - OM
    A = 2.85 * x + 0.035
    C = (-7.15 * x + 2.86) * 1e-4

    lambda_0 = (
        1.1579 / (A + C * T_fuel)
        + 2.3434e11 * T_fuel ** (-5 / 2) * np.exp(-16350 / T_fuel)
    )

    return lambda_0 * FD * FP * FM * FR


def h_gap(qp: np.ndarray) -> np.ndarray:
    """Gap conductance [W/m^2/K]."""
    qp = np.asarray(qp, dtype=float)
    return np.minimum(
        3 * (1e3 - qp / 100 + (qp / (100 * 10)) ** 2 + (qp / (100 * 100)) ** 3),
        2.3e4,
    )


def h1_Na(v_Na: np.ndarray, T_Na: np.ndarray) -> np.ndarray:
    """Single-phase sodium heat transfer coefficient [W/m^2/K]."""
    v_Na = np.asarray(v_Na, dtype=float)
    T_Na = np.asarray(T_Na, dtype=float)

    De = GEOMETRY["De"]
    Rco = GEOMETRY["Rco"]
    p = GEOMETRY["p"]

    Pe = v_Na * De * rho_Na(T_Na) * cp_Na(T_Na) / k_Na(T_Na)
    Nu = 0.047 * (1 - np.exp(-3.8 * (p / Rco - 1))) * (Pe**0.77 + 250)
    return k_Na(T_Na) * Nu / De


# --- INITIALIZATION (MATLAB equivalent) ---

def _mat_squeeze(x):
    arr = np.array(x)
    return np.squeeze(arr)


def INITIALIZATION(mat_path: str = "XS_data.mat"):
    """
    Returns:
        (BOUNDARY_CONDITIONS, GEOMETRY, DISCRETIZATION, PHYSICS_PARAMETERS, CROSS_SECTIONS_DATA)
    """
    mat = loadmat(mat_path)

    ABS_ref = _mat_squeeze(mat["ABS_ref"])
    FIS_ref = _mat_squeeze(mat["FIS_ref"])
    CHI_ref = _mat_squeeze(mat["CHI_ref"])
    NU_ref = _mat_squeeze(mat["NU_ref"])
    KAPPA_ref = _mat_squeeze(mat["KAPPA_ref"])
    SCAT_ref = _mat_squeeze(mat["SCAT_ref"])
    TR_ref = _mat_squeeze(mat["TR_ref"])

    DABS_fuel = _mat_squeeze(mat["DABS_fuel"])
    DFIS_fuel = _mat_squeeze(mat["DFIS_fuel"])
    DCHI_fuel = _mat_squeeze(mat["DCHI_fuel"])
    DNU_fuel = _mat_squeeze(mat["DNU_fuel"])
    DKAPPA_fuel = _mat_squeeze(mat["DKAPPA_fuel"])
    DSCAT_fuel = _mat_squeeze(mat["DSCAT_fuel"])
    DTR_fuel = _mat_squeeze(mat["DTR_fuel"])

    DABS_Na = _mat_squeeze(mat["DABS_Na"])
    DFIS_Na = _mat_squeeze(mat["DFIS_Na"])
    DCHI_Na = _mat_squeeze(mat["DCHI_Na"])
    DNU_Na = _mat_squeeze(mat["DNU_Na"])
    DKAPPA_Na = _mat_squeeze(mat["DKAPPA_Na"])
    DSCAT_Na = _mat_squeeze(mat["DSCAT_Na"])
    DTR_Na = _mat_squeeze(mat["DTR_Na"])

    cross_sections_data = {}
    cross_sections_data["ABS"] = np.concatenate([ABS_ref, DABS_fuel, DABS_Na], axis=-1)
    cross_sections_data["FIS"] = np.concatenate([FIS_ref, DFIS_fuel, DFIS_Na], axis=-1)
    cross_sections_data["CHI"] = np.concatenate([CHI_ref, DCHI_fuel, DCHI_Na], axis=-1)
    cross_sections_data["NU"] = np.concatenate([NU_ref, DNU_fuel, DNU_Na], axis=-1)
    cross_sections_data["KAPPA"] = np.concatenate([KAPPA_ref, DKAPPA_fuel, DKAPPA_Na], axis=-1)
    cross_sections_data["SCAT"] = np.concatenate([SCAT_ref, DSCAT_fuel, DSCAT_Na], axis=-1)
    cross_sections_data["TR"] = np.concatenate([TR_ref, DTR_fuel, DTR_Na], axis=-1)

    boundary_conditions = {
        "P_out": 1.5,
        "T_Na_in": 673.0,
        "v_Na_in": 7.5,
        "qp_ave": 3e4,
    }

    geometry = {
        "H": 1.6,
        "Rco": 8.50e-3 / 2,
        "p": 9.8e-3,
        "Rfo": 7.14e-3 / 2,
    }
    geometry["Rci"] = geometry["Rco"] - 0.565e-3

    discretization = {"NZ": 25, "NR": 5, "NG": 8}
    physics_parameters = {"g": 9.81}

    geometry["Pw"] = np.pi * geometry["Rco"]
    geometry["A"] = geometry["p"]**2 * np.sqrt(3) / 4 - np.pi * geometry["Rco"]**2 * 0.5
    geometry["De"] = 4 * geometry["A"] / geometry["Pw"]

    NZ = discretization["NZ"]
    NR = discretization["NR"]
    H = geometry["H"]
    Rfo = geometry["Rfo"]
    Rci = geometry["Rci"]
    Rco = geometry["Rco"]

    discretization["DZ"] = H / NZ
    nR = 2 * NR + 4
    DR = np.zeros(nR, dtype=float)
    R = np.zeros(nR, dtype=float)

    R[0] = np.sqrt(Rfo**2 / (NR * 2))
    for k in range(1, 2 * NR):
        R[k] = np.sqrt(R[0] ** 2 + R[k - 1] ** 2)

    R[2 * NR] = np.sqrt(R[2 * NR - 1] ** 2 + (Rci**2 - Rfo**2) / 2)
    R[2 * NR + 1] = np.sqrt(R[2 * NR] ** 2 + (Rci**2 - Rfo**2) / 2)
    R[2 * NR + 2] = np.sqrt(R[2 * NR + 1] ** 2 + (Rco**2 - Rci**2) / 2)
    R[2 * NR + 3] = np.sqrt(R[2 * NR + 2] ** 2 + (Rco**2 - Rci**2) / 2)

    DR[0] = R[0]
    DR[1:] = R[1:] - R[:-1]

    SR = 2 * np.pi * R
    VR = np.empty_like(R)
    VR[0] = np.pi * (R[0] ** 2)
    VR[1:] = np.pi * (R[1:] ** 2 - R[:-1] ** 2)

    discretization["DR"] = DR
    discretization["R"] = R
    discretization["SR"] = SR
    discretization["VR"] = VR

    return boundary_conditions, geometry, discretization, physics_parameters, cross_sections_data


# --- Residual function (MATLAB res_pin_rel.m equivalent) ---

def res_pin_rel(sol_pin: np.ndarray, qp: np.ndarray, v_Na_rel: np.ndarray, T_Na_rel: np.ndarray) -> np.ndarray:
    sol_pin = np.asarray(sol_pin, dtype=float).reshape(-1)
    qp = np.asarray(qp, dtype=float).reshape(-1)
    v_Na_rel = np.asarray(v_Na_rel, dtype=float).reshape(-1)
    T_Na_rel = np.asarray(T_Na_rel, dtype=float).reshape(-1)

    T_Na_in = BOUNDARY_CONDITIONS["T_Na_in"]
    v_Na_in = BOUNDARY_CONDITIONS["v_Na_in"]

    NZ = DISCRETIZATION["NZ"]
    NR = DISCRETIZATION["NR"]
    DZ = DISCRETIZATION["DZ"]
    DR_full = DISCRETIZATION["DR"]
    SR_full = DISCRETIZATION["SR"]
    VR_full = DISCRETIZATION["VR"]

    H = GEOMETRY["H"]
    Rfo = GEOMETRY["Rfo"]
    Rci = GEOMETRY["Rci"]

    v_Na_m = np.concatenate([[v_Na_in], v_Na_rel[:-1] * v_Na_in])
    v_Na_p = v_Na_rel * v_Na_in
    T_Na_m = np.concatenate([[T_Na_in], T_Na_rel[:-1] * T_Na_in])
    T_Na_p = T_Na_rel * T_Na_in

    v_Na_ave = 0.5 * (v_Na_p + v_Na_m)
    T_Na_ave = 0.5 * (T_Na_p + T_Na_m)

    n_total = (NR + 2) * NZ
    if sol_pin.size != n_total:
        raise ValueError(f"Expected sol_pin length {n_total}, got {sol_pin.size}")

    sol_blocks = sol_pin.reshape((NR + 2, NZ), order="C")
    sol_fuel_rel = sol_blocks[0:NR, :]
    sol_gap_rel = sol_blocks[NR, :]
    sol_ss_rel = sol_blocks[NR + 1, :]

    T_fuel = sol_fuel_rel * T_Na_in
    T_SS = sol_ss_rel * T_Na_in

    k_blocks = np.empty((NR + 2, NZ), dtype=float)
    k_blocks[0:NR, :] = k_fuel(T_fuel)
    k_blocks[NR, :] = h_gap(qp) * (Rci - GEOMETRY["Rfo"])
    k_blocks[NR + 1, :] = k_SS(T_SS)

    DRm_tmp = DR_full[0::2]
    DRp_tmp = DR_full[1::2]

    SR_even = SR_full[1::2]
    SRp_tmp = SR_even.copy()
    SRm_tmp = np.concatenate([[0.0], SR_even[:-1]])

    VR_tmp = VR_full[1::2] + VR_full[0::2]

    VRrel_tmp = VR_full[0:NR] / np.sum(VR_full[0:NR])

    DRm = DRm_tmp[:, None]
    DRp = DRp_tmp[:, None]
    SRm = SRm_tmp[:, None]
    SRp = SRp_tmp[:, None]
    VRb = VR_tmp[:, None]

    G_ip = np.zeros((NR + 2, NZ), dtype=float)
    for i in range(0, NR + 1):
        denom = k_blocks[i, :] * DRm[i + 1, 0] + k_blocks[i + 1, :] * DRp[i, 0]
        G_ip[i, :] = k_blocks[i, :] * k_blocks[i + 1, :] / denom

    G_im = np.zeros((NR + 2, NZ), dtype=float)
    for i in range(1, NR + 2):
        denom = k_blocks[i - 1, :] * DRm[i, 0] + k_blocks[i, :] * DRp[i - 1, 0]
        G_im[i, :] = k_blocks[i, :] * k_blocks[i - 1, :] / denom

    a = np.zeros((NR + 2, NZ), dtype=float)
    b = np.zeros((NR + 2, NZ), dtype=float)
    c = np.zeros((NR + 2, NZ), dtype=float)

    a[0, :] = -(SRp[0, 0] / VRb[0, 0]) * G_ip[0, :]
    b[0, :] = +(SRp[0, 0] / VRb[0, 0]) * G_ip[0, :]

    for i in range(1, NR + 1):
        a[i, :] = -(SRp[i, 0] / VRb[i, 0]) * G_ip[i, :] - (SRm[i, 0] / VRb[i, 0]) * G_im[i, :]
        b[i, :] = +(SRp[i, 0] / VRb[i, 0]) * G_ip[i, :]
        c[i, :] = +(SRm[i, 0] / VRb[i, 0]) * G_im[i, :]

    i = NR + 1
    h = h1_Na(v_Na_ave, T_Na_ave)
    conv_coeff = (SRp[i, 0] / VRb[i, 0]) * (k_blocks[i, :] * (-h) / (k_blocks[i, :] + h * DRp[i, 0]))
    a[i, :] = conv_coeff - (SRm[i, 0] / VRb[i, 0]) * G_im[i, :]
    c[i, :] = +(SRm[i, 0] / VRb[i, 0]) * G_im[i, :]

    q = np.zeros((NR + 2, NZ), dtype=float)
    for r in range(0, NR):
        q[r, :] = (VRrel_tmp[r] * qp) / VRb[r, 0]

    q[i, :] = (SRp[i, 0] / VRb[i, 0]) * (k_blocks[i, :] * h * T_Na_ave / (k_blocks[i, :] + h * DRp[i, 0]))

    res = np.zeros_like(sol_blocks)

    res[0, :] = a[0, :] * (sol_blocks[0, :] * T_Na_in) + b[0, :] * (sol_blocks[1, :] * T_Na_in) + q[0, :]

    for i in range(1, NR + 1):
        res[i, :] = (
            a[i, :] * (sol_blocks[i, :] * T_Na_in)
            + b[i, :] * (sol_blocks[i + 1, :] * T_Na_in)
            + c[i, :] * (sol_blocks[i - 1, :] * T_Na_in)
            + q[i, :]
        )

    i = NR + 1
    res[i, :] = a[i, :] * (sol_blocks[i, :] * T_Na_in) + c[i, :] * (sol_blocks[i - 1, :] * T_Na_in) + q[i, :]

    denom = (np.sum(qp) * DZ / H) / (np.pi * Rfo**2 / NR)
    res_norm = res / denom
    return res_norm.reshape(-1, order="C")


def main():
    global BOUNDARY_CONDITIONS, GEOMETRY, DISCRETIZATION, PHYSICS_PARAMETERS, CROSS_SECTIONS_DATA

    base_dir = Path(__file__).resolve().parent
    mat_path = str(base_dir / "XS_data.mat")
    BOUNDARY_CONDITIONS, GEOMETRY, DISCRETIZATION, PHYSICS_PARAMETERS, CROSS_SECTIONS_DATA = INITIALIZATION(mat_path)

    NZ = DISCRETIZATION["NZ"]
    NR = DISCRETIZATION["NR"]
    qp_ave = BOUNDARY_CONDITIONS["qp_ave"]

    x0 = np.ones(NZ * (NR + 2), dtype=float)
    v_rel = np.ones(NZ, dtype=float)
    T_rel = np.ones(NZ, dtype=float)

    qp1 = qp_ave * np.ones(NZ, dtype=float)

    sol1, info1, ier1, msg1 = fsolve(lambda x: res_pin_rel(x, qp1, v_rel, T_rel), x0, full_output=True)
    print("Case 1 (constant qp):")
    print(msg1)

    plt.figure(1)
    plt.plot(sol1)
    plt.title("Solution sol_pin (relative) - constant qp")
    plt.xlabel("Unknown index")
    plt.ylabel("sol_pin (relative)")
    plt.grid(True)

    z = (np.arange(NZ) + 0.5) / NZ * np.pi
    qp2 = qp_ave * np.sin(z)

    sol2, info2, ier2, msg2 = fsolve(lambda x: res_pin_rel(x, qp2, v_rel, T_rel), x0, full_output=True)
    print("\nCase 2 (sinusoidal qp):")
    print(msg2)

    plt.figure(2)
    plt.plot(sol2)
    plt.title("Solution sol_pin (relative) - sinusoidal qp")
    plt.xlabel("Unknown index")
    plt.ylabel("sol_pin (relative)")
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main()
