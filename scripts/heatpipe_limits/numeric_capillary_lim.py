"""
capillary_limit_temperature_sweep.py

Reactor-coupled capillary-limit solver using a target heat-pipe operating
temperature sweep.

Main idea
---------
Instead of sweeping condenser h directly, prescribe a target heat-pipe
operating temperature T_HP_target and estimate

    h_cond = Q_HP / (A_cond * (T_HP_target - T_cond))

for each reactor-power trial.

The final stored temperature is NOT T_HP_target. It is the actual simulated
temperature returned by the reactor solve.

The capillary limit is evaluated using:
    Delta_p_cap_max = 2 * sigma_l / r_pore

and a reactor-coupled liquid-pressure model using the full axial vapour
temperature profile from the solved reactor.
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np


# =============================================================================
# Adjust these imports to match your project layout
# =============================================================================

import data.dataclass as d_class

from utils import sodium_properties as s_props
from utils.sodium_properties import (
    calculate_Na_rho_l,
    calculate_Na_viscosity_l,
    calculate_Na_h_fg,
)
import json
import matplotlib.pyplot as plt
import numpy as np

import utils.sodium_properties as s_props
import data.dataclass as d_class

from pathlib import Path

from models.heatpipe.heat_pipe_limitations_model import HeatPipeLimitations
from models.heatpipe.liquid_discretised_model import LiquidDiscretised
from coupled.heatpipe import Heatpipe

from utils.solver import Solver
from utils.iso_reactor_utils import generate_config, generate_config_seq
from utils.interpolator import OpenMCTallyGridSurrogate

from coupled.vapour_reactor import VapourReactor
from coupled.iso_reactor import Reactor

# These names are assumed to exist in your project.
# Adjust import paths if needed.
#


# =============================================================================
# User-specific heat-load conversion
# =============================================================================

N_FUEL_PINS = 2970
HP_TO_FP_RATIO = 24.0 / 7.0


def thermal_power_to_heatpipe_Q(P_th: float) -> float:
    """
    Convert total reactor thermal power to heat-pipe heat load.

    User definition:
        Q_HP = P_th / 2970 * (24 / 7)
    """
    return float(P_th) / N_FUEL_PINS * HP_TO_FP_RATIO


# =============================================================================
# Heat-pipe config helpers
# =============================================================================

def get_mesh_nr_nz(cfg_HP):
    mesh = getattr(cfg_HP, "mesh", cfg_HP)

    N_R = getattr(mesh, "N_R", None)
    N_Z = getattr(mesh, "N_Z", None)

    if N_R is None:
        N_R = getattr(cfg_HP, "N_R", None)
    if N_Z is None:
        N_Z = getattr(cfg_HP, "N_Z", None)

    if N_R is None or N_Z is None:
        raise AttributeError(
            "Could not find N_R and N_Z in cfg_HP. "
            "Check your HeatpipeConfigResolved mesh attributes."
        )

    return int(N_R), int(N_Z)


def cfg_mesh_size(cfg_HP) -> int:
    N_R, N_Z = get_mesh_nr_nz(cfg_HP)
    return int(N_R * N_Z)


def build_heatpipe_config_from_data(
    data: dict,
    *,
    N_R: int,
    N_Z: int,
    h_cond: float,
):
    """
    Build a resolved heat-pipe config and Heatpipe object from reactor JSON data.
    """
    data = copy.deepcopy(data)
    data["HeatPipe"]["material"]["h_cond"] = float(h_cond)

    geom = d_class.HeatpipeGeometry(**data["HeatPipe"]["geometry"])
    mesh = d_class.HeatpipeMesh(N_R=int(N_R), N_Z=int(N_Z))
    mat = d_class.HeatpipeMaterial(**data["HeatPipe"]["material"])
    bc = d_class.HeatpipeBC(**data["HeatPipe"]["bc"])
    wick = d_class.HeatpipeWick(**data["HeatPipe"]["wick"])

    cfg = d_class.HeatpipeConfig(geom, mesh, mat, wick, bc)
    cfg_HP = cfg.resolve_geometry()

    HP = Heatpipe(cfg_HP)

    return cfg_HP, HP


def rebuild_cfg_HP_to_match_solid_temperature(
    *,
    data: dict,
    reference_cfg_HP,
    h_cond: float,
    T_solid: np.ndarray,
):
    """
    Rebuild cfg_HP so that cfg_HP.mesh.N_R * cfg_HP.mesh.N_Z matches
    T_solid.size.

    This handles cases where the geometry resolver changes N_Z.
    """
    T_solid = np.asarray(T_solid)
    target_size = int(T_solid.size)

    old_N_R, old_N_Z = get_mesh_nr_nz(reference_cfg_HP)
    old_size = old_N_R * old_N_Z

    if old_size == target_size:
        return reference_cfg_HP, None

    candidates = []

    if target_size % old_N_R == 0:
        candidates.append((old_N_R, target_size // old_N_R))

    if target_size % old_N_Z == 0:
        candidates.append((target_size // old_N_Z, old_N_Z))

    unique_candidates = []
    for candidate in candidates:
        if candidate not in unique_candidates:
            unique_candidates.append(candidate)

    for N_R_new, N_Z_new in unique_candidates:
        try:
            cfg_HP_new, HP_new = build_heatpipe_config_from_data(
                data,
                N_R=N_R_new,
                N_Z=N_Z_new,
                h_cond=h_cond,
            )

            if cfg_mesh_size(cfg_HP_new) == target_size:
                print(
                    "  Rebuilt cfg_HP to match reactor solution: "
                    f"old mesh=({old_N_R}, {old_N_Z}), "
                    f"new mesh=({N_R_new}, {N_Z_new}), "
                    f"T_solid.size={target_size}"
                )
                return cfg_HP_new, HP_new

        except Exception as exc:
            print(
                f"  Tried rebuilding cfg_HP with "
                f"N_R={N_R_new}, N_Z={N_Z_new}, but failed: {exc}"
            )

    raise ValueError(
        "Could not rebuild cfg_HP to match T_solid size. "
        f"T_solid.size={target_size}, old cfg mesh=({old_N_R}, {old_N_Z}), "
        f"old size={old_size}, candidates tried={unique_candidates}"
    )


def infer_condenser_area(cfg_HP) -> float:
    """
    Estimate condenser heat-transfer area.

    Default:
        A_cond = 2*pi*r_outer*l_cond

    The radius attribute may depend on your geometry class, so this tries
    several plausible candidates.
    """
    geom = cfg_HP.geometry

    radius_candidates = [
        "r_outer",
        "r_wall_outer",
        "r_wall",
        "r_gap",
        "r_wick",
    ]

    for name in radius_candidates:
        if hasattr(geom, name):
            radius = float(getattr(geom, name))
            if radius > 0.0:
                A_cond = 2.0 * np.pi * radius * float(geom.l_cond)
                print(
                    f"Using A_cond={A_cond:.6e} m^2 from "
                    f"geometry.{name}={radius:.6e} m"
                )
                return A_cond

    raise AttributeError(
        "Could not infer condenser area. Pass A_cond explicitly."
    )


def estimate_h_cond(
    *,
    P_th: float,
    T_HP_target: float,
    T_cond: float,
    A_cond: float,
) -> float:
    """
    Estimate condenser heat-transfer coefficient from target HP temperature.

        h_cond = Q_HP / (A_cond * (T_HP_target - T_cond))
    """
    Q_HP = thermal_power_to_heatpipe_Q(P_th)

    delta_T = float(T_HP_target) - float(T_cond)

    if delta_T <= 0.0:
        raise ValueError(
            f"T_HP_target must be greater than T_cond. "
            f"Got T_HP_target={T_HP_target}, T_cond={T_cond}."
        )

    return float(Q_HP / (float(A_cond) * delta_T))


# =============================================================================
# Liquid model using full axial vapour temperature profile
# =============================================================================

def make_liquid_input(T_HP, T_v):
    """
    Create the legacy liquid-model input format:

        [flattened solid temperature field, scalar vapour temperature]

    The scalar vapour temperature is still stored for compatibility, but the
    new get_mdot() below uses the full axial T_v profile.
    """
    T_HP_flat = np.asarray(T_HP, dtype=float).reshape(-1)

    T_v_arr = np.asarray(T_v, dtype=float)
    T_v_scalar = float(np.mean(T_v_arr))

    T_liquid_input = np.concatenate([
        T_HP_flat,
        np.array([T_v_scalar], dtype=float),
    ])

    return T_liquid_input, T_v_scalar


class LiquidDiscretisedFullTemperature:
    """
    Liquid pressure-drop model where mdot is computed from the full solved
    reactor heat-pipe temperature field.

    Expected input:
        T_HP_full = (T_solid, u_v, T_v)

    Interface index is assumed to be 0.
    """

    def __init__(self, config: d_class.HeatpipeConfigResolved, T_HP_full):
        self.cfg = config
        self.T_HP_full = T_HP_full

        self.T_solid_full = np.asarray(T_HP_full[0], dtype=float)
        self.u_v_full = np.asarray(T_HP_full[1], dtype=float)
        self.T_v_full = np.asarray(T_HP_full[2], dtype=float).reshape(-1)

        self.T_HP, self.T_v_scalar = make_liquid_input(
            self.T_solid_full,
            self.T_v_full,
        )

        self.interface_index = 0

    def get_solid_temperature_2d(self):
        T_solid = np.asarray(self.T_HP, dtype=float)[:-1]

        expected_size = self.cfg.mesh.N_Z * self.cfg.mesh.N_R

        if T_solid.size != expected_size:
            raise ValueError(
                "Temperature-size mismatch in LiquidDiscretisedFullTemperature:\n"
                f"  T_solid.size = {T_solid.size}\n"
                f"  N_Z * N_R    = {expected_size}\n"
                f"  N_Z          = {self.cfg.mesh.N_Z}\n"
                f"  N_R          = {self.cfg.mesh.N_R}"
            )

        return T_solid.reshape(self.cfg.mesh.N_Z, self.cfg.mesh.N_R)

    def get_delta_z(self):
        delta_z = np.concatenate([
            np.ones(self.cfg.mesh.N_evap)
            * self.cfg.geometry.l_evap
            / self.cfg.mesh.N_evap,

            np.ones(self.cfg.mesh.N_adiabatic)
            * self.cfg.geometry.l_adiabatic
            / self.cfg.mesh.N_adiabatic,

            np.ones(self.cfg.mesh.N_cond)
            * self.cfg.geometry.l_cond
            / self.cfg.mesh.N_cond,
        ])

        if len(delta_z) != self.cfg.mesh.N_Z:
            raise ValueError(
                "Axial mesh mismatch:\n"
                f"  len(delta_z) = {len(delta_z)}\n"
                f"  N_Z          = {self.cfg.mesh.N_Z}"
            )

        return delta_z

    def get_mdot(self):
        """
        Compute local mdot from local wick-interface and vapour temperatures:

            q_i = h_vap * A_i * (T_interface_i - T_v_i)
            dm_i = q_i / h_fg(T_v_i)

        Positive q means evaporation; negative q means condensation.
        """
        T_solid_2d = self.get_solid_temperature_2d()
        T_interface = T_solid_2d[:, self.interface_index]

        T_v = np.asarray(self.T_v_full, dtype=float).reshape(-1)

        if len(T_v) != self.cfg.mesh.N_Z:
            raise ValueError(
                "Vapour temperature length mismatch:\n"
                f"  len(T_v) = {len(T_v)}\n"
                f"  N_Z      = {self.cfg.mesh.N_Z}"
            )

        delta_z = self.get_delta_z()

        A_int = (
            2.0
            * np.pi
            * self.cfg.geometry.r_vapour
            * delta_z
        )

        q_lv = (
            self.cfg.material.h_vap
            * A_int
            * (T_interface - T_v)
        )

        h_fg = calculate_Na_h_fg(T_v)

        dm_cell = q_lv / h_fg

        mdot_faces = np.concatenate([
            np.array([0.0]),
            np.cumsum(dm_cell),
        ])

        mdot_cell_signed = 0.5 * (
            mdot_faces[:-1]
            + mdot_faces[1:]
        )

        mdot_cell = np.abs(mdot_cell_signed)

        self.T_interface = T_interface
        self.T_v_used = T_v
        self.q_lv = q_lv
        self.dm_cell = dm_cell
        self.mdot_faces = mdot_faces
        self.mdot_cell_signed = mdot_cell_signed
        self.mdot_cell = mdot_cell

        return mdot_cell

    def calculate_K_annular_wick(self):
        r_2 = self.cfg.geometry.r_wick
        r_1 = self.cfg.geometry.r_gap

        R_star = r_2 / r_1

        fRe_l = 16.0 * (1.0 - R_star) ** 2 / (
            1.0
            + R_star**2
            - (1.0 - R_star**2) / np.log(1.0 / R_star)
        )

        D_h = 2.0 * (r_1 - r_2)

        K = D_h**2 / (2.0 * fRe_l)

        return K

    def get_A_wick(self):
        if self.cfg.wick.Is_annular:
            return np.pi * (
                self.cfg.geometry.r_gap**2
                - self.cfg.geometry.r_wick**2
            )

        return np.pi * (
            self.cfg.geometry.r_wick**2
            - self.cfg.geometry.r_vapour**2
        )

    def get_K(self):
        if self.cfg.wick.Is_annular:
            return self.calculate_K_annular_wick()

        return self.cfg.wick.K

    def get_pressure_drop_profile(self):
        A_wick = self.get_A_wick()
        K = self.get_K()

        T_solid_2d = self.get_solid_temperature_2d()

        T_wick = np.mean(
            T_solid_2d[:, :self.cfg.mesh.N_wick],
            axis=1,
        )

        mu_l = calculate_Na_viscosity_l(T_wick)
        rho_l = calculate_Na_rho_l(T_wick)

        delta_z = self.get_delta_z()
        mdot = self.get_mdot()

        P = -np.cumsum(
            mu_l
            * mdot
            / (rho_l * A_wick * K)
            * delta_z
        )

        self.pressure = P
        self.mu_l = mu_l
        self.rho_l = rho_l
        self.A_wick = A_wick
        self.K = K
        self.delta_z = delta_z

        return P

    def print_diagnostics(self, label=""):
        """
        Print liquid-model diagnostics after get_pressure_drop_profile().
        """
        P = self.get_pressure_drop_profile()

        Q_evap = float(np.sum(self.q_lv[self.q_lv > 0.0]))
        Q_cond = float(np.sum(self.q_lv[self.q_lv < 0.0]))
        Q_net = float(np.sum(self.q_lv))

        print("\n" + "=" * 80)
        print(f"LIQUID FULL-TEMPERATURE DIAGNOSTIC {label}")
        print("=" * 80)

        print("\nTemperatures:")
        print(f"  T_interface min/max        = {np.nanmin(self.T_interface):.6e} / {np.nanmax(self.T_interface):.6e} K")
        print(f"  T_v min/max                = {np.nanmin(self.T_v_used):.6e} / {np.nanmax(self.T_v_used):.6e} K")
        print(f"  T_v scalar legacy          = {self.T_v_scalar:.6e} K")
        print(f"  dT local min/max           = {np.nanmin(self.T_interface - self.T_v_used):.6e} / {np.nanmax(self.T_interface - self.T_v_used):.6e} K")

        print("\nHeat exchange:")
        print(f"  Q_evap positive            = {Q_evap:.6e} W")
        print(f"  Q_cond negative            = {Q_cond:.6e} W")
        print(f"  Q_net                      = {Q_net:.6e} W")

        print("\nMass flow:")
        print(f"  mdot cell min/max          = {np.nanmin(self.mdot_cell):.6e} / {np.nanmax(self.mdot_cell):.6e} kg/s")
        print(f"  mdot faces min/max         = {np.nanmin(self.mdot_faces):.6e} / {np.nanmax(self.mdot_faces):.6e} kg/s")

        print("\nPressure-drop constants:")
        print(f"  Is_annular                 = {self.cfg.wick.Is_annular}")
        print(f"  A_wick                     = {self.A_wick:.6e} m^2")
        print(f"  K                          = {self.K:.6e} m^2")
        print(f"  A_wick*K                   = {(self.A_wick * self.K):.6e} m^4")
        print(f"  r_pore                     = {self.cfg.wick.r_pore:.6e} m")
        print(f"  mu_l min/max               = {np.nanmin(self.mu_l):.6e} / {np.nanmax(self.mu_l):.6e} Pa s")
        print(f"  rho_l min/max              = {np.nanmin(self.rho_l):.6e} / {np.nanmax(self.rho_l):.6e} kg/m^3")
        print(f"  liquid dP signed           = {(P[-1] - P[0]):.6e} Pa")
        print(f"  liquid dP abs              = {abs(P[-1] - P[0]):.6e} Pa")

        print("=" * 80 + "\n")


# =============================================================================
# Data containers
# =============================================================================

@dataclass
class CapillaryPoint:
    P_th: float
    Q_HP: float
    h_cond: float

    T_target: float
    T_actual: float

    capillary_margin: float
    capillary_pressure_required: float
    capillary_pressure_max: float

    liquid_dP: float
    vapour_dP: float
    max_mach: float

    T_v: np.ndarray
    T_solid: np.ndarray
    u_v: np.ndarray
    delta_p_profile: np.ndarray


@dataclass
class SolveResult:
    point: Optional[CapillaryPoint]
    residual: float
    status: str
    n_solves: int


# =============================================================================
# Main solver
# =============================================================================

class CapillaryTemperatureSweepSolver:
    """
    New capillary-limit solver.

    For each prescribed T_HP_target:
        1. Estimate h_cond from P_th and T_HP_target.
        2. Solve full reactor.
        3. Compute capillary margin.
        4. Adjust P_th until capillary margin changes sign.

    The saved temperature is the actual reactor result, not the target value.
    """

    def __init__(
        self,
        *,
        reactor_data_path: str | Path = "./data/reactor_data.json",
        Ns=(15, 15, 20),
        interpolator_model=None,
        T_cond: float = 300.0,
        A_cond: Optional[float] = None,
        initial_HP_mesh=(15, 20),
        output_path: str | Path = "./capillary_T_sweep_results.npz",
        store_profiles: bool = True,
        print_liquid_diagnostics: bool = False,
    ):
        self.reactor_data_path = Path(reactor_data_path)
        self.Ns = tuple(Ns)
        self.interpolator_model = interpolator_model

        self.T_cond = float(T_cond)
        self.A_cond = A_cond

        self.output_path = Path(output_path)
        self.store_profiles = bool(store_profiles)
        self.print_liquid_diagnostics = bool(print_liquid_diagnostics)

        with open(self.reactor_data_path, "r") as f:
            self.base_data = json.load(f)

        N_R0, N_Z0 = initial_HP_mesh

        self.reference_cfg_HP, self.reference_HP = build_heatpipe_config_from_data(
            self.base_data,
            N_R=N_R0,
            N_Z=N_Z0,
            h_cond=550.0,
        )

        if self.A_cond is None:
            self.A_cond = infer_condenser_area(self.reference_cfg_HP)

        self._cache: dict[tuple[float, float], CapillaryPoint] = {}

    # -------------------------------------------------------------------------
    # Reactor solve
    # -------------------------------------------------------------------------

    def solve_reactor_point(
        self,
        *,
        P_th: float,
        T_target: float,
    ) -> CapillaryPoint:
        P_th = float(P_th)
        T_target = float(T_target)

        h_cond = estimate_h_cond(
            P_th=P_th,
            T_HP_target=T_target,
            T_cond=self.T_cond,
            A_cond=self.A_cond,
        )

        cache_key = (round(P_th, 6), round(T_target, 6))
        if cache_key in self._cache:
            return self._cache[cache_key]

        data = copy.deepcopy(self.base_data)
        data["HeatPipe"]["material"]["h_cond"] = h_cond
        data["Reactor"]["power"]["thermal"] = P_th

        cfg = generate_config(data, *self.Ns)

        iso_reactor = Reactor(cfg)
        vap_reactor_800 = VapourReactor(cfg)
        vap_reactor = VapourReactor(cfg)
        vap_reactor_full = VapourReactor(cfg)

        vap_reactor.heatpipe.solid.k_temperature = 1100

        if self.interpolator_model is not None:
            vap_reactor_full.set_interpolator_model(self.interpolator_model)

        vap_reactor_full.set_variable_k(True)

        solver_vap = Solver(
            [
                vap_reactor_800,
                vap_reactor,
                vap_reactor_full,
            ],
            iterate=True,
        )

        solver_vap.fsolve()

        ((T_solid, u_v, T_v), T_FP, (phi_n_g, k)) = solver_vap.solutions[2]

        T_solid = np.asarray(T_solid, dtype=float)
        u_v = np.asarray(u_v, dtype=float).reshape(-1)
        T_v = np.asarray(T_v, dtype=float).reshape(-1)

        print(
            f"  solved: P_th={P_th:.6e} W, "
            f"h_cond={h_cond:.6e}, "
            f"T_target={T_target:.2f} K, "
            f"T_v=[{np.min(T_v):.2f}, {np.max(T_v):.2f}] K, "
            f"T_solid.size={T_solid.size}, T_v.size={T_v.size}"
        )

        cfg_HP_run, HP_run = rebuild_cfg_HP_to_match_solid_temperature(
            data=data,
            reference_cfg_HP=self.reference_cfg_HP,
            h_cond=h_cond,
            T_solid=T_solid,
        )

        if HP_run is None:
            HP_run = self.reference_HP

        # ---------------------------------------------------------------------
        # Mach diagnostic
        # ---------------------------------------------------------------------
        c_s = np.asarray(HP_run._calculate_c_s(T_v), dtype=float).reshape(-1)

        if len(c_s) == len(u_v):
            c_s_u = c_s
        elif len(c_s) == len(u_v) + 1:
            c_s_u = 0.5 * (c_s[:-1] + c_s[1:])
        elif len(c_s) + 1 == len(u_v):
            x_c = np.linspace(0.0, 1.0, len(c_s))
            x_u = np.linspace(0.0, 1.0, len(u_v))
            c_s_u = np.interp(x_u, x_c, c_s)
        else:
            x_c = np.linspace(0.0, 1.0, len(c_s))
            x_u = np.linspace(0.0, 1.0, len(u_v))
            c_s_u = np.interp(x_u, x_c, c_s)

        max_mach = float(np.nanmax(np.abs(u_v) / c_s_u))

        # ---------------------------------------------------------------------
        # Liquid pressure profile using full axial T_v
        # ---------------------------------------------------------------------
        T_HP_full = (T_solid, u_v, T_v)

        liquid = LiquidDiscretisedFullTemperature(
            cfg_HP_run,
            T_HP_full,
        )

        P_l = liquid.get_pressure_drop_profile()

        if self.print_liquid_diagnostics:
            liquid.print_diagnostics(
                label=f"P_th={P_th:.3e} W, T_target={T_target:.2f} K"
            )

        # ---------------------------------------------------------------------
        # Vapour pressure profile
        # ---------------------------------------------------------------------
        P_v = s_props.calculate_Na_pressure_v(T_v)

        P_l = np.asarray(P_l, dtype=float).reshape(-1)
        P_v = np.asarray(P_v, dtype=float).reshape(-1)

        if len(P_l) != len(P_v):
            raise ValueError(
                "Pressure-profile length mismatch: "
                f"len(P_l)={len(P_l)}, len(P_v)={len(P_v)}"
            )

        liquid_dP = float(P_l[-1] - P_l[0])
        vapour_dP = float(P_v[-1] - P_v[0])

        # ---------------------------------------------------------------------
        # Capillary pressure requirement using wet-point construction
        # ---------------------------------------------------------------------
        P_v_shift = P_v - P_v[0]
        P_l_shift = P_l - P_l[0]

        P_l_rev = P_l_shift[::-1]

        shift_touch = np.min(P_v_shift - P_l_rev)
        P_v_touch = P_v_shift - shift_touch

        diff = P_v_touch - P_l_rev

        i_contact = int(np.nanargmin(np.abs(diff)))

        shift_zero = P_v_touch[0]
        P_v_plot = P_v_touch - shift_zero
        P_l_plot = P_l_rev - shift_zero

        Delta_P_profile_v_l = np.concatenate([
            P_v_plot[:i_contact + 1],
            P_l_plot[:i_contact + 1][::-1],
        ])

        capillary_pressure_required = float(
            Delta_P_profile_v_l[0] - Delta_P_profile_v_l[-1]
        )

        T_actual = float(np.mean(T_v))
        sigma_l = s_props.calculate_Na_surface_tension(T_actual)

        capillary_pressure_max = float(
            2.0 * sigma_l / cfg_HP_run.wick.r_pore
        )

        capillary_margin = float(
            capillary_pressure_max - capillary_pressure_required
        )

        Q_HP = thermal_power_to_heatpipe_Q(P_th)

        print(
            f"  post: Q_HP={Q_HP:.6e} W, "
            f"T_actual={T_actual:.2f} K, "
            f"max_mach={max_mach:.6e}, "
            f"liquid_dP={liquid_dP:.6e} Pa, "
            f"vapour_dP={vapour_dP:.6e} Pa, "
            f"cap_required={capillary_pressure_required:.6e} Pa, "
            f"cap_max={capillary_pressure_max:.6e} Pa, "
            f"cap_margin={capillary_margin:.6e} Pa"
        )

        point = CapillaryPoint(
            P_th=P_th,
            Q_HP=Q_HP,
            h_cond=h_cond,
            T_target=T_target,
            T_actual=T_actual,
            capillary_margin=capillary_margin,
            capillary_pressure_required=capillary_pressure_required,
            capillary_pressure_max=capillary_pressure_max,
            liquid_dP=liquid_dP,
            vapour_dP=vapour_dP,
            max_mach=max_mach,
            T_v=T_v,
            T_solid=T_solid,
            u_v=u_v,
            delta_p_profile=diff,
        )

        self._cache[cache_key] = point
        return point

    # -------------------------------------------------------------------------
    # Solve one target temperature
    # -------------------------------------------------------------------------

    def solve_for_target_temperature(
        self,
        *,
        T_target: float,
        P_start: float,
        P_min: float,
        P_max: float,
        Delta_P_start: float,
        Delta_P_min: float,
        Delta_P_max: float,
        max_iter: int,
        max_start_iter: int,
        growth_factor: float,
        shrink_factor: float,
    ) -> SolveResult:
        """
        For one prescribed T_target, find P_th such that

            capillary_margin = 0.

        The condenser h value is not prescribed directly. Instead, each trial
        P_th gives

            h_cond = Q_HP / (A_cond * (T_target - T_cond)).

        If the initial capillary margin is positive, the solver increases P_th.
        If the initial capillary margin is negative, the solver decreases P_th.
        """

        T_target = float(T_target)
        P_start = float(P_start)
        Delta_P = float(Delta_P_start)

        n_solves = 0

        best_point = None
        best_margin = np.inf

        valid_point = None
        valid_margin = None
        valid_P = None

        # -----------------------------------------------------------------
        # Build start candidates around P_start.
        #
        # This makes the method more robust if the exact P_start fails.
        # -----------------------------------------------------------------
        start_candidates = [P_start]

        for k in range(1, max_start_iter + 1):
            start_candidates.append(P_start * (1.10 ** k))
            start_candidates.append(P_start / (1.10 ** k))

        start_candidates = [
            float(candidate)
            for candidate in start_candidates
            if P_min <= candidate <= P_max
        ]

        # -----------------------------------------------------------------
        # Find a converged starting point.
        # -----------------------------------------------------------------
        for j, P_try in enumerate(start_candidates):
            try:
                point = self.solve_reactor_point(
                    P_th=P_try,
                    T_target=T_target,
                )
                n_solves += 1

                margin = float(point.capillary_margin)

                print(
                    f"  start {j:03d}: P={P_try:.6e} W, "
                    f"h_cond={point.h_cond:.6e}, "
                    f"T_actual={point.T_actual:.2f} K, "
                    f"margin={margin:.6e}"
                )

                if abs(margin) < abs(best_margin):
                    best_point = point
                    best_margin = margin

                # Critical assignments. These were the missing part that made
                # previous_P become None.
                valid_point = point
                valid_margin = margin
                valid_P = float(P_try)

                break

            except Exception as exc:
                n_solves += 1

                print(
                    f"  start failed {j:03d}: "
                    f"T_target={T_target:.2f} K, "
                    f"P={P_try:.6e} W, error={exc}"
                )

        if valid_point is None or valid_margin is None or valid_P is None:
            return SolveResult(
                point=best_point,
                residual=best_margin if best_point is not None else np.nan,
                status="no converged starting point",
                n_solves=n_solves,
            )

        if abs(valid_margin) < 1.0e-8:
            return SolveResult(
                point=valid_point,
                residual=valid_margin,
                status="starting point at capillary limit",
                n_solves=n_solves,
            )

        # -----------------------------------------------------------------
        # Decide initial search direction.
        #
        # margin > 0:
        #     below capillary limit, increase P_th to increase heat transport.
        #
        # margin < 0:
        #     above capillary limit, decrease P_th to reduce heat transport.
        # -----------------------------------------------------------------
        direction = 1.0 if valid_margin > 0.0 else -1.0

        previous_P = float(valid_P)
        previous_point = valid_point
        previous_margin = float(valid_margin)

        # -----------------------------------------------------------------
        # March in P_th until sign change.
        # -----------------------------------------------------------------
        for i in range(max_iter):
            P_trial = previous_P + direction * Delta_P

            if P_trial < P_min or P_trial > P_max:
                return SolveResult(
                    point=best_point,
                    residual=best_margin,
                    status="hit P bound before crossing; returning closest point",
                    n_solves=n_solves,
                )

            try:
                point = self.solve_reactor_point(
                    P_th=P_trial,
                    T_target=T_target,
                )
                n_solves += 1

                margin = float(point.capillary_margin)

                print(
                    f"  iter {i:03d}: P={P_trial:.6e} W, "
                    f"h_cond={point.h_cond:.6e}, "
                    f"T_actual={point.T_actual:.2f} K, "
                    f"margin={margin:.6e}, "
                    f"Delta_P={Delta_P:.6e}"
                )

            except Exception as exc:
                n_solves += 1

                print(
                    f"  trial failed {i:03d}: "
                    f"P={P_trial:.6e} W, "
                    f"Delta_P={Delta_P:.6e}, error={exc}"
                )

                Delta_P *= shrink_factor

                if Delta_P < Delta_P_min:
                    return SolveResult(
                        point=best_point,
                        residual=best_margin,
                        status="failed near path; returning closest point",
                        n_solves=n_solves,
                    )

                continue

            if abs(margin) < abs(best_margin):
                best_point = point
                best_margin = margin

            # -----------------------------------------------------------------
            # Sign crossing.
            # -----------------------------------------------------------------
            if previous_margin * margin < 0.0:
                alpha = previous_margin / (previous_margin - margin)

                P_cross = previous_P + alpha * (P_trial - previous_P)

                crossing_point = copy.deepcopy(previous_point)

                crossing_point.P_th = float(P_cross)

                crossing_point.Q_HP = float(
                    previous_point.Q_HP
                    + alpha * (point.Q_HP - previous_point.Q_HP)
                )

                crossing_point.h_cond = float(
                    previous_point.h_cond
                    + alpha * (point.h_cond - previous_point.h_cond)
                )

                # This is the actual simulated temperature, interpolated between
                # the two actual reactor solutions. It is not the target temperature.
                crossing_point.T_actual = float(
                    previous_point.T_actual
                    + alpha * (point.T_actual - previous_point.T_actual)
                )

                crossing_point.T_target = float(T_target)

                crossing_point.capillary_pressure_required = float(
                    previous_point.capillary_pressure_required
                    + alpha
                    * (
                        point.capillary_pressure_required
                        - previous_point.capillary_pressure_required
                    )
                )

                crossing_point.capillary_pressure_max = float(
                    previous_point.capillary_pressure_max
                    + alpha
                    * (
                        point.capillary_pressure_max
                        - previous_point.capillary_pressure_max
                    )
                )

                crossing_point.capillary_margin = 0.0

                crossing_point.liquid_dP = float(
                    previous_point.liquid_dP
                    + alpha * (point.liquid_dP - previous_point.liquid_dP)
                )

                crossing_point.vapour_dP = float(
                    previous_point.vapour_dP
                    + alpha * (point.vapour_dP - previous_point.vapour_dP)
                )

                crossing_point.max_mach = float(
                    previous_point.max_mach
                    + alpha * (point.max_mach - previous_point.max_mach)
                )

                print(
                    f"  crossed capillary limit:\n"
                    f"    T_target = {T_target:.6f} K\n"
                    f"    T_actual = {crossing_point.T_actual:.6f} K\n"
                    f"    P_cross  = {crossing_point.P_th:.6e} W\n"
                    f"    Q_cross  = {crossing_point.Q_HP:.6e} W\n"
                    f"    h_cross  = {crossing_point.h_cond:.6e}"
                )

                return SolveResult(
                    point=crossing_point,
                    residual=0.0,
                    status="crossed capillary limit",
                    n_solves=n_solves,
                )

            # -----------------------------------------------------------------
            # Same side of the limit.
            #
            # If the new point is closer to the limit, accept it and grow step.
            # If not, shrink step and try again from previous accepted point.
            # -----------------------------------------------------------------
            if abs(margin) < abs(previous_margin):
                previous_P = float(P_trial)
                previous_point = point
                previous_margin = float(margin)

                Delta_P = min(Delta_P * growth_factor, Delta_P_max)

            else:
                print(
                    f"  not moving closer: "
                    f"old |margin|={abs(previous_margin):.6e}, "
                    f"new |margin|={abs(margin):.6e}; shrinking step."
                )

                Delta_P *= shrink_factor

                if Delta_P < Delta_P_min:
                    return SolveResult(
                        point=best_point,
                        residual=best_margin,
                        status="no crossing found; returning closest point",
                        n_solves=n_solves,
                    )

        return SolveResult(
            point=best_point,
            residual=best_margin,
            status="max iterations reached; returning closest point",
            n_solves=n_solves,
        )

    # -------------------------------------------------------------------------
    # Full curve
    # -------------------------------------------------------------------------

    def run(
        self,
        *,
        T_HP_targets,
        P_start: float = 5.0e6,
        P_min: float = 1.0e6,
        P_max: float = 40.0e6,
        Delta_P_start: float = 0.25e6,
        Delta_P_min: float = 0.025e6,
        Delta_P_max: float = 1.0e6,
        max_iter: int = 60,
        max_start_iter: int = 20,
        growth_factor: float = 1.15,
        shrink_factor: float = 0.5,
        use_T_continuation: bool = True,
        save: bool = True,
    ):
        T_HP_targets = np.asarray(T_HP_targets, dtype=float)

        points = []
        residuals = []
        statuses = []
        n_solves = []

        current_P_start = float(P_start)

        t0 = time.time()

        for j, T_target in enumerate(T_HP_targets):
            print(
                f"\n[capillary T-sweep] point {j + 1}/{len(T_HP_targets)}: "
                f"T_HP_target={T_target:.3f} K"
            )

            result = self.solve_for_target_temperature(
                T_target=float(T_target),
                P_start=current_P_start,
                P_min=P_min,
                P_max=P_max,
                Delta_P_start=Delta_P_start,
                Delta_P_min=Delta_P_min,
                Delta_P_max=Delta_P_max,
                max_iter=max_iter,
                max_start_iter=max_start_iter,
                growth_factor=growth_factor,
                shrink_factor=shrink_factor,
            )

            points.append(result.point)
            residuals.append(result.residual)
            statuses.append(result.status)
            n_solves.append(result.n_solves)

            if result.point is not None:
                print(
                    f"  status: {result.status}\n"
                    f"  T_target = {T_target:.6f} K\n"
                    f"  T_actual = {result.point.T_actual:.6f} K\n"
                    f"  P_th     = {result.point.P_th:.6e} W\n"
                    f"  Q_HP     = {result.point.Q_HP:.6e} W\n"
                    f"  h_cond   = {result.point.h_cond:.6e}\n"
                    f"  residual = {result.residual:.6e}\n"
                    f"  solves   = {result.n_solves}"
                )

                if (
                    use_T_continuation
                    and "crossed capillary limit" in result.status
                ):
                    current_P_start = result.point.P_th

            else:
                print(f"  status: {result.status}")

            elapsed = time.time() - t0
            print(f"  elapsed: {elapsed / 60.0:.2f} min")

        results = self.pack_results(
            T_HP_targets=T_HP_targets,
            points=points,
            residuals=residuals,
            statuses=statuses,
            n_solves=n_solves,
        )

        if save:
            self.save_results(results)

        return results

    def pack_results(
        self,
        *,
        T_HP_targets,
        points,
        residuals,
        statuses,
        n_solves,
    ):
        results = {
            "T_target": np.asarray(T_HP_targets, dtype=float),
            "T_actual": np.asarray(
                [p.T_actual if p is not None else np.nan for p in points],
                dtype=float,
            ),
            "Q_HP": np.asarray(
                [p.Q_HP if p is not None else np.nan for p in points],
                dtype=float,
            ),
            "P_th": np.asarray(
                [p.P_th if p is not None else np.nan for p in points],
                dtype=float,
            ),
            "h_cond": np.asarray(
                [p.h_cond if p is not None else np.nan for p in points],
                dtype=float,
            ),
            "capillary_margin": np.asarray(
                [p.capillary_margin if p is not None else np.nan for p in points],
                dtype=float,
            ),
            "capillary_pressure_required": np.asarray(
                [
                    p.capillary_pressure_required
                    if p is not None else np.nan
                    for p in points
                ],
                dtype=float,
            ),
            "capillary_pressure_max": np.asarray(
                [
                    p.capillary_pressure_max
                    if p is not None else np.nan
                    for p in points
                ],
                dtype=float,
            ),
            "liquid_dP": np.asarray(
                [p.liquid_dP if p is not None else np.nan for p in points],
                dtype=float,
            ),
            "vapour_dP": np.asarray(
                [p.vapour_dP if p is not None else np.nan for p in points],
                dtype=float,
            ),
            "max_mach": np.asarray(
                [p.max_mach if p is not None else np.nan for p in points],
                dtype=float,
            ),
            "residual": np.asarray(residuals, dtype=float),
            "status": np.asarray(statuses, dtype=str),
            "n_solves": np.asarray(n_solves, dtype=int),
        }

        if self.store_profiles:
            results["T_v_profiles"] = np.asarray(
                [p.T_v if p is not None else np.array([]) for p in points],
                dtype=object,
            )
            results["T_solid_profiles"] = np.asarray(
                [p.T_solid if p is not None else np.array([]) for p in points],
                dtype=object,
            )
            results["u_v_profiles"] = np.asarray(
                [p.u_v if p is not None else np.array([]) for p in points],
                dtype=object,
            )
            results["delta_p_profiles"] = np.asarray(
                [p.delta_p_profile if p is not None else np.array([]) for p in points],
                dtype=object,
            )

        metadata = {
            "reactor_data_path": str(self.reactor_data_path),
            "Ns": self.Ns,
            "T_cond": self.T_cond,
            "A_cond": self.A_cond,
            "Q_HP_definition": "Q_HP = P_th / 2970 * (24/7)",
            "h_cond_definition": "h_cond = Q_HP / (A_cond * (T_target - T_cond))",
            "saved_temperature": "T_actual = mean(T_v) from solved reactor, not T_target",
            "capillary_pressure_max": "2 * sigma_l(T_actual) / r_pore",
        }

        results["metadata_json"] = np.asarray(json.dumps(metadata, indent=2))

        return results

    def save_results(self, results):
        np.savez(self.output_path, **results)
        print(f"\nSaved capillary T-sweep results to: {self.output_path}")


# =============================================================================
# Example usage
# =============================================================================

if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    from utils.interpolator import OpenMCTallyGridSurrogate
    MODEL_PATH = Path("./utils/rgi_surrogate.joblib")
    interpolator_model = OpenMCTallyGridSurrogate()
    interpolator_model = interpolator_model.load(MODEL_PATH)

    solver = CapillaryTemperatureSweepSolver(
        reactor_data_path="./data/reactor_data.json",
        Ns=(15, 15, 23),

        # If you have an interpolator_model in your namespace, pass it here.
        # Otherwise set to None.
        interpolator_model=interpolator_model,

        # Condenser/environment temperature used in h estimate.
        T_cond=300.0,

        # If None, inferred from cfg_HP geometry.
        A_cond=None,

        initial_HP_mesh=(15, 23),

        output_path=f"./capillary_T_sweep_results_{timestamp}.npz",

        store_profiles=True,

        # Turn this on only for debugging. It prints a lot.
        print_liquid_diagnostics=False,
    )

    results = solver.run(
        T_HP_targets=np.linspace(750.0, 1400.0, 20),

        P_start=5.0e6,
        P_min=1.0e6,
        P_max=80.0e6,

        Delta_P_start=0.25e6,
        Delta_P_min=0.025e6,
        Delta_P_max=1.0e6,

        max_iter=60,
        max_start_iter=20,

        growth_factor=1.15,
        shrink_factor=0.5,

        use_T_continuation=True,
        save=True,
    )