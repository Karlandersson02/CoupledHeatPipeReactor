import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


# -----------------------------
# Config + IO helpers
# -----------------------------

def load_config(config_path: str | Path) -> dict[str, Any]:
    p = Path(config_path)
    with p.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg


def make_output_dir(base_dir: str | Path, model_name: str) -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(base_dir) / f"{ts}_{model_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def write_report(
    out_dir: Path,
    model_name: str,
    cfg: dict[str, Any],
    results_text: str,
    extra: dict[str, Any] | None = None,
) -> None:
    cfg_dump = json.dumps(cfg, indent=2, ensure_ascii=False)

    extra_dump = ""
    if extra:
        extra_dump = "\n\n## Derived outputs\n\n```json\n" + json.dumps(extra, indent=2, ensure_ascii=False) + "\n```\n"

    report = (
        "# Heatpipe run report\n\n"
        f"- Model: **{model_name}**\n"
        f"- Timestamp: `{datetime.now().isoformat(timespec='seconds')}`\n\n"
        "## Inputs\n\n"
        f"```json\n{cfg_dump}\n```\n\n"
        "## Outputs\n\n"
        f"```text\n{results_text}\n```\n"
        f"{extra_dump}"
    )

    (out_dir / "report.md").write_text(report, encoding="utf-8")
    (out_dir / "config_used.json").write_text(cfg_dump + "\n", encoding="utf-8")


# -----------------------------
# Model input builders
# -----------------------------

@dataclass(frozen=True)
class OutputSettings:
    base_dir: str = "outputs"
    save_figures: bool = True
    show_figures: bool = True


def get_output_settings(cfg: dict[str, Any]) -> OutputSettings:
    out = cfg.get("output", {})
    return OutputSettings(
        base_dir=str(out.get("base_dir", "outputs")),
        save_figures=bool(out.get("save_figures", True)),
        show_figures=bool(out.get("show_figures", True)),
    )


def compute_network_inputs(cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Build (k, A, lamb, T_infc, Q_total) from config, matching your original main-file logic.
    """
    g = cfg["geometry"]
    bc = cfg["boundary_conditions"]

    D_v = float(g["D_v"])
    delta_wick = float(g["delta_wick"])
    delta_wall = float(g["delta_wall"])

    l_evap = float(g["l_evap"])
    l_adiabatic = float(g["l_adiabatic"])
    l_cond = float(g["l_cond"])

    T_infc = float(bc["T_infc"])
    Q_total = float(bc["Q_total"])

    k = np.array(cfg["network_model"]["k"], dtype=float)

    A = np.array([
        np.pi * (D_v + 2 * delta_wick + 2 * delta_wall) * l_evap,
        np.pi * (D_v + 2 * delta_wick) * l_evap,
        np.pi * (D_v + 2 * delta_wick) * l_cond,
        np.pi * (D_v + 2 * delta_wick + 2 * delta_wall) * l_cond,
        np.pi * ((D_v / 2 + delta_wick) ** 2 - (D_v / 2) ** 2),
        np.pi * ((D_v / 2 + delta_wick + delta_wall) ** 2 - (D_v / 2 + delta_wick) ** 2),
        np.pi * (D_v + 2 * delta_wick + 2 * delta_wall) * l_cond,
    ], dtype=float)

    lamb = np.array([
        delta_wall,     # radial
        delta_wick,     # radial
        delta_wick,     # radial
        delta_wall,     # radial
        l_adiabatic,    # axial
        l_adiabatic,    # axial
        -1.0,           # dummy; convection handled via k[6]*A[6]
    ], dtype=float)

    return k, A, lamb, T_infc, Q_total


def build_discretised_data(cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Build the dict expected by heatpipe_discretised(...) from config,
    matching your original data_discretised construction.
    """
    g = cfg["geometry"]
    bc = cfg["boundary_conditions"]
    dm = cfg["discretised_model"]

    D_v = float(g["D_v"])
    delta_wick = float(g["delta_wick"])
    delta_wall = float(g["delta_wall"])

    l_evap = float(g["l_evap"])
    l_adiabatic = float(g["l_adiabatic"])
    l_cond = float(g["l_cond"])

    T_infc = float(bc["T_infc"])
    Q_total = float(bc["Q_total"])
    h_vap = float(bc["h_vap"])
    h_cond = float(bc["h_cond"])

    N_evap = int(dm["N_evap"])

    return {
        "r_outer": D_v / 2 + delta_wall + delta_wick,
        "delta_wick": delta_wick,
        "delta_wall": delta_wall,
        "l_evap": l_evap,
        "l_adiabatic": l_adiabatic,
        "l_cond": l_cond,
        "N_wick": int(dm["N_wick"]),
        "N_wall": int(dm["N_wall"]),
        "N_evap": N_evap,
        "N_adiabatic": int(dm["N_adiabatic"]),
        "N_cond": int(dm["N_cond"]),
        "h_vap": h_vap,
        "h_cond": h_cond,
        "T_cond": T_infc,
        "k_wall": float(dm["k_wall"]),
        "k_wick": float(dm["k_wick"]),
        "Q": np.repeat(np.array([Q_total / N_evap], dtype=float), N_evap),
    }


# -----------------------------
# Run wrappers (do IO + reporting)
# -----------------------------

def run_network(cfg: dict[str, Any], out_dir: Path) -> int:
    from models.heat_network_model import solve_heatpipe_network_model_static
    from visualisation.visualise_network_results import print_heat_pipe_temperatures

    k, A, lamb, T_infc, Q_total = compute_network_inputs(cfg)
    T = solve_heatpipe_network_model_static(k, A, lamb, T_infc, Q_total)

    table_path = out_dir / "network_temperatures.txt"
    results_text = print_heat_pipe_temperatures(T, save_path=table_path)

    extra = {"T_vector": [float(x) for x in np.asarray(T, dtype=float).tolist()]}
    write_report(out_dir, "network", cfg, results_text, extra=extra)
    return 0


def run_discretised(cfg: dict[str, Any], out_dir: Path) -> int:
    from models.heat_discretised_model import heatpipe_discretised
    from visualisation.visualise_discretised_results import display_temperature_distribution

    out_settings = get_output_settings(cfg)

    data = build_discretised_data(cfg)
    heatpipe = heatpipe_discretised(data)
    T = heatpipe.solve_heatpipe_discretised()

    fig_path = (out_dir / "temperature_distribution.png") if out_settings.save_figures else None

    display_temperature_distribution(
        T,
        data,
        save_path=str(fig_path) if fig_path else None,
        show=out_settings.show_figures,
    )

    T_arr = np.asarray(T, dtype=float)
    T_vap = float(T_arr[-1])
    T_solid = T_arr[:-1]

    summary = (
        f"T_vap = {T_vap:.6g}\n"
        f"T_solid_min = {float(T_solid.min()):.6g}\n"
        f"T_solid_max = {float(T_solid.max()):.6g}\n"
        f"All temperatures: {T.tolist()}"
    )

    extra = {
        "T_vap": T_vap,
        "T_solid_min": float(T_solid.min()),
        "T_solid_max": float(T_solid.max()),
        "figure": str(fig_path) if fig_path else None,
    }

    write_report(out_dir, "discretised", cfg, summary, extra=extra)
    return 0
