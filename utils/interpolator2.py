import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List

import joblib
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from coupled_systems.Heatpipe import Heatpipe
from models.fuel_assembly.homogenised_cell_model import (
    create_openmc_model,
    load_homogenized_xs_from_statepoint,
)

CLEAN_RUN_DIRECTORIES = True

TARGET_KEYS = [
    "total_xs",            # shape (G,)
    "scatter_matrix_xs",   # shape (G, G)
    "fission_xs",          # shape (G,)
    "nu",                  # shape (G,)
    "chi",                 # shape (G,)
    "kappa",               # shape (G,)
]

@dataclass
class TargetSpec:
    name: str
    shape: Tuple[int, ...]
    size: int
    start: int
    stop: int

def specs_to_jsonable(specs: List[TargetSpec]) -> List[dict]:
    return [
        {
            "name": s.name,
            "shape": list(s.shape),
            "size": s.size,
            "start": s.start,
            "stop": s.stop,
        }
        for s in specs
    ]

def flatten_targets(result_dict: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[TargetSpec]]:
    """
    Flatten a dictionary of tally arrays into one 1D vector.

    Returns
    -------
    y_flat : ndarray, shape (n_targets_total,)
    specs : list[TargetSpec]
        Metadata needed to reconstruct the dictionary later.
    """
    pieces = []
    specs = []

    cursor = 0
    for key in TARGET_KEYS:
        arr = np.asarray(result_dict[key], dtype=float)
        flat = arr.ravel()
        start = cursor
        stop = cursor + flat.size

        specs.append(
            TargetSpec(
                name=key,
                shape=arr.shape,
                size=flat.size,
                start=start,
                stop=stop,
            )
        )
        pieces.append(flat)
        cursor = stop

    y_flat = np.concatenate(pieces)
    return y_flat, specs

def run_openmc_case(
    T_heat_pipe: float,
    T_fuel_pin: float,
    T_moderator: float,
    num_energy_groups: int,
    case_dir: Path,
) -> Dict[str, np.ndarray]:
    """
    Run one OpenMC case and return homogenized MGXS outputs.
    """
    from coupled_systems.Heatpipe import Heatpipe
    from models.fuel_assembly.homogenised_cell_model import (
        create_openmc_model,
        load_homogenized_xs_from_statepoint,
    )

    case_dir.mkdir(parents=True, exist_ok=True)

    data_Guoju_2 = {
        "r_outer":     0.03 + 0.001 + 0.0005 + 0.0005,
        "delta_wall":  0.001,
        "delta_gap":   0.0005,
        "delta_wick":  0.0005,
        "l_evap":      0.1,
        "l_adiabatic": 0.05,
        "l_cond":      0.55,
        "N_wick":      15,
        "N_wall":      15,
        "N_evap":      30,
        "N_adiabatic": 15,
        "N_cond":      165,
        "adiabatic_radial_flux": False,
        "Temperature_BC": True,
        "h_vap":    1e6,
        "h_cond":   62.6,
        "T_cond":   300,
        "T_op":     T_heat_pipe,
        "k_wick":   66.2,
        "k_wall":   19.0,
        "P_C":      2476,
        "T_C":      856,
        "porosity": 0.7,
    }

    HP = Heatpipe(data_Guoju_2)

    old_cwd = Path.cwd()
    try:
        os.chdir(case_dir)

        model, mgxs_objects, _ = create_openmc_model(
            HP,
            num_groups=num_energy_groups,
            T_heat_pipe=T_heat_pipe,
            T_moderator=T_moderator,
            T_fuel_pin=T_fuel_pin,
        )

        statepoint_path = model.run()
        result_dict = load_homogenized_xs_from_statepoint(statepoint_path, mgxs_objects)

        result_dict = {k: np.asarray(result_dict[k], dtype=float) for k in TARGET_KEYS}
        return result_dict

    finally:
        os.chdir(old_cwd)

def generate_training_data(
    X: np.ndarray,
    num_energy_groups: int,
    run_root: Path,
) -> Tuple[np.ndarray, List[TargetSpec]]:
    """
    Run OpenMC on all temperature points in X and build training targets.
    """
    run_root.mkdir(parents=True, exist_ok=True)

    Y_rows = []
    specs_ref = None

    n_samples = len(X)

    for i, (T_hp, T_fp, T_mod) in enumerate(X):
        case_name = (
            f"case_{i:04d}"
            f"_Thp_{T_hp:.1f}"
            f"_Tfp_{T_fp:.1f}"
            f"_Tmod_{T_mod:.1f}"
        )
        case_dir = run_root / case_name

        print(f"[{i+1}/{n_samples}] Running {case_name}")

        result_dict = run_openmc_case(
            T_heat_pipe=float(T_hp),
            T_fuel_pin=float(T_fp),
            T_moderator=float(T_mod),
            num_energy_groups=num_energy_groups,
            case_dir=case_dir,
        )

        y_flat, specs = flatten_targets(result_dict)

        if specs_ref is None:
            specs_ref = specs
        else:
            for s0, s1 in zip(specs_ref, specs):
                if s0.name != s1.name or s0.shape != s1.shape:
                    raise RuntimeError(
                        f"Inconsistent target layout between cases for '{s0.name}'. "
                        f"Expected {s0.shape}, got {s1.shape}."
                    )

        Y_rows.append(y_flat)

        np.savez(
            case_dir / "result.npz",
            T_heat_pipe=T_hp,
            T_fuel_pin=T_fp,
            T_moderator=T_mod,
            **result_dict,
        )

        if CLEAN_RUN_DIRECTORIES:
            for item in case_dir.iterdir():
                if item.name != "result.npz":
                    if item.is_dir():
                        shutil.rmtree(item, ignore_errors=True)
                    else:
                        try:
                            item.unlink()
                        except OSError:
                            pass

    Y = np.vstack(Y_rows)
    return Y, specs_ref

def save_training_dataset(
    X: np.ndarray,
    Y: np.ndarray,
    specs: List[TargetSpec],
    num_energy_groups: int,
    data_file: Path,
    metadata_file: Path,
) -> None:
    data_file.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        data_file,
        X=X,
        Y=Y,
    )

    metadata = {
        "input_order": ["T_heat_pipe", "T_fuel_pin", "T_moderator"],
        "target_keys": TARGET_KEYS,
        "num_energy_groups": num_energy_groups,
        "target_specs": specs_to_jsonable(specs),
        "X_shape": list(X.shape),
        "Y_shape": list(Y.shape),
    }

    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved dataset to: {data_file}")
    print(f"Saved metadata to: {metadata_file}")