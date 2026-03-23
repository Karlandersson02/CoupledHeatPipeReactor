import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List

import joblib
import numpy as np
from scipy.interpolate import RegularGridInterpolator


# =============================================================================
# User settings
# =============================================================================

# -------------------------------------------------------------------------
# Temperature grid
# -------------------------------------------------------------------------
T_MIN = 500.0
T_MAX = 1200.0

# Number of grid points per dimension.
# Total OpenMC runs = N_T_HEAT_PIPE * N_T_FUEL_PIN * N_T_MODERATOR
N_T_HEAT_PIPE = 4
N_T_FUEL_PIN = 4
N_T_MODERATOR = 4

# -------------------------------------------------------------------------
# Output / storage
# -------------------------------------------------------------------------
BASE_OUTPUT_DIR = Path("outputs/openmc_data")
TRAINING_DATA_DIR =  BASE_OUTPUT_DIR / "training_data"
MODEL_DIR = Path("utils")

TRAINING_DATA_FILE = TRAINING_DATA_DIR / "training_dataset.npz"
METADATA_FILE = TRAINING_DATA_DIR / "training_metadata.json"
MODEL_FILE = MODEL_DIR / "rgi_surrogate.joblib"

# -------------------------------------------------------------------------
# OpenMC / MGXS settings
# -------------------------------------------------------------------------
NUM_ENERGY_GROUPS = 8

TARGET_KEYS = [
    "total_xs",            # shape (G,)
    "scatter_matrix_xs",   # shape (G, G)
    "fission_xs",          # shape (G,)
    "nu",                  # shape (G,)
    "chi",                 # shape (G,)
    "kappa",               # shape (G,)
]

CLEAN_RUN_DIRECTORIES = True


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class TargetSpec:
    name: str
    shape: Tuple[int, ...]
    size: int
    start: int
    stop: int


# =============================================================================
# Utility functions
# =============================================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def make_temperature_axes(
    t_min: float,
    t_max: float,
    n_heat_pipe: int,
    n_fuel_pin: int,
    n_moderator: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create 1D axes for the tensor-product temperature grid.
    """
    t_hp = np.linspace(t_min, t_max, n_heat_pipe)
    t_fp = np.linspace(t_min, t_max, n_fuel_pin)
    t_mod = np.linspace(t_min, t_max, n_moderator)
    return t_hp, t_fp, t_mod


def make_temperature_grid(
    t_min: float,
    t_max: float,
    n_heat_pipe: int,
    n_fuel_pin: int,
    n_moderator: int,
) -> np.ndarray:
    """
    Create a full tensor-product grid of temperatures.

    Returns
    -------
    X : ndarray, shape (n_samples, 3)
        Columns are [T_heat_pipe, T_fuel_pin, T_moderator].
    """
    t_hp, t_fp, t_mod = make_temperature_axes(
        t_min=t_min,
        t_max=t_max,
        n_heat_pipe=n_heat_pipe,
        n_fuel_pin=n_fuel_pin,
        n_moderator=n_moderator,
    )

    grid = np.array(np.meshgrid(t_hp, t_fp, t_mod, indexing="ij"))
    X = grid.reshape(3, -1).T
    return X


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


def reconstruct_targets(y_flat: np.ndarray, specs: List[TargetSpec]) -> Dict[str, np.ndarray]:
    """
    Reconstruct dictionary of arrays from flattened prediction vector.
    """
    out = {}
    for spec in specs:
        arr = y_flat[spec.start:spec.stop].reshape(spec.shape)
        out[spec.name] = arr
    return out


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


def specs_from_jsonable(data: List[dict]) -> List[TargetSpec]:
    return [
        TargetSpec(
            name=d["name"],
            shape=tuple(d["shape"]),
            size=d["size"],
            start=d["start"],
            stop=d["stop"],
        )
        for d in data
    ]


# =============================================================================
# OpenMC interface
# =============================================================================

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
    from CoupledSystems.Heatpipe import Heatpipe
    from openMC.fuelAssembly.fuel_assembly_homogenised import (
        create_openmc_model,
        load_homogenized_xs_from_statepoint,
    )

    ensure_dir(case_dir)

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


# =============================================================================
# Dataset generation
# =============================================================================

def generate_training_data(
    X: np.ndarray,
    num_energy_groups: int,
    run_root: Path,
) -> Tuple[np.ndarray, List[TargetSpec]]:
    """
    Run OpenMC on all temperature points in X and build training targets.
    """
    ensure_dir(run_root)

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
    ensure_dir(data_file.parent)

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


def load_training_dataset(
    data_file: Path,
    metadata_file: Path,
) -> Tuple[np.ndarray, np.ndarray, List[TargetSpec], dict]:
    data = np.load(data_file)
    X = data["X"]
    Y = data["Y"]

    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    specs = specs_from_jsonable(metadata["target_specs"])
    return X, Y, specs, metadata


# =============================================================================
# Regular-grid surrogate
# =============================================================================

class OpenMCTallyGridSurrogate:
    """
    Regular-grid interpolator surrogate for flattened OpenMC tally outputs.

    Assumes the data live on a tensor-product grid with axes:
        T_heat_pipe, T_fuel_pin, T_moderator
    """

    def __init__(self, method: str = "linear", bounds_error: bool = True, fill_value=None):
        self.method = method
        self.bounds_error = bounds_error
        self.fill_value = fill_value

        self.axes = None
        self.values = None
        self.model = None
        self.specs = None
        self.metadata = None

    def fit(self, X: np.ndarray, Y: np.ndarray, specs: List[TargetSpec], metadata: dict | None = None):
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)

        if X.ndim != 2 or X.shape[1] != 3:
            raise ValueError("X must have shape (n_samples, 3).")

        if Y.ndim != 2 or Y.shape[0] != X.shape[0]:
            raise ValueError("Y must have shape (n_samples, n_outputs).")

        t_hp = np.unique(X[:, 0])
        t_fp = np.unique(X[:, 1])
        t_mod = np.unique(X[:, 2])

        n_expected = len(t_hp) * len(t_fp) * len(t_mod)
        if X.shape[0] != n_expected:
            raise ValueError(
                "X does not represent a full tensor-product grid. "
                "RegularGridInterpolator requires a complete grid."
            )

        # Build a lookup from temperature triple -> row index in Y
        index_map = {}
        for i, row in enumerate(X):
            key = (float(row[0]), float(row[1]), float(row[2]))
            if key in index_map:
                raise ValueError(f"Duplicate grid point found: {key}")
            index_map[key] = i

        n_outputs = Y.shape[1]
        values = np.empty((len(t_hp), len(t_fp), len(t_mod), n_outputs), dtype=float)

        for i, hp in enumerate(t_hp):
            for j, fp in enumerate(t_fp):
                for k, mod in enumerate(t_mod):
                    key = (float(hp), float(fp), float(mod))
                    if key not in index_map:
                        raise ValueError(f"Missing grid point in dataset: {key}")
                    values[i, j, k, :] = Y[index_map[key], :]

        self.axes = (t_hp, t_fp, t_mod)
        self.values = values
        self.model = RegularGridInterpolator(
            points=self.axes,
            values=self.values,
            method=self.method,
            bounds_error=self.bounds_error,
            fill_value=self.fill_value,
        )

        self.specs = specs
        self.metadata = metadata

    def predict_flat(self, X_new: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been fit.")

        X_new = np.atleast_2d(np.asarray(X_new, dtype=float))
        Y_pred = self.model(X_new)

        # If only one point is queried, scipy may return shape (n_outputs,)
        Y_pred = np.atleast_2d(Y_pred)
        return Y_pred

    def predict_dict(self, X_new: np.ndarray) -> List[Dict[str, np.ndarray]]:
        Y_pred = self.predict_flat(X_new)
        return [reconstruct_targets(y, self.specs) for y in Y_pred]

    def save(self, model_file: Path) -> None:
        ensure_dir(model_file.parent)
        joblib.dump(
            {
                "method": self.method,
                "bounds_error": self.bounds_error,
                "fill_value": self.fill_value,
                "axes": self.axes,
                "values": self.values,
                "specs": specs_to_jsonable(self.specs),
                "metadata": self.metadata,
            },
            model_file,
        )
        print(f"Saved model to: {model_file}")

    @classmethod
    def load(cls, model_file: Path) -> "OpenMCTallyGridSurrogate":
        payload = joblib.load(model_file)

        obj = cls(
            method=payload["method"],
            bounds_error=payload["bounds_error"],
            fill_value=payload["fill_value"],
        )
        obj.axes = payload["axes"]
        obj.values = payload["values"]
        obj.model = RegularGridInterpolator(
            points=obj.axes,
            values=obj.values,
            method=obj.method,
            bounds_error=obj.bounds_error,
            fill_value=obj.fill_value,
        )
        obj.specs = specs_from_jsonable(payload["specs"])
        obj.metadata = payload["metadata"]
        return obj


# =============================================================================
# Main workflow
# =============================================================================

def build_or_load_training_data(force_recompute: bool = False) -> Tuple[np.ndarray, np.ndarray, List[TargetSpec], dict]:
    if TRAINING_DATA_FILE.exists() and METADATA_FILE.exists() and not force_recompute:
        print("Loading existing training dataset...")
        return load_training_dataset(TRAINING_DATA_FILE, METADATA_FILE)

    print("Generating new training dataset...")

    X = make_temperature_grid(
        t_min=T_MIN,
        t_max=T_MAX,
        n_heat_pipe=N_T_HEAT_PIPE,
        n_fuel_pin=N_T_FUEL_PIN,
        n_moderator=N_T_MODERATOR,
    )

    run_root = TRAINING_DATA_DIR / "openmc_runs"
    Y, specs = generate_training_data(
        X=X,
        num_energy_groups=NUM_ENERGY_GROUPS,
        run_root=run_root,
    )

    metadata = {
        "input_order": ["T_heat_pipe", "T_fuel_pin", "T_moderator"],
        "num_energy_groups": NUM_ENERGY_GROUPS,
        "temperature_range_K": [T_MIN, T_MAX],
        "grid_shape": [N_T_HEAT_PIPE, N_T_FUEL_PIN, N_T_MODERATOR],
        "n_samples": int(X.shape[0]),
        "target_keys": TARGET_KEYS,
        "target_specs": specs_to_jsonable(specs),
    }

    save_training_dataset(
        X=X,
        Y=Y,
        specs=specs,
        num_energy_groups=NUM_ENERGY_GROUPS,
        data_file=TRAINING_DATA_FILE,
        metadata_file=METADATA_FILE,
    )

    return X, Y, specs, metadata


def train_and_save_model(
    X: np.ndarray,
    Y: np.ndarray,
    specs: List[TargetSpec],
    metadata: dict,
) -> OpenMCTallyGridSurrogate:
    surrogate = OpenMCTallyGridSurrogate(
        method="linear",
        bounds_error=True,
        fill_value=None,
    )
    surrogate.fit(X, Y, specs, metadata)
    surrogate.save(MODEL_FILE)
    return surrogate


def demo_prediction(surrogate: OpenMCTallyGridSurrogate) -> None:
    x_query = np.array([[850.0, 1000.0, 900.0]])
    pred = surrogate.predict_dict(x_query)[0]

    print("\nExample prediction at:")
    print("  T_heat_pipe = 850 K")
    print("  T_fuel_pin  = 1000 K")
    print("  T_moderator = 900 K")

    for key, value in pred.items():
        print(f"\n{key}:")
        print(value)


if __name__ == "__main__":
    ensure_dir(BASE_OUTPUT_DIR)
    ensure_dir(TRAINING_DATA_DIR)
    ensure_dir(MODEL_DIR)

    FORCE_RECOMPUTE_DATA = False

    X, Y, specs, metadata = build_or_load_training_data(force_recompute=FORCE_RECOMPUTE_DATA)
    surrogate = train_and_save_model(X, Y, specs, metadata)
    demo_prediction(surrogate)