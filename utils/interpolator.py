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

SAVE_DATASET_EVERY = 10

# -------------------------------------------------------------------------
# Temperature grid
# -------------------------------------------------------------------------
T_MIN = 700
T_MAX = 1500

# -------------------------------------------------------------------------
# Data / Config
# -------------------------------------------------------------------------

import json
from data.dataclass import *

with open("./data/reactor_data.json", "r") as f:
    data = json.load(f)

#Mesh dimensions
N_R_HP, N_R_FP, N_Z = 15, 15, 50

# Heat pipe config
geom   = HeatpipeGeometry(**data["HeatPipe"]["geometry"])
mesh   = HeatpipeMesh(N_R=N_R_HP, N_Z=N_Z)
mat    = HeatpipeMaterial(**data["HeatPipe"]["material"])
wick   = HeatpipeWick(**data["HeatPipe"]["wick"])
bc     = HeatpipeBC(**data["HeatPipe"]["bc"])
cfg_HP = HeatpipeConfig(geom, mesh, mat, wick, bc)

# Fuel pin config
geom_FP   = FuelPinGeometry(**data["FuelPin"]["geometry"])
mesh_FP   = FuelPinMesh(N_R=N_R_FP, N_Z=30)
energy_FP = FuelPinEnergy(**data["FuelPin"]["energy"])
mat_FP    = FuelPinMaterial(**data["FuelPin"]["material"])
cfg_FP    = FuelPinConfig(geom_FP, mesh_FP, energy_FP, mat_FP)

# Neutronics config
mesh_N = NeutronicsMesh(
    N_R = N_R_FP,
    N_Z = 50,
    l   = data["FuelPin"]["geometry"]["l"]
)
energy = NeutronicsEnergy(
    N_G   = cfg_FP.energy.N_G,
    power = data["Reactor"]["power"]["thermal"] / data["Reactor"]["components"]["N_FP"]
)
cfg_N = NeutronicsConfig(mesh_N, energy)

# Reactor config
cfg_R = ReactorConfig(cfg_HP, cfg_FP, cfg_N)
cfg_R = cfg_R.resolve_mesh()

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

TRAINING_DATA_FILE = TRAINING_DATA_DIR / "training_dataset_4_250_50000.npz"
METADATA_FILE = TRAINING_DATA_DIR / "training_metadata_4_250_50000.json"
MODEL_FILE = MODEL_DIR / "rgi_surrogate.joblib"

# -------------------------------------------------------------------------
# OpenMC / MGXS settings
# -------------------------------------------------------------------------
NUM_ENERGY_GROUPS = 8

TARGET_KEYS = [
    "diffusion_coefficient",
    "total_xs",            # shape (G,)
    "scatter_matrix_xs",   # shape (G, G)
    "fission_xs",          # shape (G,)
    "nu",                  # shape (G,)
    "chi",                 # shape (G,)
    "kappa",               # shape (G,)
]

STD_KEYS = [
    "diffusion_coefficient_std",
    "total_xs_std",
    "scatter_matrix_xs_std",
    "fission_xs_std",
    "nu_fission_xs_std",
    "chi_std",
    "kappa_fission_xs_std",
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


def flatten_selected_targets(
    result_dict: Dict[str, np.ndarray],
    keys: List[str],
) -> Tuple[np.ndarray, List[TargetSpec]]:
    """
    Flatten selected arrays from a dictionary into one 1D vector.
    """
    pieces = []
    specs = []

    cursor = 0
    for key in keys:
        if key not in result_dict:
            raise KeyError(f"Missing key '{key}' in result_dict.")

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


def flatten_targets(result_dict: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[TargetSpec]]:
    """
    Flatten only the interpolated target arrays.
    """
    return flatten_selected_targets(result_dict, TARGET_KEYS)


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
    # from coupled_systems.heatpipe_decoupled import Heatpipe
    from models.fuel_assembly.homogenised_cell_model import (
        create_openmc_model,
        load_homogenized_xs_from_statepoint,
    )

    ensure_dir(case_dir)

    old_cwd = Path.cwd()
    try:
        os.chdir(case_dir)

        model, mgxs_objects, _ = create_openmc_model(
            cfg_R,
            num_groups=num_energy_groups,
            T_heat_pipe=T_heat_pipe,
            T_moderator=T_moderator,
            T_fuel_pin=T_fuel_pin,
        )

        statepoint_path = model.run(output=True)
        result_dict = load_homogenized_xs_from_statepoint(statepoint_path, mgxs_objects)

        result_dict = {k: np.asarray(v, dtype=float) for k, v in result_dict.items()}
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
    save_every: int = SAVE_DATASET_EVERY,
    data_file: Path = TRAINING_DATA_FILE,
    metadata_file: Path = METADATA_FILE,
) -> Tuple[np.ndarray, List[TargetSpec], np.ndarray | None, List[TargetSpec] | None]:
    """
    Run OpenMC on all temperature points in X and build training targets.

    Saves the accumulated dataset every `save_every` iterations.

    Returns
    -------
    Y : ndarray
        Flattened interpolated targets only.
    specs_ref : list[TargetSpec]
        Specs for interpolated targets.
    Y_std : ndarray or None
        Flattened standard deviation targets, saved but not interpolated.
    std_specs_ref : list[TargetSpec] or None
        Specs for std targets.
    """
    ensure_dir(run_root)

    Y_rows = []
    Y_std_rows = []

    specs_ref = None
    std_specs_ref = None

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

        # Interpolated targets only
        interp_result_dict = {k: result_dict[k] for k in TARGET_KEYS}
        y_flat, specs = flatten_targets(interp_result_dict)

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

        # Std targets: saved, but not used by interpolator
        present_std_keys = [k for k in STD_KEYS if k in result_dict]
        missing_std_keys = [k for k in STD_KEYS if k not in result_dict]

        if present_std_keys and missing_std_keys:
            raise KeyError(
                "Partial std dataset returned from OpenMC. "
                f"Present std keys: {present_std_keys}. "
                f"Missing std keys: {missing_std_keys}."
            )

        if present_std_keys:
            y_std_flat, std_specs = flatten_selected_targets(result_dict, STD_KEYS)

            if std_specs_ref is None:
                std_specs_ref = std_specs
            else:
                for s0, s1 in zip(std_specs_ref, std_specs):
                    if s0.name != s1.name or s0.shape != s1.shape:
                        raise RuntimeError(
                            f"Inconsistent std target layout between cases for '{s0.name}'. "
                            f"Expected {s0.shape}, got {s1.shape}."
                        )

            Y_std_rows.append(y_std_flat)
        elif std_specs_ref is not None:
            raise RuntimeError(
                "Earlier cases contained std data, but this case does not. "
                "Refusing to save an inconsistent Y_std dataset."
            )

        # Save full case data, including std arrays
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

        n_done = len(Y_rows)
        if n_done % save_every == 0 or n_done == n_samples:
            X_partial = X[:n_done]
            Y_partial = np.vstack(Y_rows)
            Y_std_partial = np.vstack(Y_std_rows) if Y_std_rows else None

            save_training_dataset(
                X=X_partial,
                Y=Y_partial,
                specs=specs_ref,
                num_energy_groups=num_energy_groups,
                data_file=data_file,
                metadata_file=metadata_file,
                Y_std=Y_std_partial,
                std_specs=std_specs_ref,
            )

            print(f"Saved accumulated dataset with {n_done} / {n_samples} samples.")

    Y = np.vstack(Y_rows)
    Y_std = np.vstack(Y_std_rows) if Y_std_rows else None
    return Y, specs_ref, Y_std, std_specs_ref


def save_training_dataset(
    X: np.ndarray,
    Y: np.ndarray,
    specs: List[TargetSpec],
    num_energy_groups: int,
    data_file: Path,
    metadata_file: Path,
    Y_std: np.ndarray | None = None,
    std_specs: List[TargetSpec] | None = None,
    extra_metadata: dict | None = None,
) -> None:
    ensure_dir(data_file.parent)

    save_payload = {
        "X": X,
        "Y": Y,
    }
    if Y_std is not None:
        save_payload["Y_std"] = Y_std

    np.savez(data_file, **save_payload)

    metadata = {
        "input_order": ["T_heat_pipe", "T_fuel_pin", "T_moderator"],
        "target_keys": TARGET_KEYS,
        "num_energy_groups": num_energy_groups,
        "target_specs": specs_to_jsonable(specs),
        "X_shape": list(X.shape),
        "Y_shape": list(Y.shape),
    }

    if Y_std is not None and std_specs is not None:
        metadata["std_keys"] = STD_KEYS
        metadata["std_specs"] = specs_to_jsonable(std_specs)
        metadata["Y_std_shape"] = list(Y_std.shape)

    if extra_metadata is not None:
        metadata.update(extra_metadata)

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


def load_training_std_dataset(
    data_file: Path,
    metadata_file: Path,
) -> Tuple[np.ndarray | None, List[TargetSpec] | None]:
    data = np.load(data_file)

    if "Y_std" not in data:
        return None, None

    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    std_specs_data = metadata.get("std_specs")
    if std_specs_data is None:
        return data["Y_std"], None

    std_specs = specs_from_jsonable(std_specs_data)
    return data["Y_std"], std_specs


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

        if self.axes is None:
            raise RuntimeError("Temperature axes are missing.")

        X_new = np.atleast_2d(np.asarray(X_new, dtype=float))

        if X_new.shape[1] != len(self.axes):
            raise ValueError(
                f"Expected X_new to have shape (n_samples, {len(self.axes)}), "
                f"got {X_new.shape}."
            )

        X_clipped = X_new.copy()

        for axis_idx, axis_values in enumerate(self.axes):
            t_min = np.min(axis_values)
            t_max = np.max(axis_values)
            X_clipped[:, axis_idx] = np.clip(X_clipped[:, axis_idx], t_min, t_max)

        Y_pred = self.model(X_clipped)
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
    Y, specs, Y_std, std_specs = generate_training_data(
        X=X,
        num_energy_groups=NUM_ENERGY_GROUPS,
        run_root=run_root,
        save_every=SAVE_DATASET_EVERY,
        data_file=TRAINING_DATA_FILE,
        metadata_file=METADATA_FILE,
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

    if Y_std is not None and std_specs is not None:
        metadata["std_keys"] = STD_KEYS
        metadata["std_specs"] = specs_to_jsonable(std_specs)
        metadata["Y_std_shape"] = list(Y_std.shape)


    save_training_dataset(
        X=X,
        Y=Y,
        specs=specs,
        num_energy_groups=NUM_ENERGY_GROUPS,
        data_file=TRAINING_DATA_FILE,
        metadata_file=METADATA_FILE,
        Y_std=Y_std,
        std_specs=std_specs,
        extra_metadata={
            "temperature_range_K": [T_MIN, T_MAX],
            "grid_shape": [N_T_HEAT_PIPE, N_T_FUEL_PIN, N_T_MODERATOR],
            "n_samples": int(X.shape[0]),
        },
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

    FORCE_RECOMPUTE_DATA = True

    X, Y, specs, metadata = build_or_load_training_data(force_recompute=FORCE_RECOMPUTE_DATA)
    surrogate = train_and_save_model(X, Y, specs, metadata)
    demo_prediction(surrogate)

    # Y_std, std_specs = load_training_std_dataset(TRAINING_DATA_FILE, METADATA_FILE)
    # if Y_std is not None and std_specs is not None:
    #     print("\nStd values for first sample:")
    #     for s in std_specs: print(f"\n{s.name}:\n", Y_std[0, s.start:s.stop].reshape(s.shape))