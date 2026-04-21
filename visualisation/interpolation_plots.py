import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils.interpolator import (
    OpenMCTallyGridSurrogate,
    reconstruct_targets,
    specs_from_jsonable,
)


DATA_FILE = Path("outputs/openmc_data/training_data/training_dataset.npz")
METADATA_FILE = Path("outputs/openmc_data/training_data/training_metadata.json")
MODEL_FILE = Path("utils/rgi_surrogate.joblib")


# ---------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------
QUANTITY = "fission_xs"               # e.g. "total_xs", "absorption_xs", ...
GROUP_INDEX = 7                # zero-based energy group index

VARY_AXIS = "T_hp"             # one of: "T_hp", "T_fp", "T_mod"
FIXED_SELECTION = "middle"     # "middle", "min", "max", or explicit value

N_PLOT = 300                   # number of points for smooth interpolation line


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
AXIS_TO_COLUMN = {
    "T_hp": 0,
    "T_fp": 1,
    "T_mod": 2,
}

AXIS_TO_LABEL = {
    "T_hp": "Heat pipe temperature [K]",
    "T_fp": "Fuel pin temperature [K]",
    "T_mod": "Moderator temperature [K]",
}


def load_training_data(data_file, metadata_file, model_file):
    data = np.load(data_file)
    X = data["X"]
    Y = data["Y"]

    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    specs = specs_from_jsonable(metadata["target_specs"])
    model = OpenMCTallyGridSurrogate.load(model_file)

    return X, Y, specs, model


def choose_fixed_value(values, selection="middle"):
    unique_vals = np.unique(values)

    if isinstance(selection, str):
        if selection == "middle":
            return unique_vals[len(unique_vals) // 2]
        if selection == "min":
            return unique_vals[0]
        if selection == "max":
            return unique_vals[-1]
        raise ValueError(f"Unknown FIXED_SELECTION string: {selection}")

    # explicit numeric value -> snap to nearest available grid value
    selection = float(selection)
    return unique_vals[np.argmin(np.abs(unique_vals - selection))]


def extract_quantity_from_dataset(Y, specs, quantity, group_index):
    values = np.array([
        reconstruct_targets(y_row, specs)[quantity][group_index]
        for y_row in Y
    ])
    return values


def extract_quantity_from_prediction(Y_pred, model, quantity, group_index):
    spec = next(s for s in model.specs if s.name == quantity)
    flat_index = spec.start + group_index
    return Y_pred[:, flat_index]


def build_slice(X, y_data, vary_axis, fixed_values):
    vary_col = AXIS_TO_COLUMN[vary_axis]

    mask = np.ones(len(X), dtype=bool)
    for axis_name, fixed_value in fixed_values.items():
        col = AXIS_TO_COLUMN[axis_name]
        mask &= np.isclose(X[:, col], fixed_value)

    x_slice = X[mask, vary_col]
    y_slice = y_data[mask]

    sort_idx = np.argsort(x_slice)
    return x_slice[sort_idx], y_slice[sort_idx], mask


def build_prediction_line(X, model, quantity, group_index, vary_axis, fixed_values, n_plot=300):
    vary_col = AXIS_TO_COLUMN[vary_axis]

    x_min = X[:, vary_col].min()
    x_max = X[:, vary_col].max()
    x_plot = np.linspace(x_min, x_max, n_plot)

    X_plot = np.zeros((n_plot, X.shape[1]))
    for axis_name, col in AXIS_TO_COLUMN.items():
        if axis_name == vary_axis:
            X_plot[:, col] = x_plot
        else:
            X_plot[:, col] = fixed_values[axis_name]

    Y_pred = model.predict_flat(X_plot)
    y_pred = extract_quantity_from_prediction(Y_pred, model, quantity, group_index)

    return x_plot, y_pred


def plot_comparison(
    X,
    Y,
    specs,
    model,
    quantity="total_xs",
    group_index=0,
    vary_axis="T_hp",
    fixed_selection="middle",
    n_plot=300,
):
    y_data = extract_quantity_from_dataset(Y, specs, quantity, group_index)

    other_axes = [axis for axis in AXIS_TO_COLUMN if axis != vary_axis]
    fixed_values = {}

    for axis in other_axes:
        col = AXIS_TO_COLUMN[axis]
        fixed_values[axis] = choose_fixed_value(X[:, col], fixed_selection)

    x_slice, y_slice, mask = build_slice(X, y_data, vary_axis, fixed_values)
    x_plot, y_pred = build_prediction_line(
        X, model, quantity, group_index, vary_axis, fixed_values, n_plot=n_plot
    )

    print(f"Quantity: {quantity}")
    print(f"Group index: {group_index}")
    print(f"Varying: {vary_axis}")
    for axis, value in fixed_values.items():
        print(f"Holding {axis} = {value}")
    print(f"Number of data points on slice: {mask.sum()}")

    if len(x_slice) == 0:
        raise ValueError(
            "No data points found on the requested slice. "
            "Check your grid and fixed-value selection."
        )

    title_fixed = ", ".join(f"{axis}={value:.1f} K" for axis, value in fixed_values.items())

    plt.figure(figsize=(8, 5))
    plt.scatter(x_slice, y_slice, label="OpenMC data", zorder=3)
    plt.plot(x_plot, y_pred, label="RGI interpolation")

    plt.xlabel(AXIS_TO_LABEL[vary_axis])
    plt.ylabel(f"{quantity} (group {group_index + 1})")
    plt.title(f"Surrogate vs data\n{title_fixed}")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------
if __name__ == "__main__":
    X, Y, specs, model = load_training_data(DATA_FILE, METADATA_FILE, MODEL_FILE)

    plot_comparison(
        X,
        Y,
        specs,
        model,
        quantity=QUANTITY,
        group_index=GROUP_INDEX,
        vary_axis=VARY_AXIS,
        fixed_selection=FIXED_SELECTION,
        n_plot=N_PLOT,
    )