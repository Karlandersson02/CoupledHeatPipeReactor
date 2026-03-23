import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

from openMC.fuelAssembly.interpolator import (
    specs_from_jsonable,
    OpenMCTallyGridSurrogate,
    reconstruct_targets,
)

DATA_FILE = Path("openmc_rgi_surrogate/training_data/training_dataset.npz")
METADATA_FILE = Path("openmc_rgi_surrogate/training_data/training_metadata.json")
MODEL_FILE = Path("openmc_rgi_surrogate/model/rgi_surrogate.joblib")

data = np.load(DATA_FILE)
X = data["X"]
Y = data["Y"]

with open(METADATA_FILE, "r") as f:
    metadata = json.load(f)

specs = specs_from_jsonable(metadata["target_specs"])
rgi = OpenMCTallyGridSurrogate.load(MODEL_FILE)

group_index = 0

y_total_g1 = np.array([
    reconstruct_targets(y_row, specs)["total_xs"][group_index]
    for y_row in Y
])

T_hp_data = X[:, 0]
T_fp_data = X[:, 1]
T_mod_data = X[:, 2]

# Pick actual grid values for the slice
T_fp_unique = np.unique(T_fp_data)
T_mod_unique = np.unique(T_mod_data)

T_fp_fixed = T_fp_unique[len(T_fp_unique) // 2]
T_mod_fixed = T_mod_unique[len(T_mod_unique) // 2]

print("Available T_fuel_pin values:", T_fp_unique)
print("Available T_moderator values:", T_mod_unique)
print("Using T_fuel_pin =", T_fp_fixed)
print("Using T_moderator =", T_mod_fixed)

mask = (
    np.isclose(T_fp_data, T_fp_fixed) &
    np.isclose(T_mod_data, T_mod_fixed)
)

print("Number of data points on slice:", np.sum(mask))
print("T_hp on slice:", T_hp_data[mask])
print("y on slice:", y_total_g1[mask])

# Sort the data points on the slice by heat-pipe temperature
sort_idx = np.argsort(T_hp_data[mask])
T_hp_slice = T_hp_data[mask][sort_idx]
y_slice = y_total_g1[mask][sort_idx]

# Dense line for interpolated prediction
T_hp_plot = np.linspace(T_hp_data.min(), T_hp_data.max(), 300)
X_plot = np.column_stack([
    T_hp_plot,
    np.full_like(T_hp_plot, T_fp_fixed),
    np.full_like(T_hp_plot, T_mod_fixed),
])

Y_pred = rgi.predict_flat(X_plot)

total_spec = next(s for s in rgi.specs if s.name == "total_xs")
flat_index = total_spec.start + group_index
y_interp = Y_pred[:, flat_index]

plt.figure(figsize=(8, 5))
plt.scatter(T_hp_slice, y_slice, label="OpenMC data", zorder=3)
plt.plot(T_hp_plot, y_interp, label="RGI interpolation")

plt.xlabel("Heat pipe temperature [K]")
plt.ylabel("Total macroscopic XS (group 1) [1/cm]")
plt.title(
    f"RGI surrogate vs data\n"
    f"T_fuel_pin={T_fp_fixed:.1f} K, T_moderator={T_mod_fixed:.1f} K"
)
plt.legend()
plt.tight_layout()
plt.show()