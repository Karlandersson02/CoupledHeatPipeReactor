# Heat pipe model runner

This repository contains python code that runs and visualises one of two steady-state heat pipe models: a lumped **thermal resistance network** (`network`) or a higher-resolution 2D (axial and radial) **discretised** model (`discretised`).

All constants (geometry, boundary conditions, mesh sizes, material parameters, and output settings) are stored in `config.json`.

## Requirements

- Python 3.10+
- `numpy`
- `matplotlib`

```sh
pip install numpy matplotlib
```

## How to run

Run the solver with exactly one model argument:

```sh
python solve_heatpipe.py network
python solve_heatpipe.py discretised
```

By default the script reads `config.json` in the current directory. To use another config file:

```sh
python solve_heatpipe.py network --config configs/case1.json
```

## Configuration

Edit `config.json` to change:
- **Geometry**: `D_v`, `delta_wick`, `delta_wall`, `l_evap`, `l_adiabatic`, `l_cond`
- **Boundary conditions**: `T_infc`, `Q_total`, `h_cond`, `h_vap`
- **Network model**: `network_model.k`
- **Discretised model**: `N_*` mesh sizes and `k_wall`, `k_wick`
- **Output behaviour**: `output.base_dir`, `output.save_figures`, `output.show_figures`

The file `configure.py` contains the logic to:
- load configuration
- build model inputs
- create output folders
- generate reports

`solve_heatpipe.py` is intentionally kept minimal and only handles CLI argument parsing and dispatch.

## Outputs

Each run writes a timestamped folder under `output.base_dir` (default: `outputs/`), for example:

```
outputs/2026-02-12_14-03-22_network/
outputs/2026-02-12_14-03-22_discretised/
```

Each output folder contains:
- `report.md` — summary including full input config and key outputs
- `config_used.json` — the exact config used for the run
- `network_temperatures.txt` — pretty-printed temperature table (network model)
- `temperature_distribution.png` — saved figure (discretised model, if enabled)

## Project layout (relevant files)

- `solve_heatpipe.py` — CLI entrypoint (clean wrapper)
- `configure.py` — configuration + run orchestration
- `config.json` — all tunable parameters
- `models/heat_network_model.py` — network solver
- `models/heat_discretised_model.py` — discretised solver
- `visualisation/visualise_network_results.py` — prints/saves network table
- `visualisation/visualise_discretised_results.py` — plots/saves temperature distribution
