# Heat pipe model runner

This repository contains python code that runs and visualises one of two steady-state heat pipe models: a lumped **thermal resistance network** (`network`) or a higher-resolution 2D (axial and radial) **discretised** model (`discretised`).

## Requirements

- Python 3.10+
- `numpy`, `matplotlib`

```sh
pip install numpy matplotlib
```

## How to run

Run the script with exactly one argument:

### Network model

```sh
python main.py network
```

### Discretised model

```sh
python main.py discretised
```

If you pass anything else (or nothing), the script raises `ValueError("Invalid function argument")`.

## Solution details

The script defines the heat pipe geometry and boundary conditions, then solves and visualises either the network node temperatures or the discretised temperature field.

## Project layout

- `models/heat_network_model.py` – `solve_heatpipe_network_model_static(...)`
- `models/heat_discretised_model.py` – `heatpipe_discretised` + solver
- `visualisation/visualise_network_results.py` – printing helper
- `visualisation/visualise_discretised_results.py` – plotting helper
