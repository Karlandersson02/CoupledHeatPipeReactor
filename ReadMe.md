# Coupled Modelling of a heat pipe reactor
This repository is a part of a Master's project found [here](https://odr.chalmers.se/items/b35fd56b-2b26-492e-8d28-b2f4dd06144d).

In spring of 2026 we developed a reduced/simplified model of a single fuel assembly in a heat pipe reactor, a microreactor concept currently being explored by [Westinghouse](https://westinghousenuclear.com/innovation/evinci-microreactor/) and [Antares](https://antaresindustries.com), among others. The important components of such a reactor are contained in the reactor core and consists of wicked heat pipes, a graphite moderating material, and fuel rods. Implementations of these components and the multi-physics coupling inbetween can be found in this repository.

## Usage

Install the Python dependencies from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

> [!Note]
> The full list of Python packages can be found in [requirements.txt](requirements.txt).
> Run the examples from the repository root so that the relative paths to the
> reactor data and the neutron-data surrogate, `utils/rgi_surrogate.joblib`, resolve correctly.

The [demo notebook](demo.ipynb) contains an example of the workflow below. Open it
in a notebook editor such as VS Code and select the Python environment containing
the installed dependencies, or run the following code blocks in order in a Python script.

Setting up and simulating the fuel assembly of the reactor is done in three steps:

### Step 1: Specify the reactor

The specifiers in [data/dataclass.py](data/dataclass.py) describe the geometry,
mesh, materials, boundary conditions, and energy settings of each component.
The heat pipe, fuel pin, and neutronics configurations are combined into a
`ReactorConfig`. Calling its `resolve_mesh()` method produces a resolved
configuration with compatible component meshes, including matching the fuel pin
and neutronics axial meshes to the heat pipe evaporator.

The `generate_config` helper builds these specifiers from
[data/reactor_data.json](data/reactor_data.json) and calls `resolve_mesh()` for you:

```python
import json

from utils.iso_reactor_utils import generate_config

with open("data/reactor_data.json", "r") as f:
    data = json.load(f)

# Requested radial cells in the heat pipe and fuel pin, and axial heat pipe cells.
cfg_coarse = generate_config(data, N_R_HP=15, N_R_FP=15, N_Z=30)
cfg_fine = generate_config(data, N_R_HP=30, N_R_FP=30, N_Z=60)
```

Edit the JSON values, or the loaded `data` dictionary before generating the
configurations, to change the reactor inputs. Mesh resolution can adjust the
requested cell counts to fit the component geometry; the resolved counts are
available through `cfg.HP.mesh`, `cfg.FP.mesh`, and `cfg.N.mesh`.

### Step 2: Create the models and solve

Instantiate a `Reactor` for each configuration. This couples the heat pipe,
fuel pin, and neutronics models. The example first solves a coarse model with
fixed thermal conductivities and neutron data, then enables temperature-dependent
properties on the finer mesh:

```python
from models.reactors.reactor import Reactor
from utils.solver import Solver

reactor_coarse = Reactor(cfg_coarse)
reactor_coarse.set_variable_k(False)
reactor_coarse.set_variable_neutron_data(False)

reactor_fine = Reactor(cfg_fine)
reactor_fine.set_variable_k(True)
reactor_fine.set_variable_neutron_data(True)

solver = Solver([reactor_coarse, reactor_fine], iterate=True)
solver.fsolve()
```

With `iterate=True`, the solver transfers the coarse solution to the next mesh
as its initial guess. For a single configuration, use `Solver([reactor_coarse])`.
The solver prints the SciPy convergence message and final residual norm; check
these before interpreting the results.

### Step 3: Inspect and visualise the results

`solver.solutions` contains the post-processed result for each model, in the same
order as the input list. `solver.solution` contains the final model's result.
For `Reactor`, unpack it as follows:

```python
(T_solid, u_vapour, T_vapour), T_fuel_pin, (neutron_flux, k_eff) = solver.solution
print(f"Effective multiplication factor: {k_eff:.5f}")

from utils.reactor_utils import plot_reactor_solutions

plot_reactor_solutions(solver)
```

The results include heat pipe solid and vapour temperatures, vapour velocity,
fuel pin temperatures, the neutron flux for each energy group, and the effective
multiplication factor. Temperatures are in kelvin. The plotting helper compares
temperature, velocity, and neutron flux profiles across the solved models.
