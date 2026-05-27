import numpy as np
import inspect

from typing import Sequence, Callable, Any
from scipy.optimize import fsolve, newton_krylov
from scipy.interpolate import RegularGridInterpolator

from models.component import Component
from models.heatpipe.solid_discretised_model import HeatpipeDiscretised
from models.fuel_pin.fuel_pin_model import FuelPin
from models.neutronics.axial_neutron_model import NeutronicsModel
from coupled.iso_reactor import Reactor
# from coupled.vapour_reactor import VapourReactor


def interpolate_2d(
    x_coarse: np.ndarray,
    y_coarse: np.ndarray,
    x_fine: np.ndarray,
    y_fine: np.ndarray,
    u_coarse: np.ndarray,
) -> np.ndarray:
    interp = RegularGridInterpolator(
        (x_coarse, y_coarse),
        u_coarse,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )

    x_mesh, y_mesh = np.meshgrid(x_fine, y_fine, indexing="ij")
    points_fine = np.column_stack((x_mesh.ravel(), y_mesh.ravel()))
    return interp(points_fine).reshape(len(x_fine), len(y_fine))


class Solver:
    def __init__(self, components: Sequence[Component], iterate: bool = False, save_iterates: bool = True):
        self.components = list(components)
        self.iterate = iterate
        self.solutions: list[Any] = []
        self.solution: Any | None = None
        self.save_iterates = save_iterates

    def _heatpipe_velocity_grid(self, heatpipe: Any) -> np.ndarray:
        dx = np.asarray(heatpipe.vapour.dx, dtype=float)
        faces = np.concatenate(([0.0], np.cumsum(dx)))
        return faces[1:-1]

    def _solve_component(
        self,
        component: Component,
        x_initial: np.ndarray,
        solver_fn: Callable[..., np.ndarray],
        solver_kwargs: dict[str, Any] | None = None,
    ) -> np.ndarray:
        component.assemble()
        kwargs = {} if solver_kwargs is None else solver_kwargs

        # Filter kwargs to only what the solver accepts
        sig = inspect.signature(solver_fn)
        valid_kwargs = {
            k: v for k, v in kwargs.items()
            if k in sig.parameters
        }

        x_sol, info, ier, mesg = fsolve(
            component.get_residuals,
            x_initial,
            full_output=True,
            **valid_kwargs,
        )

        r_final = info["fvec"]

        loss_l2 = np.linalg.norm(r_final)
        loss_mse = np.mean(r_final**2)
        loss_max = np.max(np.abs(r_final))

        # print("fsolve status:", ier)
        print("fsolve message:", mesg)
        # print("final residual L2 norm:", loss_l2)
        # print("final residual MSE:", loss_mse)
        # print("final residual max abs:", loss_max)
        # print("number of function evaluations:", info["nfev"])

        return x_sol

    def _transfer_iso_heatpipe(
        self,
        current: HeatpipeDiscretised,
        nxt: HeatpipeDiscretised,
        x_sol: np.ndarray,
    ) -> np.ndarray:
        x_out = current.unpack(x_sol)

        z_coarse = np.asarray(current.Z, dtype=float)
        r_coarse = np.asarray(current.R, dtype=float)

        z_fine = np.asarray(nxt.Z, dtype=float)
        r_fine = np.asarray(nxt.R, dtype=float)

        field = x_out[0].reshape(current.cfg.mesh.N_Z, current.cfg.mesh.N_R)
        field_interp = interpolate_2d(z_coarse, r_coarse, z_fine, r_fine, field)
        field_interp = field_interp.reshape(-1)

        return nxt.pack((field_interp, x_out[-1]))

    def _transfer_fuel_pin(
        self,
        current: FuelPin,
        nxt: FuelPin,
        x_sol: np.ndarray,
    ) -> np.ndarray:
        x_out = current.unpack(x_sol)

        z_coarse = np.asarray(current.Z, dtype=float)
        r_coarse = np.asarray(current.R, dtype=float)

        z_fine = np.asarray(nxt.Z, dtype=float)
        r_fine = np.asarray(nxt.R, dtype=float)

        field = x_out[0].reshape(current.cfg.mesh.N_Z, current.cfg.mesh.N_R)
        field_interp = interpolate_2d(z_coarse, r_coarse, z_fine, r_fine, field)

        return nxt.pack((field_interp,))

    def _transfer_neutronics(
        self,
        current: NeutronicsModel,
        nxt: NeutronicsModel,
        x_sol: np.ndarray,
    ) -> np.ndarray:
        x_out = current.unpack(x_sol)

        z_coarse = np.asarray(current.Z, dtype=float)
        g_coarse = np.arange(current.cfg.energy.N_G, dtype=float)

        z_fine = np.asarray(nxt.Z, dtype=float)
        g_fine = np.arange(nxt.cfg.energy.N_G, dtype=float)

        field = x_out[0].reshape(current.cfg.mesh.N_Z, current.cfg.energy.N_G)
        field_interp = interpolate_2d(z_coarse, g_coarse, z_fine, g_fine, field)

        return nxt.pack((field_interp, x_out[-1]))

    def _transfer_iso_reactor(
        self,
        current: Reactor,
        nxt: Reactor,
        x_sol: np.ndarray,
    ) -> np.ndarray:
        x_out = current.unpack(x_sol)

        hp_current  = current.heat_pipe_thermal_model
        fp_current  = current.fuel_pin_thermal_model
        neu_current = current.neutron_flux_model

        hp_next  = nxt.heat_pipe_thermal_model
        fp_next  = nxt.fuel_pin_thermal_model
        neu_next = nxt.neutron_flux_model

        x_init_hp  = self._transfer_iso_heatpipe(hp_current, hp_next, x_out[0])
        x_init_fp  = self._transfer_fuel_pin(fp_current, fp_next, x_out[1])
        x_init_neu = self._transfer_neutronics(neu_current, neu_next, x_out[2])

        return nxt.pack((x_init_hp, x_init_fp, x_init_neu))
    
    def _transfer_vap_heatpipe(
        self,
        current,
        nxt,
        x_sol: np.ndarray,
    ) -> np.ndarray:
        x_out = current.unpack(x_sol)

        z_coarse = np.asarray(current.solid.Z, dtype=float)
        z_coarse_u = self._heatpipe_velocity_grid(current)
        r_coarse = np.asarray(current.solid.R, dtype=float)

        z_fine = np.asarray(nxt.solid.Z, dtype=float)
        z_fine_u = self._heatpipe_velocity_grid(nxt)
        r_fine = np.asarray(nxt.solid.R, dtype=float)

        T_solid = x_out[0].reshape(current.cfg.mesh.N_Z, current.cfg.mesh.N_R)
        T_solid_interp = interpolate_2d(z_coarse, r_coarse, z_fine, r_fine, T_solid)
        T_solid_interp = T_solid_interp.reshape(-1)

        u_v_interp = np.interp(z_fine_u, z_coarse_u, x_out[1])
        T_v_interp = np.interp(z_fine, z_coarse, x_out[2])

        return nxt.pack((T_solid_interp, u_v_interp, T_v_interp))
    
    def _transfer_vap_reactor(
        self,
        current,
        nxt,
        x_sol: np.ndarray,
    ) -> np.ndarray:
        x_out = current.unpack(x_sol)

        hp_current  = current.heatpipe
        fp_current  = current.fuel_pin_thermal_model
        neu_current = current.neutron_flux_model

        hp_next  = nxt.heatpipe
        fp_next  = nxt.fuel_pin_thermal_model
        neu_next = nxt.neutron_flux_model

        x_init_hp  = self._transfer_vap_heatpipe(hp_current, hp_next, x_out[0])
        x_init_fp  = self._transfer_fuel_pin(fp_current, fp_next, x_out[1])
        x_init_neu = self._transfer_neutronics(neu_current, neu_next, x_out[2])

        return nxt.pack((x_init_hp, x_init_fp, x_init_neu))
    
    def _transfer_iso_to_vapour(
            self,
            current: Reactor,
            nxt,
            x_sol: np.ndarray
    ) -> np.ndarray:
        x_out = current.unpack(x_sol)

        hp_next = nxt.heatpipe

        T_init_solid = x_out[0][:-1]
        T_init_vap = np.repeat(x_out[0][-1:], hp_next.cfg.mesh.N_Z)
        hp_next.vapour.set_T_HP(T_init_solid * nxt.T_cond)
        u_init_vap = hp_next.vapour._build_initial_velocity(T_init_vap * nxt.T_cond) / nxt.u_v_ref

        x_init_hp  = np.r_[T_init_solid, u_init_vap, T_init_vap]
        x_init_fp  = x_out[1]
        x_init_neu = x_out[2]

        return nxt.pack((x_init_hp, x_init_fp, x_init_neu))

    def _is_type(self, obj: object, name: str) -> bool:
        return type(obj).__name__ == name

    def _get_next_initial_guess(
        self,
        current: Component,
        nxt: Component,
        x_sol: np.ndarray,
    ) -> np.ndarray:
        if self._is_type(current, "HeatpipeDiscretised") and self._is_type(nxt, "HeatpipeDiscretised"):
            return self._transfer_iso_heatpipe(current, nxt, x_sol)

        if self._is_type(current, "FuelPin") and self._is_type(nxt, "FuelPin"):
            return self._transfer_fuel_pin(current, nxt, x_sol)

        if self._is_type(current, "NeutronicsModel") and self._is_type(nxt, "NeutronicsModel"):
            return self._transfer_neutronics(current, nxt, x_sol)

        if self._is_type(current, "Reactor") and self._is_type(nxt, "Reactor"):
            return self._transfer_iso_reactor(current, nxt, x_sol)

        if self._is_type(current, "Reactor") and self._is_type(nxt, "VapourReactor"):
            return self._transfer_iso_to_vapour(current, nxt, x_sol)

        if self._is_type(current, "Heatpipe") and self._is_type(nxt, "Heatpipe"):
            return self._transfer_vap_heatpipe(current, nxt, x_sol)
        
        if self._is_type(current, "VapourReactor") and self._is_type(nxt, "VapourReactor"):
            return self._transfer_vap_reactor(current, nxt, x_sol)

        raise TypeError(
            f"Unsupported component transfer: {type(current).__name__} -> {type(nxt).__name__}"
        )

    def _normalize_solver_inputs(
        self,
        solvers: Callable[..., np.ndarray] | Sequence[Callable[..., np.ndarray]],
        solver_kwargs: dict[str, Any] | Sequence[dict[str, Any]] | None = None,
    ) -> tuple[list[Callable[..., np.ndarray]], list[dict[str, Any]]]:
        n_components = len(self.components)

        if callable(solvers):
            solver_list = [solvers] * n_components
        else:
            solver_list = list(solvers)
            if len(solver_list) != n_components:
                raise ValueError(
                    f"Expected {n_components} solvers, got {len(solver_list)}."
                )

        if solver_kwargs is None:
            kwargs_list = [{} for _ in range(n_components)]
        elif isinstance(solver_kwargs, dict):
            kwargs_list = [solver_kwargs.copy() for _ in range(n_components)]
        else:
            kwargs_list = list(solver_kwargs)
            if len(kwargs_list) != n_components:
                raise ValueError(
                    f"Expected {n_components} solver kwargs dictionaries, got {len(kwargs_list)}."
                )

        return solver_list, kwargs_list

    def submit_initial_guess(self, x_initial):
        self.submitted_initial_guess = x_initial

    def run(
        self,
        solvers: Callable[..., np.ndarray] | Sequence[Callable[..., np.ndarray]],
        solver_kwargs: dict[str, Any] | Sequence[dict[str, Any]] | None = None,
    ) -> None:
        solver_list, kwargs_list = self._normalize_solver_inputs(solvers, solver_kwargs)

        if hasattr(self, "submitted_initial_guess"):
            x_initial = self.submitted_initial_guess
        else:
            x_initial = self.components[0].initial_guess()

        for i, component in enumerate(self.components):
            if not self.iterate and i > 0:
                x_initial = component.initial_guess()

            if len(self.components) > 1:
                if self.iterate:
                    print(f"Running iteration {i+1} ...")
                else:
                    print(f"Running case {i+1} ...")

            x_sol = self._solve_component(
                component,
                x_initial,
                solver_list[i],
                kwargs_list[i],
            )

            self.solutions.append(component.post_process(x_sol))

            is_last = i == len(self.components) - 1

            if self.iterate and not is_last:
                next_component = self.components[i + 1]
                x_initial = self._get_next_initial_guess(component, next_component, x_sol)
            else:
                x_out = component.post_process(np.asarray(x_sol))
                self.solution = x_out

    def newton_krylov(self, verbose: bool = True, **kwargs) -> None:
        self.run(newton_krylov, solver_kwargs={"verbose": verbose, **kwargs})

    def fsolve(self, **kwargs) -> None:
        self.run(fsolve, solver_kwargs=kwargs)
