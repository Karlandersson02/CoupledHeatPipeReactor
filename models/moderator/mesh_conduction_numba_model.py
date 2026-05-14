from unittest import skip

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import meshio # type: ignore
import pyvista as pv # type: ignore

from numba import njit, prange # type: ignore

from scipy.optimize import fsolve, newton_krylov, NoConvergence

from models.moderator.triangle_mesh import UnstructuredMesh, Surface
from models.moderator.mesh_conduction_model import ModeratorDiscretisedMesh

from utils.material_properties import moderator_k

class ModeratorDiscretisedMeshFast:
    def __init__(self, data, mesh):
        self.mesh = mesh
        n_triangles = mesh.triangles.shape[0]

        self.Q_in = data.get("Q_in")

        self.HP_centers       = data.get("HP_centers")
        self.fuel_pin_centers = data.get("fuel_pin_centers")

        self.HP_radius       = data.get("HP_radius")
        self.fuel_pin_radius = data.get("fuel_pin_radius")

        self.T_HP = data.get("T_HP")
        self.T_FP = data.get("T_FP")

        self.HP_BC = data.get("HP_BC")
        self.FP_BC = data.get("FP_BC")

        self.cell_k = (
            np.ones(mesh.n_triangles, dtype=float)
            if data.get("cell_k") is None
            else np.asarray(data.get("cell_k"), dtype=float)
        )
        self.cell_T = (
            np.zeros(mesh.n_triangles, dtype=float)
            if data.get("cell_T") is None
            else np.asarray(data.get("cell_T"), dtype=float)
        )

        if self.cell_k.shape != (n_triangles,):
            raise ValueError("cell_k must have shape (n_triangles,)")
        if self.cell_T.shape != (n_triangles,):
            raise ValueError("cell_T must have shape (n_triangles,)")
        
        self.mod_disc_mesh_slow = ModeratorDiscretisedMesh(data, mesh)
        self.mod_disc_mesh_slow.cell_k = moderator_k(np.ones_like(self.cell_k) * self.T_HP)


    def solve_nonlinear_picard(
        self,
        T0: np.ndarray | None = None,
        xtol: float = 1e-8,
        rtol: float = 1e-8,
        maxfev: int = 400,
        max_outer_iter: int = 15,
        omega: float = 1.0,
        verbose: bool = True,
        use_linear_guess: bool = True,
    ):

        n_triangles = self.mesh.triangles.shape[0]

        # Boundary data that does not depend on T
        self.calculate_von_Neumann_boundary_conditions()

        # Assemble geometry/topology arrays once
        (
            areas,
            centers,
            neighbours,
            edge_lengths,
            edge_centers,
            edge_normals,
            is_boundary,
            boundary_kind,
        ) = self.assemble_mesh_data(
            mesh=self.mesh,
            HP_centers=self.HP_centers,
            fuel_pin_centers=self.fuel_pin_centers,
            HP_radius=self.HP_radius,
            fuel_pin_radius=self.fuel_pin_radius,
        )

        k_HP = moderator_k(self.T_HP)
        k_FP = moderator_k(self.T_FP)

        # Initial guess
        if T0 is None:
            if use_linear_guess:
                T0 = self.mod_disc_mesh_slow.solve_linearly(4)
            else:
                T0 = self.cell_T.copy()

        T = np.asarray(T0, dtype=np.float64).reshape(-1)

        if T.shape != (n_triangles,):
            raise ValueError(f"T0 must have shape ({n_triangles},), got {T.shape}")

        outer_ier = 2
        outer_mesg = "Maximum number of Picard iterations reached."

        for outer_iter in range(1, max_outer_iter + 1):
            # Freeze cross-diffusion correction on current Picard iterate
            cell_k_outer = np.asarray(moderator_k(T), dtype=np.float64).reshape(-1)

            cross_diffusion_corr_outer = calculate_cross_diffusion_correction(
                T=T,
                cell_k=cell_k_outer,
                centers=centers,
                areas=areas,
                neighbours=neighbours,
                edge_lengths=edge_lengths,
                edge_centers=edge_centers,
                edge_normals=edge_normals,
                boundary_kind=boundary_kind,
                T_HP=self.T_HP,
                T_FP=self.T_FP,
                HP_bc=self.HP_BC,
                FP_bc=self.FP_BC,
                vN_bc=self.vN_boundary_conditions,
            )

            call_count = {"n": 0}

            def residual_wrapper(T_inner):
                call_count["n"] += 1

                T_inner = np.asarray(T_inner, dtype=np.float64).reshape(-1)

                cell_k_inner = np.asarray(moderator_k(T_inner), dtype=np.float64).reshape(-1)

                di_bc_inner = calculate_Dirichlet_boundary_conditions(
                    cell_k=cell_k_inner,
                    centers=centers,
                    edge_lengths=edge_lengths,
                    edge_centers=edge_centers,
                    boundary_kind=boundary_kind,
                    T_HP=self.T_HP,
                    T_FP=self.T_FP,
                    HP_bc=self.HP_BC,
                    FP_bc=self.FP_BC,
                    k_HP = k_HP,
                    k_FP = k_FP,
                )

                R = get_residuals(
                    T=T_inner,
                    cell_k=cell_k_inner,
                    centers=centers,
                    neighbours=neighbours,
                    edge_lengths=edge_lengths,
                    edge_centers=edge_centers,
                    edge_normals=edge_normals,
                    is_boundary=is_boundary,
                    boundary_kind=boundary_kind,
                    vN_bc=self.vN_boundary_conditions,
                    di_bc=di_bc_inner,
                    cross_diffusion_corr=cross_diffusion_corr_outer,
                )

                return R

            def nk_callback(x, f):
                if verbose:
                    print(
                        f"Picard {outer_iter:2d} | "
                        f"NK: ||R||_inf = {np.max(np.abs(f)):.6e}, "
                        f"||R||_2 = {np.linalg.norm(f):.6e}"
                    )

            try:
                T_sol = newton_krylov(
                    F=residual_wrapper,
                    xin=T,
                    f_tol=xtol,
                    maxiter=maxfev,
                    method="lgmres",
                    callback=nk_callback,
                    verbose=False,
                )
                inner_ier = 1
                inner_mesg = "Converged"
            except NoConvergence as e:
                T_sol = np.asarray(e.args[0], dtype=np.float64)
                inner_ier = 2
                inner_mesg = "newton_krylov did not converge"

            # Picard update
            T_new = omega * T_sol + (1.0 - omega) * T

            # Recompute full updated state for convergence check
            cell_k_new = np.asarray(moderator_k(T_new), dtype=np.float64).reshape(-1)

            cross_diffusion_corr_new = calculate_cross_diffusion_correction(
                T=T_new,
                cell_k=cell_k_new,
                centers=centers,
                areas=areas,
                neighbours=neighbours,
                edge_lengths=edge_lengths,
                edge_centers=edge_centers,
                edge_normals=edge_normals,
                boundary_kind=boundary_kind,
                T_HP=self.T_HP,
                T_FP=self.T_FP,
                HP_bc=self.HP_BC,
                FP_bc=self.FP_BC,
                vN_bc=self.vN_boundary_conditions,
            )

                            
            di_bc_new = calculate_Dirichlet_boundary_conditions(
                cell_k=cell_k_new,
                centers=centers,
                edge_lengths=edge_lengths,
                edge_centers=edge_centers,
                boundary_kind=boundary_kind,
                T_HP=self.T_HP,
                T_FP=self.T_FP,
                HP_bc=self.HP_BC,
                FP_bc=self.FP_BC,
                k_HP = k_HP,
                k_FP = k_FP,
            )

            final_residual = get_residuals(
                T=T_new,
                cell_k=cell_k_new,
                centers=centers,
                neighbours=neighbours,
                edge_lengths=edge_lengths,
                edge_centers=edge_centers,
                edge_normals=edge_normals,
                is_boundary=is_boundary,
                boundary_kind=boundary_kind,
                vN_bc=self.vN_boundary_conditions,
                di_bc=di_bc_new,
                cross_diffusion_corr=cross_diffusion_corr_new,
            )

            final_norm = np.linalg.norm(final_residual, ord=2)
            rel_change = np.linalg.norm(T_new - T, ord=2) / max(np.linalg.norm(T_new, ord=2), 1.0)

            if verbose:
                print(
                    f"Picard {outer_iter:2d} | inner ier = {inner_ier}, "
                    f"inner mesg = {inner_mesg}"
                )
                print(
                    f"Picard {outer_iter:2d} | rel_change = {rel_change:.6e}, "
                    f"full ||R||_2 = {final_norm:.6e}"
                )

            T = T_new

            if (rel_change < xtol) and (final_norm < rtol):
                outer_ier = 1
                outer_mesg = (
                    f"Picard iteration converged: "
                    f"rel_change={rel_change:.3e} < {xtol:.3e} and "
                    f"||R||_2={final_norm:.3e} < {rtol:.3e}"
                )
                break

        if verbose:
            print(f"Picard ier    = {outer_ier}")
            print(f"Picard mesg   = {outer_mesg}")

        return T

    def solve_nonlinear(
        self,
        T0: np.ndarray | None = None,
        xtol: float = 1e-8,
        maxfev: int = 10000,
        verbose: bool = True,
        use_linear_guess: bool = True,
    ):
        """
        Solve the nonlinear system get_residuals(T) = 0 using scipy.optimize.fsolve.
        """
        n_triangles = self.mesh.triangles.shape[0]

        # Boundary data that does not depend on T
        self.calculate_von_Neumann_boundary_conditions()

        # Assemble geometry/topology arrays once
        (
            areas,
            centers,
            neighbours,
            edge_lengths,
            edge_centers,
            edge_normals,
            is_boundary,
            boundary_kind,
        ) = self.assemble_mesh_data(
            mesh=self.mesh,
            HP_centers=self.HP_centers,
            fuel_pin_centers=self.fuel_pin_centers,
            HP_radius=self.HP_radius,
            fuel_pin_radius=self.fuel_pin_radius,
        )

        k_HP = moderator_k(self.T_HP)
        k_FP = moderator_k(self.T_FP)

        # Initial guess
        if T0 is None:
            if use_linear_guess:
                T0 = self.mod_disc_mesh_slow.solve_linearly(1)
            else:
                T0 = self.cell_T.copy()


        T0 = np.asarray(T0, dtype=float).reshape(-1)

        if T0.shape != (n_triangles,):
            raise ValueError(f"T0 must have shape ({n_triangles},), got {T0.shape}")

        call_count = {"n": 0}

        di_bc = calculate_Dirichlet_boundary_conditions(
                cell_k=cell_k,
                centers=centers,
                edge_lengths=edge_lengths,
                edge_centers=edge_centers,
                boundary_kind=boundary_kind,
                T_HP=self.T_HP,
                T_FP=self.T_FP,
                HP_bc=self.HP_BC,
                FP_bc=self.FP_BC,
                k_HP = k_HP,
                k_FP = k_FP,
            )

        def residual_wrapper(T):
            call_count["n"] += 1

            T = np.asarray(T, dtype=np.float64).reshape(-1)
            cell_k = np.asarray(moderator_k(T), dtype=np.float64).reshape(-1)

            cross_diffusion_corr = calculate_cross_diffusion_correction(
                T=T,
                cell_k=cell_k,
                centers=centers,
                areas=areas,
                neighbours=neighbours,
                edge_lengths=edge_lengths,
                edge_centers=edge_centers,
                edge_normals=edge_normals,
                boundary_kind=boundary_kind,
                T_HP=self.T_HP,
                T_FP=self.T_FP,
                HP_bc=self.HP_BC,
                FP_bc=self.FP_BC,
                vN_bc=self.vN_boundary_conditions,
            )

            # cross_diffusion_corr = np.zeros_like(cross_diffusion_corr)

            R = get_residuals(
                T=T,
                cell_k=cell_k,
                centers=centers,
                neighbours=neighbours,
                edge_lengths=edge_lengths,
                edge_centers=edge_centers,
                edge_normals=edge_normals,
                is_boundary=is_boundary,
                boundary_kind=boundary_kind,
                vN_bc=self.vN_boundary_conditions,
                di_bc=di_bc,
                cross_diffusion_corr=cross_diffusion_corr,
            )

            return R
        
        def nk_callback(x, f):
            if verbose:
                print(f"NK: ||R||_inf = {np.max(np.abs(f)):.6e}, ||R||_2 = {np.linalg.norm(f):.6e}")

        try:
            T_sol = newton_krylov(
                F=residual_wrapper,
                xin=T0,
                f_tol=xtol,
                maxiter=maxfev,
                method="lgmres",
                callback=nk_callback,
                verbose=False,   # callback already prints
            )
            ier = 1
            mesg = "Converged"
        except NoConvergence as e:
            T_sol = np.asarray(e.args[0], dtype=np.float64)
            ier = 2
            mesg = "newton_krylov did not converge"

        # Rebuild consistent final state
        self.cell_T = np.asarray(T_sol, dtype=np.float64).copy()
        self.cell_k = np.asarray(moderator_k(self.cell_T), dtype=np.float64).copy()

        self.di_boundary_conditions = calculate_Dirichlet_boundary_conditions(
            cell_k=self.cell_k,
            centers=centers,
            edge_lengths=edge_lengths,
            edge_centers=edge_centers,
            boundary_kind=boundary_kind,
            T_HP=self.T_HP,
            T_FP=self.T_FP,
            HP_bc=self.HP_BC,
            FP_bc=self.FP_BC,
            k_HP = k_HP,
            k_FP = k_FP,
        )

        self.cross_diffusion_correction = calculate_cross_diffusion_correction(
            T=self.cell_T,
            cell_k=self.cell_k,
            centers=centers,
            areas=areas,
            neighbours=neighbours,
            edge_lengths=edge_lengths,
            edge_centers=edge_centers,
            edge_normals=edge_normals,
            boundary_kind=boundary_kind,
            T_HP=self.T_HP,
            T_FP=self.T_FP,
            HP_bc=self.HP_BC,
            FP_bc=self.FP_BC,
            vN_bc=self.vN_boundary_conditions,
        )

        final_residual = get_residuals(
            T=self.cell_T,
            cell_k=self.cell_k,
            centers=centers,
            neighbours=neighbours,
            edge_lengths=edge_lengths,
            edge_centers=edge_centers,
            edge_normals=edge_normals,
            is_boundary=is_boundary,
            boundary_kind=boundary_kind,
            vN_bc=self.vN_boundary_conditions,
            di_bc=self.di_boundary_conditions,
            cross_diffusion_corr=self.cross_diffusion_correction,
        )

        final_norm = np.linalg.norm(final_residual, ord=2)

        if verbose:
            print(f"fsolve ier    = {ier}")
            print(f"fsolve mesg   = {mesg}")
            print(f"Final ||R||_2 = {final_norm:.6e}")

        return self.cell_T
    

    def assemble_mesh_data(
        self,
        mesh,
        HP_centers,
        fuel_pin_centers,
        HP_radius,
        fuel_pin_radius,
        epsilon=1e-4,
    ):
        """
        Assemble mesh/topology/geometry arrays for fast numerical kernels.

        Returns
        -------
        centers : ndarray, shape (n_triangles, 2)
        areas : ndarray, shape (n_triangles)
        neighbours : ndarray, shape (n_triangles, 3)
            Neighbour triangle index, or -1 for boundary face.
        edge_lengths : ndarray, shape (n_triangles, 3)
        edge_centers : ndarray, shape (n_triangles, 3, 2)
        surface_normals : ndarray, shape (n_triangles, 3, 2)
            Outward unit normals for each triangle face.
        is_boundary : ndarray, shape (n_triangles, 3)
            True where neighbour == -1.
        boundary_kind : ndarray, shape (n_triangles, 3)
            0 = internal
            1 = HP boundary
            2 = fuel pin boundary
            3 = other boundary
        """

        n_triangles = mesh.n_triangles
        n_edges = 3  # triangle mesh

        centers = np.zeros((n_triangles, 2), dtype=np.float64)
        areas = np.zeros(n_triangles, dtype=np.float64)
        neighbours = np.full((n_triangles, n_edges), -1, dtype=np.int64)
        edge_lengths = np.zeros((n_triangles, n_edges), dtype=np.float64)
        edge_centers = np.zeros((n_triangles, n_edges, 2), dtype=np.float64)
        edge_normals = np.zeros((n_triangles, n_edges, 2), dtype=np.float64)
        is_boundary = np.zeros((n_triangles, n_edges), dtype=np.bool_)
        boundary_kind = np.zeros((n_triangles, n_edges), dtype=np.int64)

        for triangle_idx in range(n_triangles):
            centers[triangle_idx] = mesh._centers[triangle_idx]
            areas[triangle_idx] = mesh.triangle_area(triangle_idx)
 
            tri_center = mesh._centers[triangle_idx]
            tri_edges = mesh._triangle_to_edges[triangle_idx]

            for edge_idx, edge in enumerate(tri_edges):
                p0 = mesh.points[edge[0]]
                p1 = mesh.points[edge[1]]

                edge_center = 0.5 * (p0 + p1)
                edge_vec = p1 - p0
                edge_length = np.linalg.norm(edge_vec)

                edge_centers[triangle_idx, edge_idx] = edge_center
                edge_lengths[triangle_idx, edge_idx] = edge_length

                # --- outward unit normal ---
                n1 = np.array([edge_vec[1], -edge_vec[0]], dtype=np.float64)
                n2 = -n1

                to_edge = edge_center - tri_center
                n = n1 if np.dot(n1, to_edge) > 0.0 else n2

                norm_n = np.linalg.norm(n)
                if norm_n == 0.0:
                    raise ValueError(f"Degenerate edge found in triangle {triangle_idx}")

                edge_normals[triangle_idx, edge_idx] = n / norm_n

                # --- topology ---
                attached = self.mesh._edge_to_triangles[edge]
                if len(attached) == 2:
                    neighbour = attached[0] if attached[1] == triangle_idx else attached[1]
                    neighbours[triangle_idx, edge_idx] = neighbour
                    is_boundary[triangle_idx, edge_idx] = False
                    boundary_kind[triangle_idx, edge_idx] = 0
                elif len(attached) == 1:
                    neighbours[triangle_idx, edge_idx] = -1
                    is_boundary[triangle_idx, edge_idx] = True

                    if self.is_edge_on_circumference(edge, HP_centers, HP_radius):
                        boundary_kind[triangle_idx, edge_idx] = 1
                    elif self.is_edge_on_circumference(edge, fuel_pin_centers, fuel_pin_radius):
                        boundary_kind[triangle_idx, edge_idx] = 2
                    else:
                        boundary_kind[triangle_idx, edge_idx] = 3
                else:
                    raise ValueError(
                        f"Edge {edge} is attached to {len(attached)} triangles. "
                        f"Expected 1 or 2."
                    )

        return (
            areas,
            centers,
            neighbours,
            edge_lengths,
            edge_centers,
            edge_normals,
            is_boundary,
            boundary_kind,
        )
    

    def calculate_von_Neumann_boundary_conditions(self):

        total_length_HP = 0
        total_length_fuel_pin = 0
        HP_triangle_indices = []
        fuel_pin_triangle_indices = []

        self.vN_boundary_conditions = np.zeros(self.mesh.triangles.shape[0])

        # Iterating over all the boundary edges to see which are in contact with HP and fuel pins. 
        for boundary_edge in self.mesh.get_boundary_edges():
            triangle_idx = self.mesh._edge_to_triangles[boundary_edge]

            if ((self.is_edge_on_circumference(boundary_edge, self.HP_centers, self.HP_radius)) and
                (self.HP_BC == "vonNeumann")):

                HP_triangle_indices.append(triangle_idx)

                edge_length = np.linalg.norm(
                    self.mesh.points[boundary_edge[0]] - self.mesh.points[boundary_edge[1]])

                # Positive due to "q dot S > 0" since both S and q points outward. 
                self.vN_boundary_conditions[triangle_idx] += self.Q_in * edge_length

                total_length_HP += edge_length
            
            if ((self.is_edge_on_circumference(boundary_edge, self.fuel_pin_centers, self.fuel_pin_radius)) and
                (self.FP_BC == "vonNeumann")):

                fuel_pin_triangle_indices.append(triangle_idx)

                edge_length = np.linalg.norm(
                    self.mesh.points[boundary_edge[0]] - self.mesh.points[boundary_edge[1]])

                # Negative due to "q dot S < 0" since both S points outwards and q point inward.
                self.vN_boundary_conditions[triangle_idx] += -self.Q_in * edge_length

                total_length_fuel_pin += edge_length
                
        # Using the found triangle indecies and totalt length exposed to HPs, and fuel pins, to 
        # normalize the flux BC.        
        for triangle_idx in HP_triangle_indices:
            # Positive due to "q dot S > 0" since both S and q points outward. 
            self.vN_boundary_conditions[triangle_idx] *= 1 / total_length_HP  #Need to scale Q_in with edge length.
                
        for triangle_idx in fuel_pin_triangle_indices:
            # Negative due to "q dot S < 0" since both S points outwards and q point inward.
            self.vN_boundary_conditions[triangle_idx] *= 1 / total_length_fuel_pin

    
    def is_edge_on_circumference(self, edge, centers, radius):
        p1 = self.mesh.points[edge[0]]
        p2 = self.mesh.points[edge[1]]
        epsilon = 1e-2

        for center in centers:
            if (
                abs(np.linalg.norm(p1 - center) - radius) < epsilon
                and abs(np.linalg.norm(p2 - center) - radius) < epsilon
            ):
                return True
            
        return False


@njit(parallel=True)
def get_residuals(
        T,
        cell_k,
        centers,
        neighbours,
        edge_lengths,
        edge_centers,
        edge_normals,
        is_boundary,
        boundary_kind,
        vN_bc,
        di_bc,
        cross_diffusion_corr
):
    n_triangles = centers.shape[0]

    res = np.zeros(n_triangles, dtype=float)

    for triangle_idx in prange(n_triangles):
        triangle_center = centers[triangle_idx]
        T_C = T[triangle_idx]

        # iterating over all triangle sides
        for edge_idx in range(neighbours.shape[1]):
            neighbour_idx = neighbours[triangle_idx][edge_idx]

            if neighbour_idx == -1:
                # Boundary face
                FluxV_f_BC_vN = vN_bc[triangle_idx]
                FluxC_f_BC_Di = di_bc[triangle_idx][0]
                FluxV_f_BC_Di = di_bc[triangle_idx][1]

                # Residual = M(T) @ T - C(T)
                res[triangle_idx] += FluxC_f_BC_Di * T_C
                res[triangle_idx] += FluxV_f_BC_vN + FluxV_f_BC_Di

            else:
                # Internal face
                neighbour_center = centers[neighbour_idx]
                edge_center = edge_centers[triangle_idx][edge_idx]
                edge_length = edge_lengths[triangle_idx][edge_idx]

                d_CF = np.linalg.norm(triangle_center - neighbour_center)
                d_Cs = np.linalg.norm(triangle_center - edge_center)
                d_sF = np.linalg.norm(edge_center - neighbour_center)

                gDiff_f = edge_length / d_CF
                g_f = d_Cs / (d_Cs + d_sF)

                k_C = cell_k[triangle_idx]
                k_F = cell_k[neighbour_idx]

                k_s = k_C * k_F / ((1.0 - g_f) * k_C + g_f * k_F)

                FluxF_f = -k_s * gDiff_f
                FluxC_f =  k_s * gDiff_f

                T_F = T[neighbour_idx]
                res[triangle_idx] += FluxC_f * T_C + FluxF_f * T_F

        # Cross-diffusion correction
        FluxV_f_CD = cross_diffusion_corr[triangle_idx]
        res[triangle_idx] += FluxV_f_CD

    return res


@njit(parallel=True)
def calculate_Dirichlet_boundary_conditions(
        cell_k,
        centers,
        edge_lengths,
        edge_centers,
        boundary_kind,
        T_HP,
        T_FP,
        HP_bc,
        FP_bc,
        k_HP,
        k_FP
    ):
    n_triangles = centers.shape[0]

    # First term in the matrix represents FluxC_b, the second represents FluxV_b.
    di_bc = np.zeros((n_triangles, 2))

    # Iterating over all the boundary edges to see which are in contact with HP and fuel pins. 
    for triangle_idx in prange(n_triangles):
        triangle_center = centers[triangle_idx]

        for edge_idx in range(boundary_kind.shape[1]):
            if boundary_kind[triangle_idx][edge_idx] == 1 and HP_bc == "Dirichlet":

                edge_length = edge_lengths[triangle_idx][edge_idx]
                edge_center = edge_centers[triangle_idx][edge_idx]
                
                d_Cb = np.linalg.norm(triangle_center - edge_center)

                Diff_b = edge_length / d_Cb
                k_boundary = 2 * cell_k[triangle_idx] * k_HP / (k_HP + cell_k[triangle_idx])
                k_boundary = cell_k[triangle_idx]
                Flux_C_b = Diff_b * k_boundary

                # Positive due to "q dot S > 0" since both S and q points outward. 
                di_bc[triangle_idx][0] += Flux_C_b
                di_bc[triangle_idx][1] += -Flux_C_b * T_HP

            if boundary_kind[triangle_idx][edge_idx] == 2 and FP_bc == "Dirichlet":
                
                edge_length = edge_lengths[triangle_idx][edge_idx]
                edge_center = edge_centers[triangle_idx][edge_idx]
                
                d_Cb = np.linalg.norm(triangle_center - edge_center)

                Diff_b = edge_length / d_Cb
                k_boundary = 2 * cell_k[triangle_idx] * k_FP / (k_FP + cell_k[triangle_idx])
                Flux_C_b = Diff_b * k_boundary

                # Positive due to "q dot S > 0" since both S and q points outward. 
                di_bc[triangle_idx][0] += Flux_C_b
                di_bc[triangle_idx][1] += -Flux_C_b * T_FP

    return di_bc
    

@njit(parallel=True)
def calculate_cross_diffusion_correction(
        T,
        cell_k,
        centers,
        areas,
        neighbours,
        edge_lengths,
        edge_centers,
        edge_normals,
        boundary_kind,
        T_HP,
        T_FP,
        HP_bc,
        FP_bc,
        vN_bc
):  
    n_triangles = centers.shape[0]
    cell_gradients = np.zeros((n_triangles, 2))

    # Compute each cell gradient once
    for triangle_idx in prange(n_triangles):
        cell_gradients[triangle_idx] = calculate_cell_flux_gradient(                
            triangle_idx,
            T,
            cell_k,
            centers,
            areas,
            neighbours,
            edge_lengths,
            edge_centers,
            edge_normals,
            boundary_kind,
            T_HP,
            T_FP,
            HP_bc,
            FP_bc,
            vN_bc
            )

    # Assemble cross-diffusion correction
    cross_diffusion_correction = np.zeros(n_triangles)
    for triangle_idx in prange(n_triangles):
        C_center = centers[triangle_idx]

        for edge_idx in range(neighbours.shape[1]):
            neighbour_idx = neighbours[triangle_idx][edge_idx]

            if neighbour_idx == -1:
                continue

            grad_flux_C = cell_gradients[triangle_idx]
            grad_flux_F = cell_gradients[neighbour_idx]

            F_center = centers[neighbour_idx]

            edge_length = edge_lengths[triangle_idx][edge_idx]
            edge_center = edge_centers[triangle_idx][edge_idx]

            d_CF = np.linalg.norm(C_center - F_center)
            d_Cs = np.linalg.norm(C_center - edge_center)
            d_sF = np.linalg.norm(edge_center - F_center)

            g_F = d_Cs / (d_Cs + d_sF)
            g_C = 1.0 - g_F

            grad_flux_surface = g_C * grad_flux_C + g_F * grad_flux_F

            k_C = cell_k[triangle_idx]
            k_F = cell_k[neighbour_idx]
            k_surface = k_C * k_F / (g_C * k_C + g_F * k_F)

            n_vec = edge_normals[triangle_idx][edge_idx]
            e = (F_center - C_center) / d_CF
            T_vector_surface = (n_vec - e) * edge_length

            cross_diffusion_correction[triangle_idx] -= (
                k_surface * np.dot(grad_flux_surface, T_vector_surface)
            )
    
    return cross_diffusion_correction 
    

@njit(parallel=True)
def calculate_cell_flux_gradient(
        triangle_idx,
        T,
        cell_k,
        centers,
        areas,
        neighbours,
        edge_lengths,
        edge_centers,
        edge_normals,
        boundary_kind,
        T_HP,
        T_FP,
        HP_bc,
        FP_bc,
        vN_bc
    ):
    # Utilizing the Green-Gauss gradient theorem.

    V_C = areas[triangle_idx]
    C_center = centers[triangle_idx]

    flux_gradient_C = np.array([0., 0.])
    for edge_idx in range(neighbours.shape[1]):
        neighbour_idx = neighbours[triangle_idx][edge_idx]

        edge_length = edge_lengths[triangle_idx][edge_idx]

        if neighbour_idx == -1:
            if boundary_kind[triangle_idx][edge_idx] == 1 and HP_bc == "Dirichlet":
                T_surface = T_HP
            elif boundary_kind[triangle_idx][edge_idx] == 2 and FP_bc == "Dirichlet":
                T_surface = T_FP
            else:
                # If the neighbouring triangle is an boundary then apply note 3., eq. (8.42) on page 222 in F.Moukalled et al.
                edge_center = edge_centers[triangle_idx][edge_idx]

                q_surface = vN_bc[triangle_idx] / edge_length
                
                # Not multiplying gDiff with S due to q_surface is equal to "S * q_b" 
                T_C = T[triangle_idx]
                d_Cf = np.linalg.norm(edge_center - C_center)
                k_C = cell_k[triangle_idx]

                T_surface = T_C - q_surface * d_Cf / k_C
                
            n = edge_normals[triangle_idx][edge_idx]
            S_surface = edge_length * n

            flux_gradient_C += S_surface * T_surface

        else:
            F_center = centers[neighbour_idx]

            edge_length = edge_lengths[triangle_idx][edge_idx]
            edge_center = edge_centers[triangle_idx][edge_idx]

            d_Cs = np.linalg.norm(C_center - edge_center)
            d_sF = np.linalg.norm(edge_center - F_center)

            g_F = d_Cs / (d_Cs + d_sF)
            g_C = 1. - g_F

            T_surface = g_C * T[triangle_idx] + g_F * T[neighbour_idx]

            n = edge_normals[triangle_idx][edge_idx]
            S_surface = edge_length * n

            flux_gradient_C += S_surface * T_surface

    return flux_gradient_C / V_C

    
def plot_mesh_with_temp_profile(points, triangles, T):
    # Convert to 3D
    points_3d = np.column_stack([points, np.zeros(len(points))])

    # PyVista cell format: [3, n1, n2, n3, 3, n1, n2, n3, ...]
    cells = np.hstack([
        np.hstack([[3, *tri] for tri in triangles])
    ])

    # Cell types: 5 = triangle
    cell_types = np.full(len(triangles), pv.CellType.TRIANGLE)

    mesh_visualization = pv.UnstructuredGrid(cells, cell_types, points_3d)
    mesh_visualization.cell_data["u"] = T

    mesh_visualization.plot(
        scalars="u",
        show_edges=True,
        cmap="viridis"
    )


def calculate_effective_thermal_resistance(mesh, T):
    """
    Compute effective thermal resistance based on boundary temperatures on
    HP and fuel-pin boundaries.

    For Dirichlet boundaries, the boundary temperature is taken directly from
    the prescribed boundary value.

    For von Neumann boundaries, the boundary temperature is reconstructed from
    the adjacent cell-center temperature using
        T_surface = T_C - q_surface * d_Cf / k_C

    Parameters
    ----------
    mesh : object
        Solved moderator/result object with attributes:
        - mesh                    (underlying UnstructuredMesh)
        - HP_centers, fuel_pin_centers
        - HP_radius, fuel_pin_radius
        - HP_BC, FP_BC
        - T_HP, T_FP
        - vN_boundary_conditions
        - Q_in
        - is_edge_on_circumference(...)

    T : array-like
        Cell temperatures, shape (n_triangles,)

    Returns
    -------
    R_eff : float
        Effective thermal resistance [K/W]
    T_HP_avg : float
        Length-weighted average HP boundary temperature
    T_FP_avg : float
        Length-weighted average fuel-pin boundary temperature
    HP_boundary_temps : np.ndarray
        Reconstructed/prescribed HP boundary temperatures per boundary edge
    FP_boundary_temps : np.ndarray
        Reconstructed/prescribed fuel-pin boundary temperatures per boundary edge
    """
    geom = mesh.mesh

    T = np.asarray(T, dtype=np.float64).reshape(-1)
    n_triangles = geom.triangles.shape[0]

    if T.shape != (n_triangles,):
        raise ValueError(f"T must have shape ({n_triangles},), got {T.shape}")

    cell_k = np.asarray(moderator_k(T), dtype=np.float64).reshape(-1)

    HP_boundary_temps = []
    FP_boundary_temps = []
    HP_boundary_lengths = []
    FP_boundary_lengths = []

    for boundary_edge in geom.get_boundary_edges():
        triangle_idx = geom._edge_to_triangles[boundary_edge]

        # Boundary edges should belong to exactly one triangle
        if isinstance(triangle_idx, (list, tuple, np.ndarray)):
            triangle_idx = int(np.asarray(triangle_idx).reshape(-1)[0])

        p0 = geom.points[boundary_edge[0]]
        p1 = geom.points[boundary_edge[1]]

        edge_length = np.linalg.norm(p0 - p1)
        edge_center = 0.5 * (p0 + p1)

        T_C = T[triangle_idx]
        k_C = cell_k[triangle_idx]
        d_Cf = np.linalg.norm(edge_center - geom._centers[triangle_idx])

        # --------------------------------------------------------------
        # HP boundary
        # --------------------------------------------------------------
        if mesh.is_edge_on_circumference(boundary_edge, mesh.HP_centers, mesh.HP_radius):
            if mesh.HP_BC == "Dirichlet":
                T_surface = mesh.T_HP

            elif mesh.HP_BC == "vonNeumann":
                q_surface = mesh.vN_boundary_conditions[triangle_idx] / edge_length
                T_surface = T_C - q_surface * d_Cf / k_C

            else:
                raise ValueError(f"Unknown HP_BC: {mesh.HP_BC}")

            HP_boundary_temps.append(T_surface)
            HP_boundary_lengths.append(edge_length)

        # --------------------------------------------------------------
        # Fuel-pin boundary
        # --------------------------------------------------------------
        elif mesh.is_edge_on_circumference(
            boundary_edge, mesh.fuel_pin_centers, mesh.fuel_pin_radius
        ):
            if mesh.FP_BC == "Dirichlet":
                T_surface = mesh.T_FP

            elif mesh.FP_BC == "vonNeumann":
                q_surface = mesh.vN_boundary_conditions[triangle_idx] / edge_length
                T_surface = T_C - q_surface * d_Cf / k_C

            else:
                raise ValueError(f"Unknown FP_BC: {mesh.FP_BC}")

            FP_boundary_temps.append(T_surface)
            FP_boundary_lengths.append(edge_length)

    HP_boundary_temps = np.asarray(HP_boundary_temps, dtype=np.float64)
    FP_boundary_temps = np.asarray(FP_boundary_temps, dtype=np.float64)
    HP_boundary_lengths = np.asarray(HP_boundary_lengths, dtype=np.float64)
    FP_boundary_lengths = np.asarray(FP_boundary_lengths, dtype=np.float64)

    if HP_boundary_temps.size == 0:
        raise ValueError("No HP boundary edges were found.")
    if FP_boundary_temps.size == 0:
        raise ValueError("No fuel-pin boundary edges were found.")
    if mesh.Q_in == 0:
        raise ValueError("Q_in must be non-zero to compute effective thermal resistance.")

    # Length-weighted averages are more appropriate than plain means
    T_HP_avg = np.average(HP_boundary_temps, weights=HP_boundary_lengths)
    T_FP_avg = np.average(FP_boundary_temps, weights=FP_boundary_lengths)

    # Divide by 2 to get Q_in per fuel pin.
    R_eff = abs(T_FP_avg - T_HP_avg) / (abs(mesh.Q_in) / 2)

    return R_eff, T_HP_avg, T_FP_avg, HP_boundary_temps, FP_boundary_temps


def compute_moderator_resistance_vs_hp_temperature(mod_mesh_di, T_HPs):
    T_mod_avg = np.zeros_like(T_HPs)
    R_eff = np.zeros_like(T_HPs)

    (
        areas,
        centers,
        neighbours,
        edge_lengths,
        edge_centers,
        edge_normals,
        is_boundary,
        boundary_kind,
    ) = mod_mesh_di.assemble_mesh_data(
        mesh=mod_mesh_di.mesh,
        HP_centers=mod_mesh_di.HP_centers,
        fuel_pin_centers=mod_mesh_di.fuel_pin_centers,
        HP_radius=mod_mesh_di.HP_radius,
        fuel_pin_radius=mod_mesh_di.fuel_pin_radius,
    )

    for i in range(T_mod_avg.shape[0]):
        mod_mesh_di.T_HP = T_HPs[i]

        T = mod_mesh_di.solve_nonlinear_picard(
            xtol=1e-10,
            maxfev=10000,
            verbose=True,
            use_linear_guess=True,
            omega=0.5,
            max_outer_iter=200)

        T_mod_avg[i] = np.sum(T * areas) / np.sum(areas)

        R_eff[i], _, _, _, _  = calculate_effective_thermal_resistance(mod_mesh_di, T)

    return R_eff, T_mod_avg


def plot_temperature_profiles(T, areas, R_eff, r_HP, r_FP, Q_in):
    """
    Plot cylindrical and linear temperature profiles in the same figure,
    including horizontal lines showing their average temperatures.

    Parameters
    ----------
    result : object
        Solved mesh/result object with attributes:
        - HP_radius
        - fuel_pin_radius
        - Q_in
        - cell_T
        - mesh.triangle_area(i)
        - mesh.triangles
    R_eff : float
        Effective thermal resistance [K/W]
    r_i : float
        Inner radius / first surface location
    r_o : float
        Outer radius / second surface location
    n_points : int
        Number of points used for plotting
    """
    circumference_HP = r_HP * 7.0 / 12.0
    circumference_FP = r_FP * 2.0

    r_i = circumference_HP / 2.0
    r_o = circumference_FP / 2.0

    T = T - np.min(T)

    if R_eff <= 0:
        raise ValueError("R_eff must be positive.")
    if Q_in < 0:
        raise ValueError("Q_in must be non-negative.")
    if r_i <= 0 or r_o <= 0:
        raise ValueError("r_i and r_o must be positive.")
    if r_o <= r_i:
        raise ValueError("r_o must be greater than r_i.")

    delta_T = Q_in * R_eff
    r = np.linspace(r_i, r_o, 500)

    T_cyl = delta_T * np.log(r / r_i) / np.log(r_o / r_i)
    T_lin = delta_T * (r - r_i) / (r_o - r_i)

    numerator_cyl = np.trapezoid(T_cyl * r, r)
    denominator_cyl = np.trapezoid(r, r)
    T_cyl_avg = numerator_cyl / denominator_cyl

    T_lin_avg = np.trapezoid(T_lin, r) / (r_o - r_i)

    areas = np.array(
        [areas[i] for i in range(len(T))],
        dtype=float,
    )

    if len(areas) != len(T):
        raise ValueError("T must match number of mesh triangles.")

    T_mesh_avg = np.sum(T * areas) / np.sum(areas)

    sns.set_theme(style="whitegrid", context="talk")
    plt.figure(figsize=(9, 5.5))
    palette = sns.color_palette("deep")

    sns.lineplot(
        x=r,
        y=T_cyl,
        linewidth=2.8,
        color=palette[0],
        label="Cylindrical profile",
    )

    sns.lineplot(
        x=r,
        y=T_lin,
        linewidth=2.8,
        linestyle="--",
        color=palette[1],
        label="Linear profile",
    )

    plt.axhline(
        T_cyl_avg,
        color=palette[0],
        linestyle=":",
        linewidth=2.2,
        label=f"Cylindrical average = {T_cyl_avg:.2f} K",
    )

    plt.axhline(
        T_lin_avg,
        color=palette[1],
        linestyle=":",
        linewidth=2.2,
        label=f"Linear average = {T_lin_avg:.2f} K",
    )

    plt.axhline(
        T_mesh_avg,
        color="black",
        linestyle="-.",
        linewidth=2.5,
        label=f"Mesh vol avg = {T_mesh_avg:.2f} K",
    )

    plt.scatter(
        [r_i, r_o],
        [0.0, delta_T],
        s=60,
        color="black",
        zorder=5,
        label="Boundary temperatures",
    )

    plt.xlabel("Radial position / coordinate", fontsize=13)
    plt.ylabel("Temperature [K]", fontsize=13)
    plt.title(
        f"Temperature profiles from effective thermal resistance\n"
        f"$R_{{eff}}$ = {R_eff:.3g} K/(m W), $Q_{{in}}$ = {Q_in/2.:.3g} W/m / per FP, "
        f"$\\Delta T$ = {delta_T:.3g} K",
        fontsize=15,
    )

    plt.legend(frameon=True, fontsize=11)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    mesh = meshio.read("models/moderator/hex_mesh.msh")

    points = mesh.points[:, :2]                
    triangles = mesh.cells_dict["triangle"]     
    mesh = UnstructuredMesh(points, triangles)

    print(f"Number of elements: {triangles.shape}, Number of points: {points.shape}")

    cell_k = np.ones(triangles.shape[0])
    cell_T = np.ones(triangles.shape[0])
    l_pitch = 0.0286
    r_HP = 0.0065
    r_f = 0.008
    l_f = 1.8
    theta_hex = (np.pi / 6)

    data_di = {
        "cell_k": 20 * np.ones(triangles.shape[0]),
        "cell_T": np.ones(triangles.shape[0]),

        "Q_in": 15e6 / 2970 / l_f * 2,

        "HP_centers":       [[0., 0.], [np.tan(theta_hex) * 2. * l_pitch, 2. * l_pitch]],
        "fuel_pin_centers": [[0, l_pitch * 3./2.], [0, l_pitch * 5./2.], [np.tan(theta_hex) * 2 * l_pitch, l_pitch * 7./2.]],
        "HP_radius": r_HP,
        "fuel_pin_radius": r_f,

        "T_HP": 850,
        "T_FP": 885,

        "HP_BC": "Dirichlet",   #"Dirichlet"
        "FP_BC": "vonNeumann"   #"vonNeumann"
    }

    data_vn = {
        "cell_k": 20 * np.ones(triangles.shape[0]),
        "cell_T": np.ones(triangles.shape[0]),

        "Q_in": 15e6 / 2970 / l_f * 2,

        "HP_centers":       [[0., 0.], [np.tan(theta_hex) * 2. * l_pitch, 2. * l_pitch]],
        "fuel_pin_centers": [[0, l_pitch * 3./2.], [0, l_pitch * 5./2.], [np.tan(theta_hex) * 2 * l_pitch, l_pitch * 7./2.]],
        "HP_radius": r_HP,
        "fuel_pin_radius": r_f,

        "T_HP": 850,
        "T_FP": 885,

        "HP_BC": "vonNeumann",   #"Dirichlet"
        "FP_BC": "vonNeumann"   #"vonNeumann"
    }

    mod_mesh_di = ModeratorDiscretisedMeshFast(mesh=mesh, data=data_di)
    mod_mesh_vn = ModeratorDiscretisedMeshFast(mesh=mesh, data=data_vn)

    T_HP = np.linspace(650, 1350, 8)
    print(compute_moderator_resistance_vs_hp_temperature(mod_mesh_di, T_HPs=T_HP))

    # (
    # areas,
    # centers,
    # neighbours,
    # edge_lengths,
    # edge_centers,
    # edge_normals,
    # is_boundary,
    # boundary_kind,
    # ) = mod_mesh_di.assemble_mesh_data(
    # mesh=mod_mesh_di.mesh,
    # HP_centers=mod_mesh_di.HP_centers,
    # fuel_pin_centers=mod_mesh_di.fuel_pin_centers,
    # HP_radius=mod_mesh_di.HP_radius,
    # fuel_pin_radius=mod_mesh_di.fuel_pin_radius,
    # )

    # T_NKp_di = mod_mesh_di.solve_nonlinear_picard(
    #     xtol=1e-10,
    #     maxfev=10000,
    #     verbose=True,
    #     use_linear_guess=True,
    #     omega=0.5,
    #     max_outer_iter=200)
    
    # T_NKp_di -= np.min(T_NKp_di) - 850

    # plot_mesh_with_temp_profile(mod_mesh_di.mesh.points, mod_mesh_di.mesh.triangles, T_NKp_di)
    # R_eff_di, T_HP_avg_di, T_FP_avg_di, HP_boundary_temps_di, FP_boundary_temps_di = calculate_effective_thermal_resistance(mod_mesh_di, T_NKp_di)
    # plot_temperature_profiles(T_NKp_di, areas, R_eff_di, r_HP, r_f, data_di["Q_in"])
