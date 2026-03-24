from unittest import skip

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import meshio # type: ignore
import pyvista as pv # type: ignore

from meshHeatConduction.triangle_mesh import UnstructuredMesh, Surface

class moderator_discretised_mesh:
    def __init__(self, data, mesh):
        self.mesh = mesh
        n_triangles = mesh.triangles.shape[0]

        self.Q_in = data.get("Q_in")

        self.HP_centers       = data.get("HP_centers")
        self.fuel_pin_centers = data.get("fuel_pin_centers")

        self.HP_radius       = data.get("HP_radius")
        self.fuel_pin_radius = data.get("fuel_pin_radius")

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
        
    def solve(self, iterations: int):
        
        self.calculate_von_Neumann_boundary_conditions()

        # Iterations to allow for cross diffusion correction to converge.
        for i in range(iterations):
            n_triangles = mesh.triangles.shape[0]

            M = np.zeros((n_triangles, n_triangles))
            C = np.zeros(n_triangles)

            self.calculate_cross_diffusion_correction()

            for triangle_idx in range(n_triangles):
                triangle_center = self.mesh.get_center_point(triangle_idx)
                neighbours, surfaces = self.mesh.get_faces_and_neighbours(triangle_idx)

                # Setting the FluxF and FluxC terms, in addition to the boundary FluxV terms (BC). 
                for neighbour_idx, surface in zip(neighbours, surfaces):    
                    if neighbour_idx == -1: # Indicating a boundary triangle. 
                        FluxV_f_BC = self.vN_boundary_conditions[triangle_idx]

                        C[triangle_idx] += -FluxV_f_BC
                    
                    else:
                        neighbour_center = self.mesh.get_center_point(neighbour_idx)
                        surface_center = surface.center

                        d_CF = np.linalg.norm(triangle_center - neighbour_center)
                        d_Cs = np.linalg.norm(triangle_center - surface_center)
                        d_sF = np.linalg.norm(surface_center - neighbour_center)

                        # Utilizing orthogonal correction approach.
                        gDiff_f = surface.length / d_CF

                        g_f = d_Cs / (d_Cs + d_sF)
                        k_C = self.cell_k[triangle_idx]
                        k_F = self.cell_k[neighbour_idx]
                        
                        k_s = k_C * k_F / ( (1 - g_f) * k_C + g_f * k_F )

                        FluxF_f = -k_s * gDiff_f
                        FluxC_f =  k_s * gDiff_f

                        M[triangle_idx][neighbour_idx] =  FluxF_f
                        M[triangle_idx][triangle_idx ] += FluxC_f
                
                # Setting the FluxV terms that corresponds to the cross diffusion (CD) correction.
                FluxV_f_CD = self.cross_diffusion_correction[triangle_idx]
                C[triangle_idx] += -FluxV_f_CD

            T = np.linalg.solve(M, C)
            self.cell_T = T
            print(T - np.min(T))

        return T


    def calculate_von_Neumann_boundary_conditions(self):

        total_length_HP = 0
        total_length_fuel_pin = 0
        HP_triangle_indices = []
        fuel_pin_triangle_indices = []

        # Iterating over all the boundary edges to see which are in contact with HP and fuel pins. 
        self.vN_boundary_conditions = np.zeros(self.mesh.triangles.shape[0])
        for boundary_edge in self.mesh.get_boundary_edges():
            triangle_idx = self.mesh._edge_to_triangles[boundary_edge]

            if self.is_edge_on_circumference(boundary_edge, self.HP_centers, self.HP_radius):
                HP_triangle_indices.append(triangle_idx)

                edge_length = np.linalg.norm(
                    self.mesh.points[boundary_edge[0]] - self.mesh.points[boundary_edge[1]])

                # Positive due to "q dot S > 0" since both S and q points outward. 
                self.vN_boundary_conditions[triangle_idx] = self.Q_in * edge_length

                total_length_HP += edge_length
            
            elif self.is_edge_on_circumference(boundary_edge, self.fuel_pin_centers, self.fuel_pin_radius):
                fuel_pin_triangle_indices.append(triangle_idx)

                edge_length = np.linalg.norm(
                    self.mesh.points[boundary_edge[0]] - self.mesh.points[boundary_edge[1]])

                # Negative due to "q dot S < 0" since both S points outwards and q point inward.
                self.vN_boundary_conditions[triangle_idx] = -self.Q_in * edge_length

                total_length_fuel_pin += edge_length
                
        # Using the found triangle indecies and totalt length exposed to HPs, and fuel pins, to 
        # normalize the flux BC.        
        for triangle_idx in HP_triangle_indices:
            # Positive due to "q dot S > 0" since both S and q points outward. 
            self.vN_boundary_conditions[triangle_idx] *= 1 / total_length_HP  #Need to scale Q_in with edge length.
                
        for triangle_idx in fuel_pin_triangle_indices:
            # Negative due to "q dot S < 0" since both S points outwards and q point inward.
            self.vN_boundary_conditions[triangle_idx] *= 1 / total_length_fuel_pin


    def calculate_cross_diffusion_correction(self):

        def calculate_cell_flux_gradient(triangle_idx: int):
            # Utilizing the Green-Gauss gradient theorem.

            V_C = self.mesh.triangle_area(triangle_idx)
            C_center = self.mesh._centers[triangle_idx]
            neighbours_C, surfaces_C = self.mesh.get_faces_and_neighbours(triangle_idx)

            flux_gradient_C = np.array([0., 0.])
            for neighbour_idx, surface in zip(neighbours_C, surfaces_C):

                if neighbour_idx == -1:
                    # If the neighbouring triangle is an boundary then apply note 3., eq. (8.42) on page 222 in F.Moukalled et al.
                    T_C = self.cell_T[triangle_idx]
                    q_surface = self.vN_boundary_conditions[triangle_idx] / surface.length
                    k_C = self.cell_k[triangle_idx]
                    
                    # Not multiplying gDiff with S due to q_surface is equal to "S * q_b" 
                    d_Cf = np.linalg.norm(surface.center - C_center)

                    T_surface = T_C - q_surface * d_Cf / k_C
                    
                    n = self.mesh.get_surface_normal(triangle_idx, surface)
                    S_surface = surface.length * n

                    flux_gradient_C += S_surface * T_surface

                else:
                    F_center = self.mesh._centers[neighbour_idx]

                    d_Cs = np.linalg.norm(C_center - surface.center)
                    d_sF = np.linalg.norm(surface.center - F_center)

                    g_F = d_Cs / (d_Cs + d_sF)
                    g_C = 1. - g_F

                    T_surface = g_C * self.cell_T[triangle_idx] + g_F * self.cell_T[neighbour_idx]

                    n = self.mesh.get_surface_normal(triangle_idx, surface)
                    S_surface = surface.length * n

                    flux_gradient_C += S_surface * T_surface

            return flux_gradient_C / V_C
        

        self.cross_diffusion_correction = np.zeros(self.mesh.triangles.shape[0])
        
        # Iterating over all triangles.
        for triangle_idx in range(self.mesh.triangles.shape[0]):
            
            neighbours_C, surfaces_C = self.mesh.get_faces_and_neighbours(triangle_idx)
            grad_flux_C = calculate_cell_flux_gradient(triangle_idx)
            C_center = self.mesh._centers[triangle_idx]

            # iterating over all surfaces in all triangles. Reduntant, since most surfaces are calculated twice, but simple. 
            for neighbour_idx, surface in zip(neighbours_C, surfaces_C):
                
                # Assuming that the grad_flux is parallel with the surface and the iteration can be skipped.
                if neighbour_idx == -1:
                    continue

                grad_flux_F = calculate_cell_flux_gradient(neighbour_idx)

                F_center = self.mesh._centers[neighbour_idx]

                d_CF = np.linalg.norm(C_center - F_center)
                d_Cs = np.linalg.norm(C_center - surface.center)
                d_sF = np.linalg.norm(surface.center - F_center)

                g_F = d_Cs / (d_Cs + d_sF)
                g_C = 1. - g_F

                grad_flux_surface = g_C * grad_flux_C + g_F * grad_flux_F
                k_C = self.cell_k[triangle_idx]
                k_F = self.cell_k[neighbour_idx]
                
                k_surface = k_C * k_F / ( g_C * k_C + g_F * k_F )

                n = self.mesh.get_surface_normal(triangle_idx, surface)
                e = np.array(F_center - C_center) / d_CF

                # Using the orthogonal correction approach
                T_surface = (n - e) * surface.length

                self.cross_diffusion_correction[triangle_idx] += k_surface * np.dot(grad_flux_surface, T_surface)
                

    def calculate_effective_thermal_resistance(self):
        HP_boundary_temps = []
        FP_boundary_temps = []

        # Iterating over all the boundary edges to see which are in contact with HP and fuel pins. 
        
        for boundary_edge in self.mesh.get_boundary_edges():
            triangle_idx = self.mesh._edge_to_triangles[boundary_edge]

            if self.is_edge_on_circumference(boundary_edge, self.HP_centers, self.HP_radius):
                edge_length = np.linalg.norm(
                    self.mesh.points[boundary_edge[0]] - self.mesh.points[boundary_edge[1]])
                edge_center = (self.mesh.points[boundary_edge[0]] + self.mesh.points[boundary_edge[1]]) / 2.

                q_surface = self.vN_boundary_conditions[triangle_idx] / edge_length

                T_C = self.cell_T[triangle_idx]
                k_C = self.cell_k[triangle_idx]
                
                # Not multiplying gDiff with S due to q_surface is equal to "S * q_b" 
                d_Cf = np.linalg.norm(edge_center - self.mesh._centers[triangle_idx])

                T_surface = T_C - q_surface * d_Cf / k_C

                HP_boundary_temps.append(T_surface)
            
            elif self.is_edge_on_circumference(boundary_edge, self.fuel_pin_centers, self.fuel_pin_radius):
                edge_length = np.linalg.norm(
                    self.mesh.points[boundary_edge[0]] - self.mesh.points[boundary_edge[1]])
                edge_center = (self.mesh.points[boundary_edge[0]] + self.mesh.points[boundary_edge[1]]) / 2.

                q_surface = self.vN_boundary_conditions[triangle_idx] / edge_length

                T_C = self.cell_T[triangle_idx]
                k_C = self.cell_k[triangle_idx]
                
                # Not multiplying gDiff with S due to q_surface is equal to "S * q_b" 
                d_Cf = np.linalg.norm(edge_center - self.mesh._centers[triangle_idx])

                T_surface = T_C - q_surface * d_Cf / k_C

                FP_boundary_temps.append(T_surface)
                
        mean_HP_temp = np.mean(np.array(HP_boundary_temps))
        mean_FP_temp = np.mean(np.array(FP_boundary_temps))

        r_eff = (mean_FP_temp - mean_HP_temp) / self.Q_in

        return r_eff  


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
    

# -------------------- Rectangular test of the thermal mesh conduction code --------------------
 

class rectangular_test_discretised_mesh:
    def __init__(self, data, mesh):
        self.mesh = mesh
        n_triangles = mesh.triangles.shape[0]

        self.Q_in = data.get("Q_in")
        self.Q_out = data.get("Q_out", self.Q_in)

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
        

    def solve(self, iterations: int):

        self.calculate_von_Neumann_boundary_conditions()

        # Iterations to allow for cross diffusion correction to converge.
        for i in range(iterations):
            n_triangles = self.mesh.triangles.shape[0]

            M = np.zeros((n_triangles, n_triangles))
            C = np.zeros(n_triangles)

            self.calculate_cross_diffusion_correction()

            for triangle_idx in range(n_triangles):
                triangle_center = self.mesh.get_center_point(triangle_idx)
                neighbours, surfaces = self.mesh.get_faces_and_neighbours(triangle_idx)

                # Setting the FluxF and FluxC terms, in addition to the boundary FluxV terms (BC).
                for neighbour_idx, surface in zip(neighbours, surfaces):
                    if neighbour_idx == -1:  # Indicating a boundary triangle.
                        FluxV_f_BC = self.vN_boundary_conditions[triangle_idx]

                        C[triangle_idx] += -FluxV_f_BC

                    else:
                        neighbour_center = self.mesh.get_center_point(neighbour_idx)
                        surface_center = surface.center

                        d_CF = np.linalg.norm(triangle_center - neighbour_center)
                        d_Cs = np.linalg.norm(triangle_center - surface_center)
                        d_sF = np.linalg.norm(surface_center - neighbour_center)

                        # Utilizing orthogonal correction approach.
                        gDiff_f = surface.length / d_CF

                        g_f = d_Cs / (d_Cs + d_sF)
                        k_C = self.cell_k[triangle_idx]
                        k_F = self.cell_k[neighbour_idx]

                        k_s = k_C * k_F / ((1 - g_f) * k_C + g_f * k_F)

                        FluxF_f = -k_s * gDiff_f
                        FluxC_f =  k_s * gDiff_f

                        M[triangle_idx][neighbour_idx] = FluxF_f
                        M[triangle_idx][triangle_idx] += FluxC_f

                # Setting the FluxV terms that corresponds to the cross diffusion (CD) correction.
                FluxV_f_CD = self.cross_diffusion_correction[triangle_idx]
                C[triangle_idx] += -FluxV_f_CD

            T = np.linalg.solve(M, C)
            self.cell_T = T
            print(T - np.min(T))

        return T

    def calculate_von_Neumann_boundary_conditions(self):

        x_coords = self.mesh.points[:, 0]
        x_min = np.min(x_coords)
        x_max = np.max(x_coords)
        epsilon = 1e-6

        total_length_left = 0.0
        total_length_right = 0.0
        left_triangle_indices = []
        right_triangle_indices = []

        self.vN_boundary_conditions = np.zeros(self.mesh.triangles.shape[0])

        for boundary_edge in self.mesh.get_boundary_edges():
            triangle_idx = self.mesh._edge_to_triangles[boundary_edge]

            p1 = self.mesh.points[boundary_edge[0]]
            p2 = self.mesh.points[boundary_edge[1]]

            edge_length = np.linalg.norm(p1 - p2)

            # Left boundary: x = x_min
            if abs(p1[0] - x_min) < epsilon and abs(p2[0] - x_min) < epsilon:
                left_triangle_indices.append(triangle_idx)

                # Heat entering the domain from the left:
                # outward normal points left, heat flux points right -> q·S < 0
                self.vN_boundary_conditions[triangle_idx] = -self.Q_in * edge_length
                total_length_left += edge_length

            # Right boundary: x = x_max
            elif abs(p1[0] - x_max) < epsilon and abs(p2[0] - x_max) < epsilon:
                right_triangle_indices.append(triangle_idx)

                # Heat leaving the domain at the right:
                # outward normal points right, heat flux points right -> q·S > 0
                self.vN_boundary_conditions[triangle_idx] = self.Q_out * edge_length
                total_length_right += edge_length

        # Normalize by total boundary length on each side
        for triangle_idx in left_triangle_indices:
            self.vN_boundary_conditions[triangle_idx] *= 1 / total_length_left

        for triangle_idx in right_triangle_indices:
            self.vN_boundary_conditions[triangle_idx] *= 1 / total_length_right

    def calculate_cross_diffusion_correction(self):

        def calculate_cell_flux_gradient(triangle_idx: int):
            # Utilizing the Green-Gauss gradient theorem.

            V_C = self.mesh.triangle_area(triangle_idx)
            C_center = self.mesh._centers[triangle_idx]
            neighbours_C, surfaces_C = self.mesh.get_faces_and_neighbours(triangle_idx)

            flux_gradient_C = np.array([0., 0.])
            for neighbour_idx, surface in zip(neighbours_C, surfaces_C):

                if neighbour_idx == -1:
                    # If the neighbouring triangle is an boundary then apply note 3., eq. (8.42) on page 222 in F.Moukalled et al.
                    T_C = self.cell_T[triangle_idx]
                    q_surface = self.vN_boundary_conditions[triangle_idx] / surface.length
                    k_C = self.cell_k[triangle_idx]

                    # Not multiplying gDiff with S due to q_surface is equal to "S * q_b"
                    d_Cf = np.linalg.norm(surface.center - C_center)

                    T_surface = T_C - q_surface * d_Cf / k_C

                    n = self.mesh.get_surface_normal(triangle_idx, surface)
                    S_surface = surface.length * n

                    flux_gradient_C += S_surface * T_surface

                else:
                    F_center = self.mesh._centers[neighbour_idx]

                    d_Cs = np.linalg.norm(C_center - surface.center)
                    d_sF = np.linalg.norm(surface.center - F_center)

                    g_F = d_Cs / (d_Cs + d_sF)
                    g_C = 1. - g_F

                    T_surface = g_C * self.cell_T[triangle_idx] + g_F * self.cell_T[neighbour_idx]

                    n = self.mesh.get_surface_normal(triangle_idx, surface)
                    S_surface = surface.length * n

                    flux_gradient_C += S_surface * T_surface

            return flux_gradient_C / V_C

        self.cross_diffusion_correction = np.zeros(self.mesh.triangles.shape[0])

        # Iterating over all triangles.
        for triangle_idx in range(self.mesh.triangles.shape[0]):

            neighbours_C, surfaces_C = self.mesh.get_faces_and_neighbours(triangle_idx)
            grad_flux_C = calculate_cell_flux_gradient(triangle_idx)
            C_center = self.mesh._centers[triangle_idx]

            # iterating over all surfaces in all triangles. Reduntant, since most surfaces are calculated twice, but simple.
            for neighbour_idx, surface in zip(neighbours_C, surfaces_C):

                # Assuming that the grad_flux is parallel with the surface and the iteration can be skipped.
                if neighbour_idx == -1:
                    continue

                grad_flux_F = calculate_cell_flux_gradient(neighbour_idx)

                F_center = self.mesh._centers[neighbour_idx]

                d_CF = np.linalg.norm(C_center - F_center)
                d_Cs = np.linalg.norm(C_center - surface.center)
                d_sF = np.linalg.norm(surface.center - F_center)

                g_F = d_Cs / (d_Cs + d_sF)
                g_C = 1. - g_F

                grad_flux_surface = g_C * grad_flux_C + g_F * grad_flux_F
                k_C = self.cell_k[triangle_idx]
                k_F = self.cell_k[neighbour_idx]

                k_surface = k_C * k_F / (g_C * k_C + g_F * k_F)

                n = self.mesh.get_surface_normal(triangle_idx, surface)
                e = np.array(F_center - C_center) / d_CF

                # Using the orthogonal correction approach
                T_surface = (n - e) * surface.length

                self.cross_diffusion_correction[triangle_idx] += k_surface * np.dot(grad_flux_surface, T_surface)


def plot_rectangular_test(mod_mesh, cell_T, Q_in, k, height, lenght):
    """
    Scatter plot of numerical temperature vs x-coordinate of cell centers,
    along with analytical 1D solution and residual.

    Assumptions:
    - Steady-state 1D conduction
    - Constant conductivity k
    - Q_in = Q_out (uniform flux)
    """

    mesh = mod_mesh.mesh

    # --- Extract x positions of triangle centers ---
    centers = np.array([mesh.get_center_point(i) for i in range(mesh.triangles.shape[0])])
    x = centers[:, 0]

    # Sort by x for cleaner plotting
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    T_sorted = cell_T[sort_idx]

    # --- Domain limits ---
    x_min = np.min(x)
    x_max = np.max(x)

    # --- Analytical solution ---
    # T(x) = - (q / k) * (x - x_min)
    # Shift so T(x_min) = 0
    q_surface = Q_in / height

    T_max = (q_surface / k) * (lenght)

    T_analytical = T_max - (q_surface / k) * (x_sorted) 

    # --- Residual ---
    residual = np.mean(np.sqrt((T_sorted - T_analytical)**2))

    # --- Smooth analytical curve ---
    x_line = np.linspace(x_min, x_max, 200)
    print(q_surface, k, x_min, x_max)
    T_line = T_max - (q_surface / k) * (x_line)

    # --- Style ---
    sns.set_theme(style="whitegrid", context="talk")  # "talk" = slightly larger fonts

    # --- Plot ---
    plt.figure(figsize=(9, 5.5))

    # Use seaborn color palette
    palette = sns.color_palette("deep")

    # Numerical scatter
    sns.scatterplot(
        x=x_sorted,
        y=T_sorted,
        s=20,
        alpha=0.7,
        edgecolor=None,
        color=palette[0],
        label="Numerical"
    )

    # Analytical line
    sns.lineplot(
        x=x_line,
        y=T_line,
        linewidth=2.5,
        color=palette[1],
        label="Analytical"
    )

    # Labels & title
    plt.xlabel("x-position [m]", fontsize=13)
    plt.ylabel("Temperature [K]", fontsize=13)
    plt.title(f"1D Heat Conduction Validation\nMean residual = {residual:.2e}", fontsize=15)

    # Legend tweaks
    plt.legend(frameon=True)

    plt.xlim(30., 50.)
    plt.ylim(0., 100.)

    # Tight layout for nicer spacing
    plt.tight_layout()

    plt.show()

    # --- Print residual ---
    print(f"Mean residual: {residual:.6e}")

    return residual


def plot_temperature_profiles(R_eff, mesh, r_i=1.0, r_o=2.0, n_points=500):
    """
    Plot cylindrical and linear temperature profiles in the same figure,
    including horizontal lines showing their average temperatures.

    Parameters
    ----------
    R_eff : float
        Effective thermal resistance [K/W]
    Q_in : float
        Heat input [W]
    r_i : float
        Inner radius / first surface location
    r_o : float
        Outer radius / second surface location
    n_points : int
        Number of points used for plotting
    """

    circumference_HP = mesh.HP_radius * 7./12.
    circumference_FP = mesh.fuel_pin_radius * 2.
    r_i = circumference_HP / 2.
    r_o = circumference_FP / 2.

    Q_in = mesh.Q_in
    cell_T = mesh.cell_T - np.min(mesh.cell_T)

    if R_eff <= 0:
        raise ValueError("R_eff must be positive.")
    if Q_in < 0:
        raise ValueError("Q_in must be non-negative.")
    if r_i <= 0 or r_o <= 0:
        raise ValueError("r_i and r_o must be positive.")
    if r_o <= r_i:
        raise ValueError("r_o must be greater than r_i.")

    # Total temperature difference
    delta_T = Q_in * R_eff

    # Coordinate
    r = np.linspace(r_i, r_o, n_points)

    # Temperature profiles
    T_cyl = delta_T * np.log(r / r_i) / np.log(r_o / r_i)
    T_lin = delta_T * (r - r_i) / (r_o - r_i)

    # Volumetric averages over the interval
    # Cylindrical (weighted by r)
    numerator_cyl = np.trapezoid(T_cyl * r, r)
    denominator_cyl = np.trapezoid(r, r)
    T_cyl_avg = numerator_cyl / denominator_cyl

    # Linear (uniform volume weighting)
    T_lin_avg = np.trapezoid(T_lin, r) / (r_o - r_i)

    # Mesh volumetric average (area-weighted)
    areas = np.array([
        mesh.mesh.triangle_area(i)
        for i in range(len(cell_T))
    ])

    if len(areas) != len(cell_T):
        raise ValueError("T_cells must match number of mesh triangles.")

    T_mesh_avg = np.sum(cell_T * areas) / np.sum(areas)

    # --- Style ---
    sns.set_theme(style="whitegrid", context="talk")

    # --- Plot ---
    plt.figure(figsize=(9, 5.5))
    palette = sns.color_palette("deep")

    # Curves
    sns.lineplot(
        x=r,
        y=T_cyl,
        linewidth=2.8,
        color=palette[0],
        label="Cylindrical profile"
    )

    sns.lineplot(
        x=r,
        y=T_lin,
        linewidth=2.8,
        linestyle="--",
        color=palette[1],
        label="Linear profile"
    )

    # Average lines
    plt.axhline(
        T_cyl_avg,
        color=palette[0],
        linestyle=":",
        linewidth=2.2,
        label=f"Cylindrical average = {T_cyl_avg:.2f} K"
    )

    plt.axhline(
        T_lin_avg,
        color=palette[1],
        linestyle=":",
        linewidth=2.2,
        label=f"Linear average = {T_lin_avg:.2f} K"
    )

    plt.axhline(
        T_mesh_avg,
        color="black",
        linestyle="-.",
        linewidth=2.5,
        label=f"Mesh vol avg = {T_mesh_avg:.2f} K"
    )

    # Surface markers
    plt.scatter(
        [r_i, r_o],
        [0.0, delta_T],
        s=60,
        color="black",
        zorder=5,
        label="Boundary temperatures"
    )

    # Labels and title
    plt.xlabel("Radial position / coordinate", fontsize=13)
    plt.ylabel("Temperature [K]", fontsize=13)
    plt.title(
        f"Temperature profiles from effective thermal resistance\n"
        f"$R_{{eff}}$ = {R_eff:.3g} K/W, $Q_{{in}}$ = {Q_in:.3g} W, "
        f"$\\Delta T$ = {delta_T:.3g} K",
        fontsize=15
    )

    plt.legend(frameon=True, fontsize=11)
    plt.tight_layout()
    plt.show()


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


if __name__ == "__main__":
    mesh = meshio.read("meshHeatConduction/hex_mesh.msh")

    points = mesh.points[:, :2]                
    triangles = mesh.cells_dict["triangle"]     
    mesh = UnstructuredMesh(points, triangles)

    print(f"Number of elements: {triangles.shape}, Number of points: {points.shape}")

    cell_k = np.ones(triangles.shape[0])
    cell_T = np.ones(triangles.shape[0])
    l_pitch = 10.
    r_HP = 3.
    r_f = 1.5
    theta_hex = (np.pi / 6)

    data = {
        "cell_k": 20 * np.ones(triangles.shape[0]),
        "cell_T": np.ones(triangles.shape[0]),

        "Q_in": 1000,

        "HP_centers":       [[0., 0.], [np.tan(theta_hex) * 2. * l_pitch, 2. * l_pitch]],
        "fuel_pin_centers": [[0, l_pitch * 3./2.], [0, l_pitch * 5./2.], [np.tan(theta_hex) * 2 * l_pitch, l_pitch * 7./2.]],
        "HP_radius": r_HP,
        "fuel_pin_radius": r_f
    }

    mod_mesh = moderator_discretised_mesh(mesh=mesh, data=data)
    rect_mesh = rectangular_test_discretised_mesh(mesh=mesh, data=data)
    T = np.array(mod_mesh.solve(iterations=1))
    T = T - np.min(T)

    R_eff = mod_mesh.calculate_effective_thermal_resistance()

    plot_mesh_with_temp_profile(mod_mesh.mesh.points, mod_mesh.mesh.triangles, T)
    plot_temperature_profiles(R_eff, mod_mesh)

    #plot_rectangular_test(mod_mesh, T, data["Q_in"], data["cell_k"][0], 10., 50.)