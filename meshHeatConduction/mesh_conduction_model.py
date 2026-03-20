import numpy as np
import meshio
import pyvista as pv

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
        
    def solve(self):
        n_triangles = mesh.triangles.shape[0]

        M = np.zeros((n_triangles, n_triangles))
        C = np.zeros(n_triangles)

        self.generate_von_Neumann_boundary_conditions()

        for triangle_idx in range(n_triangles):
            triangle_center = self.mesh.get_center_point(triangle_idx)
            neighbours, surfaces = self.mesh.get_faces_and_neighbours(triangle_idx)

            for neighbour_idx, surface in zip(neighbours, surfaces):    
                if neighbour_idx == -1: # Indicating a boundary triangle. 
                    FluxV_f = self.vN_boundary_conditions[triangle_idx]

                    C[triangle_idx] = -FluxV_f
                
                else:
                    neighbour_center = self.mesh.get_center_point(neighbour_idx)
                    surface_center = surface.center

                    d_CF = np.linalg.norm(triangle_center - neighbour_center)
                    d_Cs = np.linalg.norm(triangle_center - surface_center)
                    d_sF = np.linalg.norm(surface_center - neighbour_center)

                    gDiff_f = surface.length / d_CF

                    g_f = d_Cs / (d_Cs + d_sF)
                    k_C = self.cell_k[triangle_idx]
                    k_F = self.cell_k[neighbour_idx]
                    
                    k_s = k_C * k_F / ( (1 - g_f) * k_C + g_f * k_F )

                    FluxF_f = -k_s * gDiff_f
                    FluxC_f =  k_s * gDiff_f

                    M[triangle_idx][neighbour_idx] =  FluxF_f
                    M[triangle_idx][triangle_idx ] += FluxC_f

        T = np.linalg.solve(M, C)
        self.cell_T = T
        return T

    def generate_von_Neumann_boundary_conditions(self):

        def is_edge_on_circumference(edge, centers, radius):
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


        total_length_HP = 0
        total_length_fuel_pin = 0
        HP_triangle_indices = []
        fuel_pin_triangle_indices = []

        # Iterating over all the boundary edges to see which are in contact with HP and fuel pins. 
        self.vN_boundary_conditions = np.zeros(self.mesh.triangles.shape[0])
        for boundary_edge in self.mesh.get_boundary_edges():
            triangle_idx = self.mesh._edge_to_triangles[boundary_edge]

            if is_edge_on_circumference(boundary_edge, self.HP_centers, self.HP_radius):
                HP_triangle_indices.append(triangle_idx)

                edge_length = np.linalg.norm(
                    self.mesh.points[boundary_edge[0]] - self.mesh.points[boundary_edge[1]])

                # Positive due to "q dot S > 0" since both S and q points outward. 
                self.vN_boundary_conditions[triangle_idx] = self.Q_in * edge_length

                total_length_HP += edge_length
            
            elif is_edge_on_circumference(boundary_edge, self.fuel_pin_centers, self.fuel_pin_radius):
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
        ...


if __name__ == "__main__":
    mesh = meshio.read("mesh.msh")

    points = mesh.points[:, :2]                
    triangles = mesh.cells_dict["triangle"]     
    mesh = UnstructuredMesh(points, triangles)

    cell_k = np.ones(triangles.shape[0])
    cell_T = np.ones(triangles.shape[0])
    l_pitch = 10.
    r_HP = 3.
    r_f = 1.5
    theta_hex = (np.pi / 6)
    lc = 0.5

    data = {
        "cell_k": np.ones(triangles.shape[0]),
        "cell_T": np.ones(triangles.shape[0]),

        "Q_in": 100,

        "HP_centers":       [[0., 0.], [np.tan(theta_hex) * 2. * l_pitch, 2. * l_pitch]],
        "fuel_pin_centers": [[0, l_pitch * 3./2.], [0, l_pitch * 5./2.]],
        "HP_radius": r_HP,
        "fuel_pin_radius": r_f
    }

    mod_mesh = moderator_discretised_mesh(mesh=mesh, data=data)
    T = np.array(mod_mesh.solve())

    # Convert to 3D
    points_3d = np.column_stack([points, np.zeros(len(points))])

    # PyVista cell format: [3, n1, n2, n3, 3, n1, n2, n3, ...]
    cells = np.hstack([
        np.hstack([[3, *tri] for tri in triangles])
    ])

    # Cell types: 5 = triangle
    cell_types = np.full(len(triangles), pv.CellType.TRIANGLE)

    mesh_visualization = pv.UnstructuredGrid(cells, cell_types, points_3d)
    mesh_visualization.cell_data["u"] = T - np.min(T)

    mesh_visualization.plot(
        scalars="u",
        show_edges=True,
        cmap="viridis"
    )