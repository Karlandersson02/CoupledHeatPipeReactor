from __future__ import annotations
from ast import Tuple
from collections import defaultdict
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Surface:
    nodes: tuple[int, int]
    points: np.ndarray          # shape (2, 2)
    center: np.ndarray          # shape (2,)
    length: float
    triangles: tuple[int, ...]  # 1 triangle => boundary, 2 => interior


class UnstructuredMesh:
    """
    Small 2D triangular unstructured mesh helper.

    Attributes
    ----------
    points : ndarray of shape (N, 2)
        Node coordinates.
    triangles : ndarray of shape (M, 3)
        Triangle connectivity, storing point indices.
    cell_k : ndarray of shape (M,)
        Per-cell conductivity or similar property.
    cell_T : ndarray of shape (M,)
        Per-cell temperature or similar property.
    """

    def __init__(self, points: np.ndarray, triangles: np.ndarray) -> None:
        
        self.points = np.asarray(points, dtype=float)
        self.triangles = np.asarray(triangles, dtype=int)

        if self.points.ndim != 2 or self.points.shape[1] != 2:
            raise ValueError("points must have shape (N, 2)")
        if self.triangles.ndim != 2 or self.triangles.shape[1] != 3:
            raise ValueError("triangles must have shape (M, 3)")

        self.n_points = self.points.shape[0]
        self.n_triangles = self.triangles.shape[0]

        if np.any(self.triangles < 0) or np.any(self.triangles >= self.n_points):
            raise ValueError("triangles contains invalid point indices")

        self._edge_to_triangles: dict[tuple[int, int], list[int]] = defaultdict(list)
        self._triangle_to_edges: list[list[tuple[int, int]]] = []
        self._centers = np.zeros((self.n_triangles, 2), dtype=float)

        self._build_topology()


    # ----------------------------
    # Internal helpers
    # ----------------------------
    def _build_topology(self) -> None:
        for tri_idx, tri in enumerate(self.triangles):
            edges = self._triangle_edges(tri)
            self._triangle_to_edges.append(edges)

            tri_pts = self.points[tri]
            self._centers[tri_idx] = tri_pts.mean(axis=0)

            for edge in edges:
                self._edge_to_triangles[edge].append(tri_idx)


    @staticmethod
    def _sorted_edge(i: int, j: int) -> tuple[int, int]:
        return (i, j) if i < j else (j, i)


    def _triangle_edges(self, tri: np.ndarray) -> list[tuple[int, int]]:
        a, b, c = tri
        return [
            self._sorted_edge(a, b),
            self._sorted_edge(b, c),
            self._sorted_edge(c, a),
        ]
    

    def _validate_triangle_index(self, triangle: int) -> int:
        tri_idx = int(triangle)
        if tri_idx < 0 or tri_idx >= self.n_triangles:
            raise IndexError(f"triangle index {tri_idx} out of bounds")
        return tri_idx

    # ----------------------------
    # Public API
    # ----------------------------
    def get_neighbours(self, triangle: int) -> list[int]:
        """
        Return neighboring triangle indices sharing an edge with `triangle`.
        """
        tri_idx = self._validate_triangle_index(triangle)
        neighbours: set[int] = set()

        for edge in self._triangle_to_edges[tri_idx]:
            for other in self._edge_to_triangles[edge]:
                if other != tri_idx:
                    neighbours.add(other)

        return sorted(neighbours)
    

    def get_surfaces(self, triangle: int) -> list[Surface]:
        """
        Return the 3 triangle surfaces/edges as Surface objects.
        """
        tri_idx = self._validate_triangle_index(triangle)
        surfaces: list[Surface] = []

        for edge in self._triangle_to_edges[tri_idx]:
            p0 = self.points[edge[0]]
            p1 = self.points[edge[1]]
            edge_points = np.vstack((p0, p1))
            center = 0.5 * (p0 + p1)
            length = np.linalg.norm(p1 - p0)
            attached = tuple(self._edge_to_triangles[edge])

            surfaces.append(
                Surface(
                    nodes=edge,
                    points=edge_points,
                    center=center,
                    length=float(length),
                    triangles=attached,
                )
            )

        return surfaces
    

    def get_faces_and_neighbours(self, triangle: int):
        """
        Return neighbours and corresponding surfaces in aligned order.

        Returns
        -------
        neighbours : list[int]
            Neighbour triangle indices (-1 for boundary)
        surfaces : list[Surface]
            Corresponding surfaces (same ordering)
        """
        tri_idx = self._validate_triangle_index(triangle)

        pairs = []

        for edge in self._triangle_to_edges[tri_idx]:
            # --- Geometry ---
            p0 = self.points[edge[0]]
            p1 = self.points[edge[1]]

            edge_points = np.vstack((p0, p1))
            center = 0.5 * (p0 + p1)
            length = np.linalg.norm(p1 - p0)

            # --- Topology ---
            attached = self._edge_to_triangles[edge]

            # Determine neighbour across edge
            if len(attached) == 2:
                neighbour = attached[0] if attached[1] == tri_idx else attached[1]
            else:
                neighbour = -1  # boundary

            # --- Surface object ---
            surface = Surface(
                nodes=edge,
                points=edge_points,
                center=center,
                length=float(length),
                triangles=tuple(attached),
            )

            pairs.append((neighbour, surface))

        # Sort by neighbour index (boundary = -1 comes first)
        pairs.sort(key=lambda x: (x[0] == -1, x[0]))

        neighbours = [n for n, _ in pairs]
        surfaces = [s for _, s in pairs]

        return [neighbours, surfaces]
    

    def get_triangle_surface_normals(self, triangle: int) -> np.ndarray:
        """
        Return outward unit normals for the 3 surfaces of `triangle`.

        Returns
        -------
        ndarray of shape (3, 2)
            One outward normal per triangle edge, in the same order as get_surfaces().
        """
        tri_idx = self._validate_triangle_index(triangle)
        center = self._centers[tri_idx]
        normals = []

        for edge in self._triangle_to_edges[tri_idx]:
            p0 = self.points[edge[0]]
            p1 = self.points[edge[1]]
            edge_vec = p1 - p0

            # Perpendicular candidates
            n1 = np.array([edge_vec[1], -edge_vec[0]], dtype=float)
            n2 = -n1

            edge_center = 0.5 * (p0 + p1)
            to_edge = edge_center - center

            # Choose the normal pointing away from the triangle center
            n = n1 if np.dot(n1, to_edge) > 0 else n2

            norm = np.linalg.norm(n)
            if norm == 0.0:
                raise ValueError(f"Degenerate edge found in triangle {tri_idx}")

            normals.append(n / norm)

        return np.asarray(normals)
    

    def get_surface_normal(self, triangle: int, surface: Surface) -> np.ndarray:
        """

        @dataclass(frozen=True)
        class Surface:
            nodes: tuple[int, int]
            points: np.ndarray          # shape (2, 2)
            center: np.ndarray          # shape (2,)
            length: float
            triangles: tuple[int, ...]  # 1 triangle => boundary, 2 => interior

        Return outward unit normals for the surface of a `triangle`.

        Returns
        -------
        ndarray of shape (2, )
            One outward normal per triangle edge, in the same order as get_surfaces().
        """
        tri_idx = self._validate_triangle_index(triangle)
        center = self._centers[tri_idx]

        p0 = surface.points[0]
        p1 = surface.points[1]

        edge_vec = p1 - p0

        # Perpendicular candidates
        n1 = np.array([edge_vec[1], -edge_vec[0]], dtype=float)
        n2 = -n1

        edge_center = 0.5 * (p0 + p1)
        to_edge = edge_center - center

        # Choose the normal pointing away from the triangle center
        n = n1 if np.dot(n1, to_edge) > 0 else n2

        norm = np.linalg.norm(n)
        if norm == 0.0:
            raise ValueError(f"Degenerate edge found in triangle {tri_idx}")

        return n / norm
    

    def get_center_point(self, triangle: int) -> np.ndarray:
        """
        Return centroid of triangle.
        """
        tri_idx = self._validate_triangle_index(triangle)
        return self._centers[tri_idx].copy()


    def get_boundary_triangle(self) -> list[int]:
        """
        Return a sorted list of triangle indices that have at least one boundary edge.
        """
        boundary_triangles: set[int] = set()

        for edge, tris in self._edge_to_triangles.items():
            if len(tris) == 1:
                boundary_triangles.add(tris[0])

        return sorted(boundary_triangles)
    

    # Optional convenience helpers
    def get_boundary_edges(self) -> list[tuple[int, int]]:
        """
        Return all boundary edges.
        """
        return sorted(edge for edge, tris in self._edge_to_triangles.items() if len(tris) == 1)
    

    def triangle_area(self, triangle: int) -> float:
        """
        Return area of a triangle.
        """
        tri_idx = self._validate_triangle_index(triangle)
        pts = self.points[self.triangles[tri_idx]]
        x1, y1 = pts[0]
        x2, y2 = pts[1]
        x3, y3 = pts[2]
        return 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))


if __name__ == "__main__":
    # Small example mesh: square split into two triangles
    points = np.array([
        [0.0, 0.0],  # 0
        [1.0, 0.0],  # 1
        [1.0, 1.0],  # 2
        [0.0, 1.0],  # 3
    ])

    triangles = np.array([
        [0, 1, 2],  # triangle 0
        [0, 2, 3],  # triangle 1
    ])

    cell_k = np.array([10.0, 20.0])
    cell_T = np.array([300.0, 350.0])

    mesh = UnstructuredMesh(points, triangles)

    tri = 1
    print("Center of triangle 0:", mesh.get_center_point(tri))
    print("Boundary triangles:", mesh.get_boundary_triangle())
    print("Neighbours of triangle 0:", mesh.get_neighbours(tri))
    print("Surface normals of triangle 0:\n", mesh.get_triangle_surface_normals(tri))

    for i, s in enumerate(mesh.get_surfaces(tri)):
        print(f"\nSurface {i}")
        print("  nodes:", s.nodes)
        print("  center:", s.center)
        print("  length:", s.length)
        print("  attached triangles:", s.triangles)