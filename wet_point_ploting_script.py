if __name__ == "__main__":
    import json
    import numpy as np
    import matplotlib.pyplot as plt

    from models.heatpipe.liquid_discretised_model import LiquidDiscretised
    from coupled_systems.heatpipe import Heatpipe

    from data.dataclass import *
    from utils.solver import Solver

    from utils.sodium_properties import (
        calculate_Na_h_fg,
        calculate_Na_rho_v,
        calculate_Na_viscosity_v,
    )

    # -------------------------------------------------------------------------
    # Helper functions
    # -------------------------------------------------------------------------

    def make_cfg_from_data(data, N_R, N_Z, Q_tot):
        """
        Builds a fresh resolved heat-pipe config for one operating point.
        """

        geom = HeatpipeGeometry(**data["geometry"])
        mesh = HeatpipeMesh(**data["mesh"])

        mesh.N_R = N_R
        mesh.N_Z = N_Z

        mat = HeatpipeMaterial(**data["material"])
        wick = HeatpipeWick(**data["wick"])
        bc = HeatpipeBC(**data["bc"])

        # Same convention as in coupled_systems.heatpipe.generate_cfg(...)
        bc.Q = Q_tot

        cfg = HeatpipeConfig(geom, mesh, mat, wick, bc)

        # Your coupled Heatpipe example uses resolve_mesh().
        # resolve_geometry() is kept to ensure the radii/thicknesses are consistent.
        cfg = cfg.resolve_mesh()

        return cfg


    def solve_coupled_heatpipe(cfg):
        """
        Solves the coupled heat-pipe/vapour model and returns:
            T_HP_flat : flattened solid temperature field
            u         : vapour velocity field
            T_v       : vapour temperature field
        """

        heatpipe = Heatpipe(cfg)
        solver = Solver([heatpipe], iterate=False)
        solver.fsolve()

        # Depending on your Solver implementation, the unpacked solution may be
        # stored either in solver.solutions[0] or only as solver.solution.
        if hasattr(solver, "solutions") and len(solver.solutions) > 0:
            T_HP_flat, u, T_v = solver.solutions[0]
        else:
            T_HP_flat, u, T_v = heatpipe.unpack(solver.solution)

        return heatpipe, T_HP_flat, u, T_v


    def make_liquid_input(T_HP, T_v):
        """
        LiquidDiscretised expects one flat array:

            [flattened solid temperature field, scalar vapour temperature]

        The coupled Heatpipe solver may return T_HP either as:
            - flat array of shape (N_Z * N_R,)
            - 2D array of shape (N_Z, N_R)

        Therefore, we explicitly flatten it before appending T_v.
        """

        T_HP_flat = np.asarray(T_HP, dtype=float).reshape(-1)

        T_v_arr = np.asarray(T_v, dtype=float)
        T_v_scalar = float(np.mean(T_v_arr))

        T_liquid_input = np.concatenate([
            T_HP_flat,
            np.array([T_v_scalar], dtype=float),
        ])

        return T_liquid_input, T_v_scalar


    def make_axial_cell_widths(cfg):
        """
        Cell widths in the axial direction.
        """

        dz_evap = np.ones(cfg.mesh.N_evap) * (
            cfg.geometry.l_evap / cfg.mesh.N_evap
        )

        dz_adiabatic = np.ones(cfg.mesh.N_adiabatic) * (
            cfg.geometry.l_adiabatic / cfg.mesh.N_adiabatic
        )

        dz_cond = np.ones(cfg.mesh.N_cond) * (
            cfg.geometry.l_cond / cfg.mesh.N_cond
        )

        return np.concatenate([dz_evap, dz_adiabatic, dz_cond])


    def get_discrete_section_boundaries(z, cfg):
        """
        Returns section-boundary positions consistent with the plotted pressure arrays.

        If z has length N_Z, the profiles are point-based and the boundary lies
        between the last point of one section and the first point of the next.

        If z has length N_Z + 1, the profiles are edge-based and the boundary is
        directly at the corresponding edge index.
        """

        i_evap_end = cfg.mesh.N_evap
        i_cond_start = cfg.mesh.N_evap + cfg.mesh.N_adiabatic

        if len(z) == cfg.mesh.N_Z + 1:
            z_evap_end = z[i_evap_end]
            z_cond_start = z[i_cond_start]

        elif len(z) == cfg.mesh.N_Z:
            z_evap_end = 0.5 * (z[i_evap_end - 1] + z[i_evap_end])
            z_cond_start = 0.5 * (z[i_cond_start - 1] + z[i_cond_start])

        else:
            raise ValueError(
                f"Unexpected coordinate length: len(z)={len(z)}, "
                f"N_Z={cfg.mesh.N_Z}"
            )

        return z_evap_end, z_cond_start

    def make_axial_pressure_coordinates(cfg):
        """
        Coordinates corresponding to pressure profiles computed using np.cumsum.

        Since P = cumsum(dp/dz * dz), each pressure value belongs to the
        downstream/right edge of each cell, not the cell centre.
        """

        dz_evap = np.ones(cfg.mesh.N_evap) * (
            cfg.geometry.l_evap / cfg.mesh.N_evap
        )

        dz_adiabatic = np.ones(cfg.mesh.N_adiabatic) * (
            cfg.geometry.l_adiabatic / cfg.mesh.N_adiabatic
        )

        dz_cond = np.ones(cfg.mesh.N_cond) * (
            cfg.geometry.l_cond / cfg.mesh.N_cond
        )

        dz = np.concatenate([dz_evap, dz_adiabatic, dz_cond])

        z_edges = np.concatenate([[0.0], np.cumsum(dz)])

        # Drop the first edge, because np.cumsum gives values after each cell.
        z_pressure = z_edges[1:]

        return z_pressure


    def calculate_busse_vapour_pressure_profile(cfg, liquid, T_v_scalar):
        """
        Computes the Busse vapour pressure-drop profile.

        This is the standalone replacement for the old method:

            HeatPipeLimitations.analytical_pressure_drop_Busse()
        """

        h_fg = calculate_Na_h_fg(T_v_scalar)
        rho_v = calculate_Na_rho_v(T_v_scalar)
        mu_v = calculate_Na_viscosity_v(T_v_scalar)

        R_v = cfg.geometry.r_vapour
        L_e = cfg.geometry.l_evap
        L_a = cfg.geometry.l_adiabatic
        L_c = cfg.geometry.l_cond

        # Same convention as before: Q_tot is inferred from the liquid model.
        mdot = liquid.get_mdot()
        Q_tot = mdot[cfg.mesh.N_evap] * h_fg

        print(f"Actual Q_tot from liquid model = {Q_tot:.6e} W")

        Re_re = Q_tot / (2.0 * np.pi * L_e * h_fg * mu_v)
        Re_rc = Q_tot / (2.0 * np.pi * L_c * h_fg * mu_v)

        print(f"Re_re = {Re_re:.6e}")
        print(f"Re_rc = {Re_rc:.6e}")

        # Evaporator + adiabatic, Busse-style expression.
        F = (
            7.0 / 9.0
            - 1.7
            * Re_re
            / (36.0 + 10.0 * Re_re)
            * np.exp(-7.5 * L_a / (Re_re * L_e))
        )

        dP_evap = (
            (-4.0 / np.pi)
            * (mu_v * Q_tot)
            / (rho_v * R_v**4 * h_fg)
            * (L_e * (1.0 + Re_re * F))
        )

        dP_adiabatic = (
            -(8.0 * mu_v * Q_tot * L_a)
            / (rho_v * np.pi * R_v**4 * h_fg)
        )

        dP_cond = (
            -dP_evap
            - dP_adiabatic
            - (4.0 / np.pi)
            * (mu_v * Q_tot)
            / (rho_v * R_v**4 * h_fg)
            * (L_e + 2.0 * L_a + L_c)
        )

        dz = make_axial_cell_widths(cfg)

        if len(dz) != cfg.mesh.N_Z:
            raise ValueError(
                f"Axial mesh mismatch: len(dz)={len(dz)}, "
                f"cfg.mesh.N_Z={cfg.mesh.N_Z}. "
                f"Check N_evap + N_adiabatic + N_cond."
            )

        dpdz = np.zeros(cfg.mesh.N_Z, dtype=float)

        # ---------------------------------------------------------------------
        # Evaporator pressure distribution
        # ---------------------------------------------------------------------

        i0 = 0
        i1 = cfg.mesh.N_evap

        z_evap = np.linspace(0.0, L_e, cfg.mesh.N_evap)
        profile_evap = (z_evap / L_e) ** 2

        norm_evap = np.sum(profile_evap * dz[i0:i1])
        dpdz[i0:i1] = dP_evap * profile_evap / norm_evap

        # ---------------------------------------------------------------------
        # Adiabatic pressure distribution
        # ---------------------------------------------------------------------

        i0 = cfg.mesh.N_evap
        i1 = cfg.mesh.N_evap + cfg.mesh.N_adiabatic

        dpdz[i0:i1] = dP_adiabatic / L_a

        # ---------------------------------------------------------------------
        # Condenser pressure distribution
        # ---------------------------------------------------------------------

        i0 = cfg.mesh.N_evap + cfg.mesh.N_adiabatic
        i1 = cfg.mesh.N_Z

        z_cond = np.linspace(0.0, L_c, cfg.mesh.N_cond)
        xi = 1.0 - z_cond / L_c
        profile_cond = xi**2

        norm_cond = np.sum(profile_cond * dz[i0:i1])
        dpdz[i0:i1] = dP_cond * profile_cond / norm_cond

        P_v = np.cumsum(dpdz * dz)

        return P_v
    

    def add_vertical_curly_brace(ax, x, y0, y1, width, color="black", lw=1.5):
        from matplotlib.path import Path
        from matplotlib.patches import PathPatch
        """
        Draw a vertical curly brace opening to the right.
        
        Parameters
        ----------
        ax : matplotlib axis
        x : float
            x-position of the brace
        y0, y1 : float
            lower and upper y-values
        width : float
            horizontal size of the brace
        """

        if y1 < y0:
            y0, y1 = y1, y0

        ym = 0.5 * (y0 + y1)
        dy = y1 - y0

        verts = [
            (x, y0),                              # start
            (x - width, y0),                      # control 1
            (x - width, ym - 0.20 * dy),          # control 2
            (x - 2* width, ym),                              # midpoint
            (x - width, ym + 0.20 * dy),          # control 3
            (x - width, y1),                      # control 4
            (x, y1),                              # end
        ]

        codes = [
            Path.MOVETO,
            Path.CURVE4, Path.CURVE4, Path.CURVE4,
            Path.CURVE4, Path.CURVE4, Path.CURVE4,
        ]

        patch = PathPatch(
            Path(verts, codes),
            fill=False,
            color=color,
            lw=lw,
            capstyle="round",
            joinstyle="round",
        )

        ax.add_patch(patch)
        return patch


    def compute_wet_point_profiles(data, N_R, N_Z, Q_tot):
        """
        Full workflow for one case:
            1. build cfg
            2. solve coupled heatpipe
            3. build liquid model
            4. compute liquid pressure profile
            5. compute Busse vapour pressure profile
            6. shift profiles to find wet point
        """

        cfg = make_cfg_from_data(
            data=data,
            N_R=N_R,
            N_Z=N_Z,
            Q_tot=Q_tot,
        )

        heatpipe, T_HP_flat, u, T_v = solve_coupled_heatpipe(cfg)

        T_liquid_input, T_v_scalar = make_liquid_input(T_HP_flat, T_v)

        liquid = LiquidDiscretised(cfg, T_liquid_input)

        P_l = liquid.get_pressure_drop_profile()
        P_v = calculate_busse_vapour_pressure_profile(
            cfg=cfg,
            liquid=liquid,
            T_v_scalar=T_v_scalar,
        )

        P_l = np.asarray(P_l, dtype=float)
        P_v = np.asarray(P_v, dtype=float)

        if len(P_l) != len(P_v):
            raise ValueError(
                f"Pressure-profile length mismatch: "
                f"len(P_l)={len(P_l)}, len(P_v)={len(P_v)}"
            )

        # Shift both profiles to start at zero.
        P_v = P_v - P_v[0]
        P_l = P_l - P_l[0]

        # Reverse liquid pressure profile so it can be compared against the
        # vapour profile in the same axial direction.
        P_l_rev = P_l[::-1]

        # Shift vapour profile until it just touches the liquid profile.
        shift_touch = np.min(P_v - P_l_rev)
        P_v_touch = P_v - shift_touch

        # Wet point/contact index.
        diff = P_v_touch - P_l_rev
        i_wet = int(np.argmin(np.abs(diff)))

        # Shift both profiles so the plotted vapour inlet starts from zero.
        shift_zero = P_v_touch[0]
        P_v_plot = P_v_touch - shift_zero
        P_l_plot = P_l_rev - shift_zero

        z = make_axial_pressure_coordinates(cfg)

        if len(z) != len(P_v_plot):
            raise ValueError(
                f"Coordinate/profile length mismatch: "
                f"len(z)={len(z)}, len(P_v_plot)={len(P_v_plot)}"
            )

        return {
            "cfg": cfg,
            "z": z,
            "P_v": P_v_plot,
            "P_l": P_l_plot,
            "i_wet": i_wet,
            "T_v_scalar": T_v_scalar,
            "T_v_profile": T_v,
            "T_HP_flat": T_HP_flat,
            "u": u,
        }


    def plot_wet_point_case(result, title, Q_tot, save_path_pdf=None):
        """
        Plots one wet-point case in a thesis-ready style.
        """

        cfg = result["cfg"]
        z = result["z"]
        P_v = result["P_v"]
        P_l = result["P_l"]
        i_wet = result["i_wet"]

        z_wet = z[i_wet]
        P_wet = P_v[i_wet]

        # Position and size of curly brace
        z_span = z[-1] - z[0]
        x_brace = z[0] - 0.035 * z_span
        brace_width = 0.015 * z_span

        z_evap_end, z_cond_start = get_discrete_section_boundaries(z, cfg)

        plt.rcParams.update({
            "font.size": 25,
            "axes.titlesize": 20,
            "axes.labelsize": 17,
            "legend.fontsize": 20,
            "lines.linewidth": 2.0,
        })

        fig, ax = plt.subplots(figsize=(10, 5.5))

        # Thesis-friendly colors
        # vap_color = "#2F6B5F"   # muted dark green
        # liq_color = "#8C4C5A"   # muted burgundy

        # vap_color = "#922323"     # muted dark blue
        # liq_color = "#156615"     # muted orange

        # vap_color = "#2A7F7F"   # muted teal
        # liq_color = "#A65A5A"   # muted red

        # vap_color = "#1F6F68"   # dark muted teal
        # liq_color = "#9A4F4F"   # dark muted red

        vap_color = "#EC4E20"   
        liq_color = "#016FB9"   

        boundary_color = "0.55"

        # Vapour: solid until wet point
        ax.plot(
            z[:i_wet + 1],
            P_v[:i_wet + 1],
            color=vap_color,
            linestyle="-",
            label="Vapour"
        )

        # Vapour: dashed beyond wet point
        ax.plot(
            z[i_wet:],
            P_v[i_wet:],
            color=vap_color,
            linestyle="--",
            alpha=0.55
        )

        # Liquid profile
        ax.plot(
            z,
            P_l,
            color=liq_color,
            linestyle="-",
            label="Liquid"
        )

        # Section boundaries
        ax.axvline(
            z_evap_end,
            linestyle="--",
            color=boundary_color,
            linewidth=1.5,
            label="End of evaporator",
            alpha=0.8
        )

        ax.axvline(
            z_cond_start,
            linestyle="--",
            color=boundary_color,
            linewidth=1.5,
            label="Start of condenser",
            alpha=0.8
        )

        # Wet point
        ax.scatter(
            [z_wet],
            [P_wet],
            s=110,
            color="black",
            zorder=5,
            label="Wet point"
        )

        # Total pressure-drop bracket:
        # from the lowest liquid-pressure point to the start of the vapour pressure fall
        P_top_brace = P_v[0]
        P_bottom_brace = np.min(P_l)   # or P_l[0] if that is always the minimum

        z_span = z[-1] - z[0]
        x_brace = z[0] - 0.02 * z_span
        brace_width = 0.015 * z_span

        add_vertical_curly_brace(
            ax,
            x=x_brace,
            y0=P_bottom_brace,
            y1=P_top_brace,
            width=brace_width,
            color="black",
            lw=1.8,
        )

        # Optional label next to brace
        ax.text(
            x_brace - 5 * brace_width,
            0.5 * (P_top_brace + P_bottom_brace),
            r"$\Delta P_{\text{loss}}$",
            rotation=90,
            va="center",
            ha="center",
        )

        # Minimal axes
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        # ax.set_ylabel(r"$\Delta P$")

        ax.set_xlim(x_brace - 7.0 * brace_width, z[-1] + 0.02 * z_span)

        # ax.set_title(title, pad=10)

        for spine in ax.spines.values():
            spine.set_linewidth(1.0)

        ax.grid(alpha=0.15, linewidth=0.6)

        ax.legend(
            loc="lower right",
            frameon=True,
            fancybox=False,
            framealpha=0.95,
            borderpad=0.8
        )

        fig.tight_layout()
        if save_path_pdf is not None:
            fig.savefig(save_path_pdf, format="pdf", bbox_inches="tight")
        plt.show()


    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------

    with open("./data/vapour_data.json", "r") as f:
        data_guoju = json.load(f)
        data = data_guoju["data_guoju_1000"]


    # -------------------------------------------------------------------------
    # Adjustable numerical settings
    # -------------------------------------------------------------------------

    N_R = 10
    N_Z = 200

    # These values are placeholders. Tune them until the wet point appears
    # where you want it.
    case_close_to_condenser_start = {
        "Q_tot": 1.0e3,
        "title": "Wet point close to condenser start",
    }

    case_close_to_condenser_end = {
        "Q_tot": 1.05e3,
        "title": "Wet point close to condenser end",
    }


    # -------------------------------------------------------------------------
    # Case 1: wet point close to condenser start
    # -------------------------------------------------------------------------

    result_start = compute_wet_point_profiles(
        data=data,
        N_R=N_R,
        N_Z=N_Z,
        Q_tot=case_close_to_condenser_start["Q_tot"],
    )

    plot_wet_point_case(
        result=result_start,
        title="Wet point close to condenser start",
        Q_tot=case_close_to_condenser_start["Q_tot"],
        save_path_pdf="wet_point_close_to_condenser_start.pdf",
    )

    # -------------------------------------------------------------------------
    # Case 2: wet point close to condenser end
    # -------------------------------------------------------------------------

    result_end = compute_wet_point_profiles(
        data=data,
        N_R=N_R,
        N_Z=N_Z,
        Q_tot=case_close_to_condenser_end["Q_tot"],
    )

    plot_wet_point_case(
        result=result_end,
        title="Wet point close to condenser end",
        Q_tot=case_close_to_condenser_end["Q_tot"],
        save_path_pdf="wet_point_close_to_condenser_end.pdf",
    )