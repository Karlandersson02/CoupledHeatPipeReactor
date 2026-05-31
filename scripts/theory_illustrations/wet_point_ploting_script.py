if __name__ == "__main__":
    import json
    import numpy as np
    import matplotlib.pyplot as plt

    from matplotlib.path import Path
    from matplotlib.patches import PathPatch
    import matplotlib.transforms as mtransforms

    from models.heatpipe.liquid_discretised_model import LiquidDiscretised
    from coupled.heatpipe import Heatpipe

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

        # The coupled Heatpipe model expects the resolved mesh.
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

        dz = make_axial_cell_widths(cfg)
        z_edges = np.concatenate([[0.0], np.cumsum(dz)])

        # Drop the first edge, because np.cumsum gives values after each cell.
        z_pressure = z_edges[1:]

        return z_pressure


    def calculate_busse_vapour_pressure_profile(cfg, liquid):
        """
        Computes the Busse vapour pressure-drop profile.

        Updated for the current LiquidDiscretised implementation, where the
        liquid model is initialized with the full heat-pipe solution:

            [T_solid_full, u_v_full, T_v_full]

        The vapour properties are evaluated at the mean vapour temperature, while
        the effective heat load is inferred from the maximum accumulated mass flow.
        """

        T_v_profile = np.asarray(liquid.T_v_full, dtype=float).reshape(-1)
        T_v_scalar = float(np.mean(T_v_profile))

        h_fg = calculate_Na_h_fg(T_v_scalar)
        rho_v = calculate_Na_rho_v(T_v_scalar)
        mu_v = calculate_Na_viscosity_v(T_v_scalar)

        R_v = cfg.geometry.r_vapour
        L_e = cfg.geometry.l_evap
        L_a = cfg.geometry.l_adiabatic
        L_c = cfg.geometry.l_cond

        # Trigger the current local-temperature-based mdot calculation.
        mdot_cell = np.asarray(liquid.get_mdot(), dtype=float)

        # Prefer the face mass-flow profile, since this is the accumulated flow.
        if hasattr(liquid, "mdot_faces"):
            mdot_peak = float(np.nanmax(np.abs(liquid.mdot_faces)))
        else:
            mdot_peak = float(np.nanmax(np.abs(mdot_cell)))

        Q_tot = mdot_peak * h_fg

        print(f"Actual Q_tot from current liquid model = {Q_tot:.6e} W")
        print(f"Mean vapour temperature used in Busse model = {T_v_scalar:.6e} K")

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
            -dP_adiabatic
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
        """
        Draw a vertical curly brace opening to the right.
        """

        if y1 < y0:
            y0, y1 = y1, y0

        ym = 0.5 * (y0 + y1)
        dy = y1 - y0

        verts = [
            (x, y0),
            (x - width, y0),
            (x - width, ym - 0.20 * dy),
            (x - 2.0 * width, ym),
            (x - width, ym + 0.20 * dy),
            (x - width, y1),
            (x, y1),
        ]

        codes = [
            Path.MOVETO,
            Path.CURVE4,
            Path.CURVE4,
            Path.CURVE4,
            Path.CURVE4,
            Path.CURVE4,
            Path.CURVE4,
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


    def add_section_arrows(ax, cfg, *, y_arrow, y_text, color="0.25"):
        """
        Add double-headed arrows in the lower empty part of the plot,
        below the pressure curves.

        Here both x and y are in data coordinates, which makes the placement
        robust after setting the y-limits.
        """

        z0 = 0.0
        z_evap_end = cfg.geometry.l_evap
        z_cond_start = cfg.geometry.l_evap + cfg.geometry.l_adiabatic
        z_end = (
            cfg.geometry.l_evap
            + cfg.geometry.l_adiabatic
            + cfg.geometry.l_cond
        )

        sections = [
            (z_cond_start, z_end, "Condenser"),
        ]

        for x0, x1, label in sections:
            ax.annotate(
                "",
                xy=(x1, y_arrow),
                xytext=(x0, y_arrow),
                arrowprops=dict(
                    arrowstyle="<->",
                    color=color,
                    lw=1.4,
                    shrinkA=0,
                    shrinkB=0,
                ),
                annotation_clip=False,
                zorder=6,
            )

            ax.text(
                0.5 * (x0 + x1),
                y_text,
                label,
                ha="center",
                va="center",
                fontsize=15,
                color=color,
                zorder=6,
            )


    def compute_wet_point_profiles(data, N_R, N_Z, Q_tot):
        """
        Full workflow for one case:
            1. build cfg
            2. solve coupled heatpipe
            3. build liquid model using the current full-solution interface
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

        # ------------------------------------------------------------------
        # Current LiquidDiscretised interface
        # ------------------------------------------------------------------
        # The updated liquid model expects the full heat-pipe solution:
        #
        #     [solid temperature field, vapour velocity field, vapour temperature]
        #
        # It internally converts this to the old flattened format where needed.
        # ------------------------------------------------------------------

        T_HP_full = [
            np.asarray(T_HP_flat, dtype=float),
            np.asarray(u, dtype=float),
            np.asarray(T_v, dtype=float),
        ]

        liquid = LiquidDiscretised(cfg, T_HP_full)

        P_l = liquid.get_pressure_drop_profile()

        P_v = calculate_busse_vapour_pressure_profile(
            cfg=cfg,
            liquid=liquid,
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
            "T_v_scalar": float(np.mean(np.asarray(T_v, dtype=float))),
            "T_v_profile": T_v,
            "T_HP_flat": T_HP_flat,
            "u": u,
            "liquid": liquid,
            "heatpipe": heatpipe,
        }


    def plot_wet_point_cases_1x2(
        results,
        titles,
        Q_tots,
        save_path_pdf=None,
        lower_empty_fraction=0.42,
    ):
        """
        Plots two wet-point cases in a single 1x2 thesis-ready figure.

        Parameters
        ----------
        results : list[dict]
            List containing the two result dictionaries returned by
            compute_wet_point_profiles(...).

        titles : list[str]
            Subfigure titles.

        Q_tots : list[float]
            Total heat loads. Kept as input in case you want to add them to titles
            or annotations later.

        save_path_pdf : str or None
            If given, the combined figure is saved to this path.

        lower_empty_fraction : float
            Extra y-axis space below the pressure curves, expressed as a fraction
            of the pressure-profile range. This creates room for the section
            arrows below the curves.
        """

        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "mathtext.fontset": "cm",
            "font.size": 24,
            "axes.titlesize": 20,
            "axes.labelsize": 20,
            "legend.fontsize": 18,
            "lines.linewidth": 2.2,
            "text.usetex": True,
        })

        fig, axs = plt.subplots(
            1,
            2,
            figsize=(18, 5.8),
            constrained_layout=True,
        )

        fig.set_constrained_layout_pads(
            w_pad=0.02,
            h_pad=0.06,
            wspace=0.04,
            hspace=0.04,
        )

        # Color scheme
        vap_color = "#EC4E20"
        liq_color = "#016FB9"
        boundary_color = "0.55"
        wet_point_color = "black"

        for ax, result, title, Q_tot in zip(axs, results, titles, Q_tots):
            cfg = result["cfg"]
            z = result["z"]
            P_v_plot = np.asarray(result["P_v"], dtype=float)
            P_l_plot = np.asarray(result["P_l"], dtype=float)
            i_wet = result["i_wet"]

            z_wet = z[i_wet]
            P_wet = P_v_plot[i_wet]

            # ------------------------------------------------------------------
            # Geometry and section boundaries
            # ------------------------------------------------------------------

            z_span = z[-1] - z[0]

            z_evap_end, z_cond_start = get_discrete_section_boundaries(z, cfg)

            # ------------------------------------------------------------------
            # Vapour pressure profile
            # ------------------------------------------------------------------

            ax.plot(
                z[:i_wet + 1],
                P_v_plot[:i_wet + 1],
                color=vap_color,
                linestyle="-",
                label="Vapour",
            )

            ax.plot(
                z[i_wet:],
                P_v_plot[i_wet:],
                color=vap_color,
                linestyle="--",
                alpha=0.55,
                label="Vapour, unavailable",
            )

            # ------------------------------------------------------------------
            # Liquid pressure profile
            # ------------------------------------------------------------------

            ax.plot(
                z,
                P_l_plot,
                color=liq_color,
                linestyle="-",
                label="Liquid",
            )

            # ------------------------------------------------------------------
            # Section boundaries
            # ------------------------------------------------------------------

            ax.axvline(
                z_evap_end,
                linestyle="--",
                color=boundary_color,
                linewidth=1.5,
                alpha=0.8,
                label="Section boundary",
            )

            ax.axvline(
                z_cond_start,
                linestyle="--",
                color=boundary_color,
                linewidth=1.5,
                alpha=0.8,
            )

            # ------------------------------------------------------------------
            # Wet point
            # ------------------------------------------------------------------

            ax.scatter(
                [z_wet],
                [P_wet],
                s=110,
                color=wet_point_color,
                zorder=5,
                label="Wet point",
            )

            # ------------------------------------------------------------------
            # Total pressure-drop brace
            # ------------------------------------------------------------------

            P_top_brace = P_v_plot[0]
            P_bottom_brace = np.min(P_l_plot)

            x_brace = z[0] - 0.020 * z_span
            brace_width = 0.014 * z_span

            add_vertical_curly_brace(
                ax,
                x=x_brace,
                y0=P_bottom_brace,
                y1=P_top_brace,
                width=brace_width,
                color="black",
                lw=1.8,
            )

            ax.text(
                x_brace - 5.2 * brace_width,
                0.5 * (P_top_brace + P_bottom_brace),
                r"$\Delta P_{\mathrm{loss}}$",
                rotation=90,
                va="center",
                ha="center",
                fontsize=20,
            )

            # ------------------------------------------------------------------
            # Axis limits
            # ------------------------------------------------------------------
            # Create empty space below the pressure curves. The section arrows are
            # placed in this lower empty region.
            # ------------------------------------------------------------------

            y_bottom = min(np.min(P_v_plot), np.min(P_l_plot))
            y_top = max(np.max(P_v_plot), np.max(P_l_plot))
            y_range = y_top - y_bottom

            if y_range <= 0.0:
                y_range = 1.0

            y_min_plot = y_bottom - lower_empty_fraction * y_range
            y_max_plot = y_top + 0.12 * y_range

            ax.set_ylim(y_min_plot, y_max_plot)

            # ------------------------------------------------------------------
            # Section arrows below the pressure curves
            # ------------------------------------------------------------------

            y_full_range = y_max_plot - y_min_plot

            y_arrow = y_min_plot + 0.15 * y_full_range
            y_text = y_min_plot + 0.07 * y_full_range

            add_section_arrows(
                ax=ax,
                cfg=cfg,
                y_arrow=y_arrow,
                y_text=y_text,
                color="0.25",
            )

            # ------------------------------------------------------------------
            # Axis styling
            # ------------------------------------------------------------------

            ax.set_title(title, pad=8)

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("")
            ax.set_ylabel("")

            ax.set_xlim(
                x_brace - 7.0 * brace_width,
                z[-1] + 0.025 * z_span,
            )

            for spine in ax.spines.values():
                spine.set_linewidth(1.0)

            ax.grid(alpha=0.15, linewidth=0.6)

        # ----------------------------------------------------------------------
        # Shared legend above figure
        # ----------------------------------------------------------------------

        handles, labels = [], []

        for ax in axs:
            h, l = ax.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)

        # Remove duplicate legend entries while preserving order.
        unique_handles = []
        unique_labels = []

        for handle, label in zip(handles, labels):
            if label not in unique_labels:
                unique_handles.append(handle)
                unique_labels.append(label)

        fig.legend(
            unique_handles,
            unique_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.12),
            ncol=5,
            frameon=True,
            columnspacing=1.2,
            handlelength=2.0,
            handletextpad=0.6,
            borderaxespad=0.0,
        )

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
    N_Z = 50

    case_close_to_condenser_start = {
        "Q_tot": 0.7e3,
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


    # -------------------------------------------------------------------------
    # Case 2: wet point close to condenser end
    # -------------------------------------------------------------------------

    result_end = compute_wet_point_profiles(
        data=data,
        N_R=N_R,
        N_Z=N_Z,
        Q_tot=case_close_to_condenser_end["Q_tot"],
    )


    # -------------------------------------------------------------------------
    # Combined 1x2 figure
    # -------------------------------------------------------------------------

    plot_wet_point_cases_1x2(
        results=[
            result_start,
            result_end,
        ],
        titles=[
            case_close_to_condenser_start["title"],
            case_close_to_condenser_end["title"],
        ],
        Q_tots=[
            case_close_to_condenser_start["Q_tot"],
            case_close_to_condenser_end["Q_tot"],
        ],
        save_path_pdf="wet_point_cases_1x2.pdf",
        lower_empty_fraction=0.42,
    )