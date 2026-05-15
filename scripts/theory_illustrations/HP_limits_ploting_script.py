if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    # -------------------------------------------------------------------------
    # Helper functions
    # -------------------------------------------------------------------------

    def catmull_rom_chain(points, samples_per_segment=40):
        """
        Smoothly interpolate through a set of control points using a Catmull-Rom
        spline. This avoids requiring scipy.
        """

        points = np.asarray(points, dtype=float)

        # Pad endpoints so the curve starts and ends at the first/last points.
        p = np.vstack([points[0], points, points[-1]])

        xs = []
        ys = []

        for i in range(1, len(p) - 2):
            p0, p1, p2, p3 = p[i - 1], p[i], p[i + 1], p[i + 2]

            t = np.linspace(0.0, 1.0, samples_per_segment)
            t2 = t * t
            t3 = t2 * t

            curve = 0.5 * (
                (2.0 * p1)
                + (-p0 + p2) * t[:, None]
                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2[:, None]
                + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3[:, None]
            )

            xs.append(curve[:, 0])
            ys.append(curve[:, 1])

        return np.concatenate(xs), np.concatenate(ys)


    def interpolate_curve_to_grid(x_curve, y_curve, x_grid):
        """
        Interpolates a curve onto a common x-grid.

        Values outside the curve range are returned as NaN, so they are ignored
        when constructing the limiting envelope.
        """

        y_grid = np.full_like(x_grid, np.nan, dtype=float)

        mask = (x_grid >= np.min(x_curve)) & (x_grid <= np.max(x_curve))
        y_grid[mask] = np.interp(x_grid[mask], x_curve, y_curve)

        return y_grid


    def draw_arrowed_axes(ax, x_min, x_max, y_min, y_max):
        """
        Draws simple arrow-style axes.
        """

        ax.annotate(
            "",
            xy=(x_max, y_min),
            xytext=(x_min, y_min),
            arrowprops=dict(
                arrowstyle="-|>",
                lw=1.7,
                color="black",
                shrinkA=0,
                shrinkB=0,
                mutation_scale=18,
            ),
            clip_on=False,
        )

        ax.annotate(
            "",
            xy=(x_min, y_max),
            xytext=(x_min, y_min),
            arrowprops=dict(
                arrowstyle="-|>",
                lw=1.7,
                color="black",
                shrinkA=0,
                shrinkB=0,
                mutation_scale=18,
            ),
            clip_on=False,
        )


    # -------------------------------------------------------------------------
    # Main plotting function
    # -------------------------------------------------------------------------

    def plot_heatpipe_limit_schematic(save_path_pdf=None, save_path_png=None):
        """
        Creates a schematic heat-pipe limit figure.

        The curves are purely illustrative and are not based on physical models.
        """

        # ---------------------------------------------------------------------
        # Synthetic curve definitions
        # ---------------------------------------------------------------------

        curves = {}

        curves["viscous"] = catmull_rom_chain(
            [
                (10, 17),
                (13, 27),
                (17, 41),
                (20, 53),
            ]
        )

        curves["sonic"] = catmull_rom_chain(
            [
                (14, 36),
                (23, 43),
                (32, 55),
                (39, 66),
                (42, 76),
            ]
        )

        curves["capillary"] = catmull_rom_chain(
            [
                (31, 66),
                (35, 69),
                (47, 74),
                (60, 76),
                (70, 74),
                (80, 64),
                (88, 53),
            ]
        )

        curves["entrainment"] = catmull_rom_chain(
            [
                (30, 58),
                (40, 61),
                (52, 69),
                (64, 79),
                (78, 90),
            ]
        )

        curves["boiling"] = catmull_rom_chain(
            [
                (71, 92),
                (75, 74),
                (80, 55),
                (84, 39),
                (88, 30),
            ]
        )

        # ---------------------------------------------------------------------
        # Style settings
        # ---------------------------------------------------------------------

        plt.rcParams.update({
            "font.size": 15,
            "axes.labelsize": 17,
            "axes.titlesize": 18,
            "lines.linewidth": 3.4,
            "font.family": "serif",
            "mathtext.fontset": "dejavuserif",
        })

        fig, ax = plt.subplots(figsize=(10, 7))

        colors = {
            "viscous": "#72C9D4",
            "sonic": "#425C66",
            "capillary": "#D98C18",
            "entrainment": "#39AA97",
            "boiling": "#6A6A6A",
            "fill": "#7BC67B",
        }

        # ---------------------------------------------------------------------
        # Compute tight x-limits so curves almost reach the figure edges
        # ---------------------------------------------------------------------

        all_x = np.concatenate([curve[0] for curve in curves.values()])
        x_curve_min = np.min(all_x)
        x_curve_max = np.max(all_x)

        x_span = x_curve_max - x_curve_min
        x_pad = 0.02 * x_span  # small side padding

        x_min = x_curve_min - x_pad
        x_max = x_curve_max + x_pad

        # Keep y-range as schematic
        y_min = 0
        y_max = 100

        # ---------------------------------------------------------------------
        # Fill feasible area below the limiting envelope
        # ---------------------------------------------------------------------

        x_grid = np.linspace(x_min, x_max, 900)

        y_all = []
        for x_curve, y_curve in curves.values():
            y_all.append(interpolate_curve_to_grid(x_curve, y_curve, x_grid))

        y_all = np.vstack(y_all)

        # Lowest active curve at each x-position.
        y_envelope = np.nanmin(y_all, axis=0)

        active = ~np.isnan(y_envelope)

        ax.fill_between(
            x_grid[active],
            y_min,
            y_envelope[active],
            color=colors["fill"],
            alpha=0.16,
            zorder=0,
            linewidth=0,
        )

        # ---------------------------------------------------------------------
        # Plot limit curves using the original shapes
        # ---------------------------------------------------------------------

        ax.plot(
            *curves["capillary"],
            color=colors["capillary"],
            solid_capstyle="round",
            zorder=3,
        )

        ax.plot(
            *curves["entrainment"],
            color=colors["entrainment"],
            solid_capstyle="round",
            zorder=3,
        )

        ax.plot(
            *curves["boiling"],
            color=colors["boiling"],
            solid_capstyle="round",
            zorder=3,
        )

        ax.plot(
            *curves["sonic"],
            color=colors["sonic"],
            solid_capstyle="round",
            zorder=3,
        )

        ax.plot(
            *curves["viscous"],
            color=colors["viscous"],
            solid_capstyle="round",
            zorder=3,
        )

        # ---------------------------------------------------------------------
        # Labels
        # ---------------------------------------------------------------------

        ax.text(
            15.0,
            73,
            "Capillary limit",
            color=colors["capillary"],
            ha="left",
            va="center",
            fontsize=20,
        )

        ax.text(
            77.5,
            83.5,
            "Entrainment limit",
            color=colors["entrainment"],
            ha="left",
            va="center",
            fontsize=20,
        )

        ax.text(
            60.5,
            89,
            "Boiling limit",
            color=colors["boiling"],
            ha="center",
            va="center",
            fontsize=20,
        )

        ax.text(
            28.5,
            45.5,
            "Sonic limit",
            color="black",
            ha="left",
            va="center",
            fontsize=20,
        )

        ax.text(
            13.5,
            21.5,
            "Viscous limit",
            color=colors["viscous"],
            ha="left",
            va="center",
            fontsize=20,
        )

        ax.text(
            0.5 * (x_min + x_max),
            -3.2,
            "Temperature, $T$",
            ha="center",
            va="top",
            fontsize=20,
        )

        ax.text(
            x_min - 0.04 * x_span,
            51,
            "Axial heat transfer rate, $Q$",
            rotation=90,
            ha="center",
            va="center",
            fontsize=20,
        )

        # ---------------------------------------------------------------------
        # Axes formatting
        # ---------------------------------------------------------------------

        ax.set_xlim(x_min - 0.5, x_max + 0.5)
        ax.set_ylim(-8, 103)

        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_visible(False)

        draw_arrowed_axes(
            ax=ax,
            x_min=x_min,
            x_max=x_max,
            y_min=0,
            y_max=100,
        )

        fig.tight_layout(pad=0.4)

        if save_path_pdf is not None:
            fig.savefig(
                save_path_pdf,
                format="pdf",
                bbox_inches="tight",
                pad_inches=0.03,
            )

        if save_path_png is not None:
            fig.savefig(
                save_path_png,
                dpi=300,
                bbox_inches="tight",
                pad_inches=0.03,
            )

        plt.show()


    # -------------------------------------------------------------------------
    # Create figure
    # -------------------------------------------------------------------------

    plot_heatpipe_limit_schematic(
        save_path_pdf="heatpipe_limits_schematic.pdf"
    )