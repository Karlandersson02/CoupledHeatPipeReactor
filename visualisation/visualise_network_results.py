def format_heat_pipe_temperatures(T) -> str:
    # Index mapping (from your code)
    T_ev_wall = T[0]
    T_ev_wick = T[1]
    T_co_wick = T[2]
    T_co_wall = T[3]
    T_ad_wick = T[4]
    T_ad_wall = T[5]

    lines = []
    lines.append("\n   EVAPORATOR     ADIABATIC      CONDENSER")
    lines.append("┌──────────────┬──────────────┬──────────────┐")
    lines.append("│   WICK       │    WICK      │    WICK      │")
    lines.append(f"│ {T_ev_wick:6.2f} K     │ {T_ad_wick:6.2f} K     │ {T_co_wick:6.2f} K     │")
    lines.append(f"│ {T_ev_wick - 273.15:6.2f} C     │ {T_ad_wick - 273.15:6.2f} C     │ {T_co_wick - 273.15:6.2f} C     │")
    lines.append("├──────────────┼──────────────┼──────────────┤")
    lines.append("│   WALL       │    WALL      │    WALL      │")
    lines.append(f"│ {T_ev_wall:6.2f} K     │ {T_ad_wall:6.2f} K     │ {T_co_wall:6.2f} K     │")
    lines.append(f"│ {T_ev_wall - 273.15:6.2f} C     │ {T_ad_wall - 273.15:6.2f} C     │ {T_co_wall - 273.15:6.2f} C     │")
    lines.append("└──────────────┴──────────────┴──────────────┘\n")
    return "\n".join(lines)


def print_heat_pipe_temperatures(T, save_path=None) -> str:
    """
    Prints the table, optionally saves it to save_path, and returns it as a string
    (so it can be embedded in report.md).
    """
    s = format_heat_pipe_temperatures(T)
    print(s)

    if save_path is not None:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(s)

    return s
