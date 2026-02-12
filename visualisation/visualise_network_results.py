def print_heat_pipe_temperatures(T):
    # Index mapping (from your code)
    T_ev_wall = T[0]
    T_ev_wick = T[1]
    T_co_wick = T[2]
    T_co_wall = T[3]
    T_ad_wick = T[4]
    T_ad_wall = T[5]

    print("\n   EVAPORATOR     ADIABATIC      CONDENSER")
    print("┌──────────────┬──────────────┬──────────────┐")
    print(f"│   WICK       │    WICK      │    WICK      │")
    print(f"│ {T_ev_wick:6.2f} K     │ {T_ad_wick:6.2f} K     │ {T_co_wick:6.2f} K     │")
    print(f"│ {T_ev_wick - 273.15:6.2f} C     │ {T_ad_wick - 273.15:6.2f} C     │ {T_co_wick - 273.15:6.2f} C     │")
    print("├──────────────┼──────────────┼──────────────┤")
    print(f"│   WALL       │    WALL      │    WALL      │")
    print(f"│ {T_ev_wall:6.2f} K     │ {T_ad_wall:6.2f} K     │ {T_co_wall:6.2f} K     │")
    print(f"│ {T_ev_wall - 273.15:6.2f} C     │ {T_ad_wall - 273.15:6.2f} C     │ {T_co_wall - 273.15:6.2f} C     │")
    print("└──────────────┴──────────────┴──────────────┘\n")
