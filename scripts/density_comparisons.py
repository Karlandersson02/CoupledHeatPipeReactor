import numpy as np
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt

# Sodium bible

def density_l(T):
    f     = 275.32
    g     = 511.58
    h     = 0.5
    rho_c = 219
    T_c   = 2503.7

    # kg / m^3
    return rho_c + f*(1 - T/T_c) + g*(1 - T/T_c)**h

def DeltaHg(T):
    a   = 393.37
    b   = 4398.6
    c   = 0.29302
    T_c = 2503.7

    # J / kg
    return 1e3 * (a*(1 - T/T_c) + b*(1 - T/T_c)**0.29302)

def gamma(T):
    a = 11.9463
    b = -12633.73
    c = -0.4672

    # Pa / K
    return 1e6 * (-b/T**2 + c/T)*np.exp(a + b/T + c*np.log(T))

def density_g(T):
    rho_l       = density_l(T)
    DHg         = DeltaHg(T)
    gamma_sigma = gamma(T)

    # kg / m^3
    return 1 / (DHg/(T*gamma_sigma) + 1/rho_l)

# Nasa formula

def calculate_Na_rho_v(T_vapour):
    # Constants and formula taken from:
    # MODELING OF TRANSIENT HEAT PIPE OPERATION - NASA GRANT NAG-1-392
    # BY Gene T. Colwell, George W, Woodruff 
    # page 190.

    # Should probably be for saturated vapour(?)
    
    # Unit is kg / m^3
    return 6.335e8 * (1 / T_vapour**(1.5)) * 10**(-5567 / T_vapour)
    
# CC relation

def calculate_rho_cc(T_vapour):
    pc = 1300
    Tc = 818
    R = 361.7
    hfg = 4.182e6

    rho = pc / (R * T_vapour) * np.exp(hfg/R * (1/Tc - 1/T_vapour))
    
    return rho


if __name__ == "__main__":
    T = np.linspace(1000, 1500, 100)

    density_bible = density_g(T)
    density_nasa  = calculate_Na_rho_v(T)
    density_cc    = calculate_rho_cc(T)

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111)

    ax.plot(T, density_bible, color="red"  , label="Bible")
    ax.plot(T, density_nasa , color="blue" , label="Nasa")
    ax.plot(T, density_cc   , color="green", label="CC")

    ax.grid(alpha=0.4)
    ax.legend()
    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel("Density [kg/m^3]")

    plt.show()