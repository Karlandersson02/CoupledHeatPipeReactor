import numpy as np

def calculate_Na_rho_l(T_liquid):

    # Constants and formula taken from:
    # Thermodynamic and Transport Properties of Sodium Liquid and Vapor by J. K. Fink and L. Leibowitz 
    # page 86. 

    rho_C = 219.0
    f = 275.32
    g = 511.58
    T_C = 2503.7 

    return rho_C + f * (1 - T_liquid / T_C) + g * (1 - T_liquid / T_C)**(0.5) 

def calculate_Na_viscosity_l(T_liquid):
    
    # Constants and formula taken from:
    # Thermodynamic and Transport Properties of Sodium Liquid and Vapor by J. K. Fink and L. Leibowitz 
    # page 207.

    return np.exp(-6.4406 - 0.3958 * np.log(T_liquid) + 556.835 / T_liquid)

def calculate_Na_viscosity_v(T_vapour):
    
    # Constants and formula taken from:
    # MODELING OF TRANSIENT HEAT PIPE OPERATION - NASA GRANT NAG-1-392
    # BY Gene T. Colwell, George W, Woodruff 
    # page 190.

    return 6.083e-9 * T_vapour + 1.2606e-5

def calculate_Na_h_fg(T_v):

    # Constants and formula taken from:
    # Thermodynamic and Transport Properties of Sodium Liquid and Vapor by J. K. Fink and L. Leibowitz 
    # page 65.

    T_crit_Na = 2503.7
    h_fg_Na = (393.37 * (1 - T_v / T_crit_Na) + 4398.6 * (1 - T_v / T_crit_Na)**(0.29302)) * 1e3 # kJ -> J

