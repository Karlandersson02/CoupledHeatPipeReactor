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

