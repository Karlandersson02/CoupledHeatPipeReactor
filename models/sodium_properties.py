import numpy as np

def calculate_Na_rho_l(T_liquid):

    # Constants and formula taken from:
    # Thermodynamic and Transport Properties of Sodium Liquid and Vapor by J. K. Fink and L. Leibowitz 
    # page 86. 

    rho_C = 219.0
    f = 275.32
    g = 511.58
    T_C = 2503.7 

    # units is kg / m^3
    return rho_C + f * (1 - T_liquid / T_C) + g * (1 - T_liquid / T_C)**(0.5) 

def calculate_Na_rho_v(T_vapour):
    # Constants and formula taken from:
    # MODELING OF TRANSIENT HEAT PIPE OPERATION - NASA GRANT NAG-1-392
    # BY Gene T. Colwell, George W, Woodruff 
    # page 190.

    # Should probably be for saturated vapour(?)
    
    # Unit is kg / m^3
    return 6.335e8 * (1 / T_vapour**(1.5)) * 10**(-5567 / T_vapour)

def calculate_Na_viscosity_l(T_liquid):
    
    # Constants and formula taken from:
    # Thermodynamic and Transport Properties of Sodium Liquid and Vapor by J. K. Fink and L. Leibowitz 
    # page 207.

    # unit is Pa * s
    return np.exp(-6.4406 - 0.3958 * np.log(T_liquid) + 556.835 / T_liquid)

def calculate_Na_viscosity_v(T_vapour):
    
    # Constants and formula taken from:
    # MODELING OF TRANSIENT HEAT PIPE OPERATION - NASA GRANT NAG-1-392
    # BY Gene T. Colwell, George W, Woodruff 
    # page 190.

    # Should probably be for saturated vapour(?)
    
    # Unit is N * s / m^2
    return 6.083e-9 * T_vapour + 1.2606e-5

def calculate_Na_h_fg(T_vapour):

    # Constants and formula taken from:
    # Thermodynamic and Transport Properties of Sodium Liquid and Vapor by J. K. Fink and L. Leibowitz 
    # page 65.

    T_crit_Na = 2503.7

    # Original unit is in kJ / kg, converting it to J / kg.
    return 1e3 * (393.37 * (1 - T_vapour / T_crit_Na) + 4398.6 * (1 - T_vapour / T_crit_Na)**(0.29302))

def calculate_Na_surface_tension(T_liquid):

    # Constants and formula taken from:
    # Thermodynamic and Transport Properties of Sodium Liquid and Vapor by J. K. Fink and L. Leibowitz 
    # page 65.

    sigma_0 = 240.5
    n = 1.126
    T_crit_Na = 2503.7

    # Original unit is mN / m, converting it to N / m. 
    return 1e-3 * sigma_0 * (1 - T_liquid / T_crit_Na)**(n)

    return h_fg_Na