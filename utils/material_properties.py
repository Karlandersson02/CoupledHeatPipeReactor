import numpy as np

from utils.sodium_properties import calculate_Na_thermal_conductivity_l

def HP_wall_k(T):
    # https://info.ornl.gov/sites/publications/Files/Pub114121.pdf

    # Assuming that the HP wall is made of Non-irradiated Kanthal alloy (Cr 21%, Al 5%)

    A1 = -7.223e-7 # pm 0.128e-7  
    A2 =  1.563e-2 # pm 0.211e-2
    A3 =  6.569    # pm 0.738

    return A1 * T**2 + A2 * T + A3  


def HP_gap_k(T):
    # Assuming that the gap is filled with sodium.

    k_l = calculate_Na_thermal_conductivity_l(T)

    return k_l 


def HP_wick_k(T, porosity):
    # https://info.ornl.gov/sites/publications/Files/Pub114121.pdf

    # Assuming that the HP wick is made of Non-irradiated Kanthal alloy (Cr 21%, Al 5%)
    # and sodium, in a ratio that is expressed as the porosity. 

    # Assuming that the wick is a anular wick with a screened wick.
    # Expression given in Faghri, Heat pipe science and technology, 1995 - eq. (3.49), page 140. 

    k_s = HP_wall_k(T)
    k_l = calculate_Na_thermal_conductivity_l(T)

    numerator   = k_l * ( (k_l + k_s) - (1 - porosity) * (k_l - k_s) ) 
    denominator =       ( (k_l + k_s) + (1 - porosity) * (k_l - k_s) )

    return numerator / denominator


def moderator_k(T):
    # Utilizing a isotropic graphite variant (G-348) that were intended to be used in a gas-cooled reactor.
    # https://www.osti.gov/servlets/purl/1330693.

    # Assuming that the moderator can be modelled as only graphite.

    # Original formula expressed the temperatures in deg C.

    k_graphite = 134.0 - 0.1074 * (T + 273.15) + 3.719e-5 * (T + 273.15)**2  # (W/mK)

    return k_graphite


def moderator_R_eff(T_mod):
    T_mod_data = np.array([
        682.55077538, 784.31323975, 885.81336684, 986.95822547,
        1087.67036323, 1187.90029983, 1287.63485522, 1386.89878892
    ])

    R_eff_data = np.array([
        0.01468726, 0.01547968, 0.0161522, 0.016663,
        0.01697755, 0.0170742, 0.01694783, 0.01661032
    ])

    return np.interp(T_mod, T_mod_data, R_eff_data)


def FP_fuel_k(T, porosity=0):
    # "Modelling of UO2 thermal conductivity: Improvement of the irradiation
    # defects contribution and uncertainty quantification" by Antoine Boulore,
    # Christine Struzik, Vincent Bouineau, Fabrice Gaudier, Guillaume Damblin, Stephane Bernaud 

    # Assuming that the fuel is made up of 100% unirradiated UO2.

    a = 0.035
    b = 2.165e-4
    c = 4.715e9
    d = 16361.0

    term = 1.0 / (a + b*T) + (c / (T**2.05)) * np.exp(-d / T)
    return ((1 - porosity) / (1 + 2*porosity)) * term


def FP_gap_h(T, qp):
    # Taken from TIF330 - Pin Tempereature Modelling. 
    # Source: A. Ponomarev and K. Mikityuk, "Analysis of hypothitical
    # unprotected loss of flow in Superphoenix start-up core: sensitivity to
    # modeling details," ICONE27, May 19-24, 2019, Ibaraki, Japan (2019)

    # Note qp should be in W/m.

    h_gap = np.minimum(
        3 * (1e3 - qp / 100 + (qp / (100 * 10))**2 + (qp / (100 * 100))**3),
        2.3e4)
    
    return h_gap 


def FP_clad_k(T):
    # Source: "Thermal conductivity of zirconium" by J.K. Fink, L. Leibowitz

    # Assuming an uniraddiated Zirconium cladding.
     
    return 8.8527 + 7.0820e-3 * T + 2.5329e-6 * T**2 + 2.9918e3 / T