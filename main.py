import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.interpolate import interp1d

# Constants
G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
c = 299792458  # Speed of light (m/s)
H0 = 70  # Hubble constant (km/s/Mpc)
Omega_m = 0.3  # Matter density parameter
Omega_b = 0.05  # Baryon density parameter
Omega_r = 8.4e-5  # Radiation density parameter
Omega_Lambda = 0.7  # Dark energy density parameter
Omega_k = 0  # Curvature density parameter
T_cmb = 2.7255  # CMB temperature (K)
h = H0 / 100  # Reduced Hubble constant
m_H = 1.6737236e-27  # Hydrogen mass (kg)
m_He = 6.6464764e-27  # Helium mass (kg)
X_p = 0.75  # Primordial hydrogen mass fraction
Y_p = 0.25  # Primordial helium mass fraction
n_H0 = 1.9e-7 * (Omega_b * h**2)  # Present-day hydrogen number density (cm^-3)
n_He0 = 0.1 * n_H0  # Present-day helium number density (cm^-3)
a_eq = Omega_r / Omega_m  # Scale factor at matter-radiation equality
T_eq = T_cmb / a_eq  # Temperature at matter-radiation equality
z_eq = 1 / a_eq - 1  # Redshift at matter-radiation equality
k_B = 1.380649e-23  # Boltzmann constant (J/K)
sigma_T = 6.652e-29  # Thomson scattering cross-section (m^2)
alpha = 1 / 137.035999084  # Fine-structure constant
m_e = 9.1093837015e-31  # Electron mass (kg)
eta_b = 6.1e-10  # Baryon-to-photon ratio

# Recombination parameters
a_rec = 1 / 1100  # Scale factor at recombination
z_rec = 1 / a_rec - 1  # Redshift at recombination
T_rec = T_cmb / a_rec  # Temperature at recombination
x_e0 = 1.16e-16 * (Omega_b * h**2)**(-0.44) * (Omega_m * h**2)**(-0.08)  # Initial electron fraction

# Sound speed
def sound_speed(a):
    return c * np.sqrt(1 / (3 * (1 + 3 * Omega_b / (4 * Omega_r * a))))

# Hubble parameter
def H(a):
    return H0 * np.sqrt(Omega_r / a**4 + Omega_m / a**3 + Omega_k / a**2 + Omega_Lambda)

# Optical depth
def tau(a):
    n_H = n_H0 * a**(-3)
    n_He = n_He0 * a**(-3)
    x_e = x_e0 * (a / a_rec)**(-1.5)
    return c * sigma_T * (n_H * X_p + n_He * Y_p) * x_e / H(a)

# Visibility function
def g(a):
    return np.exp(-tau(a)) * H(a) / c

# Perturbation equations
def perturbation_eqs(y, a, k):
    delta_b, v_b, delta_r, v_r = y
    c_s = sound_speed(a)
    k_p = k * c_s / H(a)
    R = 4 * Omega_r / (3 * Omega_b * a)
    dtau_da = c * sigma_T * (n_H0 * X_p + n_He0 * Y_p) * x_e0 * (a / a_rec)**(-1.5) / H(a)
    
    d_delta_b_da = v_b / a
    d_v_b_da = -v_b / a - k_p**2 * delta_b / (1 + R) + k_p**2 * delta_r / (1 + R) - dtau_da * (v_b - v_r)
    d_delta_r_da = 4 * v_r / a - 4 * delta_r / a
    d_v_r_da = -k_p**2 * delta_r / 4 + dtau_da * (v_b - v_r) / R
    
    return [d_delta_b_da, d_v_b_da, d_delta_r_da, d_v_r_da]

# Wavenumber array
k_min = 1e-4  # Minimum wavenumber (h/Mpc)
k_max = 1e2  # Maximum wavenumber (h/Mpc)
N_k = 100
k = np.logspace(np.log10(k_min), np.log10(k_max), N_k)

# Time array (scale factor)
a_start = 1e-6
a_end = 1
N_a = 1000
a = np.logspace(np.log10(a_start), np.log10(a_end), N_a)

# Initialize arrays for storing results
delta_b_final = np.zeros(N_k)
delta_r_final = np.zeros(N_k)

# Solve the perturbation equations for each wavenumber
for i, k_val in enumerate(k):
    # Initial conditions
    delta_b0 = 1e-5
    v_b0 = 0
    delta_r0 = delta_b0 / 3
    v_r0 = delta_r0 / 4
    y0 = [delta_b0, v_b0, delta_r0, v_r0]
    
    # Solve the perturbation equations
    sol = odeint(perturbation_eqs, y0, a, args=(k_val,))
    delta_b = sol[:, 0]
    delta_r = sol[:, 2]
    
    # Interpolate the solutions to get the final values
    delta_b_interp = interp1d(a, delta_b)
    delta_r_interp = interp1d(a, delta_r)
    delta_b_final[i] = delta_b_interp(a_end)
    delta_r_final[i] = delta_r_interp(a_end)

# Compute the matter power spectrum
def P_m(k):
    T_k = np.zeros_like(k)
    for i in range(len(k)):
        if k[i] < k_eq:
            T_k[i] = 1
        else:
            T_k[i] = (k_eq / k[i])**2
    return k**3 * (delta_b_final + delta_r_final)**2 * T_k

# Compute the wavenumber at matter-radiation equality
k_eq = H(a_eq) / sound_speed(a_eq)

# Plot the matter power spectrum
plt.figure(figsize=(10, 6))
plt.loglog(k, P_m(k), label='Matter Power Spectrum')
plt.xlabel('Wavenumber k (h/Mpc)')
plt.ylabel('Power Spectrum P(k)')
plt.title('Matter Power Spectrum from Baryonic Acoustic Oscillations')
plt.legend()
plt.grid(True)
plt.show()
