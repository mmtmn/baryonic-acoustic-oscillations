import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from scipy.optimize import minimize

# Constants
G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
c = 299792458  # Speed of light (m/s)
H0 = 70  # Hubble constant (km/s/Mpc)
Omega_m = 0.3  # Matter density parameter
Omega_b = 0.05  # Baryon density parameter
Omega_c = Omega_m - Omega_b  # Cold dark matter density parameter
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
N_eff = 3.046  # Effective number of neutrino species

# Recombination parameters (using RECFAST)
a_rec = 1 / 1100  # Scale factor at recombination
z_rec = 1 / a_rec - 1  # Redshift at recombination
T_rec = T_cmb / a_rec  # Temperature at recombination
x_e0 = 1.16e-16 * (Omega_b * h**2)**(-0.44) * (Omega_m * h**2)**(-0.08)  # Initial electron fraction

# Neutrino parameters
N_nu = 3  # Number of neutrino species
m_nu = np.array([0.0, 0.009, 0.05])  # Neutrino masses (eV)
Omega_nu = np.sum(m_nu) / 93.14 / h**2  # Neutrino density parameter

# Primordial power spectrum parameters
A_s = 2.1e-9  # Scalar amplitude
n_s = 0.9649  # Scalar spectral index
k_pivot = 0.05  # Pivot scale (Mpc^-1)

# Sound speed
def sound_speed(a):
    return c * np.sqrt(1 / (3 * (1 + 3 * Omega_b / (4 * Omega_r * a))))

# Hubble parameter
def H(a):
    return H0 * np.sqrt(Omega_r / a**4 + Omega_m / a**3 + Omega_k / a**2 + Omega_Lambda + Omega_nu / a**4)

# Optical depth
def tau(a):
    n_H = n_H0 * a**(-3)
    n_He = n_He0 * a**(-3)
    x_e = x_e0 * (a / a_rec)**(-1.5)
    return c * sigma_T * (n_H * X_p + n_He * Y_p) * x_e / H(a)

# Visibility function
def g(a):
    return np.exp(-tau(a)) * H(a) / c

# Primordial power spectrum
def P_primordial(k):
    return A_s * (k / k_pivot)**(n_s - 1)

# Transfer function (CDM + baryons)
def T_cb(k, a):
    q = k / (Omega_m * H0**2 / a)**0.5
    alpha_c = (46.9 * Omega_m * h**2)**0.67 * (1 + (32.1 * Omega_m * h**2)**(-0.532))
    alpha_b = (12.0 * Omega_m * h**2)**0.424 * (1 + (45.0 * Omega_m * h**2)**(-0.582))
    beta_c = (3.89 * Omega_m * h**2)**(-0.251) * (1 + (6.84 * Omega_m * h**2)**0.593)
    beta_b = (0.944 * Omega_m * h**2)**(-0.738) * (1 + (17.2 * Omega_m * h**2)**0.594)
    f_c = 1 / (1 + (q / alpha_c)**beta_c)
    f_b = 1 / (1 + (q / alpha_b)**beta_b)
    return Omega_b / Omega_m * f_b + Omega_c / Omega_m * f_c

# Perturbation equations
def perturbation_eqs(y, a, k):
    delta_b, v_b, delta_c, v_c, delta_r, v_r = y
    c_s = sound_speed(a)
    k_p = k * c_s / H(a)
    R = 4 * Omega_r / (3 * Omega_b * a)
    dtau_da = c * sigma_T * (n_H0 * X_p + n_He0 * Y_p) * x_e0 * (a / a_rec)**(-1.5) / H(a)
    
    d_delta_b_da = v_b / a
    d_v_b_da = -v_b / a - k_p**2 * delta_b / (1 + R) + k_p**2 * delta_r / (1 + R) - dtau_da * (v_b - v_r)
    d_delta_c_da = v_c / a
    d_v_c_da = -v_c / a - k_p**2 * delta_c
    d_delta_r_da = 4 * v_r / a - 4 * delta_r / a
    d_v_r_da = -k_p**2 * delta_r / 4 + dtau_da * (v_b - v_r) / R
    
    return [d_delta_b_da, d_v_b_da, d_delta_c_da, d_v_c_da, d_delta_r_da, d_v_r_da]

# Wavenumber array
k_min = 1e-4  # Minimum wavenumber (h/Mpc)
k_max = 1e2  # Maximum wavenumber (h/Mpc)
N_k = 100
k = np.logspace(np.log10(k_min), np.log10(k_max), N_k)

# Time array (scale factor)
a_start = 1e-8
a_end = 1
N_a = 1000
a = np.logspace(np.log10(a_start), np.log10(a_end), N_a)

# Initialize arrays for storing results
delta_b_final = np.zeros(N_k)
delta_c_final = np.zeros(N_k)
delta_r_final = np.zeros(N_k)

# Solve the perturbation equations for each wavenumber
for i, k_val in enumerate(k):
    # Initial conditions
    delta_b0 = P_primordial(k_val)**0.5
    v_b0 = 0
    delta_c0 = delta_b0
    v_c0 = 0
    delta_r0 = delta_b0 / 3
    v_r0 = delta_r0 / 4
    y0 = [delta_b0, v_b0, delta_c0, v_c0, delta_r0, v_r0]
    
    # Solve the perturbation equations
    sol = odeint(perturbation_eqs, y0, a, args=(k_val,), rtol=1e-8, atol=1e-10)
    delta_b = sol[:, 0]
    delta_c = sol[:, 2]
    delta_r = sol[:, 4]
    
    # Interpolate the solutions to get the final values
    delta_b_interp = interp1d(a, delta_b)
    delta_c_interp = interp1d(a, delta_c)
    delta_r_interp = interp1d(a, delta_r)
    delta_b_final[i] = delta_b_interp(a_end)
    delta_c_final[i] = delta_c_interp(a_end)
    delta_r_final[i] = delta_r_interp(a_end)

# Compute the matter power spectrum
def P_m(k):
    T_k = T_cb(k, a_end)
    return (2 * np.pi**2 / k**3) * P_primordial(k) * (delta_b_final + delta_c_final)**2 * T_k**2

# Compute the CMB angular power spectrum (temperature)
def C_l_TT(l):
    k_l = (l + 0.5) / (c / H0)
    j_l = np.sqrt(np.pi / (2 * l + 1)) * np.exp(-l * (l + 1) * sigma_T * tau(a_rec) / 2)
    integrand = P_m(k_l) * j_l**2 * g(a_rec)**2
    return np.trapz(integrand, k_l)

# Compute the CMB angular power spectrum (polarization)
def C_l_EE(l):
    k_l = (l + 0.5) / (c / H0)
    j_l = np.sqrt(np.pi / (2 * l + 1)) * np.exp(-l * (l + 1) * sigma_T * tau(a_rec) / 2)
    integrand = P_m(k_l) * j_l**2 * g(a_rec)**2 * (5 / 4 * tau(a_rec))**2
    return np.trapz(integrand, k_l)

# Compute the matter-CMB cross-power spectrum
def C_l_mT(l):
    k_l = (l + 0.5) / (c / H0)
    j_l = np.sqrt(np.pi / (2 * l + 1)) * np.exp(-l * (l + 1) * sigma_T * tau(a_rec) / 2)
    integrand = P_m(k_l) * j_l * g(a_rec)
    return np.trapz(integrand, k_l)

# Multipole array
l_min = 2
l_max = 2500
l = np.arange(l_min, l_max + 1)

# Compute the CMB power spectra
C_TT = np.array([C_l_TT(l_val) for l_val in l])
C_EE = np.array([C_l_EE(l_val) for l_val in l])
C_mT = np.array([C_l_mT(l_val) for l_val in l])

# Cosmological parameter estimation (using minimize)
def log_likelihood(params):
    # Extract parameters
    Omega_c, Omega_b, H0, A_s, n_s = params
    
    # Update cosmological parameters
    Omega_m = Omega_c + Omega_b
    Omega_Lambda = 1 - Omega_m - Omega_r - Omega_nu - Omega_k
    h = H0 / 100
    
    # Recompute power spectra
    C_TT_model = np.array([C_l_TT(l_val) for l_val in l])
    C_EE_model = np.array([C_l_EE(l_val) for l_val in l])
    C_mT_model = np.array([C_l_mT(l_val) for l_val in l])
    
    # Compute log-likelihood
    chi2 = np.sum((C_TT - C_TT_model)**2 / C_TT_model) + np.sum((C_EE - C_EE_model)**2 / C_EE_model) + np.sum((C_mT - C_mT_model)**2 / C_mT_model)
    return -0.5 * chi2

# Initial parameter values
params0 = [Omega_c, Omega_b, H0, A_s, n_s]

# Perform parameter estimation
result = minimize(log_likelihood, params0, method='Nelder-Mead')
best_params = result.x

# Print the best-fit parameters
print("Best-fit parameters:")
print("Omega_c =", best_params[0])
print("Omega_b =", best_params[1])
print("H0 =", best_params[2])
print("A_s =", best_params[3])
print("n_s =", best_params[4])

# Plot the matter power spectrum
plt.figure(figsize=(8, 6))
plt.loglog(k, P_m(k), label='Matter Power Spectrum')
plt.xlabel('Wavenumber k (h/Mpc)')
plt.ylabel('Power Spectrum P(k)')
plt.title('Matter Power Spectrum from Baryonic Acoustic Oscillations')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('matter_power_spectrum.png', dpi=300)

# Plot the CMB temperature power spectrum
plt.figure(figsize=(8, 6))
plt.loglog(l, C_TT, label='CMB Temperature Power Spectrum')
plt.xlabel('Multipole l')
plt.ylabel('Power Spectrum C_l')
plt.title('CMB Temperature Power Spectrum')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('cmb_temperature_power_spectrum.png', dpi=300)

# Plot the CMB polarization power spectrum
plt.figure(figsize=(8, 6))
plt.loglog(l, C_EE, label='CMB Polarization Power Spectrum')
plt.xlabel('Multipole l')
plt.ylabel('Power Spectrum C_l')
plt.title('CMB Polarization Power Spectrum')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('cmb_polarization_power_spectrum.png', dpi=300)

plt.show()
