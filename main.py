import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.special import spherical_jn
from numba import jit, vectorize
from joblib import Parallel, delayed
import emcee

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

# Neutrino parameters
N_nu = 3  # Number of neutrino species
m_nu = np.logspace(-3, 0, N_nu)  # Neutrino masses (eV)
Omega_nu = np.sum(m_nu) / 93.14 / h**2  # Neutrino density parameter

# Primordial power spectrum parameters
A_s = 2.1e-9  # Scalar amplitude
n_s = 0.9649  # Scalar spectral index
k_pivot = 0.05  # Pivot scale (Mpc^-1)

# Reionization parameters
z_reion = 8  # Reionization redshift
delta_z_reion = 0.5  # Width of reionization transition

# Gravitational lensing parameters
C_phi_fid = 1e-8  # Fiducial lensing potential power spectrum amplitude

# Sound speed
@jit(nopython=True)
def sound_speed(a):
   return c * np.sqrt(1 / (3 * (1 + 3 * Omega_b / (4 * Omega_r * a))))

#@vectorize
#def spherical_jn_numba(n, z):
   #return spherical_jn(n, z)

#@jit(nopython=True)
#def spherical_jn_numba(n, z):
#    return spherical_jn(int(n), z)

@jit(nopython=True)
def spherical_jn_numba(n, z):
    if n == 0:
        return np.sin(z) / z
    elif n == 1:
        return np.sin(z) / z**2 - np.cos(z) / z
    else:
        return (2 * n - 1) / z * spherical_jn_numba(n - 1, z) - spherical_jn_numba(n - 2, z)

# Precompute Omega_nu
Omega_nu_val = Omega_nu

# Hubble parameter
@jit(nopython=True)
def H(a):
   return H0 * np.sqrt(Omega_r / a**4 + Omega_m / a**3 + Omega_k / a**2 + Omega_Lambda + Omega_nu_val / a**4)


# Optical depth
@jit(nopython=True)
def tau(a):
   n_H = n_H0 * a**(-3)
   n_He = n_He0 * a**(-3)
   x_e = 1.0 - 0.5 * np.exp(-((1 - a) / (1 - a_eq))**2 / (2 * delta_z_reion**2))  # Reionization model
   return c * sigma_T * (n_H * X_p + n_He * Y_p) * x_e / H(a)

# Visibility function
def g(a):
   return np.exp(-tau(a)) * H(a) / c

# Primordial power spectrum
def P_primordial(k):
    return A_s * (k / k_pivot)**(n_s - 1)

# Transfer function (CDM + baryons)
@jit(nopython=True)
def T_cb(k, a, Omega_m, h):
    q = k / (Omega_m * H0**2 / a)**0.5
    alpha_c = (46.9 * Omega_m * h**2)**0.67 * (1 + (32.1 * Omega_m * h**2)**(-0.532))
    alpha_b = (12.0 * Omega_m * h**2)**0.424 * (1 + (45.0 * Omega_m * h**2)**(-0.582))
    beta_c = (3.89 * Omega_m * h**2)**(-0.251) * (1 + (6.84 * Omega_m * h**2)**0.593)
    beta_b = (0.944 * Omega_m * h**2)**(-0.738) * (1 + (17.2 * Omega_m * h**2)**0.594)
    f_c = 1 / (1 + (q / alpha_c)**beta_c)
    f_b = 1 / (1 + (q / alpha_b)**beta_b)
    return Omega_b / Omega_m * f_b + Omega_c / Omega_m * f_c

# Non-linear power spectrum (HALOFIT model)
@jit(nopython=True)
def P_nl(k, a, P_primordial_k, Omega_m, h):
    # HALOFIT parameters
    a_nl = 1 / (1 + z_eq) * (1 + 0.284 * (Omega_m * a**3 / (Omega_m * a**3 + Omega_Lambda))**0.693)
    delta_H = 1.68 * Omega_m * a**3 / (Omega_m * a**3 + Omega_Lambda)
    n_eff = -2 - np.log(P_primordial_k * k**3 / (2 * np.pi**2)) / np.log(k)
    C = np.exp(-((k / (3.5 * h))**4 + (k / (1.4 * h))**2)**0.5)
    
    # Linear power spectrum
    P_lin = P_primordial_k * T_cb(k, a, Omega_m, h)**2
    
    # Non-linear correction
    f_nl = (1 + (P_lin * k**3 / (2 * np.pi**2)) * a_nl**2 * delta_H**2 / (1 + (k / (5 * h))**2))**0.5
    P_nl = P_lin * f_nl**2 * C
    
    return P_nl


# Perturbation equations
@jit(nopython=True)
def perturbation_eqs(y, a, k):
   delta_b, v_b, delta_c, v_c, delta_r, v_r = y
   c_s = sound_speed(a)
   k_p = k * c_s / H(a)
   R = 4 * Omega_r / (3 * Omega_b * a)
   dtau_da = c * sigma_T * (n_H0 * X_p + n_He0 * Y_p) * (1.0 - 0.5 * np.exp(-((1 - a) / (1 - a_eq))**2 / (2 * delta_z_reion**2))) / H(a)
   
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
def solve_perturbations(k_val):
   # Initial conditions
   delta_b0 = P_primordial(k_val)**0.5
   v_b0 = 0
   delta_c0 = delta_b0
   v_c0 = 0
   delta_r0 = delta_b0 / 3
   v_r0 = delta_r0 / 4
   y0 = [delta_b0, v_b0, delta_c0, v_c0, delta_r0, v_r0]
   
   # Solve the perturbation equations
   sol, info = odeint(perturbation_eqs, y0, a, args=(k_val,), rtol=1e-8, atol=1e-10, full_output=1)
   print("ODE solver info:", info)  # Print the ODE solver information
   delta_b = sol[:, 0]
   delta_c = sol[:, 2]
   delta_r = sol[:, 4]
   
   # Interpolate the solutions to get the final values
   delta_b_interp = interp1d(a, delta_b)
   delta_c_interp = interp1d(a, delta_c)
   delta_r_interp = interp1d(a, delta_r)
   
   return delta_b_interp(a_end), delta_c_interp(a_end), delta_r_interp(a_end)

# Solve perturbations in parallel
results = Parallel(n_jobs=-1)(delayed(solve_perturbations)(k_val) for k_val in k)
delta_b_final, delta_c_final, delta_r_final = zip(*results)
delta_b_final = np.array(delta_b_final)
delta_c_final = np.array(delta_c_final)
delta_r_final = np.array(delta_r_final)

# Compute the matter power spectrum
def P_m(k):
    T_k = T_cb(k, a_end, Omega_m, h)
    P_primordial_k = P_primordial(k)
    P_lin = P_primordial_k * (delta_b_final + delta_c_final)**2 * T_k**2
    return P_nl(k, a_end, P_primordial_k, Omega_m, h) * P_lin / (P_primordial_k * T_k**2)

# Integrands for CMB power spectra
def integrand_TT(k):
   return P_m(k) * g(a_end)**2

def integrand_EE(k):
   return P_m(k) * g(a_end)**2 * (5 / 4 * tau(a_end))**2

def integrand_TE(k):
   return P_m(k) * g(a_end)**2 * (5 / 4 * tau(a_end))

def integrand_BB(k):
   return P_m(k) * (5 / 4 * tau(a_end))**2

# Compute the CMB angular power spectra
def C_l(l, P_phi, func_integrand):
   k_min_l = 1e-4  # Minimum wavenumber for integration
   k_max_l = 1e2  # Maximum wavenumber for integration
   N_k_l = 100  # Number of points for integration
   k_l_array = np.logspace(np.log10(k_min_l), np.log10(k_max_l), N_k_l)
   
   integrand = np.zeros_like(k_l_array)
   for i, k_l in enumerate(k_l_array):
       j_l = spherical_jn_numba(l, k_l * (c / H0) * tau(a_end))
       integrand[i] = func_integrand(k_l) * j_l**2
   
   integral = np.trapz(integrand, k_l_array)
   return (2 / np.pi) * integral * P_phi

def C_l_TT(l):
   return C_l(l, 1, integrand_TT)

def C_l_EE(l):
   return C_l(l, 1, integrand_EE)

def C_l_TE(l):
   return C_l(l, 1, integrand_TE)

def C_l_BB(l):
   return C_l(l, C_phi_fid, integrand_BB)

# Multipole array
l_min = 2
l_max = 2500
l = np.arange(l_min, l_max + 1)

# Compute the CMB power spectra
C_TT = np.array([C_l_TT(l_val) for l_val in l])
C_EE = np.array([C_l_EE(l_val) for l_val in l])
C_TE = np.array([C_l_TE(l_val) for l_val in l])
C_BB = np.array([C_l_BB(l_val) for l_val in l])

# Cosmological parameter estimation (using MCMC)
def log_prior(params):
   # Flat priors for cosmological parameters
   Omega_c, Omega_b, H0, A_s, n_s, m_nu_sum = params
   if 0.1 < Omega_c < 0.9 and 0.01 < Omega_b < 0.1 and 50 < H0 < 100 and 1e-10 < A_s < 1e-8 and 0.8 < n_s < 1.2 and 0 < m_nu_sum < 1:
      return 0.0
   return -np.inf

def log_likelihood(params):
   # Extract parameters
   Omega_c, Omega_b, H0, A_s, n_s, m_nu_sum = params
   m_nu = np.array([0.0, m_nu_sum / 2, m_nu_sum / 2])  # Assuming normal hierarchy
   
   # Update cosmological parameters
   Omega_m = Omega_c + Omega_b
   Omega_Lambda = 1 - Omega_m - Omega_r - Omega_nu - Omega_k
   h = H0 / 100
   
   # Recompute power spectra
   C_TT_model = np.array([C_l_TT(l_val) for l_val in l])
   C_EE_model = np.array([C_l_EE(l_val) for l_val in l])
   C_TE_model = np.array([C_l_TE(l_val) for l_val in l])
   C_BB_model = np.array([C_l_BB(l_val) for l_val in l])
   
   # Compute log-likelihood
   chi2_TT = np.sum((C_TT - C_TT_model)**2 / (C_TT_model + noise_TT**2))
   chi2_EE = np.sum((C_EE - C_EE_model)**2 / (C_EE_model + noise_EE**2))
   chi2_TE = np.sum((C_TE - C_TE_model)**2 / (C_TE_model + noise_TE**2))
   chi2_BB = np.sum((C_BB - C_BB_model)**2 / (C_BB_model + noise_BB**2))
   
   return -0.5 * (chi2_TT + chi2_EE + chi2_TE + chi2_BB)

def log_probability(params):
   lp = log_prior(params)
   if not np.isfinite(lp):
       return -np.inf
   return lp + log_likelihood(params)

# Observational data (placeholder values, replace with actual data)
noise_TT = np.ones_like(C_TT) * 1e-10
noise_EE = np.ones_like(C_EE) * 1e-10
noise_TE = np.ones_like(C_TE) * 1e-10
noise_BB = np.ones_like(C_BB) * 1e-10

# Initial parameter values
params0 = [Omega_c, Omega_b, H0, A_s, n_s, np.sum(m_nu)]

# Set up the MCMC sampler
ndim = len(params0)
nwalkers = 32
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)

# Run the MCMC sampler
nsteps = 1000
sampler.run_mcmc(params0, nsteps)

# Plot the matter power spectrum
plt.figure(figsize=(8, 6))
plt.loglog(k, np.array([P_m(k_val) for k_val in k]), label='Matter Power Spectrum')
plt.xlabel('Wavenumber k (h/Mpc)')
plt.ylabel('Power Spectrum P(k)')
plt.title('Matter Power Spectrum from Baryonic Acoustic Oscillations')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('matter_power_spectrum.png', dpi=300)

# Compute the model CMB power spectra
C_TT_model = np.array([C_l_TT(l_val) for l_val in l])
C_EE_model = np.array([C_l_EE(l_val) for l_val in l])
C_TE_model = np.array([C_l_TE(l_val) for l_val in l])
C_BB_model = np.array([C_l_BB(l_val) for l_val in l])

# Plot the CMB power spectra
fig, ax = plt.subplots(2, 2, figsize=(12, 12), sharex=True)

ax[0, 0].loglog(l, C_TT, label='TT')
ax[0, 0].loglog(l, C_TT_model, label='TT (model)')
ax[0, 0].set_ylabel(r'$C_\ell^{TT}$')
ax[0, 0].legend()

ax[0, 1].loglog(l, C_EE, label='EE')
ax[0, 1].loglog(l, C_EE_model, label='EE (model)')
ax[0, 1].set_ylabel(r'$C_\ell^{EE}$')
ax[0, 1].legend()

ax[1, 0].semilogx(l, C_TE, label='TE')
ax[1, 0].semilogx(l, C_TE_model, label='TE (model)')
ax[1, 0].set_xlabel(r'Multipole $\ell$')
ax[1, 0].set_ylabel(r'$C_\ell^{TE}$')
ax[1, 0].legend()

ax[1, 1].loglog(l, C_BB, label='BB')
ax[1, 1].loglog(l, C_BB_model, label='BB (model)')
ax[1, 1].set_xlabel(r'Multipole $\ell$')
ax[1, 1].set_ylabel(r'$C_\ell^{BB}$')
ax[1, 1].legend()

plt.tight_layout()
plt.savefig('cmb_power_spectra.png', dpi=300)

plt.show()