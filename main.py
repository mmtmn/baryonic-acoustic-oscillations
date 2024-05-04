import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Constants
G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
c = 299792458  # Speed of light (m/s)
H0 = 70  # Hubble constant (km/s/Mpc)
Omega_m = 0.3  # Matter density parameter
Omega_b = 0.05  # Baryon density parameter
Omega_r = 8.4e-5  # Radiation density parameter
Omega_Lambda = 0.7  # Dark energy density parameter

# Sound speed in the early universe
def sound_speed(a):
    return c * np.sqrt(1 / (3 * (1 + 3 * Omega_b / (4 * Omega_r * a))))

# Hubble parameter
def H(a):
    return H0 * np.sqrt(Omega_r / a**4 + Omega_m / a**3 + Omega_Lambda)

# Perturbation equations
def perturbation_eqs(y, a):
    delta_b, v_b = y
    c_s = sound_speed(a)
    dydt = [v_b / a, -v_b / a - 1.5 * Omega_m * H(a)**2 * delta_b / (a * H(a))]
    return dydt

# Time array (scale factor)
a_start = 1e-6
a_end = 1
N = 10000
a = np.logspace(np.log10(a_start), np.log10(a_end), N)

# Initial conditions
delta_b0 = 1e-5
v_b0 = 0
y0 = [delta_b0, v_b0]

# Solve the perturbation equations
sol = odeint(perturbation_eqs, y0, a)
delta_b = sol[:, 0]
v_b = sol[:, 1]

# Convert scale factor to redshift
z = 1 / a - 1

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(z, delta_b, label='Baryon Density Perturbation')
plt.xscale('log')
plt.xlabel('Redshift (z)')
plt.ylabel('Density Perturbation')
plt.title('Baryonic Acoustic Oscillations')
plt.legend()
plt.grid(True)
plt.show()
