import numpy as np
import matplotlib.pyplot as plt
import warnings

# Suppress the expected warning when we take the square root of a negative number
# (which happens whenever c < beta1)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# 1. Earth model parameters
beta1 = 2.5  
beta2 = 5.5  
rho1 = 2.8   
rho2 = 3.3   
H = 30 

# 2. Expand the grid to start phase velocity (c) near 0 
# (We start at 0.1 to avoid a literal divide-by-zero at exactly c=0)
f = np.linspace(0.0, 0.5, 300)
c = np.linspace(beta1, beta2, 400) 
F, C = np.meshgrid(f, c)
Omega = 2 * np.pi * F

# 3. Calculate terms 
mu1 = rho1 * beta1**2
mu2 = rho2 * beta2**2

# For C < beta1, this line will silently generate NaNs due to our warning filter
kz1 = np.sqrt(1/beta1**2 - 1/C**2) 
kz2 = np.sqrt(1/C**2 - 1/beta2**2)

theta = Omega * H * kz1
A = mu2 * kz2
B = mu1 * kz1

# 4. The Surface Equation (Yields NaN where Love waves cannot exist)
dispersion_surface = B * np.sin(theta) - A * np.cos(theta)

# 5. Plotting
plt.figure(figsize=(10, 6))

CS = plt.contour(F, C, dispersion_surface, levels=[0], colors='#d62728', linewidths=2)

# Formatting
plt.xlabel('Frequency (Hz)', fontsize=12)
plt.ylabel('Phase Velocity, c (km/s)', fontsize=12)
plt.title(f'Love Wave Dispersion, H = {H} km', fontsize=14)

plt.xlim(0.0, 0.6)
plt.ylim(0.0, beta2 + 1) 

# Draw a line showing the absolute physical floor for Love waves
plt.axhline(beta1, color='blue', linestyle='--', alpha=0.5, label=rf'( $\beta_1$ = {beta1} km/s)')
plt.axhline(beta2, color='purple', linestyle='--', alpha=0.5, label=rf'( $\beta_2$ = {beta2} km/s)')

plt.legend(loc='lower right')
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()

plt.savefig(f'dispersion_curve_{H}km')
plt.show()