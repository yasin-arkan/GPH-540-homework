import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# 1. Define earth model parameters
beta1 = 2.5  # Top layer shear velocity (km/s)
beta2 = 5.5  # Half-space shear velocity (km/s)
rho1 = 2.8   # Top layer density
rho2 = 3.3   # Half-space density
H = 30     # Top layer thickness (km)

# Choose a frequency (e.g., 0.25 Hz gives multiple modes)
f = 0.40
omega = 2 * np.pi * f

# 2. Create the phase velocity array (c)
# We use a small epsilon to avoid divide-by-zero errors exactly at beta1 and beta2
eps = 1e-5
c = np.linspace(beta1 + eps, beta2 - eps, 2000)

# 3. Calculate the terms
term1 = np.sqrt(1/beta1**2 - 1/c**2)
term2 = np.sqrt(1/c**2 - 1/beta2**2)
mu_ratio = (rho2 * beta2**2) / (rho1 * beta1**2)

lhs = np.tan(omega * H * term1)
rhs = mu_ratio * (term2 / term1)

def continuous_love_eq(c):
    """Stable form for root finding: B*sin(theta) - A*cos(theta) = 0"""
    omega = 2 * np.pi * f
    mu1 = rho1 * beta1**2
    mu2 = rho2 * beta2**2
    
    kz1 = np.sqrt(1/beta1**2 - 1/c**2)
    kz2 = np.sqrt(1/c**2 - 1/beta2**2)
    
    theta = omega * H * kz1
    A = mu2 * kz2
    B = mu1 * kz1
    return B * np.sin(theta) - A * np.cos(theta)

def rhs_term(c):
    """Calculate the RHS amplitude to get the exact y-coordinate for our plot"""
    mu1 = rho1 * beta1**2
    mu2 = rho2 * beta2**2
    kz1 = np.sqrt(1/beta1**2 - 1/c**2)
    kz2 = np.sqrt(1/c**2 - 1/beta2**2)
    return (mu2 / mu1) * (kz2 / kz1)

# 3. Step One: Find brackets (intervals) where roots exist
# We scan the continuous equation for sign changes to safely trap the roots
c_search = np.linspace(beta1 + 1e-5, beta2 - 1e-5, 500)
y_search = continuous_love_eq(c_search)

# A sign change means the curve crossed zero
sign_changes = np.where(np.diff(np.sign(y_search)))[0]

# 4. Step Two: Solve precisely within each bracket
roots = []
for idx in sign_changes:
    c_low = c_search[idx]
    c_high = c_search[idx + 1]
    
    # brentq finds the exact root between c_low and c_high
    root = brentq(continuous_love_eq, c_low, c_high)
    roots.append(root)

print(f"Solver found {len(roots)} roots at f = {f} Hz:")
for i, r in enumerate(roots):
    print(f" - Mode {len(roots) - 1 - i}: c = {r:.5f} km/s")

# 5. Prepare the Plotting Curves (LHS and RHS)
c_plot = np.linspace(beta1 + 1e-5, beta2 - 1e-5, 10000)
term1 = np.sqrt(1/beta1**2 - 1/c_plot**2)
term2 = np.sqrt(1/c_plot**2 - 1/beta2**2)
mu_ratio = (rho2 * beta2**2) / (rho1 * beta1**2)

lhs = np.tan(2 * np.pi * f * H * term1)
rhs = mu_ratio * (term2 / term1)

# Mask asymptotes on the tangent curve
asymptote_mask = np.append(np.diff(lhs) < 0, False)
lhs[asymptote_mask] = np.nan

# 6. Plot Everything
plt.figure(figsize=(10, 6))

# Plot the curves
plt.plot(c_plot, lhs, label=r'$tan(\omega \sqrt{1/\beta_1^2 - 1 / c^2})$', color='#1f77b4', linewidth=2)
plt.plot(c_plot, rhs, label=r'$\frac{\mu_2 \sqrt{1/c^2 - 1/\beta_2^2}}{\mu_1 \sqrt{1/\beta_1^2 - 1 / c^2}} $', color='#ff7f0e', linewidth=2, linestyle='--')

# Plot the solver's roots
for i, root in enumerate(roots):
    y_val = rhs_term(root) # The intersection height
    
    # Only add the label to the legend once
    label = 'Solver Roots' if i == 0 else "" 
    
    # Scatter the precise root
    plt.scatter(root, y_val, color='red', s=120, zorder=5, label=label, edgecolor='black')
    
    # Add a text annotation above the dot
    plt.annotate(f'{root:.3f} km/s', (root, y_val), textcoords="offset points", 
                 xytext=(-15, 12), ha='center', fontsize=10, fontweight='bold', color='red')

# Formatting
plt.axhline(0, color='black', linewidth=0.8)
plt.ylim(-40, 40)
plt.xlim(beta1 - 0.5, beta2 + 0.5)
plt.xlabel('Phase Velocity, c (km/s)', fontsize=12)
plt.ylabel(r'$\frac{\mu_2 \sqrt{1/c^2 - 1/\beta_2^2}}{\mu_1 \sqrt{1/\beta_1^2 - 1 / c^2}} $', fontsize=12, rotation='horizontal', labelpad=20)
plt.title(f'Numerical Solver Intersections (f = {f} Hz, H = {H} km)', fontsize=14)
plt.legend(loc='upper right', fontsize=15)
plt.grid(True, linestyle=':', alpha=0.7)

plt.tight_layout()
plt.savefig("equation_intersections_2")
plt.show()