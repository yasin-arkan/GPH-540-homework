import numpy as np 
import matplotlib.pyplot as plt 


alpha_1 = 4.0
beta_1 = 2.3
rho_1 = 2.5

alpha_2 = 6.0
beta_2 = 3.5
rho_2 = 2.7

if alpha_1 / alpha_2 > 1:
    critical_angle = np.pi / 2
else:
    critical_angle = np.arcsin(alpha_1 / alpha_2)



i = np.linspace(0, critical_angle, 5000)
i_deg = np.degrees(i)

def calculate_reflection_transmission(a_1, a_2, b_1, b_2, r_1, r_2):

    """
    Calculates exact Zoeppritz reflection and transmission coefficients.
    a_1, a_2: P-wave velocities (upper, lower)
    b_1, b_2: S-wave velocities (upper, lower)
    r_1, r_2: Densities (upper, lower)
    """

    p = np.sin(i) / a_1 # Ray Parameter

    nu_a_1 = np.cos(i) / a_1
    nu_b_1 = np.sqrt(1 - (p * b_1)**2) / b_1
    nu_a_2 = np.sqrt(1 - (p * a_2)**2) / a_2
    nu_b_2 = np.sqrt(1 - (p * b_2)**2) / b_2


    a = r_2 * (1 - 2 * (b_2 ** 2) * (p ** 2)) - r_1 * (1 - 2 * (b_1 ** 2) * (p ** 2))
    b = r_2 * (1 - 2 * (b_2 ** 2) * (p ** 2)) - 2 * r_1 * (b_1 ** 2) * (p ** 2)
    c = r_1 * (1 - 2 * (b_1 ** 2) * (p ** 2)) + 2 * r_2 * (b_2 ** 2) * (p ** 2)
    d = 2 * (r_2 * (b_2 ** 2) - r_1 * (b_1 ** 2))

    E = b * nu_a_1 + c * nu_a_2
    F = b * nu_b_1 + c * nu_b_2
    G = a - d * nu_a_1 * nu_b_2 
    H = a - d * nu_a_2 * nu_b_1 

    D = E * F + G * H * (p ** 2)

    R_pp = ((b * nu_a_1 - c * nu_a_2) * F - (a + d * nu_a_1 * nu_b_2) * H * (p ** 2)) / D
    R_ps = -(2 * nu_a_1 * (a * b + c * d * nu_a_2 * nu_b_2) * p * (a_1 / b_1)) / D

    T_pp = (2 * r_1 * nu_a_1 * F * (a_1 / a_2)) / D 
    T_ps = (2 * r_1 * nu_a_1 * H * p * (a_1 / b_2)) / D

    return R_pp, R_ps, T_pp, T_ps


R_pp, R_ps, T_pp, T_ps = calculate_reflection_transmission(alpha_1, alpha_2, beta_1, beta_2, rho_1, rho_2)


fig, ax = plt.subplots(2, 2, figsize=(16, 12))

fig.suptitle(
    f"Zoeppritz Reflection and Transmission Coefficients, Critical angle: {np.degrees(critical_angle):.2f}°" 
    + "\n" 
    + rf"$\alpha_1= {alpha_1}, \alpha_2 = {alpha_2}  , \beta_1 = {beta_1}, \beta_2 = {beta_2}  , \rho_1 = {rho_1}, \rho_2 = {rho_2}$",
    fontsize=14
)

ax[0, 0].plot(i_deg, R_pp)
ax[0, 0].set_title("P - P reflection coefficients")
ax[0, 0].set_xlabel("Angle")
ax[0, 0].set_ylabel(r"$R_{pp}$")

ax[0, 1].plot(i_deg, R_ps)
ax[0, 1].set_title("P - S reflection coefficients")
ax[0, 1].set_xlabel("Angle")
ax[0, 1].set_ylabel(r"$R_{ps}$")

ax[1, 0].plot(i_deg, T_pp)
ax[1, 0].set_title("P - P transmission coefficients")
ax[1, 0].set_xlabel("Angle")
ax[1, 0].set_ylabel(r"$T_{pp}$")

ax[1, 1].plot(i_deg, T_ps)
ax[1, 1].set_title("P - S transmission coefficients")
ax[1, 1].set_xlabel("Angle")
ax[1, 1].set_ylabel(r"$T_{ps}$")


# plt.savefig("coef_neg_impedance") # To save the plots
plt.show()