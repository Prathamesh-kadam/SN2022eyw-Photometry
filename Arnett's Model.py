import pandas as pd
import numpy as np

lc = pd.read_csv("/content/logL-bb_SN2022eyw_UBgVrRiz.txt",
                 delim_whitespace=True, header=None,
                 names=["JD","logL","elogL"])

# convert to linear luminosity
Lbol = 10**lc["logL"].values          # erg/s
Lerr = Lbol * np.log(10) * lc["elogL"].values

JD   = lc["JD"].values

t0 = 2459668.2   # example: set this to your B-max JD
z = 0.009
       # use actual redshift if non‑negligible

t_rest = (JD - t0) / (1.0 + z)   # rest-frame days since t0

# restrict to -20 to +50 days
mask = (t_rest > -20) & (t_rest < 50)
tB   = t_rest[mask]
LbolB = Lbol[mask]
LerrB = Lerr[mask]

# window for Arnett fit: -10 to +15 days
tph1, tph2 = -10.0, 15.0
m_ph = (tB >= tph1) & (tB <= tph2)
t_fit   = tB[m_ph]
L_fit   = LbolB[m_ph]
Lerr_fit = LerrB[m_ph]

imax = np.argmax(L_fit)
tmax     = t_fit[imax]
Lmax     = L_fit[imax]
tmax_err = 0.5          # days; refine later if needed
Lmax_err = Lerr_fit[imax]

from scipy.optimize import curve_fit

tau_Ni = 8.8
tau_Co = 111.3
eps_Ni = 3.9e10
eps_Co = 6.8e9
Msun   = 1.989e33

def arnett_lum(t, MNi, td):
    t = np.array(t, dtype=float)
    x = t / td
    phi = 1.0 - np.exp(-x**2)  # diffusion kernel approximation
    MNi_g = MNi * Msun
    term_radio = (eps_Ni * np.exp(-t/tau_Ni) +
                  eps_Co * (np.exp(-t/tau_Co) - np.exp(-t/tau_Ni)))
    return MNi_g * term_radio * phi

p0 = [0.3, 10.0]  # initial guess: 0.3 Msun, td=10d
sigma = np.where(Lerr_fit>0, Lerr_fit, np.max(Lerr_fit))

popt, pcov = curve_fit(
    arnett_lum, t_fit, L_fit,
    p0=p0, sigma=sigma, absolute_sigma=True, maxfev=10000
)

MNi_fit, td_fit = popt
MNi_err, td_err = np.sqrt(np.diag(pcov))

print(f"MNi = {MNi_fit:.3f} ± {MNi_err:.3f} Msun")
print(f"td  = {td_fit:.2f} ± {td_err:.2f} days")


import matplotlib.pyplot as plt

t_model = np.linspace(-20, 60, 400)
L_model = arnett_lum(t_model, MNi_fit, td_fit)

plt.figure(figsize=(8,6))
plt.errorbar(tB, np.log10(LbolB), yerr=LerrB/(np.log(10)*LbolB),
             fmt='o', color='k', label='SuperBol (L+BB)')
plt.plot(t_model, np.log10(L_model), color='r', label='Arnett fit')
plt.xlabel("Rest-frame days since t0")
plt.ylabel("log10 L_bol (erg/s)")
plt.xlim(-20, 60)
plt.legend()
plt.grid(alpha=0.3)
plt.show()
