import numpy as np
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
from scipy.optimize import curve_fit

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 12

def frequencyModel(Lb, omg0):
    """
    """
    return omg0 * np.sqrt(np.pi) / (2*np.pi) / np.sqrt(Lb*1e-3)

data = np.genfromtxt('flapping_freq_vs_body_length.csv', delimiter=',')
data_Lb = data[:,0]
data_f  = data[:,1]

fig, ax = plt.subplots(1, 1, figsize=(6.5,3))

marker = MarkerStyle('o', fillstyle='none')
ax.scatter(data_Lb, data_f, marker=marker, color='k')

popt, pcov = curve_fit(frequencyModel, data_Lb, data_f)
Lb = np.linspace(20, 55, 101)
ax.plot(Lb, frequencyModel(Lb, *popt), '--k', label='Optimal curve fit')
ax.plot(Lb, frequencyModel(Lb, 8*np.pi), 'k', label=r'$\widetilde{\omega}_0 = 8\pi$')
ax.legend()

print(popt)

ax.set_xlim(10,60)
ax.set_ylim(0,80)
ax.set_xlabel(r'$L_b$ (mm)')
ax.set_ylabel(r'$f$ (Hz)')

plt.tight_layout()

fig.savefig('../out/aracheloff_data.pdf', dpi=600, bbox_inches='tight')
