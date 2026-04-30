import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
from scipy.optimize import curve_fit

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 12

class DragonflySpecimen:
    def __init__(self, fore, hind, body):
        self.species = body['species'].iloc[0]
        self.sex = body['sex'].iloc[0]
        self.ID = body['ID'].iloc[0]
        self.m = body['m'].iloc[0]
        self.L = body['L'].iloc[0]

        self.R_f = fore['R'].iloc[0]
        self.R_h = hind['R'].iloc[0]
        self.S_f = fore['S'].iloc[0]
        self.S_h = hind['S'].iloc[0]

        self.units_to_si()
        self.get_adim_params()

    def units_to_si(self):
        self.m = self.m * 1e-6
        self.L = self.L * 1e-3
        self.R_f = self.R_f * 1e-3
        self.R_h = self.R_h * 1e-3
        self.S_f = self.S_f * 1e-6
        self.S_h = self.S_h * 1e-6

    def get_adim_params(self):
        self.lambda0_f = self.R_f / self.L
        self.lambda0_h = self.R_h / self.L
        Sh_f = self.S_f / (self.S_f + self.S_h)
        Sh_h = self.S_h / (self.S_f + self.S_h)
        self.lambda0 = Sh_f * self.lambda0_f + Sh_h * self.lambda0_h
        
        self.T = np.sqrt(self.L / 9.80665)
        coeff = 1.205 * (self.L / self.T)**2 / (self.m * 9.80665)

        self.mu0_f = coeff * self.S_f * self.lambda0_f
        self.mu0_h = coeff * self.S_h * self.lambda0_h
        self.mu0 = self.mu0_f + self.mu0_h

def read_wakeling1997(forewing_data, hindwing_data, body_data):
    """
    Extract morphological data of dragonfly specimens from
    J.M. Wakeling, "Odonatan Wing and Body Morphologies", 1997.
    """
    fd = pd.read_csv(forewing_data)
    hd = pd.read_csv(hindwing_data)
    bd = pd.read_csv(body_data)
    
    # only keep specimens with body + forewing + hindwing morphology data
    IDs = list(
        set(fd['ID'].values.tolist())
        .intersection(set(hd['ID'].values.tolist()))
        .intersection(set(bd['ID'].values.tolist()))
    )
    IDs.sort()
    
    specimens = []
    for ID in IDs:
        fd_i = fd.loc[fd['ID'] == ID]
        hd_i = hd.loc[hd['ID'] == ID]
        bd_i = bd.loc[bd['ID'] == ID]
        if ID[:2] != 'CS':
            specimens.append(DragonflySpecimen(fd_i, hd_i, bd_i))

    return specimens

def plot_adim_params(specimens):
    marker = {
            'Aeshna cyanea': MarkerStyle('s', fillstyle='none'),
            'Libellula quadrimaculata': MarkerStyle('o', fillstyle='none'),
            'Libellula depressa': MarkerStyle('o'),
            'Orthetrum cancellatum': MarkerStyle('x'),
            'Sympetrum striolatum': MarkerStyle('D', fillstyle='none'),
            'Sympetrum sanguineum': MarkerStyle('D'),
            'Calopteryx splendens': MarkerStyle('^')
            }

    fig, ax = plt.subplots(1, 2, figsize=(6.5,3))

    for sp in specimens:
        ax[0].scatter(sp.lambda0_f, sp.mu0_f, marker=marker[sp.species], color='k')
        ax[1].scatter(sp.lambda0_h, sp.mu0_h, marker=marker[sp.species], color='k', label=sp.species)
    # Get current handles and labels
    handles, labels = ax[1].get_legend_handles_labels()

    # Create a dictionary to store unique labels and their corresponding handles
    # The dictionary automatically handles the uniqueness of keys (labels)
    unique_labels = dict(zip(labels, handles))

    # Generate the legend using the unique handles and labels
    fig.legend(unique_labels.values(), unique_labels.keys(), loc='upper center', bbox_to_anchor=(0.5, 0), fontsize=12, ncol=2)
    ax[0].set_xlabel(r'$\lambda_0$')
    ax[0].set_ylabel(r'$\mu_0$')
    ax[1].set_xlabel(r'$\lambda_0$')
    ax[1].set_ylabel(r'$\mu_0$')
    ax[0].set_xlim(0.6, 0.9)
    ax[0].set_ylim(0, 0.2)
    ax[1].set_xlim(0.6, 0.9)
    ax[1].set_ylim(0, 0.2)
    ax[0].set_title('Forewings', fontsize=12)
    ax[1].set_title('Hindwings', fontsize=12)
    
    plt.tight_layout()
    
    fig.savefig('wakeling1997_data.pdf', dpi=600, bbox_inches='tight')

def mair(m, mu0):
    return mu0*m

def R(Lb, ld0):
    return ld0*Lb


def collect_fit_samples(specimens):
    """Collect finite fore/hind samples in the same units used for plotting."""
    samples = {
        'fore_m_g': [],
        'fore_mair_g': [],
        'fore_Lb_mm': [],
        'fore_R_mm': [],
        'hind_m_g': [],
        'hind_mair_g': [],
        'hind_Lb_mm': [],
        'hind_R_mm': [],
    }

    for sp in specimens:
        if np.isfinite(sp.m) and np.isfinite(sp.L) and np.isfinite(sp.R_f) and np.isfinite(sp.S_f):
            samples['fore_m_g'].append(sp.m * 1e3)
            samples['fore_mair_g'].append(1.205 * sp.R_f * sp.S_f * 1e3)
            samples['fore_Lb_mm'].append(sp.L * 1e3)
            samples['fore_R_mm'].append(sp.R_f * 1e3)
        if np.isfinite(sp.m) and np.isfinite(sp.L) and np.isfinite(sp.R_h) and np.isfinite(sp.S_h):
            samples['hind_m_g'].append(sp.m * 1e3)
            samples['hind_mair_g'].append(1.205 * sp.R_h * sp.S_h * 1e3)
            samples['hind_Lb_mm'].append(sp.L * 1e3)
            samples['hind_R_mm'].append(sp.R_h * 1e3)

    return {k: np.array(v, dtype=float) for k, v in samples.items()}


def fit_wakeling_trends(samples, curve_fit_fn=curve_fit):
    """Fit linear trends for fore/hind air-mass and wing-length relations."""
    mu0_fore, _ = curve_fit_fn(mair, samples['fore_m_g'], samples['fore_mair_g'])
    mu0_hind, _ = curve_fit_fn(mair, samples['hind_m_g'], samples['hind_mair_g'])
    lambda0_fore, _ = curve_fit_fn(R, samples['fore_Lb_mm'], samples['fore_R_mm'])
    lambda0_hind, _ = curve_fit_fn(R, samples['hind_Lb_mm'], samples['hind_R_mm'])

    return {
        'mu0_fore': mu0_fore,
        'mu0_hind': mu0_hind,
        'lambda0_fore': lambda0_fore,
        'lambda0_hind': lambda0_hind,
    }

def plot_adim_params_new(specimens):
    marker1 = MarkerStyle('o')
    marker2 = MarkerStyle('o', fillstyle='none')
    fore_scatter_handle = None
    hind_scatter_handle = None

    fig, ax = plt.subplots(2, 1, figsize=(6.5,5.5))

    for sp in specimens:
        if sp.m > 0.2e-3:
            ax[0].scatter(sp.m*1e3, 1.205*sp.R_f*sp.S_f*1e3, marker=marker1, color='k')
            p1 = ax[1].scatter(sp.L*1e3, sp.R_f*1e3, marker=marker1, color='k', label='Forewings')
            ax[0].scatter(sp.m*1e3, 1.205*sp.R_h*sp.S_h*1e3, marker=marker2, color='k')
            p3 = ax[1].scatter(sp.L*1e3, sp.R_h*1e3, marker=marker2, color='k', label='Hindwings')
            if fore_scatter_handle is None:
                fore_scatter_handle = p1
            if hind_scatter_handle is None:
                hind_scatter_handle = p3
        else:
            print(sp.species)

    samples = collect_fit_samples(specimens)
    trends = fit_wakeling_trends(samples)

    m_mod = np.linspace(0, 1.2, 11)
    ax[0].plot(m_mod, mair(m_mod, *trends['mu0_fore']), '--k', label='Forewings')
    ax[0].text(0.15, 0.075, r'$\mu_0^\text{fore} = %.3f$' % trends['mu0_fore'][0])

    ax[0].plot(m_mod, mair(m_mod, *trends['mu0_hind']), ':k', label='Hindwings')
    ax[0].text(0.15, 0.06, r'$\mu_0^\text{hind} = %.3f$' % trends['mu0_hind'][0])

    Lb_mod = np.linspace(0, 100, 11)
    p2 = ax[1].plot(Lb_mod, R(Lb_mod, *trends['lambda0_fore']), '--k', label='Forewings')
    ax[1].text(45, 52, r'$\lambda_0^\text{fore} = %.2f$' % trends['lambda0_fore'][0])

    p4 = ax[1].plot(Lb_mod, R(Lb_mod, *trends['lambda0_hind']), ':k', label='Hindwings')
    ax[1].text(45, 47.5, r'$\lambda_0^\text{hind} = %.2f$' % trends['lambda0_hind'][0])
    
    fore_line_handle = p2[0]
    hind_line_handle = p4[0]
    if fore_scatter_handle is not None and hind_scatter_handle is not None:
        fig.legend(
            [(fore_scatter_handle, fore_line_handle), (hind_scatter_handle, hind_line_handle)],
            ['Forewings', 'Hindwings'],
            loc='upper center',
            bbox_to_anchor=(0.5, 0),
            fontsize=12,
            ncol=2,
        )
    else:
        fig.legend(
            [fore_line_handle, hind_line_handle],
            ['Forewings', 'Hindwings'],
            loc='upper center',
            bbox_to_anchor=(0.5, 0),
            fontsize=12,
            ncol=2,
        )

    ax[0].set_xlabel(r'$m$ (g)')
    ax[0].set_ylabel(r'$\rho_\text{air}SR$ (g)')
    ax[1].set_xlabel(r'$L_b$ (mm)')
    ax[1].set_ylabel(r'$R$ (mm)')
    ax[0].set_xlim(0, 1.25)
    ax[0].set_ylim(0, 0.1)
    ax[1].set_xlim(40, 85)
    ax[1].set_ylim(30, 60)
    plt.tight_layout()
    
    fig.savefig('wakeling1997_data_new.pdf', dpi=600, bbox_inches='tight')

def main():
    specimens = read_wakeling1997('forewing_data.csv',
                                  'hindwing_data.csv',
                                  'body_data.csv')
    plot_adim_params(specimens)
    plot_adim_params_new(specimens)


if __name__ == "__main__":
    main()
