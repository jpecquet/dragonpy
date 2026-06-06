"""
plot_estimator(t, xest, Pest, x_true, plot_type="error", z=None):

This function plots a scalar state estimate and the +/- 2-sigma bounds 
from a filter along with the true value (optional) and measurement (optional)

    Note: assumes scalar state

INPUTS
    t = time vector (1 x n)
    xest = vector of state estimates (1 x n)
    Pest = vector of error variance of estimator (1 x n)
    x_true = vector of true state (1 x n)
    plot_type = 'state':  plots xest +/- 2*sigma, x_true
                'error':  plots err +/- 2*sigma
    z = measurement vector (1 x n) (optional)

MAE 6760 Model Based Estimation
Cornell University
M Campbell        
"""
import numpy as np
import matplotlib.pyplot as plt

MCcolors = {
    "blue":   np.array([4, 51, 255]) / 255,
    "purple": np.array([147, 23, 255]) / 255,
    "green":  np.array([0, 160, 0]) / 255,
    "red":    np.array([200, 0, 0]) / 255,
    "mag":    np.array([255, 64, 255]) / 255,
}

def plot_estimator(t, xest, Pest, x_true, plot_type="error", z=None, ax=None, legend=True):

    if t.ndim != 1:
        print("inputs must be 1D arrays")
        return

    if abs((t[1] - t[0]) - 1) < 1e-10:
        time_label = r"$k$"
    else:
        time_label = r"$t$ (s)"

    bd = np.squeeze(Pest)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 4))

    # z = np.asarray(z).ravel() # makes the msmt into a 1D vector - suggest changing in the future

    if plot_type == "state":

        upper = xest + 2 * np.sqrt(bd)
        lower = xest - 2 * np.sqrt(bd)

        ax.plot(t, xest, color=MCcolors["blue"],
                 label="state estimate")

        ax.fill_between(t, upper, lower,
                         color=MCcolors["blue"],
                         alpha=0.1,
                         label=r"estimate $\pm 2\sigma$")

        ax.plot(t, x_true, "-.", color=MCcolors["mag"],
                 label="true state")

        if z is not None:
            ax.plot(t, z, ".", color=MCcolors["green"],
                     markersize=2,
                     label="measurement")

        ax.set_ylabel(r"state estimate $\hat{x}(t)$")

    elif plot_type == "error":

        err = xest - x_true
        bound = 2 * np.sqrt(bd)

        ax.fill_between(t, bound, -bound,
                         color=MCcolors["blue"],
                         alpha=0.1,
                         label=r"$\pm 2\sigma$ bounds")

        if z is not None:
            ax.plot(t, z - x_true, ".",
                     color=MCcolors["red"],
                     markersize=2,
                     label="msmt noise")

        ax.plot(t, err, color=MCcolors["purple"],
                 label="error estimate")

        ax.set_ylabel(r"error estimate $e(t)$")

    ax.grid(True)
    ax.set_xlabel(time_label)
    ax.set_xlim(t[0], t[-1])

    if legend:
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=3,
                  fancybox=True, shadow=True, borderaxespad=0.)
