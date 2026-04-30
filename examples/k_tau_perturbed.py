"""
Perturbed-prey k_tilt vs sensing-delay sweep.

For each (k_tilt, tau) pair, runs N independent interception attempts with
the prey start position perturbed around a nominal point, and records the
closest approach for every (cell, perturbation) pair. The full distribution
is saved so any summary statistic (percentile, mean, P(success), ...) can
be recomputed downstream. The default plot uses the 90th percentile.

Parallelism: multiprocessing pool over the flat (k, tau, perturbation)
task list. Each worker runs one `run_case`. Perturbed prey positions are
shared across all (k, tau) cells so that controller-parameter variation is
not confounded with ensemble noise.

Output layout:
    data/k_tau/<nominal_prey_tag>/perturbed_sigma<sigma>/sweep.npz
                                                        /sweep.png
"""

import os
# Pin each worker to a single BLAS thread so the workers don't compete for
# the same cores. Must be set before numpy is imported.
for _var in (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import multiprocessing as mp
import time

import numpy as np

from examples.k_tau_sweep import plot_sweep, prey_tag
from examples.parametric import BASELINE, run_case


DATA_ROOT = "data/k_tau"


def sample_perturbations(
    nominal: np.ndarray, sigma: float, n: int, seed: int = 0,
) -> np.ndarray:
    """Gaussian perturbations of (x, z); y component held at nominal."""
    rng = np.random.default_rng(seed)
    out = np.tile(np.asarray(nominal, dtype=float), (n, 1))
    out[:, 0] += rng.normal(0.0, sigma, size=n)
    out[:, 2] += rng.normal(0.0, sigma, size=n)
    return out


def _worker(args):
    i, j, p, k, tau, prey, t_end = args
    r = run_case(
        "perturbed",
        k_tilt=k, sensing_delay=tau, t_end=t_end,
        prey_position=np.asarray(prey, dtype=float),
    )
    return i, j, p, float(r["min_dist"])


def run_perturbed_sweep(
    ks: np.ndarray,
    taus: np.ndarray,
    perturbed_preys: np.ndarray,
    t_end: float,
    n_workers: int | None = None,
    progress_every: int = 200,
):
    n_k, n_tau, n_p = len(ks), len(taus), len(perturbed_preys)
    tasks = [
        (i, j, p,
         float(ks[i]), float(taus[j]), perturbed_preys[p], float(t_end))
        for i in range(n_k)
        for j in range(n_tau)
        for p in range(n_p)
    ]
    total = len(tasks)
    print(f"  dispatching {total} runs across {n_workers or os.cpu_count()} workers")

    min_dist = np.zeros((n_k, n_tau, n_p))
    t0 = time.perf_counter()
    ctx = mp.get_context("fork")
    with ctx.Pool(n_workers) as pool:
        for c, (i, j, p, d) in enumerate(
            pool.imap_unordered(_worker, tasks, chunksize=8), start=1,
        ):
            min_dist[i, j, p] = d
            if c % progress_every == 0 or c == total:
                elapsed = time.perf_counter() - t0
                rate = c / elapsed
                eta = (total - c) / rate if rate > 0 else 0.0
                print(f"  [{c:5d}/{total}]  {rate:5.1f} runs/s  elapsed {elapsed:6.1f}s  ETA {eta:6.1f}s")
    return min_dist


def run_and_save(
    prey_nominal: np.ndarray,
    ks: np.ndarray,
    taus: np.ndarray,
    sigma: float,
    n_perturb: int,
    t_end: float,
    T_wb: float,
    seed: int = 0,
    n_workers: int | None = None,
):
    nominal_tag = prey_tag(prey_nominal)
    subdir = f"perturbed_sigma{sigma:g}_n{n_perturb}"
    outdir = os.path.join(DATA_ROOT, nominal_tag, subdir)
    os.makedirs(outdir, exist_ok=True)

    perturbed_preys = sample_perturbations(prey_nominal, sigma, n_perturb, seed=seed)

    print(f"[{nominal_tag}/{subdir}] starting")
    min_dist = run_perturbed_sweep(
        ks, taus, perturbed_preys, t_end=t_end, n_workers=n_workers,
    )

    npz_path = os.path.join(outdir, "sweep.npz")
    np.savez(
        npz_path,
        ks=ks, taus=taus,
        min_dist=min_dist,                     # (n_k, n_tau, n_perturb)
        perturbed_preys=perturbed_preys,       # (n_perturb, 3)
        prey_nominal=np.asarray(prey_nominal, dtype=float),
        sigma=float(sigma),
        seed=int(seed),
        T_wb=float(T_wb), t_end=float(t_end),
    )
    print(f"[{nominal_tag}/{subdir}] saved {npz_path}")

    # Default plot: 90th percentile over the perturbation axis.
    stat = np.percentile(min_dist, 90, axis=2)
    title = (
        fr"prey $(x, z) = ({prey_nominal[0]:g}, {prey_nominal[2]:g})$ | "
        fr"$\sigma = {sigma:g}$, $N = {n_perturb}$ | 90th percentile $d_{{\min}}$"
    )
    png_path = os.path.join(outdir, "sweep.png")
    plot_sweep(ks, taus, stat, T_wb, savepath=png_path, title=title)
    print(f"[{nominal_tag}/{subdir}] saved {png_path}")

    return min_dist, perturbed_preys


if __name__ == "__main__":
    wf = BASELINE["wing_frequency"]
    T_wb = 1.0 / wf

    ks   = np.arange(0.0, 5.0 + 1e-9, 0.5)
    taus = np.arange(0.0, 3.0 * T_wb + 1e-9, T_wb / 4.0)
    t_end = 10.0

    # Validation settings (small N for a quick smoke test).
    sigma = 0.15
    n_perturb = 8

    run_and_save(
        prey_nominal=BASELINE["prey_position"],
        ks=ks, taus=taus,
        sigma=sigma, n_perturb=n_perturb,
        t_end=t_end, T_wb=T_wb,
    )
