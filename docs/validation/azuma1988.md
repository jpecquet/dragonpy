# Azuma and Watanabe 1988

## Case and data description

The goal of this case study is to validate the simulation against forward-flight wing kinematics reported in {cite}`azuma1988` for four flight conditions (experiments 1-4). Experiments 1-3 correspond to Dragonfly A and experiment 4 corresponds to Dragonfly B. As in the Azuma 1985 case, each simulation here is run as a tethered one-wingbeat simulation with the body velocity fixed to the experimental flight condition.

Morphology values are listed separately for Dragonfly A and Dragonfly B to match the experiment grouping used on this page. In the current extracted case inputs, the same morphology values are used for both specimens.

```{raw} html
<div style="font-size:0.85em; line-height:1.2; margin-bottom:0.3rem; margin-top:1.5rem; text-align:center;">Table 1. Dragonfly A specimen morphology (experiments 1-3).</div>
```

| Quantity | Value |
|--|------:|
| Body length $L$ (mm) | 75 |
| Body mass $m$ (mg) | 790 |
| Forewing span $R_f$ (mm) | 50.0 |
| Forewing area $S_f$ (mm²) | 500 |
| Hindwing span $R_h$ (mm) | 48.5 |
| Hindwing area $S_h$ (mm²) | 600 |

```{raw} html
<div style="font-size:0.85em; line-height:1.2; margin-bottom:0.3rem; margin-top:1.5rem; text-align:center;">Table 2. Dragonfly B specimen morphology (experiment 4).</div>
```

| Quantity | Value |
|--|------:|
| Body length $L$ (mm) | 75 |
| Body mass $m$ (mg) | 790 |
| Forewing span $R_f$ (mm) | 50.0 |
| Forewing area $S_f$ (mm²) | 500 |
| Hindwing span $R_h$ (mm) | 48.5 |
| Hindwing area $S_h$ (mm²) | 600 |

```{raw} html
<div style="font-size:0.85em; line-height:1.2; margin-bottom:0.3rem; margin-top:1.5rem; text-align:center;">Table 3. Experiment conditions and tethered simulation inputs.</div>
```

| Exp. | Specimen | $f$ (Hz) | $\gamma_f$ (deg) | $\gamma_h$ (deg) | $U$ (m/s) | $\theta$ (deg) | $U^*$ | $u_x^*$ | $u_z^*$ |
|--:|:--:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | A | 26.5 | 40 | 38 | 0.7 | -12.0 | 0.816 | 0.798 | -0.170 |
| 2 | A | 28.1 | 55 | 48 | 1.5 | -1.1 | 1.749 | 1.748 | -0.034 |
| 3 | A | 29.0 | 58 | 52 | 2.3 | 4.8 | 2.681 | 2.672 | 0.224 |
| 4 | B | 27.0 | 63 | 68 | 3.2 | 0.0 | 3.731 | 3.731 | 0.000 |

The coning angles are fixed across all four experiments at $\beta_f = 8^\circ$ and $\beta_h = -2^\circ$.

## Wing kinematics

Fourier harmonic coefficients from the paper are mapped to the simulator convention using

$$
\begin{aligned}
\psi^{\text{(paper)}}(t) &= -\phi(t) \\
\theta^{\text{(paper)}}(t) &= \psi(t) + 90^\circ
\end{aligned}
$$

The plotted kinematics below show the simulator-convention flapping angle $\phi(t)$ and pitch angle $\psi(t)$ for each experiment.

## Experiment Pages

The per-experiment results are split into sub-pages:

```{toctree}
:maxdepth: 1

Experiment 1 <azuma1988/exp1>
Experiment 2 <azuma1988/exp2>
Experiment 3 <azuma1988/exp3>
Experiment 4 <azuma1988/exp4>
```

## References

```{bibliography}
:filter: docname in docnames
```
