# Pursuit Problem

## Motivation

We develop a control scheme so that our model dragonfly can intercept a moving target. This is a dual-mode control scheme with a hover phase, and an interception phase. The dragonfly starts at rest at the origin in a hovering equilibrium and must detect, pursue, and come to within a short distance of a moving target. This combines two control problems: maintaining a stable hover while scanning for targets, and switching to an aggressive pursuit mode upon detection.

## Setup

### Pursuit Controller

The pursuit controller follows the same philosophy as the model developped for the pursuit of prey by tiger beetles in {cite}`haselsteiner2014` and {cite}`noest2017`. Upon detection of the target, the controller switches from the hover wing kinematics to kinematics that allow high-speed flight, with the only variable parameter being $\gamma$. The fixed parameters are a high flapping amplitude $\phi_1 = 35°$, $\psi_0 = 0°$ for a symmetric wing stroke, a low pitch amplitude $\psi_1 = 20°$, and $\delta_0 = 90^\circ$.

$\gamma_0$ is modulated via a rate controller that drives the flight path angle $\theta_v = \text{atan2}(u_z, u_x)$ toward the line-of-sight angle $\theta_r = \text{atan2}(r_z, r_x)$:

$$\dot{\gamma}_0 = K_p \left(\theta_v - \theta_r\right)$$

When the velocity is directed above the line-of-sight ($\theta_v > \theta_r$), $\gamma_0$ increases to redirect the flight path forward. When the velocity is below the line-of-sight, $\gamma_0$ decreases to steer the dragonfly upward. The rate formulation allows $\gamma$ to integrate toward whatever value balances the aerodynamic forces for the current flight condition, rather than relying on a pre-calibrated mapping from error angle to stroke plane angle.

### Target Detection and Interception

Target detection uses an angular field-of-view model: the target must lie within a $60°$ half-cone centered on the forward ($+x$) body axis. Interception is declared when the distance drops below $0.5\,\tilde{L}$. After interception, the controller returns to hover mode and the simulation runs for 30 additional wingbeats before ending.

## Results

### Run 1: Horizontal Target

```{raw} html
<div style="margin-bottom:1.5rem;">
  <video
    class="case-study-video"
    loop
    autoplay
    muted
    playsinline
    preload="metadata"
  >
    <source src="../_static/media/pursuit/pursuit_animation.dark.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
  <div style="font-size:0.85em; line-height:1.2; margin-top:0.3rem; text-align:center;">Fig. 1. Animation for Run 1.</div>
</div>
```

```{image} ../_static/media/pursuit/pursuit_control.dark.png
:align: center
:width: 80%
```
<div style="font-size:0.85em; line-height:1.2; margin-top:0.3rem; text-align:center; margin-bottom:1.5rem;">Fig. 2. Controller state for Run 1.</div>

### Run 2: Descending Target

```{raw} html
<div style="margin-bottom:1.5rem;">
  <video
    class="case-study-video"
    loop
    autoplay
    muted
    playsinline
    preload="metadata"
  >
    <source src="../_static/media/pursuit/pursuit_descending.dark.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
  <div style="font-size:0.85em; line-height:1.2; margin-top:0.3rem; text-align:center;">Fig. 3. Animation for Run 2.</div>
</div>
```

```{image} ../_static/media/pursuit/pursuit_descending_control.dark.png
:align: center
:width: 80%
```
<div style="font-size:0.85em; line-height:1.2; margin-top:0.3rem; text-align:center; margin-bottom:1.5rem;">Fig. 4. Controller state for Run 2.</div>

### Run 3: Fast Pursuit

Same target as Run 1, but with a higher pursuit flapping amplitude $\phi_1 = 45°$ (vs. $35°$ in Runs 1–2) to achieve higher flight speed.

```{raw} html
<div style="margin-bottom:1.5rem;">
  <video
    class="case-study-video"
    loop
    autoplay
    muted
    playsinline
    preload="metadata"
  >
    <source src="../_static/media/pursuit/pursuit_fast.dark.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
  <div style="font-size:0.85em; line-height:1.2; margin-top:0.3rem; text-align:center;">Fig. 5. Fast pursuit with higher flapping amplitude.</div>
</div>
```

```{image} ../_static/media/pursuit/pursuit_fast_control.dark.png
:align: center
:width: 80%
```
<div style="font-size:0.85em; line-height:1.2; margin-top:0.3rem; text-align:center; margin-bottom:1.5rem;">Fig. 6. Controller state for Run 3.</div>

### Run 4: Fast Target

Same configuration as Run 3 ($\phi_1 = 45°$), but the target moves at $2\,\tilde{v}$ along $x$.

```{raw} html
<div style="margin-bottom:1.5rem;">
  <video
    class="case-study-video"
    loop
    autoplay
    muted
    playsinline
    preload="metadata"
  >
    <source src="../_static/media/pursuit/pursuit_fast_2x.dark.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
  <div style="font-size:0.85em; line-height:1.2; margin-top:0.3rem; text-align:center;">Fig. 7. Pursuit of a fast-moving target.</div>
</div>
```

```{image} ../_static/media/pursuit/pursuit_fast_2x_control.dark.png
:align: center
:width: 80%
```
<div style="font-size:0.85em; line-height:1.2; margin-top:0.3rem; text-align:center; margin-bottom:1.5rem;">Fig. 8. Controller state for Run 4.</div>

### Run 5: Faster Target

Same configuration as Run 3 ($\phi_1 = 45°$), but the target moves at $4\,\tilde{v}$ along $x$.

```{raw} html
<div style="margin-bottom:1.5rem;">
  <video
    class="case-study-video"
    loop
    autoplay
    muted
    playsinline
    preload="metadata"
  >
    <source src="../_static/media/pursuit/pursuit_fast_4x.dark.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
  <div style="font-size:0.85em; line-height:1.2; margin-top:0.3rem; text-align:center;">Fig. 9. Pursuit of a fast target ($4\,\tilde{v}$).</div>
</div>
```

```{image} ../_static/media/pursuit/pursuit_fast_4x_control.dark.png
:align: center
:width: 80%
```
<div style="font-size:0.85em; line-height:1.2; margin-top:0.3rem; text-align:center; margin-bottom:1.5rem;">Fig. 10. Controller state for Run 5.</div>

## References

```{bibliography}
:filter: docname in docnames
```
