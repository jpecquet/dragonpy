# Wing Kinematics

## Model of wing motion

Dragonflies are equipped with four wings arranged in pairs and attached to the thorax. When wing motion is observed from the side, the wingtips appear to be mostly constrained to a straight line (Fig. 1A). Generalizing this to three dimensions, the wingtips appear to be bound to a plane called the stroke plane (Fig. 1B). The orientation of the stroke plane is defined by the angle $\gamma$ which can vary during flight. Higher values of $\gamma$ (closer to the vertical) are generally associated with higher flight velocities, since the net aerodynamic force vector is roughly normal to the stroke plane. For this reason it is convenient to define $\gamma$ relative to the horizontal axis, not the body axis.

<!--
Fig. 1A and 1B side to side
-->

```{raw} html
<div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(280px, 1fr)); gap:0.75rem; align-items:start; margin-bottom:1.5rem;">
  <div>
    <video
      class="case-study-video"
      style="width:100%;"
      loop
      autoplay
      muted
      playsinline
      preload="metadata"
      data-light-src="../_static/media/modeling/wing_kinematics/wing_kinematics_section1_anim1_side_y.light.mp4"
      data-dark-src="../_static/media/modeling/wing_kinematics/wing_kinematics_section1_anim1_side_y.dark.mp4"
    >
      <source src="../_static/media/modeling/wing_kinematics/wing_kinematics_section1_anim1_side_y.dark.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    <div style="font-size:0.85em; line-height:1.2; margin-top:0.3rem; text-align:center;">Fig. 1A. Wingtip stroke lines.</div>
  </div>
  <div>
    <video
      class="case-study-video"
      style="width:100%;"
      loop
      autoplay
      muted
      playsinline
      preload="metadata"
      data-light-src="../_static/media/modeling/wing_kinematics/wing_kinematics_section1_anim1_p111_stroke_planes.light.mp4"
      data-dark-src="../_static/media/modeling/wing_kinematics/wing_kinematics_section1_anim1_p111_stroke_planes.dark.mp4"
    >
      <source src="../_static/media/modeling/wing_kinematics/wing_kinematics_section1_anim1_p111_stroke_planes.dark.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    <div style="font-size:0.85em; line-height:1.2; margin-top:0.3rem; text-align:center;">Fig. 1B. Stroke planes.</div>
  </div>
</div>
```

If the wingtip is bound to a plane, it follows from simple geometry that the line extending from the wing root joint to the wingtip must be bound to a cone whose top vertex coincides with the root joint, and whose axis is normal to the stroke plane. A *coning angle* $\beta$ {cite}`azuma1988` can be defined from the stroke plane to the wing cone. $\beta$ generally has a small positive value for the forewings, and a small negative value for the hindwings (Fig. 2A). The flapping of the wings within the wing cone is described by the flapping angle $\phi$ (Fig. 2B). For $\gamma = 0$, $\phi > 0$ means the wings are raised above the horizontal plane.

```{raw} html
<div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(280px, 1fr)); gap:0.75rem; align-items:start; margin-bottom:1.5rem;">
  <div>
    <video
      class="case-study-video"
      style="width:100%;"
      loop
      autoplay
      muted
      playsinline
      preload="metadata"
      data-light-src="../_static/media/modeling/wing_kinematics/wing_kinematics_section1_anim1_side_y_cones.light.mp4"
      data-dark-src="../_static/media/modeling/wing_kinematics/wing_kinematics_section1_anim1_side_y_cones.dark.mp4"
    >
      <source src="../_static/media/modeling/wing_kinematics/wing_kinematics_section1_anim1_side_y_cones.dark.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    <div style="font-size:0.85em; line-height:1.2; margin-top:0.3rem; text-align:center;">Fig. 2A. Wing coning angles.</div>
  </div>
  <div>
    <video
      class="case-study-video"
      style="width:100%;"
      loop
      autoplay
      muted
      playsinline
      preload="metadata"
      data-light-src="../_static/media/modeling/wing_kinematics/wing_kinematics_section1_anim1_fore_left_cone_normal.light.mp4"
      data-dark-src="../_static/media/modeling/wing_kinematics/wing_kinematics_section1_anim1_fore_left_cone_normal.dark.mp4"
    >
      <source src="../_static/media/modeling/wing_kinematics/wing_kinematics_section1_anim1_fore_left_cone_normal.dark.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    <div style="font-size:0.85em; line-height:1.2; margin-top:0.3rem; text-align:center;">Fig. 2B. Wing flapping angle.</div>
  </div>
</div>
```

<!--
Fig. 2A and 2B side to side
-->

In addition to flapping, the wing motion has a pitching component. Here we take the approach of {cite}`azuma1988` and assume that the root-to-tip line introduced to describe flapping is also the pitching axis. Following {cite}`azuma1988`, we further assume that it corresponds to the quarter-chord line (the quarter-chord point at each spanwise location being the chordwise aerodynamic center). Pitching about this spanwise axis is described by the pitching angle $\psi$. $\psi = 0$ means the wing chord axis is normal to the wing cone surface. $\psi > 0$ is pitched up and $\psi < 0$ is pitched down. A classical way to visualize wing pitch, and wing motion in general, is the stick plot (Fig. 3), showing the wing crossection as a line segment with a small circle at the leading edge. Here sticks are shown for the wingtips, but they could in principle be shown for any spanwise station. Note that this plot is not strictly a 2D projection: the wing pitch orientation is shown as if viewed from the wingtip, looking down the spanwise direction.

```{raw} html
<div style="margin-bottom:1.5rem;">
  <video
    class="case-study-video"
    style="width:100%;"
    loop
    autoplay
    muted
    playsinline
    preload="metadata"
    data-light-src="../_static/media/modeling/wing_kinematics/wing_kinematics_section1_anim1_stick.light.mp4"
    data-dark-src="../_static/media/modeling/wing_kinematics/wing_kinematics_section1_anim1_stick.dark.mp4"
  >
    <source src="../_static/media/modeling/wing_kinematics/wing_kinematics_section1_anim1_stick.dark.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
  <div style="font-size:0.85em; line-height:1.2; margin-top:0.3rem; text-align:center;">Fig. 3. Stick plot.</div>
</div>
```

## Wing angle representation

In the model, the motion of each wing can be described by four angular quantities: the stroke plane angle $\gamma(t)$, the coning angle $\beta(t)$, the flapping angle $\phi(t)$, and the pitching angle $\psi(t)$. Two waveform options are available for $\phi$ and $\psi$.

### Fourier series

The default representation is a harmonic series of the form

$$
\theta(t) = \theta_0 + \sum_{k=1}^{N} \theta_k \cos{(k \omega t + \delta_k)}
$$

The stroke plane angle is simply $\gamma(t) = \gamma_0$. The coning angle may be represented by $\beta(t) = \beta_0$, but in some cases, the wingtip does not follow the stroke plane exactly. Instead it draws a narrow ellipse whose major axis is the stroke plane as viewed from the side {cite}`azuma1985` {cite}`azuma1988`. In this case, a single-harmonic ($N=1$) representation $\beta(t) = \beta_0 + \beta_1 \cos{(\omega t + \delta_1)}$ is appropriate, with the wing cone widening on the downstroke and narrowing on the upstroke. Higher-order representations may also be appropriate to capture more irregular motion in the stroke plane normal direction.

The flapping angle $\phi(t)$ has been found to be well represented by a single harmonic. The time evolution of the pitching angle $\psi(t)$ is somewhat more complex, and 3 to 4 harmonics may be required to adequately capture it {cite}`azuma1985` {cite}`azuma1988`.

### Berman waveforms

{cite}`berman2007` introduce alternative functional forms for $\phi$ and $\psi$ that span a wider range of waveform shapes with a single parameter. For the flapping angle, a smoothed triangular waveform is used:

$$
\phi(t) = \phi_0 + \frac{\phi_1}{\sin^{-1} K_\phi} \sin^{-1}\!\left[K_\phi \cos(\omega t + \delta_1)\right], \quad K_\phi \in (0, 1)
$$

As $K_\phi \to 0$ the waveform approaches a cosine; as $K_\phi \to 1$ it approaches a triangular wave. For the pitching angle, a hyperbolic tangent waveform is used:

$$
\psi(t) = \psi_0 + \frac{\psi_1}{\tanh K_\psi} \tanh\!\left[K_\psi \cos(\omega t + \delta_1)\right], \quad K_\psi > 0
$$

As $K_\psi \to 0$ the waveform approaches a cosine; as $K_\psi \to \infty$ it approaches a square wave, corresponding to an instantaneous pitch reversal. These waveforms are selected in the config with `phi_waveform = berman` and `psi_waveform = berman`, with the shape parameters specified as `phi_k` and `psi_k` respectively. The amplitude $\phi_1$ (or $\psi_1$) and phase $\delta_1$ are taken from the first harmonic of the corresponding angle series.

## References

```{bibliography}
:filter: docname in docnames
```
