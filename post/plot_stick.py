"""
Stick-plot visualization of right forewing/hindwing motion in the (X,Z) plane.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani

from post.animation import save_animation
from post.style import apply_matplotlib_style, figure_size, resolve_style

STICK_LENGTH = 0.1
WING_ROOT_DISTANCE = 0.1
WING_ROOT_MIDPOINT_Z = 0.04
WING_ROOT_TILT_DEG = 15.0
STICK_FPS = 20
STICK_BITRATE = 4000
STICK_DPI = 200

SILHOUETTE_PATH = Path(__file__).resolve().parent.parent / "assets" / "dragonfly_silhouette.csv"


def wing_root_positions(distance=WING_ROOT_DISTANCE, midpoint_z=WING_ROOT_MIDPOINT_Z, tilt_deg=WING_ROOT_TILT_DEG):
    """Compute fore and hind wing root (x, z) from distance, midpoint z, and tilt angle."""
    half = 0.5 * float(distance)
    tilt = np.radians(float(tilt_deg))
    dx, dz = half * np.cos(tilt), half * np.sin(tilt)
    fore_root = np.array([dx, midpoint_z + dz], dtype=float)
    hind_root = np.array([-dx, midpoint_z - dz], dtype=float)
    return fore_root, hind_root


def load_silhouette():
    """Load the processed dragonfly silhouette as (N, 2) array of (x, z) coords."""
    return np.loadtxt(SILHOUETTE_PATH, delimiter=",")


def _stroke_plane_normal_xz(gamma):
    """Return the stroke-plane normal projected into the (X, Z) stick-plot plane."""
    return np.array([np.sin(float(gamma)), np.cos(float(gamma))], dtype=float)


def _mean_stroke_plane_segment_xz(root_offset, lambda0, gamma, beta_mean, station=1.0):
    """Return endpoints of the selected-station constant-beta center-path line in XZ."""
    gamma = float(gamma)
    beta_mean = float(beta_mean)
    lam = float(lambda0) * float(station)
    root = np.asarray(root_offset, dtype=float)
    d = np.array([-np.cos(gamma), np.sin(gamma)], dtype=float)
    n = np.array([np.sin(gamma), np.cos(gamma)], dtype=float)
    p0 = root + (lam * np.sin(beta_mean)) * n - (lam * np.cos(beta_mean)) * d
    p1 = root + (lam * np.sin(beta_mean)) * n + (lam * np.cos(beta_mean)) * d
    return p0, p1


def _clip_segment_to_projected_path_extent(segment, path_points):
    """Clip a line segment to the min/max projection of a path onto that line."""
    if segment is None:
        return None
    p0, p1 = (np.asarray(segment[0], dtype=float), np.asarray(segment[1], dtype=float))
    v = p1 - p0
    v_norm = float(np.linalg.norm(v))
    if v_norm < 1e-12:
        return segment
    d_hat = v / v_norm
    center = 0.5 * (p0 + p1)
    pts = np.asarray(path_points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] == 0:
        return segment
    s = (pts - center) @ d_hat
    return center + np.min(s) * d_hat, center + np.max(s) * d_hat


def _draw_pitch_glyph_2d(ax, center, chord_dir_2d, stroke_normal_2d, label_text,
                          *, color, alpha, line_width, fontsize,
                          ref_length=0.12, arc_radius=0.10):
    """Draw a pitch-angle arc glyph in 2D and return the created artists."""
    artists = []
    cn = float(np.linalg.norm(chord_dir_2d))
    sn = float(np.linalg.norm(stroke_normal_2d))
    if cn < 1e-12 or sn < 1e-12:
        return artists
    c_hat = np.asarray(chord_dir_2d, dtype=float) / cn
    s_hat = np.asarray(stroke_normal_2d, dtype=float) / sn

    # Reference lines from center.
    ref_end = center + ref_length * s_hat
    chord_end = center + ref_length * c_hat
    artists.append(ax.plot(
        [center[0], ref_end[0]], [center[1], ref_end[1]],
        "--", color=color, linewidth=line_width * 0.7, alpha=alpha,
    )[0])
    artists.append(ax.plot(
        [center[0], chord_end[0]], [center[1], chord_end[1]],
        "--", color=color, linewidth=line_width * 0.7, alpha=alpha,
    )[0])

    # Signed angle from stroke normal to chord direction.
    angle_ref = float(np.arctan2(s_hat[1], s_hat[0]))
    angle_chord = float(np.arctan2(c_hat[1], c_hat[0]))
    delta = angle_chord - angle_ref
    delta = (delta + np.pi) % (2.0 * np.pi) - np.pi

    if abs(delta) < 1e-6:
        # Angle ≈ 0 — just show the label near the reference end.
        artists.append(ax.text(
            float(ref_end[0]), float(ref_end[1]), label_text,
            color=color, fontsize=fontsize, ha="center", va="center",
        ))
        return artists

    sign = 1.0 if delta >= 0.0 else -1.0

    # Arrowhead sizing: fixed chord-length of 0.014, derive angular gap
    # so the triangle always bridges exactly from arc end to head tip.
    max_head_chord = 0.014
    head_angle = 2.0 * float(np.arcsin(np.clip(
        max_head_chord / (2.0 * arc_radius), 0.0, 1.0,
    )))
    head_angle = min(head_angle, 0.45 * abs(delta))
    arc_end_angle = delta - sign * head_angle
    if sign * arc_end_angle <= 0.0:
        arc_end_angle = 0.6 * delta

    theta = np.linspace(0.0, arc_end_angle, 48) + angle_ref
    arc_x = center[0] + arc_radius * np.cos(theta)
    arc_y = center[1] + arc_radius * np.sin(theta)
    artists.append(ax.plot(arc_x, arc_y, "-", color=color, linewidth=line_width, alpha=alpha)[0])

    # Arrowhead triangle — base is the arc end, tip is the angle endpoint.
    head_tip_angle = angle_ref + delta
    head_tip = center + arc_radius * np.array([np.cos(head_tip_angle), np.sin(head_tip_angle)])
    head_base = np.array([arc_x[-1], arc_y[-1]], dtype=float)
    head_vec = head_tip - head_base
    head_len = float(np.linalg.norm(head_vec))
    if head_len > 1e-12:
        t_hat = head_vec / head_len
        n_hat = np.array([-t_hat[1], t_hat[0]], dtype=float)
        head_w = 0.25 * min(head_len, max_head_chord)
        tri_x = [float(head_tip[0]), float(head_base[0] + head_w * n_hat[0]), float(head_base[0] - head_w * n_hat[0])]
        tri_y = [float(head_tip[1]), float(head_base[1] + head_w * n_hat[1]), float(head_base[1] - head_w * n_hat[1])]
        artists.append(ax.fill(tri_x, tri_y, color=color, alpha=alpha)[0])

    # Label at midpoint of arc, offset outward.
    mid_angle = angle_ref + 0.5 * delta
    label_r = arc_radius + 0.045
    label_pos = center + label_r * np.array([np.cos(mid_angle), np.sin(mid_angle)])
    artists.append(ax.text(
        float(label_pos[0]), float(label_pos[1]), label_text,
        color=color, fontsize=fontsize, ha="center", va="center",
    ))
    return artists


def project_xz(vec):
    return np.array([vec[0], vec[2]], dtype=float)


def compute_stick_endpoints(
    span_dir,
    chord_dir,
    root_offset,
    stick_length,
    station,
    lambda0,
):
    """Compute stick center, leading edge, and trailing edge in nondimensional (X,Z).

    span_dir:   (3,) spanwise unit vector in body frame (R_body_wing column 0).
    chord_dir:  (3,) chord vector toward leading edge in body frame.
    """
    if not 0.0 <= float(station) <= 1.0:
        raise ValueError("station must be in [0, 1]")
    span_xz = project_xz(span_dir)
    center = float(station) * float(lambda0) * span_xz + np.asarray(root_offset, dtype=float)

    chord_xz = project_xz(chord_dir)
    chord_norm = np.linalg.norm(chord_xz)
    if chord_norm < 1e-9:
        chord_hat = np.array([1.0, 0.0], dtype=float)
    else:
        chord_hat = chord_xz / chord_norm

    length = float(stick_length)
    leading = center + 0.25 * length * chord_hat
    trailing = center - 0.75 * length * chord_hat
    return center, leading, trailing


def animate_stroke(
    time,
    wings,
    fore_wing_name,
    hind_wing_name,
    outfile,
    omega,
    style=None,
    stations=(2.0 / 3.0,),
    fore_lambda0=1.0,
    hind_lambda0=1.0,
    show_axes=True,
    show_grid=True,
    show_timestamp=True,
    show_pitch_angle=False,
    params=None,
    stroke_plane_beta_mode="mean",
):
    """Create animation of right fore/hind stick motion in the nondimensional (X,Z) plane."""
    style = resolve_style(style)
    apply_matplotlib_style(style)
    omega = float(omega)
    time_scale = omega / (2.0 * np.pi)

    fig, ax = plt.subplots(figsize=figure_size(0.6 if show_axes else 0.4))
    stations = tuple(float(s) for s in stations)
    n_frames = len(time)
    n_stations = len(stations)
    trace_station = max(stations) if stations else (2.0 / 3.0)
    trace_station_idx = int(np.argmax(np.asarray(stations, dtype=float))) if n_stations > 0 else 0
    fore_centers = np.zeros((n_stations, n_frames, 2), dtype=float)
    hind_centers = np.zeros((n_stations, n_frames, 2), dtype=float)
    fore_leading = np.zeros((n_stations, n_frames, 2), dtype=float)
    fore_trailing = np.zeros((n_stations, n_frames, 2), dtype=float)
    hind_leading = np.zeros((n_stations, n_frames, 2), dtype=float)
    hind_trailing = np.zeros((n_stations, n_frames, 2), dtype=float)

    fore_wing = wings[fore_wing_name]
    hind_wing = wings[hind_wing_name]
    fore_root, hind_root = wing_root_positions()
    fore_beta_mean = None
    hind_beta_mean = None
    fore_mean_plane_segment = None
    hind_mean_plane_segment = None
    beta_mode = str(stroke_plane_beta_mode).strip().lower()
    if beta_mode not in {"mean", "actual"}:
        raise ValueError(f"stroke_plane_beta_mode must be 'mean' or 'actual' (got {stroke_plane_beta_mode!r})")
    if params is not None and beta_mode == "mean":
        gamma_map = params.get("wing_gamma_mean", {})
        cone_mean_map = params.get("wing_cone_mean", {})
        cone_static_map = params.get("wing_cone_angle", {})

        def _resolve_total_beta_mean(wing_name):
            beta = 0.0
            have_beta = False
            if wing_name in cone_static_map:
                beta += float(cone_static_map[wing_name])
                have_beta = True
            if wing_name in cone_mean_map:
                beta += float(cone_mean_map[wing_name])
                have_beta = True
            return beta if have_beta else None

        fore_beta_mean = _resolve_total_beta_mean(fore_wing_name)
        hind_beta_mean = _resolve_total_beta_mean(hind_wing_name)
        fore_gamma_mean = gamma_map.get(fore_wing_name)
        hind_gamma_mean = gamma_map.get(hind_wing_name)
        if fore_gamma_mean is not None and fore_beta_mean is not None:
            fore_mean_plane_segment = _mean_stroke_plane_segment_xz(
                fore_root, fore_lambda0, fore_gamma_mean, fore_beta_mean, trace_station
            )
        if hind_gamma_mean is not None and hind_beta_mean is not None:
            hind_mean_plane_segment = _mean_stroke_plane_segment_xz(
                hind_root, hind_lambda0, hind_gamma_mean, hind_beta_mean, trace_station
            )

    for station_idx, station in enumerate(stations):
        for i in range(n_frames):
            fore_center, fore_le, fore_te = compute_stick_endpoints(
                fore_wing["e_r"][i],
                fore_wing["e_c"][i],
                root_offset=fore_root,
                stick_length=STICK_LENGTH,
                station=station,
                lambda0=fore_lambda0,
            )
            hind_center, hind_le, hind_te = compute_stick_endpoints(
                hind_wing["e_r"][i],
                hind_wing["e_c"][i],
                root_offset=hind_root,
                stick_length=STICK_LENGTH,
                station=station,
                lambda0=hind_lambda0,
            )
            fore_centers[station_idx, i] = fore_center
            hind_centers[station_idx, i] = hind_center
            fore_leading[station_idx, i] = fore_le
            fore_trailing[station_idx, i] = fore_te
            hind_leading[station_idx, i] = hind_le
            hind_trailing[station_idx, i] = hind_te

    if beta_mode == "mean":
        if fore_mean_plane_segment is not None and 0 <= trace_station_idx < n_stations:
            fore_mean_plane_segment = _clip_segment_to_projected_path_extent(
                fore_mean_plane_segment, fore_centers[trace_station_idx]
            )
        if hind_mean_plane_segment is not None and 0 <= trace_station_idx < n_stations:
            hind_mean_plane_segment = _clip_segment_to_projected_path_extent(
                hind_mean_plane_segment, hind_centers[trace_station_idx]
            )

    all_centers = np.vstack([fore_centers.reshape(-1, 2), hind_centers.reshape(-1, 2)])
    half_len = 0.5 * STICK_LENGTH
    pad = 0.08
    ax.set_xlim([
        float(np.min(all_centers[:, 0]) - half_len - pad),
        float(np.max(all_centers[:, 0]) + half_len + pad),
    ])
    if show_axes:
        ax.set_xlim([-1, 0.6])
        ax.set_ylim([-0.8, 0.8])
    else:
        all_pts = np.vstack([
            fore_leading.reshape(-1, 2), fore_trailing.reshape(-1, 2),
            hind_leading.reshape(-1, 2), hind_trailing.reshape(-1, 2),
        ])
        if SILHOUETTE_PATH.exists():
            all_pts = np.vstack([all_pts, load_silhouette()])
        if show_pitch_angle:
            # Include pitch-glyph extent (label_r = arc_radius + 0.045).
            glyph_r = 0.145
            si = n_stations - 1
            glyph_centers = np.vstack([fore_centers[si], hind_centers[si]])
            offsets = np.array([[glyph_r, 0], [-glyph_r, 0], [0, glyph_r], [0, -glyph_r]], dtype=float)
            glyph_pts = glyph_centers[:, None, :] + offsets[None, :, :]
            all_pts = np.vstack([all_pts, glyph_pts.reshape(-1, 2)])
        ax.set_xlim([float(np.min(all_pts[:, 0]) - pad), float(np.max(all_pts[:, 0]) + pad)])
        ax.set_ylim([float(np.min(all_pts[:, 1]) - pad), float(np.max(all_pts[:, 1]) + pad)])
    ax.set_aspect("equal")
    if show_axes:
        ax.set_xlabel(r"$\tilde{X}$")
        ax.set_ylabel(r"$\tilde{Z}$")
    else:
        ax.set_axis_off()
    if show_grid:
        ax.grid(True, alpha=0.3)

    if SILHOUETTE_PATH.exists():
        silhouette = load_silhouette()
        ax.plot(
            silhouette[:, 0],
            silhouette[:, 1],
            "-",
            color=style.muted_text_color,
            linewidth=0.5,
        )

    for root in (fore_root, hind_root):
        ax.plot(root[0], root[1], ".", color=style.muted_text_color, markersize=4)

    if beta_mode == "actual":
        for station_idx in range(n_stations):
            ax.plot(
                fore_centers[station_idx, :, 0],
                fore_centers[station_idx, :, 1],
                "-",
                color=style.muted_text_color,
                linewidth=0.5,
            )
            ax.plot(
                hind_centers[station_idx, :, 0],
                hind_centers[station_idx, :, 1],
                "-",
                color=style.muted_text_color,
                linewidth=0.5,
            )
    else:
        for seg in (fore_mean_plane_segment, hind_mean_plane_segment):
            if seg is None:
                continue
            p0, p1 = seg
            ax.plot(
                [float(p0[0]), float(p1[0])],
                [float(p0[1]), float(p1[1])],
                "-",
                color=style.muted_text_color,
                linewidth=0.7,
            )

    le_marker = dict(
        marker="o",
        markersize=4,
        markerfacecolor=style.axes_facecolor,
        markeredgewidth=1.8,
        linestyle="none",
    )

    fore_sticks = [ax.plot([], [], "-", color=style.body_color, linewidth=2.2)[0] for _ in range(n_stations)]
    hind_sticks = [ax.plot([], [], "-", color=style.body_color, linewidth=2.2)[0] for _ in range(n_stations)]
    fore_le_markers = [ax.plot([], [], markeredgecolor=style.body_color, **le_marker)[0] for _ in range(n_stations)]
    hind_le_markers = [ax.plot([], [], markeredgecolor=style.body_color, **le_marker)[0] for _ in range(n_stations)]
    time_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left", color=style.text_color) if show_timestamp else None

    # Pitch-angle glyph state.
    pitch_glyph_cfg = None
    if show_pitch_angle and params is not None:
        gamma_map = params.get("wing_gamma_mean", {})
        fore_gamma = gamma_map.get(fore_wing_name)
        hind_gamma = gamma_map.get(hind_wing_name)
        if fore_gamma is not None and hind_gamma is not None:
            pitch_glyph_cfg = {
                "fore_normal": _stroke_plane_normal_xz(fore_gamma),
                "hind_normal": _stroke_plane_normal_xz(hind_gamma),
                "station_idx": n_stations - 1,
            }
    _pitch_artists = []

    fig.tight_layout()

    def update(frame):
        nonlocal _pitch_artists
        # Remove previous pitch-angle artists.
        for a in _pitch_artists:
            a.remove()
        _pitch_artists.clear()

        artists = []
        for station_idx in range(n_stations):
            fore_sticks[station_idx].set_data(
                [fore_trailing[station_idx, frame, 0], fore_leading[station_idx, frame, 0]],
                [fore_trailing[station_idx, frame, 1], fore_leading[station_idx, frame, 1]],
            )
            hind_sticks[station_idx].set_data(
                [hind_trailing[station_idx, frame, 0], hind_leading[station_idx, frame, 0]],
                [hind_trailing[station_idx, frame, 1], hind_leading[station_idx, frame, 1]],
            )
            fore_le_markers[station_idx].set_data(
                [fore_leading[station_idx, frame, 0]],
                [fore_leading[station_idx, frame, 1]],
            )
            hind_le_markers[station_idx].set_data(
                [hind_leading[station_idx, frame, 0]],
                [hind_leading[station_idx, frame, 1]],
            )
            artists.extend(
                [
                    fore_sticks[station_idx],
                    hind_sticks[station_idx],
                    fore_le_markers[station_idx],
                    hind_le_markers[station_idx],
                ]
            )
        if time_text is not None:
            time_text.set_text(r"$t/T_{wb} = %.2f$" % (time[frame] * time_scale))
            artists.append(time_text)

        if pitch_glyph_cfg is not None:
            si = pitch_glyph_cfg["station_idx"]
            fore_chord = fore_leading[si, frame] - fore_trailing[si, frame]
            _pitch_artists.extend(_draw_pitch_glyph_2d(
                ax, fore_centers[si, frame], fore_chord,
                pitch_glyph_cfg["fore_normal"],
                r"$\psi_f$",
                color=style.text_color, alpha=0.9, line_width=1.0,
                fontsize=style.font_size,
            ))
            hind_chord = hind_leading[si, frame] - hind_trailing[si, frame]
            _pitch_artists.extend(_draw_pitch_glyph_2d(
                ax, hind_centers[si, frame], hind_chord,
                pitch_glyph_cfg["hind_normal"],
                r"$\psi_h$",
                color=style.text_color, alpha=0.9, line_width=1.0,
                fontsize=style.font_size,
            ))
            artists.extend(_pitch_artists)

        return tuple(artists)

    # Create animation
    anim = ani.FuncAnimation(fig, update, frames=n_frames, interval=50, blit=False)

    save_animation(
        anim,
        outfile,
        fps=STICK_FPS,
        bitrate=STICK_BITRATE,
        dpi=STICK_DPI,
        progress_label="Matplotlib",
    )

    plt.close()
    print(f"Saved: {outfile}")


