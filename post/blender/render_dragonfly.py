#!/usr/bin/env python3
"""
Blender script: render dragonfly body + wings to PNG frames.

Invoked as:
    blender --background --python post/blender/render_dragonfly.py -- \
        --data frame_data.json --config config.json --output-dir /tmp/frames \
        [--start 0] [--end -1]

Self-contained — all geometry helpers are inlined so this script has no
dependencies beyond bpy and numpy.
"""

import argparse
import json
import math
import sys
from pathlib import Path

try:
    import bpy
    import bmesh
    from mathutils import Matrix, Vector
except ImportError:
    pass

import numpy as np

# ---------------------------------------------------------------------------
# Inlined geometry helpers
# ---------------------------------------------------------------------------

_WING_ROOT_DISTANCE = 0.1
_WING_ROOT_MIDPOINT_Z = 0.04
_WING_ROOT_TILT_DEG = 15.0

_half = 0.5 * _WING_ROOT_DISTANCE
_tilt = math.radians(_WING_ROOT_TILT_DEG)
_FW_X0 = _half * math.cos(_tilt)
_HW_X0 = -_half * math.cos(_tilt)
_FW_Z0 = _WING_ROOT_MIDPOINT_Z + _half * math.sin(_tilt)
_HW_Z0 = _WING_ROOT_MIDPOINT_Z - _half * math.sin(_tilt)

_RIGHT_Y = -0.04
_LEFT_Y = 0.04

_DEFAULT_LB0 = 0.75
_DEFAULT_ASPECT_RATIO = 5
_DEFAULT_ROOT_CHORD_RATIO = 4.0 / (math.pi * _DEFAULT_ASPECT_RATIO)


def _wing_offsets(wname):
    xo = _FW_X0 if "fore" in wname else _HW_X0
    zo = _FW_Z0 if "fore" in wname else _HW_Z0
    yo = _RIGHT_Y if "right" in wname else _LEFT_Y
    return [xo, yo, zo]


def _wing_polygon_local(span, root_chord, n=32):
    eta = np.linspace(0.0, 1.0, n + 1)
    chord_scale = np.sqrt(np.clip(1.0 - (2.0 * eta - 1.0) ** 2, 0.0, None))
    x_le = 0.25 * root_chord * chord_scale
    x_te = -0.75 * root_chord * chord_scale
    y = -span * eta  # tip at -Y (Blender convention)
    leading = np.column_stack([x_le, y, np.zeros_like(y)])
    trailing = np.column_stack([x_te[-2::-1], y[-2::-1], np.zeros_like(y[:-1])])
    return np.vstack([leading, trailing])


# ---------------------------------------------------------------------------
# Blender helpers
# ---------------------------------------------------------------------------

def _hex_to_rgba(h, alpha=1.0):
    h = h.lstrip("#")
    return (int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255, alpha)


def setup_scene(width, height, bg_color):
    scene = bpy.context.scene
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    transparent = bg_color is None
    scene.render.film_transparent = transparent
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA" if transparent else "RGB"

    scene.display.shading.light = "STUDIO"
    scene.display.shading.studio_light = "Default"
    scene.display.shading.color_type = "MATERIAL"
    if not transparent:
        scene.display.shading.background_type = "VIEWPORT"
        scene.display.shading.background_color = bg_color[:3]

    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = False


def setup_camera(elevation, azimuth, center, ortho_scale, width, height,
                 shift_x=0.0, shift_y=0.0):
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = ortho_scale
    cam_data.shift_x = shift_x
    cam_data.shift_y = shift_y
    cam_data.clip_start = 0.1
    cam_data.clip_end = ortho_scale * 9

    cam_obj = bpy.data.objects.new("Camera", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    elev = math.radians(elevation)
    azim = math.radians(azimuth)
    dist = ortho_scale * 3

    x = dist * math.cos(elev) * math.cos(azim)
    y = dist * math.cos(elev) * math.sin(azim)
    z = dist * math.sin(elev)

    center_vec = Vector(center)
    cam_pos = center_vec + Vector((x, y, z))
    cam_obj.location = cam_pos

    direction = center_vec - cam_pos
    rot_quat = direction.to_track_quat("-Z", "Y")
    cam_obj.rotation_euler = rot_quat.to_euler()
    return cam_obj


def setup_lighting():
    key_data = bpy.data.lights.new("KeyLight", "SUN")
    key_data.energy = 3.0
    key_obj = bpy.data.objects.new("KeyLight", key_data)
    bpy.context.scene.collection.objects.link(key_obj)
    key_obj.rotation_euler = (math.radians(45), 0, math.radians(-45))

    fill_data = bpy.data.lights.new("FillLight", "SUN")
    fill_data.energy = 1.5
    fill_obj = bpy.data.objects.new("FillLight", fill_data)
    bpy.context.scene.collection.objects.link(fill_obj)
    fill_obj.rotation_euler = (math.radians(30), 0, math.radians(135))


def load_body_mesh(filepath, body_color):
    bpy.ops.wm.obj_import(filepath=str(filepath))
    body = bpy.context.selected_objects[0]
    body.name = "Body"

    bpy.context.view_layer.objects.active = body
    saved_cursor = bpy.context.scene.cursor.location.copy()
    bpy.context.scene.cursor.location = (0, 0, 0)
    bpy.ops.object.origin_set(type="ORIGIN_CURSOR")
    bpy.context.scene.cursor.location = saved_cursor

    mat = bpy.data.materials.new(name="DragonBody")
    mat.diffuse_color = body_color
    body.data.materials.clear()
    body.data.materials.append(mat)
    return body


def create_wing_mesh(name, span, root_chord, wing_color):
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)

    poly = _wing_polygon_local(span, root_chord, n=32)
    bm = bmesh.new()
    verts = [bm.verts.new((float(v[0]), float(v[1]), float(v[2]))) for v in poly]
    bm.verts.ensure_lookup_table()
    bm.faces.new(verts)
    bm.to_mesh(mesh)
    bm.free()

    mat = bpy.data.materials.new(name=f"{name}_mat")
    mat.diffuse_color = wing_color
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    return obj


def transform_wing(wing_obj, origin, e_r, e_c):
    e_r = np.array(e_r, dtype=float)
    e_c = np.array(e_c, dtype=float)
    e_n = np.cross(e_c, e_r)
    rotation = np.column_stack([e_c, e_r, e_n])

    rot_mat = Matrix([
        [rotation[0, 0], rotation[0, 1], rotation[0, 2], 0],
        [rotation[1, 0], rotation[1, 1], rotation[1, 2], 0],
        [rotation[2, 0], rotation[2, 1], rotation[2, 2], 0],
        [0, 0, 0, 1],
    ])
    mirror_mat = Matrix([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])
    trans_mat = Matrix.Translation(Vector(origin))
    wing_obj.matrix_world = trans_mat @ rot_mat @ mirror_mat


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    args = parser.parse_args(argv)

    with open(args.config) as f:
        config = json.load(f)
    with open(args.data) as f:
        frame_data = json.load(f)

    n_frames = frame_data["n_frames"]
    states = frame_data["states"]
    wing_names = frame_data["wing_names"]
    wing_vectors = frame_data["wing_vectors"]
    wing_lb0 = frame_data.get("wing_lb0", {})

    start = args.start
    end = args.end if args.end >= 0 else n_frames

    # Config
    camera = config.get("camera", {})
    elevation = camera.get("elevation", 30.0)
    azimuth = camera.get("azimuth", -60.0)
    ortho_scale = camera.get("ortho_scale", 3.0)
    shift_x = camera.get("shift_x", 0.0)
    shift_y = camera.get("shift_y", 0.0)
    center = config.get("center", [0.0, 0.0, 0.0])
    width = config.get("width", 1950)
    height = config.get("height", 1300)
    raw_bg = config.get("bg_color")
    bg_color = None if raw_bg is None else _hex_to_rgba(raw_bg)
    body_color = _hex_to_rgba(config.get("body_color", "#f2f5f7"))
    wing_color = _hex_to_rgba(config.get("wing_color", "#8f9aa6"))

    # Scene setup
    setup_scene(width, height, bg_color)
    setup_camera(elevation, azimuth, center, ortho_scale, width, height,
                 shift_x=shift_x, shift_y=shift_y)
    setup_lighting()

    # Body mesh
    assets_dir = Path(__file__).resolve().parent.parent.parent / "assets"
    body_obj = load_body_mesh(assets_dir / "dragonfly_body.obj", body_color)

    # Wing meshes
    wing_info = {}
    wing_objects = {}
    for wname in wing_names:
        offset = _wing_offsets(wname)
        lb0 = float(wing_lb0.get(wname, _DEFAULT_LB0))
        root_chord = _DEFAULT_ROOT_CHORD_RATIO * lb0
        wing_info[wname] = {"offset": offset, "span": lb0}
        wing_objects[wname] = create_wing_mesh(wname, lb0, root_chord, wing_color)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for fi in range(start, end):
        body_pos = states[fi][0:3]
        body_obj.location = Vector(body_pos)

        for wname, wing_obj in wing_objects.items():
            info = wing_info[wname]
            origin = [body_pos[j] + info["offset"][j] for j in range(3)]
            e_r = wing_vectors[wname]["e_r"][fi]
            e_c = wing_vectors[wname]["e_c"][fi]
            transform_wing(wing_obj, origin, e_r, e_c)

        filepath = str(output_dir / f"frame_{fi:06d}.png")
        bpy.context.scene.render.filepath = filepath
        bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()
