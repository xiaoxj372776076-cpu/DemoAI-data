#!/usr/bin/env python3
"""Build and render the first review scene for the Blender scene skill."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import sys

import bpy
from mathutils import Vector


def parse_args() -> argparse.Namespace:
    arguments = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(arguments)


def srgb(hex_color: str) -> tuple[float, float, float, float]:
    value = hex_color.lstrip("#")
    channels = [int(value[index : index + 2], 16) / 255 for index in (0, 2, 4)]
    linear = [
        channel / 12.92
        if channel <= 0.04045
        else ((channel + 0.055) / 1.055) ** 2.4
        for channel in channels
    ]
    return tuple(linear) + (1.0,)


def reset_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in (
        bpy.data.meshes,
        bpy.data.curves,
        bpy.data.materials,
        bpy.data.cameras,
        bpy.data.lights,
    ):
        for block in list(collection):
            if block.users == 0:
                collection.remove(block)


def make_wood_material(
    name: str,
    light_color: str,
    dark_color: str,
    grain_scale: float,
    roughness: float,
    mapping_scale: tuple[float, float, float] = (0.34, 4.8, 4.8),
) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    texture = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    noise = nodes.new("ShaderNodeTexNoise")
    color_ramp = nodes.new("ShaderNodeValToRGB")
    bump = nodes.new("ShaderNodeBump")

    texture.location = (-900, 0)
    mapping.location = (-720, 0)
    noise.location = (-510, 20)
    color_ramp.location = (-270, 90)
    bump.location = (-250, -150)
    shader.location = (20, 60)
    output.location = (280, 60)

    mapping.inputs["Scale"].default_value = mapping_scale
    noise.inputs["Scale"].default_value = grain_scale
    noise.inputs["Detail"].default_value = 5.0
    noise.inputs["Roughness"].default_value = 0.64
    noise.inputs["Distortion"].default_value = 0.45

    color_ramp.color_ramp.elements[0].position = 0.22
    color_ramp.color_ramp.elements[0].color = srgb(dark_color)
    color_ramp.color_ramp.elements[1].position = 0.78
    color_ramp.color_ramp.elements[1].color = srgb(light_color)
    shader.inputs["Roughness"].default_value = roughness
    shader.inputs["IOR"].default_value = 1.46
    bump.inputs["Strength"].default_value = 0.07
    bump.inputs["Distance"].default_value = 0.025

    links.new(texture.outputs["Generated"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], noise.inputs["Vector"])
    links.new(noise.outputs["Fac"], color_ramp.inputs["Fac"])
    links.new(color_ramp.outputs["Color"], shader.inputs["Base Color"])
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], shader.inputs["Normal"])
    links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    return material


def make_simple_material(
    name: str,
    color: str,
    roughness: float,
) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.diffuse_color = srgb(color)
    material.use_nodes = True
    shader = material.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = srgb(color)
    shader.inputs["Roughness"].default_value = roughness
    return material


def beveled_cube(
    name: str,
    dimensions: tuple[float, float, float],
    location: tuple[float, float, float],
    material: bpy.types.Material,
    bevel: float,
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(location=location, rotation=rotation)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = dimensions
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    modifier = obj.modifiers.new(name="Softened edges", type="BEVEL")
    modifier.width = bevel
    modifier.segments = 5
    modifier.limit_method = "ANGLE"
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    obj.data.materials.append(material)
    return obj


def look_at(obj: bpy.types.Object, target: tuple[float, float, float]) -> None:
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def add_area_light(
    name: str,
    location: tuple[float, float, float],
    target: tuple[float, float, float],
    color: tuple[float, float, float],
    energy: float,
    size: float,
) -> bpy.types.Object:
    light_data = bpy.data.lights.new(name=name, type="AREA")
    light_data.energy = energy
    light_data.color = color
    light_data.shape = "DISK"
    light_data.size = size
    light = bpy.data.objects.new(name, light_data)
    bpy.context.collection.objects.link(light)
    light.location = location
    look_at(light, target)
    return light


def add_table() -> None:
    top_materials = [
        make_wood_material("Oak 01", "#B97B43", "#58301B", 3.4, 0.36),
        make_wood_material("Oak 02", "#C1874F", "#633A22", 3.7, 0.34),
        make_wood_material("Oak 03", "#A96B3D", "#4A2817", 3.2, 0.38),
    ]
    frame_material = make_wood_material(
        "Dark oak frame", "#85512F", "#382014", 3.3, 0.42
    )
    leg_material = make_wood_material(
        "Vertical dark oak", "#85512F", "#382014", 3.3, 0.42,
        mapping_scale=(4.8, 4.8, 0.34),
    )

    top_length = 3.8
    top_width = 2.1
    top_height = 1.75
    plank_count = 6
    gap = 0.008
    plank_width = (top_width - gap * (plank_count - 1)) / plank_count
    first_y = -top_width / 2 + plank_width / 2
    for index in range(plank_count):
        y = first_y + index * (plank_width + gap)
        plank = beveled_cube(
            name=f"Tabletop plank {index + 1:02d}",
            dimensions=(top_length, plank_width, 0.18),
            location=(0.0, y, top_height),
            material=top_materials[index % len(top_materials)],
            bevel=0.028,
        )
        plank["part"] = "wooden tabletop"

    leg_x = 1.52
    leg_y = 0.73
    leg_height = 1.58
    leg_z = leg_height / 2
    for x_sign in (-1, 1):
        for y_sign in (-1, 1):
            leg = beveled_cube(
                name=f"Leg {'R' if x_sign > 0 else 'L'}{'F' if y_sign < 0 else 'B'}",
                dimensions=(0.22, 0.22, leg_height),
                location=(x_sign * leg_x, y_sign * leg_y, leg_z),
                material=leg_material,
                bevel=0.045,
                rotation=(math.radians(y_sign * 1.6), math.radians(-x_sign * 1.6), 0.0),
            )
            leg["part"] = "table leg"

    apron_z = 1.54
    beveled_cube(
        "Front apron", (3.15, 0.12, 0.28), (0.0, -0.78, apron_z), frame_material, 0.035
    )
    beveled_cube(
        "Back apron", (3.15, 0.12, 0.28), (0.0, 0.78, apron_z), frame_material, 0.035
    )
    beveled_cube(
        "Left apron", (0.12, 1.42, 0.28), (-1.57, 0.0, apron_z), frame_material, 0.035
    )
    beveled_cube(
        "Right apron", (0.12, 1.42, 0.28), (1.57, 0.0, apron_z), frame_material, 0.035
    )


def add_studio() -> None:
    floor_material = make_simple_material("Warm gray floor", "#77736B", 0.72)
    wall_material = make_simple_material("Soft backdrop", "#3D4144", 0.82)
    beveled_cube(
        "Floor",
        dimensions=(30.0, 30.0, 0.08),
        location=(0.0, 0.0, -0.08),
        material=floor_material,
        bevel=0.02,
    )
    beveled_cube(
        "Backdrop",
        dimensions=(30.0, 0.08, 9.0),
        location=(0.0, 4.8, 3.42),
        material=wall_material,
        bevel=0.02,
    )


def add_camera_and_lighting(scene: bpy.types.Scene) -> None:
    camera_data = bpy.data.cameras.new("Right-front camera")
    camera = bpy.data.objects.new("Right-front camera", camera_data)
    bpy.context.collection.objects.link(camera)
    camera.location = (5.0, -5.8, 3.7)
    camera_data.lens = 54
    camera_data.sensor_width = 36
    look_at(camera, (0.0, 0.0, 0.92))
    scene.camera = camera

    add_area_light(
        "Warm key",
        location=(3.4, -3.6, 5.8),
        target=(0.0, 0.0, 0.95),
        color=(1.0, 0.47, 0.22),
        energy=980,
        size=3.2,
    )
    add_area_light(
        "Soft fill",
        location=(-3.8, -1.2, 3.5),
        target=(0.0, 0.0, 1.0),
        color=(0.54, 0.65, 0.78),
        energy=260,
        size=4.5,
    )
    add_area_light(
        "Edge light",
        location=(-1.0, 3.5, 4.6),
        target=(0.0, 0.0, 1.1),
        color=(1.0, 0.74, 0.48),
        energy=420,
        size=2.4,
    )

    world = bpy.data.worlds.new("Studio world")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (
        0.025,
        0.03,
        0.035,
        1.0,
    )
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.22
    scene.world = world


def configure_render(scene: bpy.types.Scene, output_path: Path) -> None:
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 960
    scene.render.resolution_y = 720
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.filepath = str(output_path)
    scene.render.film_transparent = False
    scene.render.image_settings.color_depth = "8"
    scene.render.use_file_extension = True
    scene.view_settings.look = "AgX - Medium High Contrast"


def save_artifacts(args: argparse.Namespace, output_dir: Path) -> None:
    prompt_path = output_dir / "prompt.txt"
    prompt_path.write_text(args.prompt.strip() + "\n", encoding="utf-8")
    source_path = Path(__file__).resolve()
    archived_path = output_dir / "scene.py"
    if source_path != archived_path:
        shutil.copy2(source_path, archived_path)
    manifest = {
        "prompt": args.prompt,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "blender_version": bpy.app.version_string,
        "files": {
            "scene": "wooden_table.blend",
            "render": "wooden_table.png",
            "prompt": "prompt.txt",
            "code": "scene.py",
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    render_path = output_dir / "wooden_table.png"
    blend_path = output_dir / "wooden_table.blend"

    reset_scene()
    add_studio()
    add_table()
    scene = bpy.context.scene
    add_camera_and_lighting(scene)
    configure_render(scene, render_path)
    save_artifacts(args, output_dir)
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    bpy.ops.render.render(write_still=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    print(f"BLEND_FILE={blend_path}")
    print(f"RENDER_FILE={render_path}")


if __name__ == "__main__":
    main()
