"""Build a runnable MuJoCo scene for Vega + f5d6 tracking.

Pipeline (mirrors the proven approach in dexmate-vega-sim/scene_builder.py):
  1. Parse the Vega-1U f5d6 URDF and drop every visual/collision geom that
     references a `.glb` mesh — MuJoCo 3.x has no glb decoder. The `.obj`
     collision meshes that remain are enough for physics *and* visualisation.
  2. Compile the stripped URDF with MuJoCo, save it back out as MJCF, and
     rewrite mesh `file=` paths to absolute so the scene loads from anywhere.
  3. Wrap that body tree in a full scene: floor, table, a manipulable object
     with a free joint, position actuators for the controlled joints, lights,
     and a camera.

The result is written to `assets_gen/vega_f5d6_scene.xml` and returned as a
string. Everything downstream (env, rewards) keys off joint/body *names*, which
are stable across this transform.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

from . import config as C


def _strip_glb(urdf_path: Path) -> str:
    """Return URDF XML text with all glb-referencing geoms removed."""
    root = ET.parse(urdf_path).getroot()

    def uses_glb(elem) -> bool:
        return any(m.get("filename", "").lower().endswith(".glb")
                   for m in elem.iter("mesh"))

    for link in root.iter("link"):
        for tag in ("visual", "collision"):
            for geom in link.findall(tag):
                if uses_glb(geom):
                    link.remove(geom)
    return ET.tostring(root, encoding="unicode")


def _compile_to_mjcf(urdf_path: Path) -> str:
    """Compile URDF (glb stripped) and return MJCF text with absolute meshes."""
    stripped = _strip_glb(urdf_path)
    # MuJoCo resolves relative mesh paths against the model file's directory,
    # so we must write the stripped URDF *beside* the original.
    tmp = urdf_path.with_name("_dextrack_stripped.urdf")
    tmp.write_text(stripped)
    try:
        model = mujoco.MjModel.from_xml_path(str(tmp))
        raw_xml = tmp.with_name("_dextrack_raw.xml")
        mujoco.mj_saveLastXML(str(raw_xml), model)
        mjcf = raw_xml.read_text()
        raw_xml.unlink()
    finally:
        tmp.unlink()

    urdf_dir = urdf_path.parent

    def absolutize(match: re.Match) -> str:
        rel = match.group(1)
        if rel.startswith("/"):
            return match.group(0)
        return f'file="{(urdf_dir / rel).resolve()}"'

    return re.sub(r'file="([^"]+)"', absolutize, mjcf)


def _extract_blocks(mjcf: str) -> tuple[str, str, str]:
    """Pull <asset>, <worldbody>, and <default> inner text from compiled MJCF."""
    def inner(tag: str) -> str:
        m = re.search(rf"<{tag}>(.*?)</{tag}>", mjcf, re.S)
        return m.group(1).strip() if m else ""
    return inner("asset"), inner("worldbody"), inner("default")


def _actuator_block(sides: list[str]) -> str:
    """Position actuators for arm + hand joints of the given side(s)."""
    lines = []
    for side in sides:
        for j in C.ARM_JOINTS[side]:
            lines.append(f'    <position name="act_{j}" joint="{j}" '
                         f'kp="{C.ARM_KP}" kv="{C.ARM_KV}"/>')
        for j in C.HAND_JOINTS[side]:
            lines.append(f'    <position name="act_{j}" joint="{j}" '
                         f'kp="{C.HAND_KP}" kv="{C.HAND_KV}"/>')
    # Keep posture joints held at home with stiff position actuators.
    for j in C.POSTURE_JOINTS:
        lines.append(f'    <position name="act_{j}" joint="{j}" '
                     f'kp="{C.POSTURE_KP}" kv="{C.POSTURE_KV}"/>')
    return "\n".join(lines)


def _object_block(
    obj_type: str,
    obj_dims: tuple[float, ...],
    obj_pos: tuple[float, float, float],
    obj_mass: float = 0.08,
    obj_friction: str = "2.0 0.05 0.002",
) -> str:
    """A free-floating object to be manipulated/tracked.

    obj_type/obj_dims:
      "box"      -> (half_x, half_y, half_z)
      "cylinder" -> (radius, half_height)   # wrappable for a power grasp/lift
      "sphere"   -> (radius,)
    """
    x, y, z = obj_pos
    size = " ".join(str(d) for d in obj_dims)
    return f"""
    <body name="object" pos="{x} {y} {z}">
      <freejoint name="object_free"/>
      <geom name="object_geom" type="{obj_type}" size="{size}"
            rgba="0.8 0.3 0.2 1" mass="{obj_mass}" friction="{obj_friction}"/>
      <site name="object_site" pos="0 0 0" size="0.005"/>
    </body>"""


def build_scene(
    sides: list[str] | None = None,
    table_z: float = 0.75,
    obj_size: float = 0.03,
    obj_pos: tuple[float, float, float] = (0.6, -0.15, 0.80),
    obj_type: str = "box",
    obj_dims: tuple[float, ...] | None = None,
    obj_mass: float = 0.08,
    obj_friction: str = "2.0 0.05 0.002",
    write: bool = True,
) -> str:
    """Assemble the full scene MJCF. `sides` = which arms get actuators.

    obj_type/obj_dims select the manipulable object (see `_object_block`).
    Defaults to the box (obj_dims falls back to a cube of `obj_size`).
    """
    sides = sides or ["R"]
    if obj_dims is None:
        obj_dims = (obj_size, obj_size, obj_size)
    asset, worldbody, default = _extract_blocks(_compile_to_mjcf(C.VEGA_F5D6_URDF))

    scene = f"""<mujoco model="vega_f5d6_tracking">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{C.SIM_DT}" integrator="implicitfast" cone="elliptic"/>

  <default>
{_indent(default, 4)}
  </default>

  <asset>
{_indent(asset, 4)}
    <texture name="grid" type="2d" builtin="checker" rgb1="0.2 0.3 0.4"
             rgb2="0.1 0.15 0.2" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="6 6" reflectance="0.1"/>
  </asset>

  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" size="5 5 0.1" material="grid"/>
    <body name="table" pos="0.3 0 {table_z}">
      <geom name="table_top" type="box" size="0.55 0.4 0.02" rgba="0.6 0.5 0.4 1"/>
    </body>
    <camera name="track" pos="1.4 -1.0 1.6" xyaxes="0.7 0.7 0 -0.4 0.4 0.8"/>
{_indent(worldbody, 4)}
{_object_block(obj_type, obj_dims, obj_pos, obj_mass, obj_friction)}
  </worldbody>

  <actuator>
{_actuator_block(sides)}
  </actuator>
</mujoco>
"""
    if write:
        C.GENERATED_SCENE.parent.mkdir(parents=True, exist_ok=True)
        C.GENERATED_SCENE.write_text(scene)
    return scene


def _indent(text: str, n: int) -> str:
    pad = " " * n
    return "\n".join(pad + ln if ln.strip() else ln for ln in text.splitlines())


if __name__ == "__main__":
    build_scene()
    model = mujoco.MjModel.from_xml_path(str(C.GENERATED_SCENE))
    print(f"built {C.GENERATED_SCENE}")
    print(f"  nq={model.nq} nv={model.nv} nu={model.nu} nbody={model.nbody}")
