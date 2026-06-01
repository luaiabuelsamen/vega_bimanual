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


def _convert_cylinders_to_boxes(mjcf: str) -> str:
    """Rewrite cylinder geoms as boxes (MJX has no cylinder<->box collision).

    The URDF has two cylinder wrist-collision geoms on {L,R}_arm_l7; approximate
    each (radius r, half-height h) by a box of half-extents (r, r, h). Applies to
    both scene builds; the wrist volume barely changes.
    """
    root = ET.fromstring(mjcf)
    n = 0
    for geom in root.iter("geom"):
        if geom.get("type") == "cylinder":
            size = geom.get("size", "").split()
            if len(size) == 2:                       # (radius, half_height)
                r, h = size
                geom.set("type", "box")
                geom.set("size", f"{r} {r} {h}")
                n += 1
    if n:
        mjcf = ET.tostring(root, encoding="unicode")
    return mjcf


def _simplify_collision(mjcf: str, sides: list[str], ftip_radius: float = 0.008) -> str:
    """Replace the robot's mesh collision with primitives for MJX.

    MJX's all-pairs convex-mesh collision makes the ~40-mesh hand compile for
    minutes; this collapses it to ~10 fingertip contact spheres so it compiles
    in seconds. Each fingertip distal-link mesh becomes a small sphere; every
    other mesh geom is deleted (link mass comes from the URDF <inertial>, so it's
    safe) and remaining primitives have collision disabled. Joints also get
    armature + damping so the stiff arm stays stable without the mesh contact
    that used to damp it. Behind build_scene(collision="primitive"); the default
    mesh scene is untouched.
    """
    ftips = {b for s in sides for b in C.FINGERTIP_BODIES[s]}
    root = ET.fromstring(mjcf)
    worldbody = root.find("worldbody")
    if worldbody is None:
        return mjcf

    def walk(body):
        name = body.get("name", "")
        is_ftip = name in ftips
        # armature (reflected actuator inertia) + damping keep the stiff arm
        # stable under exploration once the mesh contact that damped it is gone.
        for joint in body.findall("joint"):
            arm = "_arm_" in joint.get("name", "")
            joint.set("armature", "0.1" if arm else "0.01")
            joint.set("damping", "1.0")
        for geom in list(body.findall("geom")):
            if is_ftip and geom.get("type") == "mesh":
                geom.set("type", "sphere"); geom.set("size", f"{ftip_radius}")
                geom.attrib.pop("mesh", None)
                geom.set("contype", "1"); geom.set("conaffinity", "1")
                geom.set("friction", "2.0 0.05 0.002")
            elif geom.get("type") == "mesh":
                body.remove(geom)            # delete: mesh data bloats MJX compile
            else:
                geom.set("contype", "0"); geom.set("conaffinity", "0")
        for child in body.findall("body"):
            walk(child)

    for b in worldbody.findall("body"):
        walk(b)

    # prune now-unreferenced <mesh> assets so MJX never uploads them.
    asset = root.find("asset")
    if asset is not None:
        used = {g.get("mesh") for g in root.iter("geom") if g.get("mesh")}
        for mesh in list(asset.findall("mesh")):
            if mesh.get("name") not in used:
                asset.remove(mesh)
    return ET.tostring(root, encoding="unicode")


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
    collision: str = "mesh",
    write: bool = True,
) -> str:
    """Assemble the full scene MJCF. `sides` = which arms get actuators.

    obj_type/obj_dims select the manipulable object (see `_object_block`).
    Defaults to the box (obj_dims falls back to a cube of `obj_size`).

    collision:
      "mesh"      -> keep the URDF's full mesh collision (default; what the
                     existing CPU demos / BC checkpoints were tuned on).
      "primitive" -> fingertip spheres + collision disabled elsewhere, so the
                     scene is MJX-compatible and compiles in seconds
                     (see `_simplify_collision`).
    """
    sides = sides or ["R"]
    if obj_dims is None:
        obj_dims = (obj_size, obj_size, obj_size)
    mjcf = _convert_cylinders_to_boxes(_compile_to_mjcf(C.VEGA_F5D6_URDF))
    if collision == "primitive":
        mjcf = _simplify_collision(mjcf, sides)
    asset, worldbody, default = _extract_blocks(mjcf)

    # Primitive (MJX) scene uses CG + pyramidal cone: Newton's inner iterations
    # nest badly under jax.lax.scan and blow up rollout compile time. Mesh scene
    # keeps the more accurate Newton + elliptic.
    if collision == "primitive":
        opt = (f'<option timestep="{C.SIM_DT}" integrator="implicitfast" '
               f'cone="pyramidal" solver="CG" iterations="10" ls_iterations="8"/>')
    else:
        opt = f'<option timestep="{C.SIM_DT}" integrator="implicitfast" cone="elliptic"/>'

    scene = f"""<mujoco model="vega_f5d6_tracking">
  <compiler angle="radian" autolimits="true"/>
  {opt}

  <default>
{_indent(default, 4)}
  </default>

  <visual>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.6 0.6 0.6" specular="0.0 0.0 0.0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global offwidth="1280" offheight="960" elevation="-20" azimuth="120"/>
  </visual>

  <asset>
{_indent(asset, 4)}
    <texture name="skybox" type="skybox" builtin="gradient"
             rgb1="0.30 0.35 0.45" rgb2="0.05 0.07 0.10" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.55 0.55 0.55"
             rgb2="0.40 0.40 0.40" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="6 6" reflectance="0.05"/>
  </asset>

  <worldbody>
    <light name="top"    pos="0.5 0.0 2.5" dir="0 0 -1" diffuse="0.7 0.7 0.7" specular="0.0 0.0 0.0"/>
    <light name="frontL" pos="1.2 0.6 1.3" dir="-0.6 -0.3 -0.5" diffuse="0.45 0.45 0.45"/>
    <light name="frontR" pos="1.2 -0.6 1.3" dir="-0.6  0.3 -0.5" diffuse="0.45 0.45 0.45"/>
    <geom name="floor" type="plane" size="5 5 0.1" material="grid"/>
    <body name="table" pos="0.3 0 {table_z}">
      <geom name="table_top" type="box" size="0.55 0.4 0.02" rgba="0.78 0.65 0.50 1"/>
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
