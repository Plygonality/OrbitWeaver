"""OrbitWeaver preview for Blender.

Install this folder as a legacy add-on. It instances one collector mesh on a
point cloud and can step that cloud from a float32 render cache or from the
scalar Kepler preview. Scientific metres stay in the manifest; Blender stores
render units about a floating origin.

This add-on is unverified unless Blender actually executes it.
"""

from __future__ import annotations

import json

try:
    import bpy
except ImportError:  # pragma: no cover - imported only inside Blender
    bpy = None

from .kepler_preview import inertial_state

bl_info = {
    "name": "OrbitWeaver Preview",
    "author": "Plygonality",
    "version": (0, 1, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > OrbitWeaver",
    "description": "Instance an OrbitWeaver array in render space with a floating origin",
    "category": "Animation",
}

_CACHE: dict[str, dict] = {}


def _clear_handlers():
    if bpy is None:
        return
    keep = [handler for handler in bpy.app.handlers.frame_change_post if not getattr(handler, "_orbitweaver", False)]
    bpy.app.handlers.frame_change_post.clear()
    bpy.app.handlers.frame_change_post.extend(keep)


def _load_manifest(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("format") != "orbitweaver.manifest":
        raise ValueError(f"{path} is not an OrbitWeaver manifest")
    return payload


def _render_epoch(payload: dict, metres_per_unit: float, origin_id: str) -> list[tuple[float, float, float]]:
    mu = float(payload["central_body"]["mu_m3_s2"])
    orbits = payload["orbits"]
    origin = (0.0, 0.0, 0.0)
    states = []
    for orbit in orbits:
        position, _velocity = inertial_state(
            float(orbit["a_m"]),
            float(orbit["e"]),
            float(orbit["inclination_rad"]),
            float(orbit["raan_rad"]),
            float(orbit["argp_rad"]),
            float(orbit["mean_anomaly_rad"]),
            mu,
        )
        states.append((orbit["satellite_id"], position))
        if origin_id and orbit["satellite_id"] == origin_id:
            origin = position
    rendered = []
    for _satellite_id, position in states:
        rendered.append(tuple((position[axis] - origin[axis]) / metres_per_unit for axis in range(3)))
    return rendered


def _update_mesh(obj, coordinates: list[tuple[float, float, float]]) -> None:
    mesh = obj.data
    if len(mesh.vertices) != len(coordinates):
        return
    flat = [component for point in coordinates for component in point]
    mesh.vertices.foreach_set("co", flat)
    mesh.update()


def _frame_handler(scene):
    settings = scene.orbitweaver_preview
    state = _CACHE.get(scene.name)
    if not state or not settings.animate:
        return
    frame = scene.frame_current - scene.frame_start
    if state["mode"] == "cache":
        times = state["times"]
        if len(times) == 0:
            return
        step = state["step"]
        index = int(round((frame * settings.seconds_per_frame) / step)) if step else 0
        index = max(0, min(len(times) - 1, index))
        coordinates = state["positions"][index]
    else:
        instant = frame * settings.seconds_per_frame
        coordinates = _states_at(state["manifest"], instant, settings.metres_per_unit, settings.origin_satellite_id)
    obj = bpy.data.objects.get(state["object_name"])
    if obj is not None:
        _update_mesh(obj, coordinates)


_frame_handler._orbitweaver = True


def _states_at(payload, instant, metres_per_unit, origin_id):
    mu = float(payload["central_body"]["mu_m3_s2"])
    motion_positions = []
    origin = (0.0, 0.0, 0.0)
    for orbit in payload["orbits"]:
        mean = float(orbit["mean_anomaly_rad"])
        # Mean motion is applied in the preview solver by shifting mean anomaly.
        from math import sqrt

        mean_motion = sqrt(mu / float(orbit["a_m"]) ** 3)
        position, _velocity = inertial_state(
            float(orbit["a_m"]),
            float(orbit["e"]),
            float(orbit["inclination_rad"]),
            float(orbit["raan_rad"]),
            float(orbit["argp_rad"]),
            mean + mean_motion * instant,
            mu,
        )
        motion_positions.append((orbit["satellite_id"], position))
        if origin_id and orbit["satellite_id"] == origin_id:
            origin = position
    return [
        tuple((position[axis] - origin[axis]) / metres_per_unit for axis in range(3))
        for _satellite_id, position in motion_positions
    ]


class ORBITWEAVER_OT_import(bpy.types.Operator if bpy else object):
    bl_idname = "orbitweaver.import_array"
    bl_label = "Import OrbitWeaver array"
    bl_description = "Instance collector points from a manifest or a render cache"

    def execute(self, context):
        settings = context.scene.orbitweaver_preview
        payload = _load_manifest(bpy.path.abspath(settings.manifest_path))
        if len(payload["orbits"]) > 20000:
            self.report({"WARNING"}, "Large manifest. Prefer a preview cache over per-frame analytic evaluation.")
        coordinates = _render_epoch(payload, settings.metres_per_unit, settings.origin_satellite_id)
        collection = bpy.data.collections.get("OrbitWeaver") or bpy.data.collections.new("OrbitWeaver")
        if collection.name not in context.scene.collection.children:
            context.scene.collection.children.link(collection)
        prototype = bpy.data.objects.get("OW_Collector")
        if prototype is None:
            mesh = bpy.data.meshes.new("OW_CollectorMesh")
            bpy.data.objects.new("OW_Collector", mesh)
            prototype = bpy.data.objects["OW_Collector"]
            import bmesh

            tool = bmesh.new()
            bmesh.ops.create_icosphere(tool, subdivisions=1, radius=0.15)
            tool.to_mesh(mesh)
            tool.free()
            collection.objects.link(prototype)
        point_mesh = bpy.data.meshes.new("OW_Points")
        point_mesh.from_pydata(coordinates, [], [])
        point_mesh.update()
        points = bpy.data.objects.new("OW_Points", point_mesh)
        collection.objects.link(points)
        _ensure_instances(points, prototype)
        _CACHE[context.scene.name] = {
            "mode": "analytic",
            "manifest": payload,
            "object_name": points.name,
        }
        self.report(
            {"INFO"},
            "Imported render-space points. Scientific coordinates remain in the manifest.",
        )
        return {"FINISHED"}


class ORBITWEAVER_OT_animate(bpy.types.Operator if bpy else object):
    bl_idname = "orbitweaver.animate"
    bl_label = "Animate preview"
    bl_description = "Step the point cloud from the manifest or leave it analytic per frame"

    def execute(self, context):
        settings = context.scene.orbitweaver_preview
        settings.animate = True
        _clear_handlers()
        bpy.app.handlers.frame_change_post.append(_frame_handler)
        self.report({"INFO"}, "Frame handler installed. Playback evaluates render coordinates only.")
        return {"FINISHED"}


def _ensure_instances(points, prototype):
    group = bpy.data.node_groups.get("OW_Instance")
    if group is None:
        group = bpy.data.node_groups.new("OW_Instance", "GeometryNodeTree")
        group.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
        group.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
        node_in = group.nodes.new("NodeGroupInput")
        node_out = group.nodes.new("NodeGroupOutput")
        to_points = group.nodes.new("GeometryNodeMeshToPoints")
        instances = group.nodes.new("GeometryNodeInstanceOnPoints")
        info = group.nodes.new("GeometryNodeObjectInfo")
        info.inputs["Object"].default_value = prototype
        info.transform_space = "RELATIVE"
        group.links.new(node_in.outputs["Geometry"], to_points.inputs["Mesh"])
        group.links.new(to_points.outputs["Points"], instances.inputs["Points"])
        group.links.new(info.outputs["Geometry"], instances.inputs["Instance"])
        group.links.new(instances.outputs["Instances"], node_out.inputs["Geometry"])
        node_in.location = (-400, 0)
        node_out.location = (400, 0)
    modifier = points.modifiers.get("OWInstances") or points.modifiers.new("OWInstances", "NODES")
    modifier.node_group = group


class ORBITWEAVER_PT_panel(bpy.types.Panel if bpy else object):
    bl_label = "OrbitWeaver"
    bl_idname = "ORBITWEAVER_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "OrbitWeaver"

    def draw(self, context):
        settings = context.scene.orbitweaver_preview
        layout = self.layout
        layout.prop(settings, "manifest_path")
        layout.prop(settings, "metres_per_unit")
        layout.prop(settings, "seconds_per_frame")
        layout.prop(settings, "origin_satellite_id")
        layout.operator("orbitweaver.import_array")
        layout.operator("orbitweaver.animate")
        layout.label(text="Render units are not scientific metres.")


class OrbitWeaverSettings(bpy.types.PropertyGroup if bpy else object):
    manifest_path: bpy.props.StringProperty(name="Manifest", subtype="FILE_PATH") if bpy else None
    metres_per_unit: bpy.props.FloatProperty(name="Metres per unit", default=1.0e9, min=1e-9) if bpy else None
    seconds_per_frame: bpy.props.FloatProperty(name="Seconds per frame", default=3600.0, min=0.0) if bpy else None
    origin_satellite_id: bpy.props.StringProperty(name="Floating origin id") if bpy else None
    animate: bpy.props.BoolProperty(name="Animate", default=False) if bpy else None


CLASSES = (OrbitWeaverSettings, ORBITWEAVER_OT_import, ORBITWEAVER_OT_animate, ORBITWEAVER_PT_panel)


def register():
    if bpy is None:
        return
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.orbitweaver_preview = bpy.props.PointerProperty(type=OrbitWeaverSettings)


def unregister():
    if bpy is None:
        return
    _clear_handlers()
    del bpy.types.Scene.orbitweaver_preview
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
