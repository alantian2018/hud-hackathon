from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


_STATUS_COLORS = {
    "decision": "#475569",
    "repositioning": "#16a34a",
    "to_pickup": "#2563eb",
    "to_dropoff": "#dc2626",
}


@dataclass(frozen=True)
class RenderConfig:
    width: int = 1280
    height: int = 800
    hud_width: int = 340
    scale: int = 1


class RichRenderer:
    def __init__(self, *, width: int = 1280, height: int = 800, scale: int = 1) -> None:
        self.config = RenderConfig(width=width, height=height, scale=max(1, int(scale)))
        self.last_metadata: dict[str, Any] | None = None
        self._base_cache_key: tuple[Any, ...] | None = None
        self._base_canvas: Image.Image | None = None
        self.base_cache_hits = 0

    def render(self, scene: dict[str, Any]) -> np.ndarray:
        key = _base_cache_key(scene, self.config)
        has_static_edges = bool(scene.get("congestion"))
        if self._base_canvas is not None and (key == self._base_cache_key or not has_static_edges):
            base_canvas = self._base_canvas
            self.base_cache_hits += 1
        else:
            base_canvas = _render_base_canvas(scene, self.config)
            self._base_canvas = base_canvas
            self._base_cache_key = key
        frame, metadata = _compose_scene_array(
            scene,
            self.config,
            base_canvas=base_canvas,
            return_metadata=True,
        )
        self.last_metadata = metadata
        return frame

    def needs_static_scene(self, time_seconds: float | None = None) -> bool:
        if self._base_canvas is None or self._base_cache_key is None:
            return True
        if time_seconds is None:
            return False
        return _hour_from_seconds(time_seconds) != int(self._base_cache_key[6])


class PygletWindowRenderer:
    def __init__(
        self,
        *,
        width: int,
        height: int,
        title: str = "JAX Fleet live render",
        fullscreen: bool = False,
    ) -> None:
        import pyglet

        self.configure_pyglet(pyglet)
        self._pyglet = pyglet
        self.window = pyglet.window.Window(
            width=width,
            height=height,
            caption=title,
            resizable=True,
            fullscreen=False,
        )
        self._closed = False
        if fullscreen:
            self.window.maximize()

        @self.window.event
        def on_close():
            self._closed = True
            self.window.close()

    @staticmethod
    def configure_pyglet(pyglet_module) -> None:
        # On macOS Hi-DPI displays, pyglet's platform mode can expose a larger
        # backing framebuffer than the requested window, which makes ImageData
        # blits appear in a corner. Stretch keeps the content coordinate system
        # matched to the requested window size.
        pyglet_module.options["dpi_scaling"] = "stretch"

    @property
    def closed(self) -> bool:
        return self._closed or bool(getattr(self.window, "has_exit", False))

    def render(self, frame: np.ndarray) -> bool:
        if self.closed:
            return False
        frame = np.ascontiguousarray(frame.astype(np.uint8, copy=False))
        height, width = frame.shape[:2]
        self.window.switch_to()
        self.window.dispatch_events()
        image = self._pyglet.image.ImageData(
            width,
            height,
            "RGB",
            frame.tobytes(),
            pitch=-width * 3,
        )
        target_width, target_height = self.window.get_size()
        self.window.clear()
        image.blit(0, 0, width=target_width, height=target_height)
        self.window.flip()
        return not self.closed

    def close(self) -> None:
        if not self.closed:
            self.window.close()
        self._closed = True


def render_scene_to_array(
    scene: dict[str, Any],
    *,
    width: int = 1280,
    height: int = 800,
    scale: int = 1,
    return_metadata: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    config = RenderConfig(width=width, height=height, scale=max(1, int(scale)))
    return _compose_scene_array(scene, config, base_canvas=None, return_metadata=return_metadata)


def _compose_scene_array(
    scene: dict[str, Any],
    config: RenderConfig,
    *,
    base_canvas: Image.Image | None,
    return_metadata: bool,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    scale = max(1, int(config.scale))
    if base_canvas is None:
        canvas = _render_base_canvas(scene, config)
    else:
        canvas = base_canvas.copy()
    draw = ImageDraw.Draw(canvas, "RGBA")
    map_box, hud_box = _layout_boxes(config)
    mapper = _CoordinateMapper(scene, map_box)
    route_preview_count = _draw_route_previews(draw, scene, mapper, scale)
    _draw_edge_progress(draw, scene, mapper, scale)
    _draw_requests(draw, scene, mapper, scale)
    _draw_cars(draw, scene, mapper, scale)
    hud_lines = _draw_hud(draw, scene, hud_box, scale)

    if scale != 1:
        canvas = canvas.resize((config.width, config.height), Image.Resampling.LANCZOS)
    frame = np.asarray(canvas, dtype=np.uint8)
    metadata = {
        "hud_lines": hud_lines,
        "map_box": tuple(int(v / scale) for v in map_box),
        "hud_box": tuple(int(v / scale) for v in hud_box),
        "route_preview_count": route_preview_count,
    }
    if return_metadata:
        return frame, metadata
    return frame


def _render_base_canvas(scene: dict[str, Any], config: RenderConfig) -> Image.Image:
    scale = max(1, int(config.scale))
    canvas = Image.new("RGB", (config.width * scale, config.height * scale), "#f8fafc")
    draw = ImageDraw.Draw(canvas, "RGBA")
    map_box, _ = _layout_boxes(config)
    mapper = _CoordinateMapper(scene, map_box)
    _draw_map_background(draw, map_box)
    _draw_roads(draw, scene, mapper, scale)
    return canvas


def _layout_boxes(config: RenderConfig) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    scale = max(1, int(config.scale))
    hud_width = min(config.hud_width, max(260, int(config.width * 0.34)))
    map_box = (
        22 * scale,
        22 * scale,
        max(24 * scale, (config.width - hud_width - 18) * scale),
        (config.height - 22) * scale,
    )
    hud_box = ((config.width - hud_width) * scale, 0, config.width * scale, config.height * scale)
    return map_box, hud_box


def _base_cache_key(scene: dict[str, Any], config: RenderConfig) -> tuple[Any, ...]:
    graph = scene.get("graph", {})
    bounds = tuple(round(float(value), 7) for value in graph.get("bounds", []))
    hour = _hour_from_seconds(float(scene.get("time_seconds", 0.0)))
    edges = scene.get("congestion", [])
    edge_fingerprint: tuple[Any, ...]
    if edges:
        first = edges[0]
        last = edges[-1]
        edge_fingerprint = (
            len(edges),
            tuple(first.get("source") or ()),
            tuple(first.get("target") or ()),
            tuple(last.get("source") or ()),
            tuple(last.get("target") or ()),
        )
    else:
        edge_fingerprint = (0,)
    return (
        int(config.width),
        int(config.height),
        int(config.hud_width),
        int(config.scale),
        int(graph.get("num_nodes", 0)),
        int(graph.get("num_edges", 0)),
        hour,
        bounds,
        edge_fingerprint,
    )


def _hour_from_seconds(time_seconds: float) -> int:
    return int(float(time_seconds) // 3600) % 24


class _CoordinateMapper:
    def __init__(self, scene: dict[str, Any], map_box: tuple[int, int, int, int]) -> None:
        self.left, self.top, self.right, self.bottom = map_box
        self.bounds = _scene_bounds(scene)
        min_lon, min_lat, max_lon, max_lat = self.bounds
        span_lon = max(1.0e-6, max_lon - min_lon)
        span_lat = max(1.0e-6, max_lat - min_lat)
        map_w = max(1, self.right - self.left)
        map_h = max(1, self.bottom - self.top)
        self.scale = min(map_w / span_lon, map_h / span_lat)
        fitted_w = span_lon * self.scale
        fitted_h = span_lat * self.scale
        self.x_pad = (map_w - fitted_w) / 2.0
        self.y_pad = (map_h - fitted_h) / 2.0

    def xy(self, lonlat: list[float] | tuple[float, float] | np.ndarray | None) -> tuple[int, int] | None:
        if lonlat is None:
            return None
        min_lon, min_lat, max_lon, max_lat = self.bounds
        lon = float(lonlat[0])
        lat = float(lonlat[1])
        x = self.left + self.x_pad + (lon - min_lon) * self.scale
        y = self.top + self.y_pad + (max_lat - lat) * self.scale
        return int(round(x)), int(round(y))


def _draw_map_background(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    left, top, right, bottom = box
    draw.rounded_rectangle(box, radius=10, fill="#eef2f7", outline="#cbd5e1", width=1)
    for i in range(1, 5):
        x = left + (right - left) * i / 5
        y = top + (bottom - top) * i / 5
        draw.line([(x, top), (x, bottom)], fill=(148, 163, 184, 55), width=1)
        draw.line([(left, y), (right, y)], fill=(148, 163, 184, 55), width=1)


def _draw_roads(
    draw: ImageDraw.ImageDraw,
    scene: dict[str, Any],
    mapper: _CoordinateMapper,
    scale: int,
) -> None:
    for edge in scene.get("congestion", []):
        start = mapper.xy(edge.get("source"))
        end = mapper.xy(edge.get("target"))
        if start is None or end is None:
            continue
        color = _congestion_rgba(float(edge.get("congestion", 1.0)), alpha=135)
        draw.line([start, end], fill=color, width=max(1, scale))


def _draw_edge_progress(
    draw: ImageDraw.ImageDraw,
    scene: dict[str, Any],
    mapper: _CoordinateMapper,
    scale: int,
) -> None:
    for progress in scene.get("edge_progress", []):
        start = mapper.xy(progress.get("from"))
        end = mapper.xy(progress.get("to"))
        if start is None or end is None:
            continue
        color = _status_rgba(str(progress.get("status", "")), alpha=225)
        draw.line([start, end], fill=color, width=max(2, 3 * scale))


def _draw_route_previews(
    draw: ImageDraw.ImageDraw,
    scene: dict[str, Any],
    mapper: _CoordinateMapper,
    scale: int,
) -> int:
    drawn = 0
    for preview in scene.get("route_previews", []):
        status = str(preview.get("status", ""))
        if status not in {"to_pickup", "to_dropoff"}:
            continue
        points = [mapper.xy(point) for point in preview.get("points", [])]
        xy = [point for point in points if point is not None]
        if len(xy) < 2:
            continue
        draw.line(xy, fill=_status_rgba(status, alpha=70), width=max(6, 8 * scale), joint="curve")
        draw.line(xy, fill=_status_rgba(status, alpha=230), width=max(2, 3 * scale), joint="curve")
        end = xy[-1]
        radius = 5 * scale
        draw.ellipse(
            (end[0] - radius, end[1] - radius, end[0] + radius, end[1] + radius),
            fill=_status_rgba(status, alpha=245),
            outline=(255, 255, 255, 235),
            width=max(1, scale),
        )
        drawn += 1
    return drawn


def _draw_requests(
    draw: ImageDraw.ImageDraw,
    scene: dict[str, Any],
    mapper: _CoordinateMapper,
    scale: int,
) -> None:
    for request in scene.get("requests", []):
        status = request.get("status", "unknown")
        origin = mapper.xy(request.get("origin"))
        destination = mapper.xy(request.get("destination"))
        if origin is not None and status in {"queued", "assigned"}:
            radius = 6 * scale
            fill = "#f59e0b" if status == "queued" else "#d97706"
            _ellipse(draw, origin, radius, fill=fill, outline="#78350f", width=max(1, scale))
        if destination is not None and status in {"queued", "assigned", "onboard"}:
            radius = 7 * scale
            x, y = destination
            color = "#7c3aed"
            draw.line([(x - radius, y - radius), (x + radius, y + radius)], fill=color, width=max(2, scale))
            draw.line([(x - radius, y + radius), (x + radius, y - radius)], fill=color, width=max(2, scale))


def _draw_cars(
    draw: ImageDraw.ImageDraw,
    scene: dict[str, Any],
    mapper: _CoordinateMapper,
    scale: int,
) -> None:
    font = _font(12 * scale, bold=True)
    for car in scene.get("cars", []):
        xy = mapper.xy(car.get("position"))
        if xy is None:
            continue
        x, y = xy
        radius = 9 * scale
        fill = _status_color(str(car.get("status", "")))
        _ellipse(draw, (x, y), radius, fill=fill, outline="#ffffff", width=max(2, scale))
        label = str(car.get("id", "?"))
        bbox = draw.textbbox((0, 0), label, font=font)
        draw.text(
            (x - (bbox[2] - bbox[0]) / 2, y - (bbox[3] - bbox[1]) / 2 - scale),
            label,
            fill="#ffffff",
            font=font,
        )


def _draw_hud(
    draw: ImageDraw.ImageDraw,
    scene: dict[str, Any],
    box: tuple[int, int, int, int],
    scale: int,
) -> list[str]:
    left, top, right, bottom = box
    draw.rectangle(box, fill="#0f172a")
    draw.line([(left, top), (left, bottom)], fill="#334155", width=max(1, scale))

    title_font = _font(18 * scale, bold=True)
    label_font = _font(11 * scale, bold=False)
    lines = _hud_lines(scene)
    x = left + 18 * scale
    y = top + 18 * scale
    draw.text((x, y), "JAX Fleet", fill="#f8fafc", font=title_font)
    y += 31 * scale
    draw.text((x, y), "live random-step environment", fill="#94a3b8", font=label_font)
    y += 29 * scale

    for line in lines:
        if y > bottom - 24 * scale:
            break
        color = "#e2e8f0"
        if line.startswith("time") or line.startswith("current car") or line.startswith("reward"):
            color = "#f8fafc"
        if "invalid" in line or "dropped" in line:
            color = "#fca5a5"
        draw.text((x, y), line, fill=color, font=label_font)
        y += 18 * scale

    mask = list(scene.get("action_mask", []))
    if mask and y < bottom - 42 * scale:
        y += 8 * scale
        draw.text((x, y), "valid outgoing slots", fill="#94a3b8", font=label_font)
        y += 18 * scale
        slot_size = 18 * scale
        for idx, valid in enumerate(mask[:12]):
            sx = x + idx * (slot_size + 5 * scale)
            fill = "#22c55e" if valid else "#475569"
            draw.rounded_rectangle(
                (sx, y, sx + slot_size, y + slot_size),
                radius=3 * scale,
                fill=fill,
                outline="#cbd5e1" if valid else "#64748b",
                width=max(1, scale),
            )
            text = str(idx)
            bbox = draw.textbbox((0, 0), text, font=label_font)
            draw.text(
                (sx + (slot_size - (bbox[2] - bbox[0])) / 2, y + 1 * scale),
                text,
                fill="#052e16" if valid else "#cbd5e1",
                font=label_font,
            )
    return lines


def _hud_lines(scene: dict[str, Any]) -> list[str]:
    events = scene.get("recent_events", {})
    metrics = scene.get("metrics", events)
    graph = scene.get("graph", {})
    counts = scene.get("status_counts", {})
    cars = counts.get("cars", {})
    requests = counts.get("requests", {})
    current = _current_car(scene)
    action_mask = "".join("1" if value else "0" for value in scene.get("action_mask", []))
    return [
        f"time: {_format_time(float(scene.get('time_seconds', 0.0)))}",
        f"step: {int(scene.get('step_count', 0))}   done: {bool(scene.get('done', False))}",
        f"current car: {scene.get('current_car_id', -1)}   decision: {bool(scene.get('decision_required', False))}",
        f"reward: {float(events.get('reward', 0.0)):.3f}",
        f"dt seconds: {float(events.get('dt_seconds', 0.0)):.2f}",
        f"discount: {float(scene.get('discount', 1.0)):.5f}",
        f"action mask: {action_mask or 'none'}",
        "",
        f"cars: {len(scene.get('cars', []))}   edges: {int(graph.get('num_edges', 0))}",
        f"nodes: {int(graph.get('num_nodes', 0))}   max degree: {int(graph.get('max_degree', 0))}",
        f"decision cars: {int(cars.get('decision', 0))}",
        f"repositioning: {int(cars.get('repositioning', 0))}",
        f"to pickup: {int(cars.get('to_pickup', 0))}",
        f"to dropoff: {int(cars.get('to_dropoff', 0))}",
        "",
        f"requests shown: {len(scene.get('requests', []))}",
        f"active target: {int(metrics.get('active_requests', 0))}/{int(metrics.get('target_active_requests', 0))}",
        f"queued: {int(metrics.get('queued_requests', requests.get('queued', 0)))}",
        f"assigned: {int(requests.get('assigned', 0))}",
        f"onboard: {int(requests.get('onboard', 0))}",
        f"completed: {int(metrics.get('completed_requests', 0))}",
        f"dropped: {int(metrics.get('dropped_requests', 0))}",
        f"invalid actions: {int(metrics.get('invalid_actions', 0))}",
        f"pickup wait total: {float(metrics.get('pickup_wait_seconds', 0.0)):.1f}s",
        "",
        f"car node: {current.get('compact_node_id', 'n/a')}",
        f"car status: {current.get('status', 'n/a')}",
        f"car edge: {current.get('edge_id', 'n/a')}",
        f"car request: {current.get('request_id', 'n/a')}",
    ]


def _current_car(scene: dict[str, Any]) -> dict[str, Any]:
    current_id = int(scene.get("current_car_id", -1))
    for car in scene.get("cars", []):
        if int(car.get("id", -2)) == current_id:
            return car
    return {}


def _scene_bounds(scene: dict[str, Any]) -> tuple[float, float, float, float]:
    graph_bounds = scene.get("graph", {}).get("bounds")
    if graph_bounds and len(graph_bounds) == 4:
        min_lon, min_lat, max_lon, max_lat = [float(value) for value in graph_bounds]
    else:
        points: list[list[float]] = []
        for edge in scene.get("congestion", []):
            if edge.get("source") is not None:
                points.append(edge["source"])
            if edge.get("target") is not None:
                points.append(edge["target"])
        for car in scene.get("cars", []):
            if car.get("position") is not None:
                points.append(car["position"])
        for request in scene.get("requests", []):
            if request.get("origin") is not None:
                points.append(request["origin"])
            if request.get("destination") is not None:
                points.append(request["destination"])
        for preview in scene.get("route_previews", []):
            points.extend(point for point in preview.get("points", []) if point is not None)
        if not points:
            points = [[0.0, 0.0], [1.0, 1.0]]
        values = np.asarray(points, dtype=np.float64)
        min_lon, min_lat = values.min(axis=0)
        max_lon, max_lat = values.max(axis=0)

    lon_pad = max(1.0e-5, (max_lon - min_lon) * 0.05)
    lat_pad = max(1.0e-5, (max_lat - min_lat) * 0.05)
    return min_lon - lon_pad, min_lat - lat_pad, max_lon + lon_pad, max_lat + lat_pad


def _congestion_rgba(value: float, *, alpha: int) -> tuple[int, int, int, int]:
    if value <= 1.25:
        rgb = _mix((34, 197, 94), (234, 179, 8), (value - 1.0) / 0.25)
    else:
        rgb = _mix((234, 179, 8), (220, 38, 38), min(1.0, (value - 1.25) / 1.25))
    return (*rgb, alpha)


def _status_color(status: str) -> str:
    return _STATUS_COLORS.get(status, "#475569")


def _status_rgba(status: str, *, alpha: int) -> tuple[int, int, int, int]:
    value = _status_color(status).lstrip("#")
    return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16), int(alpha))


def _mix(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    t = float(np.clip(t, 0.0, 1.0))
    return tuple(int(round(a[i] * (1.0 - t) + b[i] * t)) for i in range(3))


def _ellipse(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    radius: int,
    *,
    fill: str,
    outline: str,
    width: int,
) -> None:
    x, y = xy
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline=outline, width=width)


def _format_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    minutes, sec = divmod(int(round(seconds)), 60)
    hours, minute = divmod(minutes, 60)
    return f"{hours:02d}:{minute:02d}:{sec:02d}"


@lru_cache(maxsize=32)
def _font(size: int, *, bold: bool = False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()
