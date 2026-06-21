import React, {useEffect, useMemo, useState} from "react";
import {createRoot} from "react-dom/client";
import DeckGL from "@deck.gl/react";
import {GeoJsonLayer, ScatterplotLayer} from "@deck.gl/layers";
import {StaticMap} from "react-map-gl";
import "mapbox-gl/dist/mapbox-gl.css";

const MAPTILER_KEY = "4WonZ3glTzG3MWfQd6gQ";
const DAY_MINUTES = 24 * 60;
const PRE_FRAME_MINUTE = -1;
const SPEEDS = [0.5, 1, 4, 12, 30, 60];
const INITIAL_VIEW_STATE = {
  longitude: -122.4194,
  latitude: 37.7749,
  zoom: 12.7,
  pitch: 52,
  bearing: -18
};
const GREEDY = {
  id: "greedy",
  label: "Greedy",
  route: [66, 133, 244, 245],
  casing: [232, 240, 254, 235],
  active: [52, 168, 83, 245],
  idle: [189, 193, 198, 225],
  panel: "rgba(4,8,14,0.88)",
  accent: "#7dd3fc"
};
const RL = {
  id: "rl",
  label: "Agentic Fleet",
  route: [66, 133, 244, 245],
  casing: [232, 240, 254, 235],
  reposition: [20, 184, 166, 230],
  active: [168, 85, 247, 245],
  idle: [203, 213, 225, 220],
  panel: "rgba(12,8,22,0.88)",
  accent: "#c4b5fd"
};
const MAX_VISIBLE_ROUTE_SEGMENT_METERS = 650;
const PICKUP_BLUE = [66, 133, 244, 245];
const PICKUP_BLUE_FILL = [66, 133, 244, 160];
const DESTINATION_RED = [234, 67, 53, 245];
const DESTINATION_RED_FILL = [234, 67, 53, 150];
const PEOPLE_GRID_EMPTY_FILL = [18, 34, 52, 0];
const PEOPLE_GRID_EMPTY_LINE = [138, 180, 248, 34];
const PEOPLE_GRID_BOTH_FILL = [168, 85, 247, 155];
const PEOPLE_GRID_BOTH_LINE = [216, 180, 254, 230];
const FEATURE_GRID_CELLS_CACHE = new WeakMap();

function buildStyleUrl() {
  return `https://api.maptiler.com/maps/streets-v2-dark/style.json?key=${MAPTILER_KEY}`;
}

function hideBaseTransportationLayers(map) {
  const styleLayers = map.getStyle()?.layers ?? [];
  for (const layer of styleLayers) {
    const id = layer.id ?? "";
    const sourceLayer = layer["source-layer"] ?? "";
    const isRoadGeometry =
      sourceLayer === "transportation" &&
      (layer.type === "line" || layer.type === "fill");
    if (isRoadGeometry && map.getLayer(id)) map.setLayoutProperty(id, "visibility", "none");
  }
}

function clamp(v, min, max) {
  return Math.max(min, Math.min(max, v));
}

function formatClock(minute) {
  if (minute < 0) return "00:00";
  const wrapped = clamp(minute, 0, DAY_MINUTES - 0.001);
  const hour = Math.floor(wrapped / 60);
  const min = Math.floor(wrapped % 60);
  return `${String(hour).padStart(2, "0")}:${String(min).padStart(2, "0")}`;
}

function formatMoney(value) {
  return `$${Math.round(Number(value) || 0).toLocaleString()}`;
}

function formatPct(value) {
  return `${(Number(value) || 0).toFixed(1)}%`;
}

function rgba(color) {
  const [r, g, b, a = 255] = color;
  return `rgba(${r},${g},${b},${a / 255})`;
}

async function fetchJson(paths) {
  let lastError = null;
  for (const path of paths) {
    try {
      const response = await fetch(`${path}?v=${Date.now()}`, {cache: "no-store"});
      if (!response.ok) throw new Error(`${path} ${response.status}`);
      return await response.json();
    } catch (error) {
      lastError = error;
    }
  }
  throw lastError;
}

function zeroStats() {
  return {
    completed_trips: 0,
    revenue: 0,
    demand_served_pct: 0,
    wait_time_min: 0,
    fleet_utilization_pct: 0,
    avg_fleet_utilization_pct: 0,
    active_cars: 0,
    repositioning_cars: 0,
    stalled_cars: 0,
    unassigned_people: 0,
    canceled_requests: 0,
    total_requests: 0,
    reposition_cost: 0
  };
}

function zeroDispatchSummary() {
  return {
    num_assignments: 0,
    num_new_assignments: 0,
    num_queued_assignments: 0,
    num_new_queued_assignments: 0,
    num_repositions: 0,
    num_new_repositions: 0,
    num_unassigned_people: 0,
    num_stalled_cars: 0,
    num_active_cars: 0
  };
}

function zeroPreFrame(snapshot, policyId) {
  const statsKey = policyId === "greedy" ? "map_greedy_stats" : "map_orchestrator_stats";
  const cars = (snapshot?.map_dispatch?.cars ?? []).map(car => ({
    ...car,
    status: "idle",
    assigned_person_id: null,
    pickup_node_id: null,
    dropoff_node_id: null,
    route_elapsed: 0
  }));
  const stats = zeroStats();
  const dispatchSummary = zeroDispatchSummary();
  return {
    ...(snapshot ?? {}),
    timestep: PRE_FRAME_MINUTE,
    is_pre_frame: true,
    new_people: [],
    summary: {
      ...(snapshot?.summary ?? {}),
      num_new_people: 0,
      top_demand_cells: [],
      traffic_bottlenecks: [],
      dispatch: dispatchSummary,
      greedy_stats: stats,
      orchestrator_stats: stats
    },
    map_dispatch: {
      ...(snapshot?.map_dispatch ?? {}),
      assignments: [],
      new_assignments: [],
      queued_assignments: [],
      new_queued_assignments: [],
      repositions: [],
      new_repositions: [],
      cars,
      summary: dispatchSummary
    },
    map_people: [],
    [statsKey]: stats
  };
}

function scenarioSnapshots(world, scenarioId, policyId) {
  if (!world) return [];
  let snapshots = world.snapshots ?? [];
  if (scenarioId && world.event_scenarios?.[scenarioId]?.snapshots?.length) {
    snapshots = world.event_scenarios[scenarioId].snapshots;
  }
  if (!snapshots.length) return [];
  if (snapshots[0]?.is_pre_frame) return snapshots;
  return [zeroPreFrame(snapshots[0], policyId), ...snapshots];
}

function scenarioStepMinutes(world, scenarioId) {
  if (scenarioId && world?.event_scenarios?.[scenarioId]?.step_minutes) {
    return world.event_scenarios[scenarioId].step_minutes;
  }
  return world?.step_minutes ?? 5;
}

function snapshotAt(snapshots, clockMinute) {
  if (!snapshots.length) return null;
  let active = snapshots[0];
  for (const snapshot of snapshots) {
    if ((snapshot.timestep ?? 0) <= clockMinute) active = snapshot;
    else break;
  }
  return active;
}

function elapsedSeconds(snapshot, clockMinute, stepMinutes) {
  const start = Number(snapshot?.timestep ?? 0);
  return clamp(clockMinute - start, 0, Math.max(1, stepMinutes)) * 60;
}

function jobElapsedSeconds(job, car, snapshot, clockMinute, stepMinutes) {
  return Number(car?.route_elapsed ?? job?.route_elapsed ?? 0) + elapsedSeconds(snapshot, clockMinute, stepMinutes);
}

function resolveRoute(route, routeIndex) {
  if (!route) return null;
  if (Array.isArray(route.coordinates)) return route;
  const id = route.id ?? route.route_id;
  const stored = id ? routeIndex?.[id] : null;
  if (!stored) return route;
  return {...stored, ...route, coordinates: route.coordinates ?? stored.coordinates};
}

function routeSegmentMeters(a, b) {
  if (!a || !b) return 0;
  const meanLat = (((a[1] ?? 0) + (b[1] ?? 0)) / 2) * Math.PI / 180;
  const dx = ((b[0] ?? 0) - (a[0] ?? 0)) * Math.cos(meanLat) * 111320;
  const dy = ((b[1] ?? 0) - (a[1] ?? 0)) * 111320;
  return Math.hypot(dx, dy);
}

function routeCoordinatesAreUsable(coordinates) {
  if (!Array.isArray(coordinates) || coordinates.length < 2) return false;
  for (let i = 0; i < coordinates.length - 1; i++) {
    if (routeSegmentMeters(coordinates[i], coordinates[i + 1]) > MAX_VISIBLE_ROUTE_SEGMENT_METERS) return false;
  }
  return true;
}

function pointAlongRoute(coordinates, progress) {
  if (!Array.isArray(coordinates) || !coordinates.length) return null;
  if (coordinates.length === 1) return coordinates[0];
  const lengths = [];
  let total = 0;
  for (let i = 0; i < coordinates.length - 1; i++) {
    const length = Math.hypot(
      coordinates[i + 1][0] - coordinates[i][0],
      coordinates[i + 1][1] - coordinates[i][1]
    );
    lengths.push(length);
    total += length;
  }
  if (total <= 0) return coordinates[0];
  let remaining = clamp(progress, 0, 1) * total;
  for (let i = 0; i < lengths.length; i++) {
    if (remaining > lengths[i]) {
      remaining -= lengths[i];
      continue;
    }
    const t = lengths[i] <= 0 ? 0 : remaining / lengths[i];
    const a = coordinates[i];
    const b = coordinates[i + 1];
    return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
  }
  return coordinates[coordinates.length - 1];
}

function remainingRoute(coordinates, progress) {
  const point = pointAlongRoute(coordinates, progress);
  if (!point || !coordinates?.length) return [];
  if (progress >= 1) return [coordinates[coordinates.length - 1]];
  const target = clamp(progress, 0, 1);
  const splitIndex = Math.floor(target * Math.max(1, coordinates.length - 1));
  return [point, ...coordinates.slice(Math.min(coordinates.length - 1, splitIndex + 1))];
}

function activeJobs(snapshot) {
  return [
    ...(snapshot?.map_dispatch?.assignments ?? []).map(job => ({...job, kind: "trip"})),
    ...(snapshot?.map_dispatch?.repositions ?? []).map(job => ({...job, kind: "reposition"}))
  ];
}

function jobProgress(job, car, snapshot, clockMinute, stepMinutes) {
  const routeCost = Number(job?.route?.cost ?? job?.total_cost ?? 0);
  if (!routeCost) return 0;
  const elapsed = jobElapsedSeconds(job, car, snapshot, clockMinute, stepMinutes);
  return clamp(elapsed / routeCost, 0, 1);
}

function animatedCars(snapshot, routeIndex, clockMinute, stepMinutes) {
  const cars = snapshot?.map_dispatch?.cars ?? [];
  const jobByCar = new Map(activeJobs(snapshot).map(job => [job.car_id, job]));
  return cars.map(car => {
    const job = jobByCar.get(car.id);
    const route = resolveRoute(job?.route, routeIndex);
    const coordinates = route?.coordinates ?? [];
    const elapsed = jobElapsedSeconds(job, car, snapshot, clockMinute, stepMinutes);
    const pickupCost = Number(job?.pickup_route?.cost ?? 0);
    const progress = job && routeCoordinatesAreUsable(coordinates)
      ? jobProgress(job, car, snapshot, clockMinute, stepMinutes)
      : 0;
    return {
      ...car,
      active_job_kind: job?.kind,
      status: job?.kind === "reposition"
        ? "repositioning"
        : job?.kind === "trip"
        ? elapsed < pickupCost
          ? "to_pickup"
          : "to_dropoff"
        : car.status,
      position: job && routeCoordinatesAreUsable(coordinates)
        ? pointAlongRoute(coordinates, progress) ?? car.position
        : car.position
    };
  });
}

function makeRouteFeatures(snapshot, routeIndex, clockMinute, stepMinutes, includeRepositions) {
  const carById = new Map((snapshot?.map_dispatch?.cars ?? []).map(car => [car.id, car]));
  const features = [];
  for (const job of activeJobs(snapshot)) {
    if (job.kind === "reposition" && !includeRepositions) continue;
    const route = resolveRoute(job.route, routeIndex);
    const coordinates = route?.coordinates ?? [];
    if (!routeCoordinatesAreUsable(coordinates)) continue;
    const car = carById.get(job.car_id);
    const progress = jobProgress(job, car, snapshot, clockMinute, stepMinutes);
    const line = remainingRoute(coordinates, progress);
    if (line.length < 2) continue;
    features.push({
      type: "Feature",
      geometry: {type: "LineString", coordinates: line},
      properties: {
        kind: job.kind,
        car_id: job.car_id,
        person_id: job.person_id ?? null,
        cost: Number(job.route?.cost ?? job.total_cost ?? 0) * (1 - progress)
      }
    });
  }
  return {type: "FeatureCollection", features};
}

function activeAssignmentContext(snapshot) {
  const assignments = snapshot?.map_dispatch?.assignments ?? [];
  const carById = new Map((snapshot?.map_dispatch?.cars ?? []).map(car => [car.id, car]));
  return {
    assignmentByPersonId: new Map(assignments.map(assignment => [assignment.person_id, assignment])),
    carById
  };
}

function visibleMarkerKinds(person, assignment, car, snapshot, clockMinute, stepMinutes) {
  if (!assignment) return {pickup: true, dropoff: true};
  const elapsed = jobElapsedSeconds(assignment, car, snapshot, clockMinute, stepMinutes);
  const pickupCost = Number(assignment.pickup_route?.cost ?? 0);
  const routeCost = Number(assignment.route?.cost ?? assignment.total_cost ?? 0);
  return {
    pickup: elapsed < pickupCost,
    dropoff: elapsed < routeCost
  };
}

function makePeopleFeatures(snapshot, clockMinute, stepMinutes) {
  const features = [];
  const {assignmentByPersonId, carById} = activeAssignmentContext(snapshot);
  for (const person of snapshot?.map_people ?? []) {
    const assignment = assignmentByPersonId.get(person.id);
    const car = assignment ? carById.get(assignment.car_id) : null;
    const visible = visibleMarkerKinds(person, assignment, car, snapshot, clockMinute, stepMinutes);
    if (person.pickup_position && visible.pickup) {
      features.push({
        type: "Feature",
        geometry: {type: "Point", coordinates: person.pickup_position},
        properties: {kind: "pickup", person_id: person.id, status: person.status}
      });
    }
    if (person.dropoff_position && visible.dropoff) {
      features.push({
        type: "Feature",
        geometry: {type: "Point", coordinates: person.dropoff_position},
        properties: {kind: "dropoff", person_id: person.id, status: person.status}
      });
    }
  }
  return {type: "FeatureCollection", features};
}

function gridKey(row, col) {
  return `${row}:${col}`;
}

function makeGridFeatureCollection(grid) {
  if (!grid?.bounds) return null;
  const {
    rows,
    cols,
    bounds: {min_lon: minLon, max_lon: maxLon, min_lat: minLat, max_lat: maxLat},
    values
  } = grid;
  if (!rows || !cols) return null;
  const dLon = (maxLon - minLon) / cols;
  const dLat = (maxLat - minLat) / rows;
  const features = [];
  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      const west = minLon + col * dLon;
      const east = west + dLon;
      const south = minLat + row * dLat;
      const north = south + dLat;
      features.push({
        type: "Feature",
        geometry: {
          type: "Polygon",
          coordinates: [
            [
              [west, south],
              [east, south],
              [east, north],
              [west, north],
              [west, south]
            ]
          ]
        },
        properties: {
          row,
          col,
          population_density: values?.[row]?.[col] ?? 0
        }
      });
    }
  }
  return {type: "FeatureCollection", features};
}

function gridCellForPosition(position, grid) {
  if (!position || !grid?.bounds) return null;
  const {
    rows,
    cols,
    bounds: {min_lon: minLon, max_lon: maxLon, min_lat: minLat, max_lat: maxLat}
  } = grid;
  const [lon, lat] = position;
  if (lon < minLon || lon > maxLon || lat < minLat || lat > maxLat) return null;
  const col = clamp(Math.floor(((lon - minLon) / (maxLon - minLon)) * cols), 0, cols - 1);
  const row = clamp(Math.floor(((lat - minLat) / (maxLat - minLat)) * rows), 0, rows - 1);
  return {row, col, key: gridKey(row, col)};
}

function edgeCentroid(feature) {
  const coords = feature?.geometry?.coordinates ?? [];
  if (!coords.length) return null;
  const flatCoords = feature?.geometry?.type === "MultiLineString" ? coords.flat() : coords;
  if (!flatCoords.length) return null;
  const sum = flatCoords.reduce(
    (acc, coord) => {
      acc.longitude += coord[0] ?? 0;
      acc.latitude += coord[1] ?? 0;
      return acc;
    },
    {longitude: 0, latitude: 0}
  );
  return {
    longitude: sum.longitude / flatCoords.length,
    latitude: sum.latitude / flatCoords.length
  };
}

function registerGridCell(cells, coord, grid) {
  const cell = gridCellForPosition(coord, grid);
  if (cell) cells.set(cell.key, cell);
}

function gridCellsForFeature(feature, grid) {
  if (!feature || !grid) return [];
  const cached = FEATURE_GRID_CELLS_CACHE.get(feature);
  if (cached) return cached;

  const geometry = feature.geometry ?? {};
  const lines =
    geometry.type === "MultiLineString"
      ? geometry.coordinates ?? []
      : geometry.type === "LineString"
        ? [geometry.coordinates ?? []]
        : [];
  const cells = new Map();
  for (const line of lines) {
    for (let i = 0; i < line.length; i++) {
      registerGridCell(cells, line[i], grid);
      if (i === 0) continue;
      const prev = line[i - 1];
      const current = line[i];
      const samples = Math.max(1, Math.ceil(routeSegmentMeters(prev, current) / 220));
      for (let step = 1; step < samples; step++) {
        const t = step / samples;
        registerGridCell(
          cells,
          [
            prev[0] + (current[0] - prev[0]) * t,
            prev[1] + (current[1] - prev[1]) * t
          ],
          grid
        );
      }
    }
  }

  if (!cells.size) {
    const centroid = edgeCentroid(feature);
    if (centroid) registerGridCell(cells, [centroid.longitude, centroid.latitude], grid);
  }

  const result = [...cells.values()];
  FEATURE_GRID_CELLS_CACHE.set(feature, result);
  return result;
}

function makeGeneratedTrafficPressureByCell(snapshot, grid) {
  const hotspots = snapshot?.summary?.traffic_bottlenecks ?? [];
  if (!hotspots.length || !grid?.rows || !grid?.cols) return null;

  const rows = grid.rows;
  const cols = grid.cols;
  const values = new Float32Array(rows * cols);
  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      let influence = 0;
      for (const hotspot of hotspots) {
        const hotRow = Number(hotspot.row);
        const hotCol = Number(hotspot.col);
        const value = Number(hotspot.value);
        if (!Number.isFinite(hotRow) || !Number.isFinite(hotCol) || !Number.isFinite(value)) continue;
        const dist = Math.hypot(row - hotRow, col - hotCol);
        influence = Math.max(influence, value * Math.exp(-(dist * dist) / (2 * 3.2 * 3.2)));
      }
      const normalized = clamp((influence - 0.45) / 0.55, 0, 1);
      values[row * cols + col] = 1 + normalized * 2.8;
    }
  }
  return {rows, cols, values};
}

function generatedTrafficMultiplierForEdge(feature, pressureByCell, grid) {
  if (!pressureByCell || !grid) return 1;
  const cells = gridCellsForFeature(feature, grid);
  let multiplier = 1;
  for (const cell of cells) {
    multiplier = Math.max(multiplier, pressureByCell.values[cell.row * pressureByCell.cols + cell.col] || 1);
  }
  return multiplier;
}

function makeNodeCountsByGridCell(nodes, grid) {
  const counts = new Map();
  for (const node of nodes ?? []) {
    const hasGridCell = Number.isFinite(node.grid_row) && Number.isFinite(node.grid_col);
    const cell = hasGridCell
      ? {row: node.grid_row, col: node.grid_col, key: gridKey(node.grid_row, node.grid_col)}
      : gridCellForPosition([node.lon, node.lat], grid);
    if (!cell) continue;
    counts.set(cell.key, (counts.get(cell.key) ?? 0) + 1);
  }
  return counts;
}

function makeCarPresenceGridFeatureCollection(grid, cars, nodeCountsByCell) {
  const collection = makeGridFeatureCollection(grid);
  if (!collection) return null;
  const presence = new Map();
  for (const car of cars ?? []) {
    const providedCell = car.grid_cell ?? car.grid_position;
    const cellFromPosition = gridCellForPosition(car.position, grid);
    const cell = cellFromPosition ?? (providedCell
      ? {row: providedCell[0], col: providedCell[1], key: gridKey(providedCell[0], providedCell[1])}
      : null);
    if (!cell) continue;
    const current = presence.get(cell.key) ?? {
      row: cell.row,
      col: cell.col,
      car_ids: [],
      active_count: 0,
      idle_count: 0,
      repositioning_count: 0
    };
    current.car_ids.push(car.id);
    if (car.status === "idle") current.idle_count += 1;
    else if (car.status === "repositioning") current.repositioning_count += 1;
    else current.active_count += 1;
    presence.set(cell.key, current);
  }
  return {
    ...collection,
    features: collection.features.map(feature => {
      const {row, col} = feature.properties;
      const cell = presence.get(gridKey(row, col));
      const carCount = cell?.car_ids.length ?? 0;
      return {
        ...feature,
        properties: {
          ...feature.properties,
          ...(cell ?? {}),
          has_car: carCount > 0,
          car_count: carCount,
          intersection_count: nodeCountsByCell.get(gridKey(row, col)) ?? 0
        }
      };
    })
  };
}

function makePeopleGridFeatureCollection(snapshot, grid, clockMinute, stepMinutes) {
  const collection = makeGridFeatureCollection(grid);
  if (!collection) return null;
  const cells = new Map();
  const register = (cell, kind, person) => {
    if (!cell) return;
    const [row, col] = cell;
    const key = gridKey(row, col);
    const current = cells.get(key) ?? {
      row,
      col,
      pickup_count: 0,
      dropoff_count: 0,
      pickup_ids: [],
      dropoff_ids: [],
      pickup_nodes: [],
      dropoff_nodes: []
    };
    if (kind === "pickup") {
      current.pickup_count += 1;
      current.pickup_ids.push(person.id);
      if (person.pickup_node_id) current.pickup_nodes.push(person.pickup_node_id);
    } else {
      current.dropoff_count += 1;
      current.dropoff_ids.push(person.id);
      if (person.dropoff_node_id) current.dropoff_nodes.push(person.dropoff_node_id);
    }
    cells.set(key, current);
  };
  const {assignmentByPersonId, carById} = activeAssignmentContext(snapshot);
  for (const person of snapshot?.map_people ?? []) {
    const assignment = assignmentByPersonId.get(person.id);
    const car = assignment ? carById.get(assignment.car_id) : null;
    const visible = visibleMarkerKinds(person, assignment, car, snapshot, clockMinute, stepMinutes);
    if (visible.pickup) register(person.pickup_grid_cell ?? person.origin, "pickup", person);
    if (visible.dropoff) register(person.dropoff_grid_cell ?? person.destination, "dropoff", person);
  }
  return {
    ...collection,
    features: collection.features.map(feature => {
      const {row, col} = feature.properties;
      const cell = cells.get(gridKey(row, col));
      const pickupCount = cell?.pickup_count ?? 0;
      const dropoffCount = cell?.dropoff_count ?? 0;
      return {
        ...feature,
        properties: {
          ...feature.properties,
          ...(cell ?? {}),
          has_people: pickupCount + dropoffCount > 0,
          has_pickup: pickupCount > 0,
          has_dropoff: dropoffCount > 0,
          people_count: pickupCount + dropoffCount
        }
      };
    })
  };
}

function circlePolygon(center, radiusM, steps = 96) {
  if (!Array.isArray(center) || center.length < 2) return [];
  const [lon, lat] = center.map(Number);
  const latRadians = lat * Math.PI / 180;
  const metersPerDegreeLon = Math.max(1, 111320 * Math.cos(latRadians));
  const dLat = Number(radiusM || 1000) / 111320;
  const dLon = Number(radiusM || 1000) / metersPerDegreeLon;
  const coordinates = [];
  for (let i = 0; i <= steps; i++) {
    const angle = (i / steps) * Math.PI * 2;
    coordinates.push([lon + Math.cos(angle) * dLon, lat + Math.sin(angle) * dLat]);
  }
  return coordinates;
}

function makeEventFeatureCollection(event) {
  if (!event?.center) return {type: "FeatureCollection", features: []};
  return {
    type: "FeatureCollection",
    features: [
      {
        type: "Feature",
        geometry: {
          type: "Polygon",
          coordinates: [circlePolygon(event.center, event.radius_m ?? 1000)]
        },
        properties: {
          id: event.id,
          label: event.label,
          description: event.description
        }
      }
    ]
  };
}

function trafficColor(value) {
  if (value < 1.25) return [55, 124, 92, 104];
  if (value < 1.75) return [185, 143, 54, 112];
  if (value < 2.35) return [192, 103, 48, 124];
  return [196, 58, 56, 138];
}

function roadCongestionForFeature(feature, currentHour, trafficPressureByCell, grid) {
  const base = Number(feature?.properties?.hourly_congestion_factor?.[currentHour] ?? 1);
  return (Number.isFinite(base) ? base : 1) * generatedTrafficMultiplierForEdge(feature, trafficPressureByCell, grid);
}

function metricsFor(snapshot, policyId) {
  const raw = policyId === "greedy" ? snapshot?.map_greedy_stats : snapshot?.map_orchestrator_stats;
  return raw ?? {};
}

function avgWaitMinutes(stats) {
  const completed = Number(stats?.completed_trips ?? 0);
  const servedByPct = Number(stats?.total_requests ?? 0) * Number(stats?.demand_served_pct ?? 0) / 100;
  return Number(stats?.wait_time_min ?? 0) / Math.max(1, completed, servedByPct);
}

function finalTotalRequests(snapshots, policyId) {
  if (!snapshots?.length) return 0;
  const finalMetrics = metricsFor(snapshots[snapshots.length - 1], policyId);
  return Number(finalMetrics?.total_requests ?? 0);
}

function metricDelta(rl, greedy, key, lowerIsBetter = false) {
  const a = Number(rl?.[key] ?? 0);
  const b = Number(greedy?.[key] ?? 0);
  const delta = a - b;
  const good = lowerIsBetter ? delta < 0 : delta > 0;
  return {delta, good};
}

function eventChoices(greedyWorld, rlWorld) {
  const greedyScenarioIds = new Set(Object.keys(greedyWorld?.event_scenarios ?? {}));
  const rlScenarioIds = new Set(Object.keys(rlWorld?.event_scenarios ?? {}));
  return (greedyWorld?.events ?? [])
    .filter(event => greedyScenarioIds.has(event.id) && rlScenarioIds.has(event.id))
    .map(event => ({
      id: event.id,
      label: event.label,
      short_label: event.short_label ?? event.label,
      description: event.description,
      center: event.center,
      radius_m: event.radius_m
    }));
}

function MapPanel({policy, world, network, grid, nodes, snapshot, routeIndex, clockMinute, stepMinutes, viewState, setViewState, height, event}) {
  const cars = useMemo(
    () => animatedCars(snapshot, routeIndex, clockMinute, stepMinutes),
    [snapshot, routeIndex, clockMinute, stepMinutes]
  );
  const nodeCountsByCell = useMemo(
    () => makeNodeCountsByGridCell(nodes, grid),
    [nodes, grid]
  );
  const carGridData = useMemo(
    () => makeCarPresenceGridFeatureCollection(grid, cars, nodeCountsByCell),
    [grid, cars, nodeCountsByCell]
  );
  const peopleGridData = useMemo(
    () => makePeopleGridFeatureCollection(snapshot, grid, clockMinute, stepMinutes),
    [snapshot, grid, clockMinute, stepMinutes]
  );
  const trafficPressureByCell = useMemo(
    () => makeGeneratedTrafficPressureByCell(snapshot, grid),
    [snapshot, grid]
  );
  const metrics = metricsFor(snapshot, policy.id);
  const currentHour = Math.floor(clamp(clockMinute, 0, DAY_MINUTES - 1) / 60);
  const layers = useMemo(() => {
    const roads = network
      ? new GeoJsonLayer({
          id: `${policy.id}-roads`,
          data: network,
          stroked: true,
          filled: false,
          lineWidthMinPixels: 0.85,
          lineWidthMaxPixels: 3.65,
          getLineWidth: f => {
            const congestion = roadCongestionForFeature(f, currentHour, trafficPressureByCell, grid);
            return 0.7 + clamp((congestion - 1) / 3.4, 0, 1) * 2.75;
          },
          getLineColor: f => trafficColor(roadCongestionForFeature(f, currentHour, trafficPressureByCell, grid)),
          lineCapRounded: true,
          lineJointRounded: true,
          parameters: {depthTest: false},
          pickable: true,
          updateTriggers: {
            getLineWidth: [currentHour, trafficPressureByCell, grid],
            getLineColor: [currentHour, trafficPressureByCell, grid]
          }
        })
      : null;
    const carGrid = carGridData
      ? new GeoJsonLayer({
          id: `${policy.id}-car-grid`,
          data: carGridData,
          stroked: true,
          filled: true,
          pickable: true,
          autoHighlight: true,
          highlightColor: [255, 255, 255, 70],
          lineWidthMinPixels: 0.35,
          lineWidthMaxPixels: 2.5,
          getLineWidth: f => (f.properties?.has_car ? 2 : 0.55),
          getFillColor: f =>
            f.properties?.has_car ? [0, 210, 165, 92] : [16, 31, 48, 16],
          getLineColor: f =>
            f.properties?.has_car ? [0, 255, 190, 225] : [130, 170, 205, 48]
        })
      : null;
    const peopleGrid = peopleGridData
      ? new GeoJsonLayer({
          id: `${policy.id}-people-grid`,
          data: peopleGridData,
          stroked: true,
          filled: true,
          pickable: true,
          autoHighlight: true,
          highlightColor: [255, 255, 255, 65],
          lineWidthMinPixels: 0.65,
          getLineWidth: f => (f.properties?.has_people ? 3 : 0.45),
          getFillColor: f => {
            const p = f.properties ?? {};
            if (p.has_pickup && p.has_dropoff) return PEOPLE_GRID_BOTH_FILL;
            if (p.has_pickup) return PICKUP_BLUE_FILL;
            if (p.has_dropoff) return DESTINATION_RED_FILL;
            return PEOPLE_GRID_EMPTY_FILL;
          },
          getLineColor: f => {
            const p = f.properties ?? {};
            if (p.has_pickup && p.has_dropoff) return PEOPLE_GRID_BOTH_LINE;
            if (p.has_pickup) return PICKUP_BLUE;
            if (p.has_dropoff) return DESTINATION_RED;
            return PEOPLE_GRID_EMPTY_LINE;
          }
        })
      : null;
    const routeCasing = new GeoJsonLayer({
      id: `${policy.id}-route-casing`,
      data: makeRouteFeatures(snapshot, routeIndex, clockMinute, stepMinutes, true),
      stroked: true,
      filled: false,
      lineWidthMinPixels: 10,
      lineWidthMaxPixels: 14,
      getLineColor: policy.casing,
      pickable: false
    });
    const routes = new GeoJsonLayer({
      id: `${policy.id}-routes`,
      data: makeRouteFeatures(snapshot, routeIndex, clockMinute, stepMinutes, true),
      stroked: true,
      filled: false,
      lineWidthMinPixels: 5,
      lineWidthMaxPixels: 9,
      getLineColor: f => f.properties?.kind === "reposition" ? policy.reposition ?? [20, 184, 166, 230] : policy.route,
      pickable: true
    });
    const people = new GeoJsonLayer({
      id: `${policy.id}-people`,
      data: makePeopleFeatures(snapshot, clockMinute, stepMinutes),
      stroked: true,
      filled: true,
      pointType: "circle",
      pointRadiusUnits: "pixels",
      pointRadiusMinPixels: 7,
      pointRadiusMaxPixels: 15,
      getPointRadius: f => f.properties?.kind === "pickup" ? 9 : 8,
      getFillColor: f => f.properties?.kind === "pickup" ? [66, 133, 244, 210] : [234, 67, 53, 205],
      getLineColor: [255, 255, 255, 230],
      lineWidthMinPixels: 1.5,
      pickable: true
    });
    const carLayer = new ScatterplotLayer({
      id: `${policy.id}-cars`,
      data: cars,
      pickable: true,
      radiusUnits: "meters",
      radiusMinPixels: 4,
      radiusMaxPixels: 12,
      getRadius: d => d.status === "idle" ? 30 : 44,
      getPosition: d => d.position,
      getFillColor: d => d.status === "idle" ? policy.idle : d.status === "repositioning" ? policy.reposition ?? policy.active : policy.active,
      getLineColor: [255, 255, 255, 225],
      lineWidthMinPixels: 1.2,
      stroked: true
    });
    const surgeArea = event
      ? new GeoJsonLayer({
          id: `${policy.id}-surge-area`,
          data: makeEventFeatureCollection(event),
          stroked: true,
          filled: true,
          lineWidthMinPixels: 3,
          lineWidthMaxPixels: 6,
          getFillColor: [239, 68, 68, 46],
          getLineColor: [248, 113, 113, 245],
          pickable: true
        })
      : null;
    const surgeCore = event
      ? new ScatterplotLayer({
          id: `${policy.id}-surge-core`,
          data: [event],
          radiusUnits: "meters",
          getRadius: d => Math.max(90, Number(d.radius_m ?? 1000) * 0.09),
          getPosition: d => d.center,
          getFillColor: [239, 68, 68, 190],
          getLineColor: [254, 226, 226, 245],
          lineWidthMinPixels: 2,
          stroked: true,
          pickable: false
        })
      : null;
    return [carGrid, peopleGrid, roads, surgeArea, routeCasing, routes, people, carLayer, surgeCore].filter(Boolean);
  }, [network, grid, carGridData, peopleGridData, policy, snapshot, routeIndex, clockMinute, stepMinutes, cars, currentHour, event, trafficPressureByCell]);

  return (
    <section style={{position: "relative", height, minHeight: height, overflow: "hidden"}}>
      <DeckGL
        viewState={viewState}
        onViewStateChange={({viewState: next}) => setViewState(next)}
        controller
        layers={layers}
        getTooltip={({object, layer}) => {
          if (!object || !layer) return null;
          if (layer.id.endsWith("-cars")) {
            return {text: `${object.id}\nstatus: ${object.status}\nperson: ${object.assigned_person_id ?? "none"}`};
          }
          if (layer.id.endsWith("-routes")) {
            const p = object.properties ?? {};
            return {text: `${p.kind === "reposition" ? "Reposition" : "Trip"} route\ncar: ${p.car_id}\nperson: ${p.person_id ?? "none"}\nremaining cost: ${Number(p.cost ?? 0).toFixed(1)}`};
          }
          if (layer.id.endsWith("-people")) {
            const p = object.properties ?? {};
            return {text: `${p.kind}\nperson: ${p.person_id}\nstatus: ${p.status}`};
          }
          if (layer.id.endsWith("-car-grid")) {
            const p = object.properties ?? {};
            return {text: `grid cell [${p.row}, ${p.col}]\ncars: ${p.car_count ?? 0}\nactive: ${p.active_count ?? 0}\nidle: ${p.idle_count ?? 0}\nrepositioning: ${p.repositioning_count ?? 0}\nintersection nodes: ${p.intersection_count ?? 0}`};
          }
          if (layer.id.endsWith("-people-grid")) {
            const p = object.properties ?? {};
            return {text: `grid cell [${p.row}, ${p.col}]\npickups: ${p.pickup_count ?? 0}\ndestinations: ${p.dropoff_count ?? 0}`};
          }
          if (layer.id.endsWith("-roads")) {
            const p = object.properties ?? {};
            const congestion = roadCongestionForFeature(object, currentHour, trafficPressureByCell, grid);
            return {text: `${p.name ?? "road"}\ntraffic: ${congestion.toFixed(2)}x\nhour: ${currentHour}:00`};
          }
          return null;
        }}
        style={{width: "100%", height: "100%"}}
      >
        <StaticMap
          mapStyle={buildStyleUrl()}
          onLoad={event => {
            const map = event.target;
            hideBaseTransportationLayers(map);
            map.on("styledata", () => hideBaseTransportationLayers(map));
            setTimeout(() => hideBaseTransportationLayers(map), 250);
          }}
        />
      </DeckGL>
      <div style={{
        position: "absolute",
        top: 12,
        left: 12,
        width: 320,
        maxWidth: "calc(100% - 24px)",
        padding: 12,
        borderRadius: 8,
        background: policy.panel,
        border: `1px solid ${policy.accent}66`,
        color: "white",
        fontFamily: "system-ui, sans-serif",
        boxShadow: "0 18px 52px rgba(0,0,0,0.38)",
        pointerEvents: "none"
      }}>
        <div style={{fontSize: 12, opacity: 0.68}}>Same map, same request stream</div>
        <div style={{fontSize: 20, fontWeight: 800, color: policy.accent, marginTop: 2}}>{policy.label}</div>
        <div style={{display: "grid", gridTemplateColumns: "repeat(2, minmax(0,1fr))", gap: 8, marginTop: 12}}>
          <Metric label="Trips" value={Number(metrics.completed_trips ?? 0).toLocaleString()} />
          <Metric label="Profit" value={formatMoney(metrics.revenue)} />
          <Metric label="Demand Served" value={formatPct(metrics.demand_served_pct)} />
          <Metric label="Avg Wait" value={`${avgWaitMinutes(metrics).toFixed(1)}m`} />
          <Metric label="Util." value={formatPct(metrics.avg_fleet_utilization_pct ?? metrics.fleet_utilization_pct)} />
          <Metric label="Active" value={Number(metrics.active_cars ?? 0).toLocaleString()} />
        </div>
        {policy.id === "rl" && (
          <div style={{fontSize: 12, opacity: 0.76, marginTop: 10}}>
            Repositioning cars: {metrics.repositioning_cars ?? 0}
          </div>
        )}
      </div>
    </section>
  );
}

function Metric({label, value}) {
  return (
    <div style={{padding: "8px 9px", borderRadius: 7, background: "rgba(255,255,255,0.07)", border: "1px solid rgba(255,255,255,0.09)"}}>
      <div style={{fontSize: 11, opacity: 0.62}}>{label}</div>
      <div style={{fontSize: 18, fontWeight: 800, lineHeight: 1.1, marginTop: 4}}>{value}</div>
    </div>
  );
}

function TrafficLegend() {
  const items = [
    ["Light", [55, 124, 92, 104]],
    ["Moderate", [185, 143, 54, 112]],
    ["Heavy", [192, 103, 48, 124]],
    ["Severe", [196, 58, 56, 138]]
  ];
  return (
    <div style={{display: "flex", alignItems: "center", gap: 7, flexWrap: "wrap", justifyContent: "flex-end"}}>
      <span style={{fontSize: 12, opacity: 0.62, fontWeight: 700}}>Traffic</span>
      {items.map(([label, color]) => (
        <span
          key={label}
          style={{
            display: "inline-flex",
            alignItems: "center",
            gap: 5,
            padding: "5px 7px",
            borderRadius: 7,
            background: "rgba(15,23,42,0.72)",
            border: "1px solid rgba(148,163,184,0.24)",
            fontSize: 12,
            color: "rgba(255,255,255,0.82)"
          }}
        >
          <span style={{width: 20, height: 3, borderRadius: 99, background: rgba(color), display: "inline-block"}} />
          {label}
        </span>
      ))}
    </div>
  );
}

function DeltaBar({greedySnapshot, rlSnapshot}) {
  const greedy = metricsFor(greedySnapshot, "greedy");
  const rl = metricsFor(rlSnapshot, "rl");
  const trips = metricDelta(rl, greedy, "completed_trips");
  const revenue = metricDelta(rl, greedy, "revenue");
  const demand = metricDelta(rl, greedy, "demand_served_pct");
  const avgWait = {
    delta: avgWaitMinutes(rl) - avgWaitMinutes(greedy),
    good: avgWaitMinutes(rl) < avgWaitMinutes(greedy)
  };
  return (
    <div style={{display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap", fontSize: 13}}>
      <Delta label="Trips" value={trips.delta} good={trips.good} />
      <Delta label="Profit" value={revenue.delta} good={revenue.good} money />
      <Delta label="Demand Served" value={demand.delta} good={demand.good} suffix="pp" />
      <Delta label="Avg Wait" value={avgWait.delta} good={avgWait.good} suffix="m" />
    </div>
  );
}

function Delta({label, value, good, money, suffix = ""}) {
  const sign = value > 0 ? "+" : "";
  const shown = money ? `${sign}${formatMoney(value).replace("$-", "-$")}` : `${sign}${value.toFixed(1)}${suffix}`;
  const neutral = Math.abs(Number(value) || 0) < 0.001;
  return (
    <span style={{
      display: "inline-flex",
      gap: 6,
      alignItems: "center",
      padding: "6px 8px",
      borderRadius: 999,
      background: neutral ? "rgba(148,163,184,0.12)" : good ? "rgba(20,184,166,0.16)" : "rgba(248,113,113,0.15)",
      color: neutral ? "#cbd5e1" : good ? "#99f6e4" : "#fecaca",
      border: neutral ? "1px solid rgba(148,163,184,0.26)" : good ? "1px solid rgba(45,212,191,0.32)" : "1px solid rgba(248,113,113,0.28)"
    }}>
      <strong>{label}</strong> {shown}
    </span>
  );
}

function formatSignedInteger(value) {
  const numeric = Number(value) || 0;
  const sign = numeric > 0 ? "+" : numeric < 0 ? "-" : "";
  return `${sign}${Math.abs(Math.round(numeric)).toLocaleString()}`;
}

function formatSignedDollars(value) {
  const numeric = Number(value) || 0;
  const sign = numeric > 0 ? "+" : numeric < 0 ? "-" : "";
  return `${sign}$${Math.floor(Math.abs(numeric)).toLocaleString()}`;
}

function formatSignedDecimal(value, digits = 1) {
  const numeric = Number(value) || 0;
  const sign = numeric > 0 ? "+" : numeric < 0 ? "-" : "";
  return `${sign}${Math.abs(numeric).toFixed(digits)}`;
}

function BusinessImpactOverlay({greedySnapshot, rlSnapshot, visible}) {
  if (!visible) return null;
  const greedy = metricsFor(greedySnapshot, "greedy");
  const rl = metricsFor(rlSnapshot, "rl");
  const rows = [
    {
      value: formatSignedInteger(Number(rl.completed_trips ?? 0) - Number(greedy.completed_trips ?? 0)),
      label: "completed trips"
    },
    {
      value: formatSignedDollars(Number(rl.revenue ?? 0) - Number(greedy.revenue ?? 0)),
      label: "revenue"
    },
    {
      value: formatSignedDecimal(Number(rl.demand_served_pct ?? 0) - Number(greedy.demand_served_pct ?? 0), 1),
      label: "percentage points demand served"
    },
    {
      value: formatSignedDecimal(avgWaitMinutes(rl) - avgWaitMinutes(greedy), 1),
      label: "minutes average wait"
    }
  ];

  return (
    <div style={{
      position: "absolute",
      left: "50%",
      bottom: 24,
      transform: "translateX(-50%)",
      width: "min(620px, calc(100vw - 36px))",
      padding: 16,
      borderRadius: 8,
      background: "rgba(3,7,18,0.92)",
      border: "1px solid rgba(45,212,191,0.34)",
      color: "white",
      fontFamily: "system-ui, sans-serif",
      boxShadow: "0 22px 60px rgba(0,0,0,0.46)",
      pointerEvents: "none",
      zIndex: 20
    }}>
      <div style={{fontSize: 12, opacity: 0.62, fontWeight: 800, textTransform: "uppercase", letterSpacing: 0}}>
        Business Impact
      </div>
      <div style={{display: "grid", gridTemplateColumns: "repeat(2, minmax(0, 1fr))", gap: 10, marginTop: 10}}>
        {rows.map(row => (
          <div
            key={row.label}
            style={{
              minWidth: 0,
              padding: "10px 11px",
              borderRadius: 7,
              background: "rgba(15,23,42,0.76)",
              border: "1px solid rgba(148,163,184,0.16)"
            }}
          >
            <div style={{fontSize: 22, lineHeight: 1, fontWeight: 900, color: row.value.startsWith("-") ? "#99f6e4" : "#bae6fd"}}>
              {row.value}
            </div>
            <div style={{fontSize: 12, opacity: 0.74, marginTop: 5, fontWeight: 700}}>
              {row.label}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function useComparisonData() {
  const [state, setState] = useState({status: "loading"});
  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const [greedyWorld, rlWorld, network, grid, nodeData] = await Promise.all([
          fetchJson(["/data/mobility_world.json"]),
          fetchJson(["/data/mobility_orchestrator_world.json"]),
          fetchJson(["/data/osmnx_edges.geojson", "/dist/data/osmnx_edges.geojson"]),
          fetchJson(["/data/population_density_grid.json", "/dist/data/population_density_grid.json"]),
          fetchJson(["/data/ppo_nodes.json", "/dist/data/ppo_nodes.json"])
        ]);
        const nodes = Array.isArray(nodeData?.nodes) ? nodeData.nodes : Array.isArray(nodeData) ? nodeData : [];
        if (!cancelled) setState({status: "ready", greedyWorld, rlWorld, network, grid, nodes});
      } catch (error) {
        console.error(error);
        if (!cancelled) setState({status: "error", error});
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, []);
  return state;
}

function ControlBar({
  running,
  onStart,
  onPause,
  onReset,
  speed,
  setSpeed,
  clockLabel,
  greedySnapshot,
  rlSnapshot,
  compare,
  events,
  activeEventId,
  onEventChange,
  baseDemandRequests
}) {
  return (
    <header style={{
      minHeight: compare ? 112 : 104,
      display: "flex",
      alignItems: "center",
      justifyContent: "space-between",
      gap: 14,
      padding: "10px 14px",
      boxSizing: "border-box",
      background: "rgba(4,8,14,0.96)",
      color: "white",
      borderBottom: "1px solid rgba(148,163,184,0.22)",
      fontFamily: "system-ui, sans-serif"
    }}>
      <div style={{display: "flex", flexDirection: "column", gap: 4}}>
        <div style={{fontSize: 12, opacity: 0.66}}>{compare ? "Synchronized comparison" : "Agentic Fleet page"}</div>
        <div style={{fontSize: 20, fontWeight: 850}}>{compare ? "Greedy vs Agentic Fleet" : "Agentic Fleet"}</div>
        {compare && (
          <div style={{fontSize: 12, opacity: 0.68}}>
            Base demand: {Number(baseDemandRequests ?? 0).toLocaleString()} requests, same stream for both policies
          </div>
        )}
        {compare && (
          <DeltaBar
            greedySnapshot={greedySnapshot}
            rlSnapshot={rlSnapshot}
          />
        )}
      </div>
      <div style={{display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 8}}>
        <div style={{display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap", justifyContent: "flex-end"}}>
          <a href="/" style={navButtonStyle}>Greedy Page</a>
          <a href="/rl.html" style={navButtonStyle}>Agentic Fleet</a>
          <a href="/compare.html" style={navButtonStyle}>Compare</a>
          <span style={{padding: "8px 10px", border: "1px solid rgba(148,163,184,0.32)", borderRadius: 7, minWidth: 60, textAlign: "center"}}>
            {clockLabel}
          </span>
          <button type="button" onClick={() => setSpeed(SPEEDS[(SPEEDS.indexOf(speed) + 1) % SPEEDS.length])} style={buttonStyle}>
            x{speed}
          </button>
          <button type="button" onClick={running ? onPause : onStart} style={{...buttonStyle, background: running ? "#facc15" : "#14b8a6", color: "#061016"}}>
            {running ? "Pause" : compare ? "Start Both" : "Start"}
          </button>
          <button type="button" onClick={onReset} style={buttonStyle}>Reset</button>
        </div>
        {events.length > 0 && (
          <div style={{display: "flex", alignItems: "center", gap: 7, flexWrap: "wrap", justifyContent: "flex-end"}}>
            <span style={{fontSize: 12, opacity: 0.62, fontWeight: 700}}>Events</span>
            <button
              type="button"
              onClick={() => onEventChange(null)}
              style={activeEventId ? eventButtonStyle : activeEventButtonStyle}
            >
              Base
            </button>
            {events.map(event => {
              const active = activeEventId === event.id;
              return (
                <button
                  key={event.id}
                  type="button"
                  title={event.description}
                  onClick={() => onEventChange(event.id)}
                  style={active ? activeEventButtonStyle : eventButtonStyle}
                >
                  {event.short_label}
                </button>
              );
            })}
          </div>
        )}
        <TrafficLegend />
      </div>
    </header>
  );
}

const buttonStyle = {
  padding: "9px 11px",
  border: "1px solid rgba(148,163,184,0.34)",
  borderRadius: 7,
  background: "black",
  color: "white",
  cursor: "pointer",
  fontWeight: 700
};

const navButtonStyle = {
  ...buttonStyle,
  textDecoration: "none",
  display: "inline-flex",
  alignItems: "center"
};

const eventButtonStyle = {
  ...buttonStyle,
  padding: "7px 9px",
  fontSize: 12,
  background: "rgba(15,23,42,0.82)"
};

const activeEventButtonStyle = {
  ...eventButtonStyle,
  background: "#7dd3fc",
  color: "#061016",
  border: "1px solid rgba(255,255,255,0.65)"
};

function ComparisonShell({compare}) {
  const data = useComparisonData();
  const [clockMinute, setClockMinute] = useState(PRE_FRAME_MINUTE);
  const [running, setRunning] = useState(false);
  const [speed, setSpeed] = useState(0.5);
  const [activeEventId, setActiveEventId] = useState(null);
  const [viewState, setViewState] = useState(INITIAL_VIEW_STATE);

  const events = useMemo(
    () => eventChoices(data.greedyWorld, data.rlWorld),
    [data.greedyWorld, data.rlWorld]
  );
  const activeEvent = events.find(event => event.id === activeEventId) ?? null;
  const greedySnapshots = useMemo(
    () => scenarioSnapshots(data.greedyWorld, activeEventId, "greedy"),
    [data.greedyWorld, activeEventId]
  );
  const rlSnapshots = useMemo(
    () => scenarioSnapshots(data.rlWorld, activeEventId, "rl"),
    [data.rlWorld, activeEventId]
  );
  const greedyStep = scenarioStepMinutes(data.greedyWorld, activeEventId);
  const rlStep = scenarioStepMinutes(data.rlWorld, activeEventId);
  const mapHeight = "100%";
  const endMinute = useMemo(() => {
    const greedyEnd = greedySnapshots.at(-1)?.timestep ?? DAY_MINUTES;
    const rlEnd = rlSnapshots.at(-1)?.timestep ?? DAY_MINUTES;
    return Math.min(greedyEnd, rlEnd) + Math.min(greedyStep, rlStep) - 0.001;
  }, [greedySnapshots, rlSnapshots, greedyStep, rlStep]);
  const baseDemandRequests = useMemo(
    () => Math.max(finalTotalRequests(greedySnapshots, "greedy"), finalTotalRequests(rlSnapshots, "rl")),
    [greedySnapshots, rlSnapshots]
  );

  useEffect(() => {
    const timer = setInterval(() => {
      if (!running) return;
      setClockMinute(current => {
        const next = current + speed;
        if (next >= endMinute) {
          setRunning(false);
          return endMinute;
        }
        return next;
      });
    }, 80);
    return () => clearInterval(timer);
  }, [running, speed, endMinute]);

  useEffect(() => {
    if (!activeEventId) return;
    if (!events.some(event => event.id === activeEventId)) setActiveEventId(null);
  }, [activeEventId, events]);

  if (data.status !== "ready") {
    return (
      <div style={{height: "100vh", display: "grid", placeItems: "center", background: "#04080e", color: "white", fontFamily: "system-ui, sans-serif"}}>
        {data.status === "error" ? "Could not load comparison data." : "Loading comparison data..."}
      </div>
    );
  }

  const greedySnapshot = snapshotAt(greedySnapshots, clockMinute);
  const rlSnapshot = snapshotAt(rlSnapshots, clockMinute);
  const finalGreedySnapshot = greedySnapshots.at(-1);
  const finalRlSnapshot = rlSnapshots.at(-1);
  const showBusinessImpact = compare && clockMinute >= endMinute - 0.01;
  const onStart = () => {
    if (clockMinute >= endMinute - 0.01) setClockMinute(PRE_FRAME_MINUTE);
    setRunning(true);
  };
  const onReset = () => {
    setClockMinute(PRE_FRAME_MINUTE);
    setRunning(false);
  };
  const onEventChange = nextEventId => {
    setActiveEventId(nextEventId);
    setClockMinute(PRE_FRAME_MINUTE);
    setRunning(false);
  };

  return (
    <main style={{width: "100vw", height: "100vh", overflow: "hidden", background: "#04080e", display: "grid", gridTemplateRows: "auto minmax(0,1fr)", position: "relative"}}>
      <ControlBar
        running={running}
        onStart={onStart}
        onPause={() => setRunning(false)}
        onReset={onReset}
        speed={speed}
        setSpeed={setSpeed}
        clockLabel={formatClock(clockMinute)}
        greedySnapshot={greedySnapshot}
        rlSnapshot={rlSnapshot}
        compare={compare}
        events={events}
        activeEventId={activeEventId}
        onEventChange={onEventChange}
        baseDemandRequests={baseDemandRequests}
      />
      {compare ? (
        <div style={{display: "grid", gridTemplateColumns: "1fr 1fr", height: mapHeight, minHeight: 0}}>
          <MapPanel
            policy={GREEDY}
            world={data.greedyWorld}
            network={data.network}
            grid={data.grid}
            nodes={data.nodes}
            snapshot={greedySnapshot}
            routeIndex={data.greedyWorld.routes}
            clockMinute={clockMinute}
            stepMinutes={greedyStep}
            viewState={viewState}
            setViewState={setViewState}
            height={mapHeight}
            event={activeEvent}
          />
          <MapPanel
            policy={RL}
            world={data.rlWorld}
            network={data.network}
            grid={data.grid}
            nodes={data.nodes}
            snapshot={rlSnapshot}
            routeIndex={data.rlWorld.routes}
            clockMinute={clockMinute}
            stepMinutes={rlStep}
            viewState={viewState}
            setViewState={setViewState}
            height={mapHeight}
            event={activeEvent}
          />
        </div>
      ) : (
        <MapPanel
          policy={RL}
          world={data.rlWorld}
          network={data.network}
          grid={data.grid}
          nodes={data.nodes}
          snapshot={rlSnapshot}
          routeIndex={data.rlWorld.routes}
          clockMinute={clockMinute}
          stepMinutes={rlStep}
          viewState={viewState}
          setViewState={setViewState}
          height={mapHeight}
          event={activeEvent}
        />
      )}
      <BusinessImpactOverlay
        greedySnapshot={finalGreedySnapshot}
        rlSnapshot={finalRlSnapshot}
        visible={showBusinessImpact}
      />
    </main>
  );
}

document.body.style.margin = 0;
const root = document.createElement("div");
document.body.appendChild(root);
createRoot(root).render(<ComparisonShell compare={!window.location.pathname.includes("rl.html")} />);
