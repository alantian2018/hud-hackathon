import React, {useEffect, useMemo, useState} from "react";
import {createRoot} from "react-dom/client";
import DeckGL from "@deck.gl/react";
import {TripsLayer} from "@deck.gl/geo-layers";
import {GeoJsonLayer, ScatterplotLayer, IconLayer} from "@deck.gl/layers";
import {StaticMap} from "react-map-gl";
import "mapbox-gl/dist/mapbox-gl.css";


const MAPTILER_KEY = "4WonZ3glTzG3MWfQd6gQ";
const POPULATION_HOURLY_MULTIPLIER = [
  0.82, 0.78, 0.74, 0.72, 0.74, 0.82, 0.96, 1.1, 1.18, 1.2, 1.16, 1.1,
  1.06, 1.02, 1.0, 1.04, 1.12, 1.2, 1.14, 1.04, 0.96, 0.9, 0.86, 0.84
];
const ROAD_DISPLAY_SENSITIVITY = {
  motorway: 2.4,
  trunk: 2.0,
  primary: 1.65,
  secondary: 1.1,
  tertiary: 0.8,
  residential: 0.32,
  service: 0.25,
  living_street: 0.2
};
const DOWNTOWN_CONGESTION_CENTERS = [
  {longitude: -122.399, latitude: 37.794, weight: 1.15}, // Financial District
  {longitude: -122.408, latitude: 37.785, weight: 1.0}, // SoMa / Market
  {longitude: -122.419, latitude: 37.775, weight: 0.85} // Civic / Mission corridor
];
const COMMUTE_CORE = {longitude: -122.407, latitude: 37.787};
const DAY_MINUTES = 24 * 60;
const SIM_RESET_MINUTE = 0;
const SIM_START_MINUTE = 7 * 60;
const SIM_END_MINUTE = DAY_MINUTES - 0.001;
const SIM_SPEED_STEPS = [0.1, 0.5, 1, 4, 12, 30];
const TIME_PRESETS = [
  {label: "02:00", minute: 2 * 60},
  {label: "08:00", minute: 8 * 60},
  {label: "12:00", minute: 12 * 60},
  {label: "17:00", minute: 17 * 60},
  {label: "21:00", minute: 21 * 60}
];
const TRAFFIC_GREEN = [52, 168, 83, 245];
const TRAFFIC_YELLOW = [251, 188, 4, 245];
const TRAFFIC_ORANGE = [251, 140, 0, 245];
const TRAFFIC_RED = [234, 67, 53, 245];
const FEATURE_GRID_CELL_CACHE = new WeakMap();
const TRAFFIC_DARK_RED = [165, 14, 14, 250];
const PICKUP_BLUE = [66, 133, 244, 245];
const PICKUP_BLUE_FILL = [66, 133, 244, 175];
const DESTINATION_RED = [234, 67, 53, 245];
const DESTINATION_RED_FILL = [234, 67, 53, 170];
const PEOPLE_GRID_EMPTY_FILL = [18, 34, 52, 0];
const PEOPLE_GRID_EMPTY_LINE = [138, 180, 248, 38];
const PEOPLE_GRID_BOTH_FILL = [168, 85, 247, 190];
const PEOPLE_GRID_BOTH_LINE = [216, 180, 254, 245];
const ROUTE_BLUE = [66, 133, 244, 245];
const ROUTE_CASING = [232, 240, 254, 245];
const MAX_VISIBLE_ROUTE_SEGMENT_METERS = 350;
const ACTIVE_CAR_GREEN = [52, 168, 83, 245];
const IDLE_CAR_GRAY = [189, 193, 198, 230];
const UBER_CAR_ICON_SVG = encodeURIComponent(
  `<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 64 64">
    <rect x="8" y="22" width="48" height="22" rx="7" fill="#0c0c0c" stroke="#f0f0f0" stroke-width="2"/>
    <rect x="18" y="18" width="28" height="10" rx="4" fill="#191919" stroke="#f0f0f0" stroke-width="1.5"/>
    <circle cx="20" cy="46" r="6" fill="#111" stroke="#d9d9d9" stroke-width="2"/>
    <circle cx="44" cy="46" r="6" fill="#111" stroke="#d9d9d9" stroke-width="2"/>
    <rect x="23" y="28" width="18" height="8" rx="2" fill="#00c2ff"/>
  </svg>`
);
const UBER_CAR_ICON_URL = `data:image/svg+xml;charset=utf-8,${UBER_CAR_ICON_SVG}`;

// 🌉 Initial cinematic SF view
const INITIAL_VIEW_STATE = {
  longitude: -122.4194,
  latitude: 37.7749,
  zoom: 13.8,
  pitch: 60,
  bearing: -20
};

const FALLBACK_TRIPS = [
  {
    path: [
      [-122.4194, 37.7749],
      [-122.418, 37.776],
      [-122.416, 37.778],
      [-122.413, 37.779],
      [-122.410, 37.780],
      [-122.407, 37.782]
    ],
    timestamps: [420, 428, 436, 444, 452, 460],
    start_hour: 7
  }
];

function buildStyleUrl() {
  return `https://api.maptiler.com/maps/streets-v2-dark/style.json?key=${MAPTILER_KEY}`;
}

async function fetchFirstJson(paths) {
  let lastError = null;
  for (const path of paths) {
    try {
      const response = await fetch(path);
      if (!response.ok) throw new Error(`${path} ${response.status}`);
      return await response.json();
    } catch (error) {
      lastError = error;
    }
  }
  throw lastError;
}

function hideBaseTransportationLayers(map) {
  const styleLayers = map.getStyle()?.layers ?? [];
  for (const layer of styleLayers) {
    const id = layer.id ?? "";
    const sourceLayer = layer["source-layer"] ?? "";
    const isRoadGeometry =
      sourceLayer === "transportation" &&
      (layer.type === "line" || layer.type === "fill");

    if (!isRoadGeometry || !map.getLayer(id)) continue;
    map.setLayoutProperty(id, "visibility", "none");
  }
}

function clamp(v, min, max) {
  return Math.max(min, Math.min(max, v));
}

function densityColor(value, min, max, alpha = 170) {
  const t = clamp((value - min) / Math.max(1e-9, max - min), 0, 1);
  const r = Math.round(40 + 215 * t);
  const g = Math.round(60 + 110 * (1 - t));
  const b = Math.round(235 - 200 * t);
  return [r, g, b, alpha];
}

function trafficColorFromT(t, alpha = 225) {
  const x = clamp(t, 0, 1);
  if (x <= 0.5) {
    const k = x / 0.5;
    // Green -> Yellow
    return [Math.round(50 + 205 * k), Math.round(200 + 20 * k), 60, alpha];
  }
  const k = (x - 0.5) / 0.5;
  // Yellow -> Red
  return [255, Math.round(220 - 170 * k), Math.round(60 - 20 * k), alpha];
}

function trafficColorFromCongestion(congestion) {
  if (congestion < 1.45) return TRAFFIC_GREEN;
  if (congestion < 2.05) return TRAFFIC_YELLOW;
  if (congestion < 2.85) return TRAFFIC_ORANGE;
  if (congestion < 4.2) return TRAFFIC_RED;
  return TRAFFIC_DARK_RED;
}

function hourlyValue(values, hour) {
  if (!Array.isArray(values) || values.length === 0) return 1;
  const wrapped = ((hour % 24) + 24) % 24;
  const h0 = Math.floor(wrapped);
  const h1 = (h0 + 1) % values.length;
  const t = wrapped - h0;
  return (values[h0] ?? values[0] ?? 1) * (1 - t) + (values[h1] ?? values[0] ?? 1) * t;
}

function timeWave(hour, peakHour, spreadHours) {
  const d = Math.min(Math.abs(hour - peakHour), 24 - Math.abs(hour - peakHour));
  return Math.exp(-(d * d) / (2 * spreadHours * spreadHours));
}

function populationTimeFactor(hour, row, col, rows, cols) {
  const baseHour = POPULATION_HOURLY_MULTIPLIER[hour] ?? 1;
  const cx = (cols - 1) / 2;
  const cy = (rows - 1) / 2;
  const nx = (col - cx) / Math.max(1, cols / 2);
  const ny = (row - cy) / Math.max(1, rows / 2);
  const centerWeight = clamp(1 - Math.sqrt(nx * nx + ny * ny), 0, 1);

  // Midday pulls activity to core, late evening shifts some activity outward.
  const middayBoost = timeWave(hour, 13, 3.4);
  const eveningResidential = timeWave(hour, 21, 2.8);
  const spatialShift = 1 + 0.35 * centerWeight * middayBoost - 0.18 * centerWeight * eveningResidential;
  return clamp(baseHour * spatialShift, 0.4, 1.75);
}

function timeTrafficPressureFactor(hour) {
  // Normal city traffic never fully disappears; rush hour layers on top gradually.
  return (
    0.65 +
    0.85 * timeWave(hour, 8, 2.45) +
    0.95 * timeWave(hour, 17, 2.7) +
    0.42 * timeWave(hour, 12, 3.0)
  );
}

function roadDisplaySensitivity(highway) {
  const h = String(highway || "");
  for (const [roadType, sensitivity] of Object.entries(ROAD_DISPLAY_SENSITIVITY)) {
    if (h.includes(roadType)) return sensitivity;
  }
  return 0.55;
}

function edgeCentroid(feature) {
  const coords = feature?.geometry?.coordinates ?? [];
  if (!coords.length) return null;
  const sum = coords.reduce(
    (acc, coord) => {
      acc.longitude += coord[0] ?? 0;
      acc.latitude += coord[1] ?? 0;
      return acc;
    },
    {longitude: 0, latitude: 0}
  );
  return {
    longitude: sum.longitude / coords.length,
    latitude: sum.latitude / coords.length
  };
}

function edgeEndpoints(feature) {
  const coords = feature?.geometry?.coordinates ?? [];
  if (coords.length < 2) return null;
  return {
    start: {longitude: coords[0][0], latitude: coords[0][1]},
    end: {
      longitude: coords[coords.length - 1][0],
      latitude: coords[coords.length - 1][1]
    }
  };
}

function distanceToCore(point) {
  if (!point) return 0;
  const dx = (point.longitude - COMMUTE_CORE.longitude) / 0.03;
  const dy = (point.latitude - COMMUTE_CORE.latitude) / 0.024;
  return Math.sqrt(dx * dx + dy * dy);
}

function downtownActivityFactor(hour) {
  const morning = 0.48 * timeWave(hour, 8, 2.55);
  const midday = 0.4 * timeWave(hour, 12, 3.0);
  const evening = 0.58 * timeWave(hour, 17, 2.85);
  return clamp(0.22 + morning + midday + evening, 0.22, 1.05);
}

function commuteWaveBoost(feature, hour) {
  const endpoints = edgeEndpoints(feature);
  const centroid = edgeCentroid(feature);
  if (!endpoints || !centroid) return 0;

  const centroidDistance = distanceToCore(centroid);
  const direction = distanceToCore(endpoints.end) - distanceToCore(endpoints.start);
  const outboundness = clamp(direction / 0.12, -1, 1);
  const inboundness = -outboundness;

  const morningProgress = clamp((hour - 6.0) / 3.5, 0, 1);
  const eveningProgress = clamp((hour - 16.0) / 3.75, 0, 1);
  const morningWaveCenter = 2.65 - 2.3 * morningProgress;
  const eveningWaveCenter = 0.35 + 2.45 * eveningProgress;

  const waveWidth = 0.55;
  const morningRing = Math.exp(
    -((centroidDistance - morningWaveCenter) ** 2) / (2 * waveWidth * waveWidth)
  );
  const eveningRing = Math.exp(
    -((centroidDistance - eveningWaveCenter) ** 2) / (2 * waveWidth * waveWidth)
  );

  const morningWindow = timeWave(hour, 8, 2.2);
  const eveningWindow = timeWave(hour, 17.4, 2.35);
  const morningDirectional = Math.max(0.16, Math.max(0, inboundness));
  const eveningDirectional = Math.max(0.16, Math.max(0, outboundness));

  return 1.25 * morningWindow * morningRing * morningDirectional +
    1.45 * eveningWindow * eveningRing * eveningDirectional;
}

function downtownCongestionBoost(feature) {
  const centroid = edgeCentroid(feature);
  if (!centroid) return 0;

  let boost = 0;
  for (const center of DOWNTOWN_CONGESTION_CENTERS) {
    const dx = (centroid.longitude - center.longitude) / 0.022;
    const dy = (centroid.latitude - center.latitude) / 0.018;
    boost = Math.max(boost, center.weight * Math.exp(-(dx * dx + dy * dy) / 2));
  }
  return boost;
}

function circlePolygon(center, radiusMeters, steps = 72) {
  if (!center) return [];
  const [lon, lat] = center;
  const latRadius = radiusMeters / 111320;
  const lonRadius = radiusMeters / (111320 * Math.cos((lat * Math.PI) / 180));
  const coords = [];
  for (let i = 0; i <= steps; i++) {
    const angle = (i / steps) * Math.PI * 2;
    coords.push([lon + Math.cos(angle) * lonRadius, lat + Math.sin(angle) * latRadius]);
  }
  return coords;
}

function makeEventFeatureCollection(event) {
  if (!event) return {type: "FeatureCollection", features: []};
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

function commuteDirectionBoost(feature, hour) {
  const endpoints = edgeEndpoints(feature);
  if (!endpoints) return 0;

  const startDistance = distanceToCore(endpoints.start);
  const endDistance = distanceToCore(endpoints.end);
  const centroidDistance = distanceToCore(edgeCentroid(feature));
  const direction = endDistance - startDistance;
  const outboundness = clamp(direction / 0.12, -1, 1);
  const inboundness = -outboundness;
  const outerNeighborhoodWeight = clamp((centroidDistance - 0.75) / 1.35, 0, 1);

  const morningInbound = timeWave(hour, 8, 2.45) * Math.max(0, inboundness);
  const eveningOutbound = timeWave(hour, 17, 2.7) * Math.max(0, outboundness);
  const middayDowntownCirculation = timeWave(hour, 12, 3.0) * 0.24;
  const outerCommute = outerNeighborhoodWeight * (0.65 * morningInbound + 0.78 * eveningOutbound);

  return 0.65 * morningInbound + 0.78 * eveningOutbound + outerCommute + middayDowntownCirculation;
}

function persistentTrafficFloor(feature, roadSensitivity, coreBoost, hour) {
  const offPeakLocalActivity =
    0.14 * timeWave(hour, 11, 4.2) +
    0.16 * timeWave(hour, 20, 3.8) +
    0.08 * timeWave(hour, 2, 4.5);
  const majorRoadCarryover = 0.2 * roadSensitivity;
  const downtownCarryover = 0.18 * coreBoost;
  return clamp(0.1 + majorRoadCarryover + downtownCarryover + offPeakLocalActivity, 0.1, 0.85);
}

function effectiveCongestionForEdge(feature, hour) {
  const props = feature?.properties ?? {};
  const base = hourlyValue(props?.hourly_congestion_factor, hour);
  const roadSensitivity = roadDisplaySensitivity(props?.highway);
  const coreBoost = downtownCongestionBoost(feature);
  const commuteBoost = commuteDirectionBoost(feature, hour);
  const waveBoost = commuteWaveBoost(feature, hour);
  const trafficFloor = persistentTrafficFloor(feature, roadSensitivity, coreBoost, hour);
  const displaySensitivity =
    roadSensitivity * 0.72 + coreBoost * 0.5 * downtownActivityFactor(hour) + commuteBoost + waveBoost;
  return 1 + trafficFloor + (base - 1) * displaySensitivity;
}

function commuteWaveIntensity(feature, hour) {
  const base = hourlyValue(feature?.properties?.hourly_congestion_factor, hour);
  const intensity = (base - 1) * commuteWaveBoost(feature, hour);
  return clamp(intensity / 2.2, 0, 1);
}

function formatSpeed(value) {
  const speed = Number(value) || 0;
  return speed < 1 ? speed.toFixed(1) : speed.toFixed(speed % 1 === 0 ? 0 : 1);
}

function nextSimSpeed(current) {
  const idx = SIM_SPEED_STEPS.findIndex(speed => Math.abs(speed - current) < 0.001);
  if (idx < 0) return SIM_SPEED_STEPS[0];
  return SIM_SPEED_STEPS[(idx + 1) % SIM_SPEED_STEPS.length];
}

function trafficFlowSpeedFactor(hour) {
  const pressure = timeTrafficPressureFactor(hour);
  return clamp(1 / (0.6 + pressure), 0.2, 1.1);
}

function trafficStateLabel(hour) {
  if (hour >= 7 && hour <= 9) return "Morning rush";
  if (hour >= 16 && hour <= 18) return "Evening rush";
  if (hour >= 0 && hour <= 4) return "Late night low traffic";
  if (hour >= 11 && hour <= 14) return "Midday";
  return "Off peak";
}

function commuteFlowLabel(hour) {
  if (hour >= 7 && hour <= 9) return "inbound toward downtown";
  if (hour >= 16 && hour <= 18) return "outbound toward neighborhoods/suburbs";
  if (hour >= 11 && hour <= 14) return "downtown circulation";
  return "low directional pressure";
}

function commuteWaveLabel(hour) {
  if (hour >= 6 && hour < 10) return "inbound wave moving toward downtown";
  if (hour >= 16 && hour < 20) return "outbound wave moving away from downtown";
  return "no active commute wave";
}

function makeGridFeatureCollection(grid) {
  if (!grid) return null;
  const {
    rows,
    cols,
    bounds: {min_lon: minLon, max_lon: maxLon, min_lat: minLat, max_lat: maxLat},
    values
  } = grid;

  const dLon = (maxLon - minLon) / cols;
  const dLat = (maxLat - minLat) / rows;
  const features = [];

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const west = minLon + c * dLon;
      const east = west + dLon;
      const south = minLat + r * dLat;
      const north = south + dLat;
      const density = values?.[r]?.[c] ?? 0;

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
          row: r,
          col: c,
          population_density: density
        }
      });
    }
  }

  return {type: "FeatureCollection", features};
}

function gridKey(row, col) {
  return `${row}:${col}`;
}

function gridCellForPosition(position, grid) {
  if (!position || !grid) return null;
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

function gridCellForFeature(feature, grid) {
  if (!feature || !grid) return null;
  const cached = FEATURE_GRID_CELL_CACHE.get(feature);
  if (cached) return cached;
  const centroid = edgeCentroid(feature);
  const cell = centroid ? gridCellForPosition([centroid.longitude, centroid.latitude], grid) : null;
  if (cell) FEATURE_GRID_CELL_CACHE.set(feature, cell);
  return cell;
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
        influence = Math.max(influence, value * Math.exp(-(dist * dist) / (2 * 2.8 * 2.8)));
      }
      values[row * cols + col] = 1 + clamp((influence - 0.42) * 0.9, 0, 0.75);
    }
  }
  return {rows, cols, values};
}

function generatedTrafficMultiplierForEdge(feature, pressureByCell, grid) {
  if (!pressureByCell || !grid) return 1;
  const cell = gridCellForFeature(feature, grid);
  if (!cell) return 1;
  return pressureByCell.values[cell.row * pressureByCell.cols + cell.col] || 1;
}

function nearestNodeForPosition(position, nodes) {
  if (!position || !nodes.length) return null;
  const [lon, lat] = position;
  const lonScale = Math.cos((lat * Math.PI) / 180);
  let bestNode = null;
  let bestDist = Infinity;

  for (const node of nodes) {
    const dx = ((node.lon ?? lon) - lon) * lonScale;
    const dy = (node.lat ?? lat) - lat;
    const dist = dx * dx + dy * dy;
    if (dist < bestDist) {
      bestDist = dist;
      bestNode = node;
    }
  }

  return bestNode;
}

function makeNodeCountsByGridCell(nodes, grid) {
  const counts = new Map();
  for (const node of nodes) {
    const row = Number.isFinite(node.grid_row) ? node.grid_row : null;
    const col = Number.isFinite(node.grid_col) ? node.grid_col : null;
    const cell =
      row !== null && col !== null
        ? {row, col, key: gridKey(row, col)}
        : gridCellForPosition([node.lon, node.lat], grid);
    if (!cell) continue;
    counts.set(cell.key, (counts.get(cell.key) ?? 0) + 1);
  }
  return counts;
}

function makeCarPresenceByGridCell(cars, nodes, grid, options = {}) {
  const {snapToNearestNode = true, useProvidedCell = false} = options;
  const presence = new Map();

  for (const car of cars) {
    const snappedNode = snapToNearestNode ? nearestNodeForPosition(car.position, nodes) : null;
    const providedCell = useProvidedCell ? car.grid_cell ?? car.grid_position : null;
    const cell = providedCell
      ? {
          row: providedCell[0],
          col: providedCell[1],
          key: gridKey(providedCell[0], providedCell[1])
        }
      : snappedNode
      ? {
          row: snappedNode.grid_row,
          col: snappedNode.grid_col,
          key: gridKey(snappedNode.grid_row, snappedNode.grid_col)
        }
      : gridCellForPosition(car.position, grid);
    if (!cell || cell.row === undefined || cell.col === undefined) continue;

    const existing = presence.get(cell.key) ?? {
      row: cell.row,
      col: cell.col,
      carIds: [],
      nodeIds: []
    };
    existing.carIds.push(car.id);
    if (snappedNode?.node_id !== undefined) existing.nodeIds.push(snappedNode.node_id);
    presence.set(cell.key, existing);
  }

  return presence;
}

function makeCarPresenceGridFeatureCollection(grid, carPresenceByCell, nodeCountsByCell) {
  const collection = makeGridFeatureCollection(grid);
  if (!collection) return null;

  return {
    ...collection,
    features: collection.features.map(feature => {
      const row = feature.properties.row;
      const col = feature.properties.col;
      const key = gridKey(row, col);
      const presence = carPresenceByCell.get(key);
      return {
        ...feature,
        properties: {
          ...feature.properties,
          has_car: Boolean(presence),
          car_count: presence?.carIds.length ?? 0,
          car_ids: presence?.carIds ?? [],
          snapped_node_ids: presence?.nodeIds ?? [],
          intersection_count: nodeCountsByCell.get(key) ?? 0
        }
      };
    })
  };
}

function gridCellCenter(cell, grid) {
  if (!cell || !grid?.bounds) return null;
  const [row, col] = cell;
  const {
    rows,
    cols,
    bounds: {min_lon: minLon, max_lon: maxLon, min_lat: minLat, max_lat: maxLat}
  } = grid;
  const dLon = (maxLon - minLon) / cols;
  const dLat = (maxLat - minLat) / rows;
  return [minLon + (col + 0.5) * dLon, minLat + (row + 0.5) * dLat];
}

function gridCellPolygonFeature(cell, grid, properties) {
  if (!cell || !grid?.bounds) return null;
  const [row, col] = cell;
  const {
    rows,
    cols,
    bounds: {min_lon: minLon, max_lon: maxLon, min_lat: minLat, max_lat: maxLat}
  } = grid;
  if (row < 0 || row >= rows || col < 0 || col >= cols) return null;

  const dLon = (maxLon - minLon) / cols;
  const dLat = (maxLat - minLat) / rows;
  const west = minLon + col * dLon;
  const east = west + dLon;
  const south = minLat + row * dLat;
  const north = south + dLat;
  return {
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
    properties: {...properties, row, col}
  };
}

function activeAssignmentContext(snapshot) {
  const assignments = activeGreedyAssignments(snapshot);
  const carById = new Map((snapshot?.map_dispatch?.cars ?? []).map(car => [car.id, car]));
  return {
    assignmentByPersonId: new Map(assignments.map(assignment => [assignment.person_id, assignment])),
    carById
  };
}

function assignmentElapsedSeconds(assignment, car, snapshot, clockMinute, stepMinutes) {
  const snapshotSeconds = snapshotElapsedMinutes(snapshot, clockMinute, stepMinutes) * 60;
  const routeElapsed = Number(car?.route_elapsed ?? assignment?.route_elapsed ?? 0);
  return routeElapsed + snapshotSeconds;
}

function visibleMarkerKinds(person, assignment, car, snapshot, clockMinute, stepMinutes) {
  if (!assignment) return {pickup: true, dropoff: true};
  const elapsed = assignmentElapsedSeconds(assignment, car, snapshot, clockMinute, stepMinutes);
  const pickupCost = Number(assignment.pickup_route?.cost ?? 0);
  const routeCost = Number(assignment.route?.cost ?? assignment.total_cost ?? 0);
  return {
    pickup: elapsed < pickupCost,
    dropoff: elapsed < routeCost
  };
}

function makeGreedyPeopleFeatureCollection(snapshot, grid, clockMinute = 0, stepMinutes = 15) {
  const mapPeople = activeGreedyPeople(snapshot);
  if (mapPeople.length) {
    const features = [];
    const {assignmentByPersonId, carById} = activeAssignmentContext(snapshot);
    for (const person of mapPeople) {
      const assignment = assignmentByPersonId.get(person.id);
      const car = assignment ? carById.get(assignment.car_id) : null;
      const visible = visibleMarkerKinds(person, assignment, car, snapshot, clockMinute, stepMinutes);
      if (person.pickup_position && visible.pickup) {
        features.push({
          type: "Feature",
          geometry: {type: "Point", coordinates: person.pickup_position},
          properties: {
            kind: "pickup",
            person_id: person.id,
            node_id: person.pickup_node_id,
            grid_cell: person.pickup_grid_cell ?? person.origin,
            request_cell: person.request_origin ?? person.origin,
            paired_node_id: person.dropoff_node_id,
            paired_position: person.dropoff_position
          }
        });
      }
      if (person.dropoff_position && visible.dropoff) {
        features.push({
          type: "Feature",
          geometry: {type: "Point", coordinates: person.dropoff_position},
          properties: {
            kind: "dropoff",
            person_id: person.id,
            node_id: person.dropoff_node_id,
            grid_cell: person.dropoff_grid_cell ?? person.destination,
            request_cell: person.request_destination ?? person.destination,
            paired_node_id: person.pickup_node_id,
            paired_position: person.pickup_position
          }
        });
      }
    }
    return {type: "FeatureCollection", features};
  }

  const markers = snapshot?.people_grid?.markers ?? [];
  const features = [];
  for (const marker of markers) {
    const pickup = gridCellPolygonFeature(marker.pickup, grid, {
      kind: "pickup",
      person_id: marker.person_id,
      paired_cell: marker.dropoff
    });
    const dropoff = gridCellPolygonFeature(marker.dropoff, grid, {
      kind: "dropoff",
      person_id: marker.person_id,
      paired_cell: marker.pickup
    });
    if (pickup) features.push(pickup);
    if (dropoff) features.push(dropoff);
  }
  return {type: "FeatureCollection", features};
}

function makeGreedyPeopleGridFeatureCollection(snapshot, grid, clockMinute = 0, stepMinutes = 15) {
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
      dropoff_nodes: [],
      paired_cells: []
    };
    if (kind === "pickup") {
      current.pickup_count += 1;
      current.pickup_ids.push(person.id ?? person.person_id);
      if (person.pickup_node_id) current.pickup_nodes.push(person.pickup_node_id);
      if (person.dropoff_grid_cell ?? person.destination) {
        current.paired_cells.push(person.dropoff_grid_cell ?? person.destination);
      }
    } else {
      current.dropoff_count += 1;
      current.dropoff_ids.push(person.id ?? person.person_id);
      if (person.dropoff_node_id) current.dropoff_nodes.push(person.dropoff_node_id);
      if (person.pickup_grid_cell ?? person.origin) {
        current.paired_cells.push(person.pickup_grid_cell ?? person.origin);
      }
    }
    cells.set(key, current);
  };

  const mapPeople = activeGreedyPeople(snapshot);
  if (mapPeople.length) {
    const {assignmentByPersonId, carById} = activeAssignmentContext(snapshot);
    for (const person of mapPeople) {
      const assignment = assignmentByPersonId.get(person.id);
      const car = assignment ? carById.get(assignment.car_id) : null;
      const visible = visibleMarkerKinds(person, assignment, car, snapshot, clockMinute, stepMinutes);
      if (visible.pickup) register(person.pickup_grid_cell ?? person.origin, "pickup", person);
      if (visible.dropoff) register(person.dropoff_grid_cell ?? person.destination, "dropoff", person);
    }
  } else {
    for (const marker of snapshot?.people_grid?.markers ?? []) {
      register(marker.pickup, "pickup", marker);
      register(marker.dropoff, "dropoff", marker);
    }
  }

  return {
    ...collection,
    features: collection.features.map(feature => {
      const row = feature.properties.row;
      const col = feature.properties.col;
      const presence = cells.get(gridKey(row, col));
      const pickupCount = presence?.pickup_count ?? 0;
      const dropoffCount = presence?.dropoff_count ?? 0;
      return {
        ...feature,
        properties: {
          ...feature.properties,
          ...(presence ?? {}),
          has_people: pickupCount + dropoffCount > 0,
          has_pickup: pickupCount > 0,
          has_dropoff: dropoffCount > 0,
          people_count: pickupCount + dropoffCount
        }
      };
    })
  };
}

function routePathToCoordinates(path, grid) {
  return (path ?? [])
    .map(cell => gridCellCenter(cell, grid))
    .filter(Boolean);
}

function interpolatePathPosition(coordinates, progress) {
  if (!Array.isArray(coordinates) || coordinates.length === 0) return null;
  if (coordinates.length === 1) return coordinates[0];
  const targetProgress = clamp(progress, 0, 1);
  if (targetProgress <= 0) return coordinates[0];
  if (targetProgress >= 1) return coordinates[coordinates.length - 1];

  const lengths = [];
  let totalLength = 0;
  for (let i = 0; i < coordinates.length - 1; i++) {
    const a = coordinates[i];
    const b = coordinates[i + 1];
    const dx = (b[0] - a[0]) * Math.cos(((a[1] + b[1]) / 2) * Math.PI / 180);
    const dy = b[1] - a[1];
    const length = Math.hypot(dx, dy);
    lengths.push(length);
    totalLength += length;
  }
  if (totalLength <= 0) return coordinates[0];

  let distance = totalLength * targetProgress;
  for (let i = 0; i < lengths.length; i++) {
    const length = lengths[i];
    if (distance > length) {
      distance -= length;
      continue;
    }
    const a = coordinates[i];
    const b = coordinates[i + 1];
    const t = length <= 0 ? 0 : distance / length;
    return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
  }

  return coordinates[coordinates.length - 1];
}

function routePathFromProgress(coordinates, progress) {
  if (!Array.isArray(coordinates) || coordinates.length < 2) return coordinates ?? [];
  const targetProgress = clamp(progress, 0, 1);
  if (targetProgress >= 1) return [coordinates[coordinates.length - 1]];

  const lengths = [];
  let totalLength = 0;
  for (let i = 0; i < coordinates.length - 1; i++) {
    const a = coordinates[i];
    const b = coordinates[i + 1];
    const length = Math.hypot(b[0] - a[0], b[1] - a[1]);
    lengths.push(length);
    totalLength += length;
  }
  if (totalLength <= 0) return coordinates;

  let distance = totalLength * targetProgress;
  for (let i = 0; i < lengths.length; i++) {
    const length = lengths[i];
    if (distance > length) {
      distance -= length;
      continue;
    }
    const a = coordinates[i];
    const b = coordinates[i + 1];
    const t = length <= 0 ? 0 : distance / length;
    const current = [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
    return [current, ...coordinates.slice(i + 1)];
  }

  return [coordinates[coordinates.length - 1]];
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
    if (routeSegmentMeters(coordinates[i], coordinates[i + 1]) > MAX_VISIBLE_ROUTE_SEGMENT_METERS) {
      return false;
    }
  }
  return true;
}

function resolveRoute(route, routeIndex = {}) {
  if (!route) return null;
  if (Array.isArray(route.coordinates)) return route;
  const routeId = route.id ?? route.route_id;
  const stored = routeId ? routeIndex[routeId] : null;
  if (!stored) return route;
  return {
    ...stored,
    ...route,
    coordinates: route.coordinates ?? stored.coordinates
  };
}

function snapshotElapsedMinutes(snapshot, clockMinute, stepMinutes) {
  if (!snapshot) return 0;
  const start = snapshot.timestep ?? 0;
  const elapsed = Math.max(0, clockMinute - start);
  return clamp(elapsed, 0, Math.max(1, stepMinutes));
}

function snapshotProgress(snapshot, clockMinute, stepMinutes) {
  return snapshotElapsedMinutes(snapshot, clockMinute, stepMinutes) / Math.max(1, stepMinutes);
}

function activeRouteProgress(assignment, car, snapshot, clockMinute, stepMinutes) {
  const routeCost = Number(assignment?.route?.cost ?? assignment?.total_cost ?? assignment?.dropoff_route?.cost ?? 0);
  if (!assignment || routeCost <= 0) return snapshotProgress(snapshot, clockMinute, stepMinutes);
  const snapshotSeconds = snapshotElapsedMinutes(snapshot, clockMinute, stepMinutes) * 60;
  const routeElapsed = Number(car?.route_elapsed ?? assignment?.route_elapsed ?? 0);
  return clamp((routeElapsed + snapshotSeconds) / routeCost, 0, 1);
}

function activeGreedyAssignments(snapshot) {
  return snapshot?.map_dispatch?.assignments ?? snapshot?.dispatch?.assignments ?? [];
}

function activeGreedyPeople(snapshot) {
  return snapshot?.map_people ?? [];
}

function makeGreedyRouteFeatureCollection(
  snapshot,
  grid,
  kindFilter = null,
  clockMinute = 0,
  stepMinutes = 15,
  routeIndex = {}
) {
  const assignments = activeGreedyAssignments(snapshot);
  const carById = new Map((snapshot?.map_dispatch?.cars ?? []).map(car => [car.id, car]));
  const features = [];
  for (const assignment of assignments) {
    const route =
      resolveRoute(assignment.route, routeIndex) ??
      resolveRoute(assignment.active_route, routeIndex) ??
      resolveRoute(assignment.dropoff_route, routeIndex);
    const activeCoords =
      route?.coordinates ??
      routePathToCoordinates(assignment.dropoff_route?.path, grid);
    if (!routeCoordinatesAreUsable(activeCoords)) continue;
    const car = carById.get(assignment.car_id);
    const progress = activeRouteProgress(assignment, car, snapshot, clockMinute, stepMinutes);
    const remainingCoords = routePathFromProgress(activeCoords, progress);
    if ((!kindFilter || kindFilter === "to_dropoff") && remainingCoords.length >= 2) {
      features.push({
        type: "Feature",
        geometry: {type: "LineString", coordinates: remainingCoords},
        properties: {
          kind: "active_route",
          car_id: assignment.car_id,
          person_id: assignment.person_id,
          cost: (assignment.route?.cost ?? assignment.total_cost ?? assignment.dropoff_route?.cost ?? 0) * (1 - progress)
        }
      });
    }
  }
  return {type: "FeatureCollection", features};
}

function greedyCarPoints(snapshot, grid, clockMinute = 0, stepMinutes = 15, routeIndex = {}) {
  const mapCars = snapshot?.map_dispatch?.cars ?? [];
  if (mapCars.length) {
    const assignmentByCarId = new Map(
      activeGreedyAssignments(snapshot).map(assignment => [assignment.car_id, assignment])
    );
    return mapCars
      .map(car => {
        if (!car.position) return null;
        const assignment = assignmentByCarId.get(car.id);
        const route =
          resolveRoute(assignment?.route, routeIndex) ??
          resolveRoute(assignment?.active_route, routeIndex) ??
          resolveRoute(assignment?.dropoff_route, routeIndex);
        const routeCoords = route?.coordinates ?? [];
        const progress = activeRouteProgress(assignment, car, snapshot, clockMinute, stepMinutes);
        const animatedPosition =
          assignment && routeCoordinatesAreUsable(routeCoords)
            ? interpolatePathPosition(routeCoords, progress)
            : null;
        return {
          ...car,
          position: animatedPosition ?? car.position,
          route_progress: assignment ? progress : 0,
          grid_position: null
        };
      })
      .filter(Boolean);
  }

  return (snapshot?.dispatch?.cars ?? [])
    .map(car => {
      const position = gridCellCenter(car.position, grid);
      if (!position) return null;
      return {...car, grid_position: car.position, position};
    })
    .filter(Boolean);
}

function tripPositionAtTime(trip, clockMinute) {
  const path = trip?.path ?? [];
  const timestamps = trip?.timestamps ?? [];
  if (path.length < 2 || timestamps.length < 2 || path.length !== timestamps.length) return null;

  const start = timestamps[0];
  const end = timestamps[timestamps.length - 1];
  const duration = Math.max(1e-6, end - start);
  const wrapped = start + ((((clockMinute - start) % duration) + duration) % duration);

  for (let i = 0; i < timestamps.length - 1; i++) {
    const t0 = timestamps[i];
    const t1 = timestamps[i + 1];
    if (wrapped < t0 || wrapped > t1) continue;
    const p0 = path[i];
    const p1 = path[i + 1];
    const frac = (wrapped - t0) / Math.max(1e-6, t1 - t0);
    return [p0[0] + (p1[0] - p0[0]) * frac, p0[1] + (p1[1] - p0[1]) * frac];
  }

  return path[path.length - 1];
}

function formatCurrency(value) {
  return `$${Math.round(value).toLocaleString()}`;
}

function formatMinutes(value) {
  const minutes = Number.isFinite(value) ? Math.max(0, value) : 0;
  return minutes >= 100 ? `${Math.round(minutes).toLocaleString()} min` : `${minutes.toFixed(1)} min`;
}

function snapshotWaitTimeMinutes(snapshot) {
  const greedy = snapshot?.map_greedy_stats ?? snapshot?.greedy_stats;
  if (Number.isFinite(greedy?.wait_time_min)) return greedy.wait_time_min;

  const assignmentTotal = activeGreedyAssignments(snapshot).reduce(
    (sum, assignment) => sum + (Number(assignment.wait_time_min) || 0),
    0
  );
  if (assignmentTotal > 0) return assignmentTotal;
  return 0;
}

function App() {
  const [clockMinute, setClockMinute] = useState(SIM_START_MINUTE);
  const [simSpeed, setSimSpeed] = useState(0.5);
  const [paused, setPaused] = useState(false);
  const [keyMissing, setKeyMissing] = useState(MAPTILER_KEY === "YOUR_MAPTILER_KEY");
  const [network, setNetwork] = useState(null);
  const [populationGrid, setPopulationGrid] = useState(null);
  const [ppoNodes, setPpoNodes] = useState([]);
  const [trips, setTrips] = useState(FALLBACK_TRIPS);
  const [datasetReady, setDatasetReady] = useState(false);
  const [showNodeDensity, setShowNodeDensity] = useState(false);
  const [showCarGrid, setShowCarGrid] = useState(true);
  const [showGreedySim, setShowGreedySim] = useState(true);
  const [showPeopleGrid, setShowPeopleGrid] = useState(true);
  const [mobilityWorld, setMobilityWorld] = useState(null);
  const [mobilityWorldStatus, setMobilityWorldStatus] = useState("loading");
  const [activeEventId, setActiveEventId] = useState(null);
  const [telemetryCollapsed, setTelemetryCollapsed] = useState(true);

  // Load OSMnx-generated artifacts.
  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const [edgeData, tripData, gridData, nodeData] = await Promise.all([
          fetchFirstJson(["/data/osmnx_edges.geojson", "/dist/data/osmnx_edges.geojson"]),
          fetchFirstJson(["/data/sample_trips.json", "/dist/data/sample_trips.json"]),
          fetchFirstJson(["/data/population_density_grid.json", "/dist/data/population_density_grid.json"]),
          fetchFirstJson(["/data/ppo_nodes.json", "/dist/data/ppo_nodes.json"])
        ]);
        if (cancelled) return;
        setNetwork(edgeData);
        setTrips(Array.isArray(tripData) && tripData.length > 0 ? tripData : FALLBACK_TRIPS);
        setPopulationGrid(gridData);
        setPpoNodes(Array.isArray(nodeData?.nodes) ? nodeData.nodes : []);
        setDatasetReady(true);
      } catch (_) {
        // fallback remains active; nothing else needed
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;

    async function loadMobilityWorld() {
      try {
        setMobilityWorldStatus("loading");
        const res = await fetch(`/data/mobility_world.json?v=${Date.now()}`, {
          cache: "no-store"
        });
        if (!res.ok) {
          throw new Error(`mobility_world.json ${res.status}`);
        }
        const text = await res.text();
        const data = JSON.parse(text);
        if (!cancelled) {
          setMobilityWorld(data);
          setMobilityWorldStatus("ready");
        }
      } catch (error) {
        console.error("Failed to load mobility_world.json", error);
        if (!cancelled) setMobilityWorldStatus("error");
      }
    }

    loadMobilityWorld();
    return () => {
      cancelled = true;
    };
  }, []);

  const normalizedClockMinute = clamp(clockMinute, SIM_RESET_MINUTE, SIM_END_MINUTE);
  const currentHour = Math.floor(normalizedClockMinute / 60);
  const currentMinute = Math.floor(normalizedClockMinute % 60);
  const currentHourFloat = normalizedClockMinute / 60;
  const trafficDisplayHour = Math.round(currentHourFloat * 4) / 4;
  const clockLabel = `${String(currentHour).padStart(2, "0")}:${String(
    currentMinute
  ).padStart(2, "0")}`;
  const flowFactor = trafficFlowSpeedFactor(currentHourFloat);
  const tripClockMinute = normalizedClockMinute * flowFactor;
  const routeIndex = mobilityWorld?.routes ?? {};
  const mobilityEvents = mobilityWorld?.events ?? [];
  const activeEvent = mobilityEvents.find(event => event.id === activeEventId) ?? null;
  const activeScenario = activeEventId ? mobilityWorld?.event_scenarios?.[activeEventId] : null;
  const activeSnapshots = useMemo(() => {
    const snapshots = activeScenario?.snapshots ?? mobilityWorld?.snapshots ?? [];
    return [...snapshots].sort((a, b) => (a.timestep ?? 0) - (b.timestep ?? 0));
  }, [activeScenario, mobilityWorld]);

  const mobilitySnapshot = useMemo(() => {
    if (!activeSnapshots.length) return null;
    let active = activeSnapshots[activeSnapshots.length - 1];
    for (const snapshot of activeSnapshots) {
      if ((snapshot.timestep ?? 0) <= clockMinute) {
        active = snapshot;
      } else {
        break;
      }
    }
    return active;
  }, [activeSnapshots, clockMinute]);
  const mobilityStepMinutes = activeScenario?.step_minutes ?? mobilityWorld?.step_minutes ?? 15;
  const simulationEndMinute = useMemo(() => {
    if (!activeSnapshots.length) return SIM_END_MINUTE;
    const lastTimestep = activeSnapshots.reduce(
      (latest, snapshot) => Math.max(latest, Number(snapshot.timestep) || 0),
      0
    );
    return clamp(lastTimestep + mobilityStepMinutes - 0.001, SIM_RESET_MINUTE, SIM_END_MINUTE);
  }, [activeSnapshots, mobilityStepMinutes]);
  const atSimulationEnd = clockMinute >= simulationEndMinute - 0.01;

  // Simulated clock runs once through the exported timeline. It pauses at the end
  // so cumulative metrics stay on the final snapshot until the user restarts.
  useEffect(() => {
    const i = setInterval(() => {
      if (paused) return;
      setClockMinute(current => {
        if (current >= simulationEndMinute) {
          setPaused(true);
          return simulationEndMinute;
        }
        const next = current + simSpeed;
        if (next >= simulationEndMinute) {
          setPaused(true);
          return simulationEndMinute;
        }
        return next;
      });
    }, 50);
    return () => clearInterval(i);
  }, [simSpeed, paused, simulationEndMinute]);
  const generatedTrafficPressureByCell = useMemo(
    () => makeGeneratedTrafficPressureByCell(mobilitySnapshot, populationGrid),
    [mobilitySnapshot, populationGrid]
  );

  const nodeCountsByCell = useMemo(
    () => makeNodeCountsByGridCell(ppoNodes, populationGrid),
    [ppoNodes, populationGrid]
  );

  const densityStats = useMemo(() => {
    const rawValues = populationGrid?.values?.flat?.() ?? [];
    const nodeValues = ppoNodes.map(n => n.population_density_noisy);
    if (!rawValues.length && !nodeValues.length) return {min: 0, max: 1};
    const all = rawValues.length ? rawValues : nodeValues;
    const minBase = Math.min(...all);
    const maxBase = Math.max(...all);
    // Keep a stable color legend while values vary by time.
    return {min: minBase * 0.7, max: maxBase * 1.35};
  }, [populationGrid, ppoNodes]);

  const globalCongestionScale = useMemo(() => {
    const features = network?.features ?? [];
    if (!features.length) return {min: 1, max: 2, p05: 1, p50: 1.5, p95: 2};
    const vals = [];
    for (const f of features) {
      for (let h = 0; h < 24; h++) {
        const c = effectiveCongestionForEdge(f, h);
        if (typeof c !== "number" || Number.isNaN(c)) continue;
        vals.push(c);
      }
    }
    if (vals.length < 4) return {min: 1, max: 2, p05: 1, p50: 1.5, p95: 2};
    vals.sort((a, b) => a - b);
    const q = p => vals[Math.floor((vals.length - 1) * p)];
    const min = vals[0];
    const max = vals[vals.length - 1];
    const p05 = q(0.05);
    const p50 = q(0.5);
    const p95 = q(0.95);
    if (!Number.isFinite(min) || !Number.isFinite(max) || min === max) {
      return {min: 1, max: 2, p05: 1, p50: 1.5, p95: 2};
    }
    return {min, max, p05, p50, p95};
  }, [network]);

  const currentHourCongestion = useMemo(() => {
    if (!network?.features?.length) return {mean: 1, p50: 1, p90: 1, hotShare: 0};
    const vals = network.features
      .map(
        f =>
          effectiveCongestionForEdge(f, trafficDisplayHour) *
          generatedTrafficMultiplierForEdge(f, generatedTrafficPressureByCell, populationGrid)
      )
      .filter(v => typeof v === "number" && !Number.isNaN(v))
      .sort((a, b) => a - b);
    if (!vals.length) return {mean: 1, p50: 1, p90: 1, hotShare: 0};
    const q = p => vals[Math.floor((vals.length - 1) * p)];
    const mean = vals.reduce((acc, v) => acc + v, 0) / vals.length;
    const hotShare = vals.filter(v => v >= 2.5).length / vals.length;
    return {mean, p50: q(0.5), p90: q(0.9), hotShare};
  }, [network, trafficDisplayHour, generatedTrafficPressureByCell, populationGrid]);

  const edgeLayer = useMemo(() => {
    if (!network) return null;

    return new GeoJsonLayer({
      id: "osmnx-edges",
      data: network,
      stroked: true,
      filled: false,
      lineWidthMinPixels: showGreedySim ? 0.75 : 1.25,
      lineWidthMaxPixels: showGreedySim ? 5 : 8,
      getLineWidth: f => {
        const congestion =
          effectiveCongestionForEdge(f, trafficDisplayHour) *
          generatedTrafficMultiplierForEdge(f, generatedTrafficPressureByCell, populationGrid);
        const t = clamp((congestion - 1.0) / (4.8 - 1.0), 0, 1);
        return showGreedySim ? 0.65 + t * 2.7 : 0.9 + t * 5.4;
      },
      getLineColor: f => {
        const congestion =
          effectiveCongestionForEdge(f, trafficDisplayHour) *
          generatedTrafficMultiplierForEdge(f, generatedTrafficPressureByCell, populationGrid);
        const [r, g, b] = trafficColorFromCongestion(congestion);
        return [r, g, b, showGreedySim ? 130 : 245];
      },
      updateTriggers: {
        getLineWidth: [trafficDisplayHour, showGreedySim, generatedTrafficPressureByCell, populationGrid],
        getLineColor: [trafficDisplayHour, showGreedySim, generatedTrafficPressureByCell, populationGrid]
      },
      pickable: true,
      autoHighlight: true,
      highlightColor: [250, 250, 250, 220],
      getTooltip: ({object}) => {
        if (!object) return null;
        const p = object.properties ?? {};
        const speed = p.hourly_speed_kph?.[currentHour];
        const ttime = p.hourly_travel_time_s?.[currentHour];
        const baseCong = p.hourly_congestion_factor?.[currentHour] ?? 1;
        const simCong =
          effectiveCongestionForEdge(object, trafficDisplayHour) *
          generatedTrafficMultiplierForEdge(object, generatedTrafficPressureByCell, populationGrid);
        const multiplier = simCong / Math.max(0.05, baseCong);
        const simSpeed = typeof speed === "number" ? speed / multiplier : speed;
        const simTime = typeof ttime === "number" ? ttime * multiplier : ttime;
        return {
          text:
            `${p.name || p.highway || "road"}\n` +
            `hour ${currentHour} base: ${speed?.toFixed?.(1) ?? speed ?? "?"} kph\n` +
            `simulated: ${simSpeed?.toFixed?.(1) ?? simSpeed ?? "?"} kph\n` +
            `travel time: ${simTime?.toFixed?.(1) ?? simTime ?? "?"} s\n` +
            `length: ${p.length_m ?? "?"} m`
        };
      }
    });
  }, [network, currentHour, trafficDisplayHour, globalCongestionScale, showGreedySim, generatedTrafficPressureByCell, populationGrid]);

  const commuteWaveLayer = useMemo(() => {
    if (!network || showGreedySim) return null;

    return new GeoJsonLayer({
      id: "commute-wave",
      data: network,
      stroked: true,
      filled: false,
      lineWidthMinPixels: 0,
      lineWidthMaxPixels: 14,
      getLineWidth: f => {
        const intensity = commuteWaveIntensity(f, trafficDisplayHour);
        return intensity < 0.08 ? 0 : 2 + intensity * 8;
      },
      getLineColor: f => {
        const intensity = commuteWaveIntensity(f, trafficDisplayHour);
        if (intensity < 0.08) return [0, 0, 0, 0];
        return [
          255,
          Math.round(120 - 70 * intensity),
          Math.round(35 - 20 * intensity),
          Math.round(80 + 160 * intensity)
        ];
      },
      updateTriggers: {
        getLineWidth: [trafficDisplayHour],
        getLineColor: [trafficDisplayHour]
      },
      pickable: false
    });
  }, [network, trafficDisplayHour, showGreedySim]);

  const nodeDensityLayer = useMemo(() => {
    if (!showNodeDensity || !ppoNodes.length) return null;
    const rows = populationGrid?.rows ?? 50;
    const cols = populationGrid?.cols ?? 50;
    return new ScatterplotLayer({
      id: "node-density",
      data: ppoNodes,
      pickable: true,
      radiusUnits: "meters",
      radiusMinPixels: 1,
      radiusMaxPixels: 16,
      getRadius: d => {
        const factor = populationTimeFactor(
          currentHour,
          d.grid_row ?? 0,
          d.grid_col ?? 0,
          rows,
          cols
        );
        return 6 + clamp(((d.population_density_noisy ?? 0) * factor) / 900, 0, 26);
      },
      getPosition: d => [d.lon, d.lat],
      getFillColor: d => {
        const factor = populationTimeFactor(
          currentHour,
          d.grid_row ?? 0,
          d.grid_col ?? 0,
          rows,
          cols
        );
        return densityColor(
          (d.population_density_noisy ?? 0) * factor,
          densityStats.min,
          densityStats.max,
          220
        );
      }
    });
  }, [showNodeDensity, ppoNodes, densityStats, currentHour, populationGrid]);

  const uberCars = useMemo(
    () =>
      trips
        .map((trip, idx) => {
          const position = tripPositionAtTime(trip, tripClockMinute);
          if (!position) return null;
          return {id: `uber-${idx}`, position};
        })
        .filter(Boolean),
    [trips, tripClockMinute]
  );

  const greedyCars = useMemo(
    () =>
      showGreedySim && mobilitySnapshot && populationGrid
        ? greedyCarPoints(mobilitySnapshot, populationGrid, clockMinute, mobilityStepMinutes, routeIndex)
        : [],
    [showGreedySim, mobilitySnapshot, populationGrid, clockMinute, mobilityStepMinutes, routeIndex]
  );
  const gridClockMinute = Math.floor(clockMinute / 5) * 5;
  const greedyCarsForGrid = useMemo(
    () =>
      showGreedySim && mobilitySnapshot && populationGrid
        ? greedyCarPoints(mobilitySnapshot, populationGrid, gridClockMinute, mobilityStepMinutes, routeIndex)
        : [],
    [showGreedySim, mobilitySnapshot, populationGrid, gridClockMinute, mobilityStepMinutes, routeIndex]
  );

  const carsForPresenceGrid = showGreedySim ? greedyCarsForGrid : uberCars;

  const carPresenceByCell = useMemo(
    () =>
      makeCarPresenceByGridCell(carsForPresenceGrid, ppoNodes, populationGrid, {
        snapToNearestNode: !showGreedySim
      }),
    [carsForPresenceGrid, ppoNodes, populationGrid, showGreedySim]
  );

  const carGridStats = useMemo(() => {
    let carsInGrid = 0;
    for (const cell of carPresenceByCell.values()) {
      carsInGrid += cell.carIds.length;
    }
    return {
      carsInGrid,
      occupiedCells: carPresenceByCell.size,
      totalCells: (populationGrid?.rows ?? 0) * (populationGrid?.cols ?? 0),
      snapMode: showGreedySim ? "visible greedy car position" : ppoNodes.length ? "nearest intersection" : "grid position"
    };
  }, [carPresenceByCell, populationGrid, ppoNodes.length, showGreedySim]);

  const demoStats = useMemo(() => {
    const greedy = mobilitySnapshot?.map_greedy_stats ?? mobilitySnapshot?.greedy_stats;
    if (greedy) {
      return {
        completedTrips: greedy.completed_trips ?? 0,
        revenue: greedy.revenue ?? 0,
        waitTimeMin: snapshotWaitTimeMinutes(mobilitySnapshot),
        fleetUtilization: greedy.avg_fleet_utilization_pct ?? greedy.fleet_utilization_pct ?? 0,
        activeCars: greedy.active_cars ?? 0,
        stalledCars: greedy.stalled_cars ?? 0,
        unassignedPeople: greedy.unassigned_people ?? 0,
        source: "greedy"
      };
    }

    const fleetSize = Math.max(uberCars.length, trips.length, 1);
    const dayProgress = clamp(clockMinute / DAY_MINUTES, 0, 1);
    const congestionPenalty = clamp((currentHourCongestion.mean - 1.2) / 3, 0, 0.32);
    const activeShare = clamp(carGridStats.carsInGrid / fleetSize, 0, 1);
    const completedTrips = Math.round(
      fleetSize * (18 + dayProgress * 74) * (0.88 + flowFactor * 0.16)
    );
    const averageFare = 18.5 + 3.2 * clamp(currentHourCongestion.mean - 1, 0, 2);
    const revenue = completedTrips * averageFare;
    const estimatedWaitPerTrip = clamp(
      2.5 + congestionPenalty * 18 + currentHourCongestion.hotShare * 8 + (1 - activeShare) * 2,
      2,
      24
    );
    const waitTimeMin = completedTrips * estimatedWaitPerTrip;
    const fleetUtilization = clamp(
      52 + activeShare * 31 + (1 - flowFactor) * 16 + timeTrafficPressureFactor(currentHourFloat) * 3,
      48,
      96
    );

    return {
      completedTrips,
      revenue,
      waitTimeMin,
      fleetUtilization,
      activeCars: uberCars.length,
      stalledCars: 0,
      unassignedPeople: 0,
      source: "estimate"
    };
  }, [
    carGridStats.carsInGrid,
    clockMinute,
    currentHourCongestion,
    currentHourFloat,
    flowFactor,
    mobilitySnapshot,
    trips.length,
    uberCars.length
  ]);

  const peopleGridStats = useMemo(() => {
    const totalCells = (populationGrid?.rows ?? 0) * (populationGrid?.cols ?? 0);
    const pickupCells = new Set();
    const dropoffCells = new Set();
    const visiblePeople = new Set();
    const mapPeople = activeGreedyPeople(mobilitySnapshot);
    const {assignmentByPersonId, carById} = activeAssignmentContext(mobilitySnapshot);
    for (const person of mapPeople) {
      const assignment = assignmentByPersonId.get(person.id);
      const car = assignment ? carById.get(assignment.car_id) : null;
      const visible = visibleMarkerKinds(person, assignment, car, mobilitySnapshot, clockMinute, mobilityStepMinutes);
      const pickupCell = person.pickup_grid_cell ?? person.origin;
      const dropoffCell = person.dropoff_grid_cell ?? person.destination;
      if (pickupCell && visible.pickup) {
        pickupCells.add(gridKey(pickupCell[0], pickupCell[1]));
        visiblePeople.add(person.id);
      }
      if (dropoffCell && visible.dropoff) {
        dropoffCells.add(gridKey(dropoffCell[0], dropoffCell[1]));
        visiblePeople.add(person.id);
      }
    }
    const assignments = activeGreedyAssignments(mobilitySnapshot);
    return {
      totalCells,
      people: visiblePeople.size,
      pickupCells: pickupCells.size,
      dropoffCells: dropoffCells.size,
      assignments: assignments.length
    };
  }, [mobilitySnapshot, populationGrid, clockMinute, mobilityStepMinutes]);

  const carPresenceGridGeoJson = useMemo(
    () =>
      showCarGrid
        ? makeCarPresenceGridFeatureCollection(
            populationGrid,
            carPresenceByCell,
            nodeCountsByCell
          )
        : null,
    [showCarGrid, populationGrid, carPresenceByCell, nodeCountsByCell]
  );

  const carPresenceGridLayer = useMemo(() => {
    if (!showCarGrid || !carPresenceGridGeoJson) return null;

    return new GeoJsonLayer({
      id: "car-presence-grid",
      data: carPresenceGridGeoJson,
      stroked: true,
      filled: true,
      pickable: true,
      autoHighlight: true,
      highlightColor: [255, 255, 255, 80],
      lineWidthMinPixels: 0.35,
      lineWidthMaxPixels: 2.5,
      getLineWidth: f => (f.properties?.has_car ? 2 : 0.6),
      getFillColor: f =>
        f.properties?.has_car ? [0, 210, 165, 105] : [16, 31, 48, 18],
      getLineColor: f =>
        f.properties?.has_car ? [0, 255, 190, 235] : [130, 170, 205, 55]
    });
  }, [showCarGrid, carPresenceGridGeoJson]);

  const eventAreaLayer = useMemo(() => {
    if (!activeEvent) return null;
    return new GeoJsonLayer({
      id: "active-event-area",
      data: makeEventFeatureCollection(activeEvent),
      stroked: true,
      filled: true,
      pickable: true,
      autoHighlight: true,
      highlightColor: [255, 255, 255, 70],
      lineWidthMinPixels: 2,
      getFillColor: [239, 68, 68, 46],
      getLineColor: [248, 113, 113, 245],
      getTooltip: ({object}) => {
        if (!object) return null;
        return {
          text: `${object.properties?.label ?? "Event"}\n${object.properties?.description ?? ""}`
        };
      }
    });
  }, [activeEvent]);

  const greedyPeopleLayer = useMemo(() => {
    if (!showGreedySim || !mobilitySnapshot || !populationGrid) return null;
    return new GeoJsonLayer({
      id: "greedy-people-grid",
      data: makeGreedyPeopleFeatureCollection(
        mobilitySnapshot,
        populationGrid,
        clockMinute,
        mobilityStepMinutes
      ),
      stroked: true,
      filled: true,
      pointType: "circle",
      pickable: true,
      autoHighlight: true,
      highlightColor: [255, 255, 255, 85],
      pointRadiusUnits: "pixels",
      getPointRadius: f => (f.properties?.kind === "pickup" ? 16 : 15),
      pointRadiusMinPixels: 12,
      pointRadiusMaxPixels: 24,
      lineWidthMinPixels: 3,
      getFillColor: f =>
        f.properties?.kind === "pickup" ? PICKUP_BLUE : DESTINATION_RED,
      getLineColor: f =>
        f.properties?.kind === "pickup" ? [232, 240, 254, 250] : [252, 232, 230, 250]
    });
  }, [showGreedySim, mobilitySnapshot, populationGrid, clockMinute, mobilityStepMinutes]);

  const greedyPeopleGridLayer = useMemo(() => {
    if (!showGreedySim || !showPeopleGrid || !mobilitySnapshot || !populationGrid) return null;
    return new GeoJsonLayer({
      id: "greedy-people-grid-cells",
      data: makeGreedyPeopleGridFeatureCollection(
        mobilitySnapshot,
        populationGrid,
        clockMinute,
        mobilityStepMinutes
      ),
      stroked: true,
      filled: true,
      pickable: true,
      autoHighlight: true,
      highlightColor: [255, 255, 255, 70],
      lineWidthMinPixels: 0.7,
      getLineWidth: f => (f.properties?.has_people ? 3.4 : 0.45),
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
    });
  }, [showGreedySim, showPeopleGrid, mobilitySnapshot, populationGrid, clockMinute, mobilityStepMinutes]);

  const greedyRouteCasingLayer = useMemo(() => {
    if (!showGreedySim || !mobilitySnapshot || !populationGrid) return null;
    return new GeoJsonLayer({
      id: "greedy-dispatch-route-casing",
      data: makeGreedyRouteFeatureCollection(
        mobilitySnapshot,
        populationGrid,
        "to_dropoff",
        clockMinute,
        mobilityStepMinutes,
        routeIndex
      ),
      stroked: true,
      filled: false,
      pickable: false,
      lineWidthMinPixels: 11,
      lineWidthMaxPixels: 18,
      getLineColor: ROUTE_CASING
    });
  }, [showGreedySim, mobilitySnapshot, populationGrid, clockMinute, mobilityStepMinutes, routeIndex]);

  const greedyRouteLayer = useMemo(() => {
    if (!showGreedySim || !mobilitySnapshot || !populationGrid) return null;
    return new GeoJsonLayer({
      id: "greedy-dispatch-routes",
      data: makeGreedyRouteFeatureCollection(
        mobilitySnapshot,
        populationGrid,
        "to_dropoff",
        clockMinute,
        mobilityStepMinutes,
        routeIndex
      ),
      stroked: true,
      filled: false,
      pickable: true,
      lineWidthMinPixels: 7,
      lineWidthMaxPixels: 13,
      getLineColor: ROUTE_BLUE
    });
  }, [showGreedySim, mobilitySnapshot, populationGrid, clockMinute, mobilityStepMinutes, routeIndex]);

  const greedyCarLayer = useMemo(() => {
    if (!showGreedySim || !mobilitySnapshot || !populationGrid) return null;
    return new ScatterplotLayer({
      id: "greedy-cars",
      data: greedyCars,
      pickable: true,
      radiusUnits: "meters",
      radiusMinPixels: 5,
      radiusMaxPixels: 13,
      getRadius: d => (d.status === "idle" ? 34 : 46),
      getPosition: d => d.position,
      getFillColor: d => (d.status === "idle" ? IDLE_CAR_GRAY : ACTIVE_CAR_GREEN),
      getLineColor: [255, 255, 255, 230],
      lineWidthMinPixels: 1.5,
      stroked: true
    });
  }, [showGreedySim, mobilitySnapshot, populationGrid, greedyCars]);

  const uberLayer = useMemo(
    () =>
      showGreedySim
        ? null
        :
      new IconLayer({
        id: "uber-cars",
        data: uberCars,
        pickable: true,
        sizeScale: 1,
        getPosition: d => d.position,
        getIcon: () => ({
          url: UBER_CAR_ICON_URL,
          width: 64,
          height: 64,
          anchorY: 32
        }),
        getSize: 24
      }),
    [showGreedySim, uberCars]
  );

  const layers = [
    carPresenceGridLayer,
    greedyPeopleGridLayer,
    eventAreaLayer,
    edgeLayer,
    commuteWaveLayer,
    greedyRouteCasingLayer,
    greedyRouteLayer,
    nodeDensityLayer,
    !showGreedySim && new TripsLayer({
      id: "glow",
      data: trips,
      getPath: d => d.path,
      getTimestamps: d => d.timestamps,
      getColor: [0, 255, 255],
      opacity: 0.12,
      widthMinPixels: 14,
      trailLength: 55,
      currentTime: tripClockMinute
    }),

    !showGreedySim && new TripsLayer({
      id: "trips",
      data: trips,
      getPath: d => d.path,
      getTimestamps: d => d.timestamps,
      getColor: [0, 180, 255],
      opacity: 0.9,
      widthMinPixels: 4,
      trailLength: 55,
      currentTime: tripClockMinute
    }),
    greedyPeopleLayer,
    greedyCarLayer,
    uberLayer
  ].filter(Boolean);

  const controlButtonStyle = {
    padding: "8px 12px",
    background: "black",
    color: "white",
    border: "1px solid #333"
  };
  const telemetryPanelStyle = {
    position: "absolute",
    zIndex: 10,
    margin: 12,
    marginTop: 158,
    width: 390,
    maxWidth: "calc(100vw - 24px)",
    overflow: "hidden",
    background: "rgba(5,10,16,0.84)",
    color: "white",
    border: "1px solid rgba(120,150,180,0.38)",
    borderRadius: 8,
    boxShadow: "0 18px 50px rgba(0,0,0,0.38)",
    backdropFilter: "blur(10px)",
    fontFamily: "sans-serif",
    fontSize: 13,
    pointerEvents: "auto"
  };
  const demoStatsPanelStyle = {
    position: "absolute",
    zIndex: 10,
    top: keyMissing ? 148 : 12,
    right: 12,
    width: 360,
    maxWidth: "calc(100vw - 24px)",
    padding: 14,
    background: "rgba(4,8,14,0.84)",
    color: "white",
    border: "1px solid rgba(0,210,165,0.34)",
    borderRadius: 8,
    boxShadow: "0 18px 50px rgba(0,0,0,0.4)",
    backdropFilter: "blur(12px)",
    fontFamily: "sans-serif",
    pointerEvents: "auto"
  };
  const greedyLegendStyle = {
    position: "absolute",
    zIndex: 11,
    left: 12,
    bottom: 12,
    display: "flex",
    alignItems: "center",
    gap: 12,
    padding: "9px 11px",
    borderRadius: 8,
    background: "rgba(4,8,14,0.82)",
    color: "white",
    border: "1px solid rgba(138,180,248,0.34)",
    boxShadow: "0 14px 40px rgba(0,0,0,0.34)",
    backdropFilter: "blur(10px)",
    fontFamily: "sans-serif",
    fontSize: 12,
    pointerEvents: "auto"
  };
  const legendBadge = (label, color) => (
    <span style={{display: "inline-flex", alignItems: "center", gap: 6}}>
      <span
        style={{
          width: 22,
          height: 22,
          borderRadius: 999,
          background: color,
          border: "2px solid rgba(255,255,255,0.92)",
          display: "inline-flex"
        }}
      />
      <span>{label}</span>
    </span>
  );
  const demoStatItems = [
    {
      label: "Completed Trips",
      value: demoStats.completedTrips.toLocaleString(),
      detail:
        demoStats.source === "greedy"
          ? `${demoStats.activeCars} active, ${demoStats.stalledCars} stalled`
          : `${uberCars.length} active vehicles`,
      accent: "#00d2a5"
    },
    {
      label: "Profit",
      value: formatCurrency(demoStats.revenue),
      detail: `avg fare ${formatCurrency(demoStats.revenue / Math.max(1, demoStats.completedTrips))}`,
      accent: "#7dd3fc"
    },
    {
      label: "Wait Time",
      value: formatMinutes(demoStats.waitTimeMin),
      detail:
        demoStats.source === "greedy"
          ? "total customer minutes"
          : "estimated customer minutes",
      progress: clamp((demoStats.waitTimeMin / 1800) * 100, 0, 100),
      accent: "#facc15"
    },
    {
      label: "Avg Fleet Utilization",
      value: `${demoStats.fleetUtilization.toFixed(0)}%`,
      detail: demoStats.source === "greedy" ? "mean assigned fleet" : "estimated mean active",
      progress: demoStats.fleetUtilization,
      accent: "#fb7185"
    }
  ];

  return (
    <>
      <div
        style={{
          position: "absolute",
          zIndex: 10,
          top: 12,
          left: 12,
          display: "flex",
          flexWrap: "wrap",
          gap: 8,
          maxWidth: "calc(100vw - 24px)",
          pointerEvents: "auto"
        }}
      >
        <button
          type="button"
          onClick={() => setShowNodeDensity(v => !v)}
          style={controlButtonStyle}
        >
          Nodes: {showNodeDensity ? "ON" : "OFF"}
        </button>

        <button
          type="button"
          onClick={() => setShowCarGrid(v => !v)}
          style={controlButtonStyle}
        >
          Cars Grid: {showCarGrid ? "ON" : "OFF"}
        </button>

        <button
          type="button"
          onClick={() => setShowGreedySim(v => !v)}
          style={controlButtonStyle}
        >
          Greedy: {showGreedySim ? "ON" : "OFF"}
        </button>

        <button
          type="button"
          onClick={() => setShowPeopleGrid(v => !v)}
          style={controlButtonStyle}
        >
          People Grid: {showPeopleGrid ? "ON" : "OFF"}
        </button>

        <a href="/rl.html" style={{...controlButtonStyle, textDecoration: "none"}}>
          Agentic Fleet
        </a>

        <a href="/" style={{...controlButtonStyle, textDecoration: "none"}}>
          FleetForge
        </a>

        <button
          type="button"
          onClick={() => setSimSpeed(s => nextSimSpeed(s))}
          style={controlButtonStyle}
        >
          Speed: x{formatSpeed(simSpeed)}
        </button>

        <button
          type="button"
          onClick={() => {
            if (paused && atSimulationEnd) {
              setClockMinute(SIM_RESET_MINUTE);
              setPaused(false);
              return;
            }
            setPaused(v => !v);
          }}
          style={controlButtonStyle}
        >
          {paused ? "Resume" : "Pause"}
        </button>
      </div>

      <div
        style={{
          position: "absolute",
          zIndex: 10,
          top: 68,
          left: 12,
          display: "flex",
          gap: 8,
          pointerEvents: "auto"
        }}
      >
        {TIME_PRESETS.map(preset => (
          <button
            type="button"
            key={preset.label}
            onMouseDown={e => e.stopPropagation()}
            onClick={e => {
              e.stopPropagation();
              setPaused(true);
              setClockMinute(preset.minute);
            }}
            style={{
              padding: "7px 10px",
              background: "black",
              color: "white",
              border: "1px solid #333"
            }}
          >
            {preset.label}
          </button>
        ))}
      </div>

      {mobilityEvents.length > 0 && (
        <div
          style={{
            position: "absolute",
            zIndex: 10,
            top: 112,
            left: 12,
            display: "flex",
            flexWrap: "wrap",
            alignItems: "center",
            gap: 8,
            maxWidth: "calc(100vw - 24px)",
            pointerEvents: "auto"
          }}
        >
          <span
            style={{
              padding: "7px 9px",
              background: "rgba(4,8,14,0.82)",
              color: "white",
              border: "1px solid rgba(120,150,180,0.34)",
              fontFamily: "sans-serif",
              fontSize: 13
            }}
          >
            Events
          </span>
          {mobilityEvents.map(event => {
            const active = activeEventId === event.id;
            return (
              <button
                type="button"
                key={event.id}
                title={event.description}
                onClick={() => setActiveEventId(id => (id === event.id ? null : event.id))}
                style={{
                  ...controlButtonStyle,
                  background: active ? "#facc15" : "black",
                  color: active ? "#111827" : "white",
                  border: active ? "1px solid #fde68a" : "1px solid #333"
                }}
              >
                {event.short_label ?? event.label}
              </button>
            );
          })}
        </div>
      )}

      <div style={telemetryPanelStyle}>
        <button
          type="button"
          aria-expanded={!telemetryCollapsed}
          onClick={() => setTelemetryCollapsed(v => !v)}
          style={{
            width: "100%",
            border: 0,
            padding: "10px 12px",
            background: "rgba(255,255,255,0.04)",
            color: "white",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            gap: 12,
            fontFamily: "sans-serif",
            cursor: "pointer"
          }}
        >
          <span style={{display: "flex", flexDirection: "column", alignItems: "flex-start", gap: 2}}>
            <span style={{fontSize: 11, letterSpacing: 0, opacity: 0.58}}>Telemetry</span>
            <span style={{fontSize: 15, fontWeight: 700}}>Simulated Time: {clockLabel}</span>
          </span>
          <span style={{display: "flex", alignItems: "center", gap: 10, fontSize: 12}}>
            <span
              style={{
                padding: "4px 8px",
                borderRadius: 999,
                background: paused ? "rgba(250,204,21,0.16)" : "rgba(0,210,165,0.16)",
                color: paused ? "#fde68a" : "#99f6e4",
                border: paused ? "1px solid rgba(250,204,21,0.34)" : "1px solid rgba(0,210,165,0.34)"
              }}
            >
              {paused ? "Paused" : "Running"}
            </span>
            <span style={{opacity: 0.74}}>{telemetryCollapsed ? "Show" : "Hide"}</span>
          </span>
        </button>

        {!telemetryCollapsed && (
          <div style={{padding: "10px 12px 12px", lineHeight: 1.42}}>
            <div style={{opacity: 0.8}}>Traffic state: {trafficStateLabel(currentHour)}</div>
            <div style={{opacity: 0.75}}>Commute flow: {commuteFlowLabel(currentHour)}</div>
            <div style={{opacity: 0.75}}>Commute wave: {commuteWaveLabel(currentHour)}</div>
            <div style={{opacity: 0.8}}>
              Traffic profile: Google-style green, yellow, orange, red
            </div>
            <div style={{opacity: 0.75}}>
              Traffic congestion range (global): {globalCongestionScale.min.toFixed(2)} -{" "}
              {globalCongestionScale.max.toFixed(2)}
            </div>
            <div style={{opacity: 0.75}}>
              Traffic color scale (global p05-p95): {globalCongestionScale.p05.toFixed(2)} -{" "}
              {globalCongestionScale.p95.toFixed(2)}
            </div>
            <div style={{opacity: 0.72}}>
              Legend: green &lt;1.5, yellow ~2.1, orange ~2.8, red &gt;3.2
            </div>
            <div style={{opacity: 0.8}}>
              Population: blue (low) to red (high), grid 50x50
            </div>
            <div style={{opacity: 0.78}}>
              Cars grid: {showCarGrid ? "ON" : "OFF"} ({carGridStats.snapMode})
            </div>
            <div style={{opacity: 0.78}}>
              Greedy dispatch: {mobilitySnapshot ? `loaded snapshot ${String(Math.floor((mobilitySnapshot.timestep ?? 0) / 60))
                .padStart(2, "0")}:${String((mobilitySnapshot.timestep ?? 0) % 60).padStart(2, "0")}` : mobilityWorldStatus}
            </div>
            <div style={{opacity: 0.78}}>
              Event scenario: {activeEvent ? activeEvent.label : "none"}
            </div>
            <div style={{opacity: 0.72}}>
              People grid: {showPeopleGrid ? "ON" : "OFF"} | {peopleGridStats.people} people | pickups{" "}
              {peopleGridStats.pickupCells}/{peopleGridStats.totalCells || "?"} | destinations{" "}
              {peopleGridStats.dropoffCells}/{peopleGridStats.totalCells || "?"}
            </div>
            <div style={{opacity: 0.72}}>
              Dispatch path: {peopleGridStats.assignments} greedy Dijkstra assignments drawn
            </div>
            <div style={{opacity: 0.65}}>
              Occupied cells: {carGridStats.occupiedCells}/{carGridStats.totalCells || "?"} | cars mapped:{" "}
              {carGridStats.carsInGrid}/{uberCars.length}
            </div>
            <div style={{opacity: 0.75}}>
              Population time factor: {(POPULATION_HOURLY_MULTIPLIER[currentHour] ?? 1).toFixed(2)}x
            </div>
            <div style={{opacity: 0.75}}>
              Dataset: {datasetReady ? "OSMnx loaded" : "fallback trips only"}
            </div>
            <div style={{opacity: 0.65}}>
              Density range: {Math.round(densityStats.min)} - {Math.round(densityStats.max)}
            </div>
            <div style={{opacity: 0.65}}>
              Flow speed x{flowFactor.toFixed(2)}
            </div>
            <div style={{opacity: 0.65}}>
              Current hour congestion: mean {currentHourCongestion.mean.toFixed(2)} | p90{" "}
              {currentHourCongestion.p90.toFixed(2)}
            </div>
            <div style={{opacity: 0.65}}>
              Hot roads: {(currentHourCongestion.hotShare * 100).toFixed(0)}%
            </div>
          </div>
        )}
      </div>

      <div style={demoStatsPanelStyle}>
        <div style={{display: "flex", justifyContent: "space-between", gap: 12, marginBottom: 12}}>
          <div>
            <div style={{fontSize: 11, opacity: 0.58}}>
              {demoStats.source === "greedy" ? "Greedy Stats" : "Demo Stats"}
            </div>
            <div style={{fontSize: 17, fontWeight: 800}}>Dispatch Performance</div>
          </div>
          <div
            style={{
              alignSelf: "flex-start",
              padding: "5px 8px",
              borderRadius: 999,
              background: "rgba(0,210,165,0.14)",
              color: "#99f6e4",
              border: "1px solid rgba(0,210,165,0.3)",
              fontSize: 12
            }}
          >
            Live
          </div>
        </div>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(2, minmax(0, 1fr))",
            gap: 10
          }}
        >
          {demoStatItems.map(item => (
            <div
              key={item.label}
              style={{
                minHeight: 104,
                padding: 12,
                borderRadius: 8,
                background: "rgba(255,255,255,0.055)",
                border: "1px solid rgba(255,255,255,0.09)",
                boxShadow: `inset 3px 0 0 ${item.accent}`
              }}
            >
              <div style={{fontSize: 11, opacity: 0.6, marginBottom: 7}}>{item.label}</div>
              <div style={{fontSize: 25, fontWeight: 800, lineHeight: 1, color: item.accent}}>
                {item.value}
              </div>
              <div style={{fontSize: 12, opacity: 0.64, marginTop: 8}}>{item.detail}</div>
              {item.progress !== undefined && (
                <div
                  style={{
                    height: 6,
                    borderRadius: 999,
                    background: "rgba(255,255,255,0.12)",
                    overflow: "hidden",
                    marginTop: 12
                  }}
                >
                  <div
                    style={{
                      width: `${item.progress}%`,
                      height: "100%",
                      borderRadius: 999,
                      background: item.accent
                    }}
                  />
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {showGreedySim && (
        <div style={greedyLegendStyle}>
          {legendBadge("Pickup", "#4285f4")}
          {legendBadge("Dropoff", "#ea4335")}
          {legendBadge("Active car node", "#34a853")}
          {legendBadge("Idle car node", "#bdbfc6")}
          {activeEvent && legendBadge("Event zone", "#facc15")}
          <span style={{display: "inline-flex", alignItems: "center", gap: 7}}>
            <span
              style={{
                width: 34,
                height: 0,
                borderTop: "6px solid #4285f4",
                boxShadow: "0 0 0 3px rgba(232,240,254,0.86)",
                borderRadius: 999
              }}
            />
            <span>Route</span>
          </span>
          <span style={{opacity: 0.72}}>
            {mobilityWorldStatus === "ready"
              ? `${peopleGridStats.people} people, ${peopleGridStats.assignments} paths`
              : `world ${mobilityWorldStatus}`}
          </span>
        </div>
      )}

      {keyMissing && (
        <div
          style={{
            position: "absolute",
            zIndex: 10,
            top: 12,
            right: 12,
            maxWidth: 320,
            padding: "10px 14px",
            background: "rgba(20,0,0,0.85)",
            color: "white",
            border: "1px solid #800",
            borderRadius: 4,
            fontFamily: "sans-serif",
            fontSize: 13,
            lineHeight: 1.4
          }}
        >
          ⚠️ Set <code>MAPTILER_KEY</code> at the top of this file to a free key
          from{" "}
          <a
            href="https://cloud.maptiler.com/account/keys/"
            target="_blank"
            rel="noreferrer"
            style={{color: "#9cf"}}
          >
            cloud.maptiler.com
          </a>{" "}
          to load the map and 3D buildings.
        </div>
      )}

      <DeckGL
        initialViewState={INITIAL_VIEW_STATE}
        controller={true}
        layers={layers}
        getTooltip={({object, layer}) => {
          if (!object || !layer) return null;
          if (layer.id === "greedy-people-grid-cells") {
            const p = object.properties ?? {};
            const cellType =
              p.has_pickup && p.has_dropoff
                ? "Pickup + destination grid cell"
                : p.has_pickup
                  ? "Pickup grid cell"
                  : p.has_dropoff
                    ? "Destination grid cell"
                    : "People grid cell";
            return {
              text:
                `${cellType} [${p.row}, ${p.col}]\n` +
                `pickups: ${p.pickup_count ?? 0}\n` +
                `destinations: ${p.dropoff_count ?? 0}\n` +
                `pickup IDs: ${(p.pickup_ids ?? []).join(", ") || "none"}\n` +
                `destination IDs: ${(p.dropoff_ids ?? []).join(", ") || "none"}`
            };
          }
          if (layer.id === "greedy-people-grid") {
            const p = object.properties ?? {};
            return {
              text:
                `${p.kind === "pickup" ? "Pickup" : "Dropoff"} node ${p.node_id ?? "?"}\n` +
                `person: ${p.person_id}\n` +
                `source grid: [${p.grid_cell?.[0] ?? p.row ?? "?"}, ${p.grid_cell?.[1] ?? p.col ?? "?"}]\n` +
                `paired node: ${p.paired_node_id ?? "?"}`
            };
          }
          if (layer.id === "greedy-dispatch-routes") {
            const p = object.properties ?? {};
            return {
              text:
                `Active car route\n` +
                `car: ${p.car_id}\n` +
                `person: ${p.person_id}\n` +
                `traffic-weighted cost: ${Number(p.cost ?? 0).toFixed(2)}`
            };
          }
          if (layer.id === "greedy-cars") {
            return {
              text:
                `${object.id}\n` +
                `status: ${object.status}\n` +
                `node: ${object.node_id ?? "?"}\n` +
                `person: ${object.assigned_person_id ?? "none"}\n` +
                `stall ticks: ${object.stall_ticks ?? 0}`
            };
          }
          if (layer.id === "car-presence-grid") {
            const p = object.properties ?? {};
            const nodeIds = p.snapped_node_ids ?? [];
            return {
              text:
                `grid cell [${p.row}, ${p.col}]\n` +
                `cars present: ${p.has_car ? "yes" : "no"}\n` +
                `car count: ${p.car_count ?? 0}\n` +
                `intersection nodes in cell: ${p.intersection_count ?? 0}\n` +
                `snapped intersections: ${nodeIds.length ? nodeIds.join(", ") : "none"}`
            };
          }
          if (layer.id === "population-grid") {
            const row = object.properties?.row ?? 0;
            const col = object.properties?.col ?? 0;
            const base = object.properties?.population_density ?? 0;
            const factor = populationTimeFactor(
              currentHour,
              row,
              col,
              populationGrid?.rows ?? 50,
              populationGrid?.cols ?? 50
            );
            return {
              text:
                `grid cell [${row}, ${col}]\n` +
                `base density: ${Math.round(base)}\n` +
                `time-adjusted: ${Math.round(base * factor)}`
            };
          }
          if (layer.id === "node-density") {
            const factor = populationTimeFactor(
              currentHour,
              object.grid_row ?? 0,
              object.grid_col ?? 0,
              populationGrid?.rows ?? 50,
              populationGrid?.cols ?? 50
            );
            return {
              text:
                `node: ${object.node_id}\n` +
                `grid: [${object.grid_row}, ${object.grid_col}]\n` +
                `base: ${Math.round(object.population_density_base ?? 0)}\n` +
                `noisy: ${Math.round(object.population_density_noisy ?? 0)}\n` +
                `time-adjusted: ${Math.round((object.population_density_noisy ?? 0) * factor)}`
            };
          }
          if (layer.id === "osmnx-edges") {
            const p = object.properties ?? {};
            const speed = p.hourly_speed_kph?.[currentHour];
            const ttime = p.hourly_travel_time_s?.[currentHour];
            const baseCong = p.hourly_congestion_factor?.[currentHour] ?? 1;
            const simCong =
              effectiveCongestionForEdge(object, trafficDisplayHour) *
              generatedTrafficMultiplierForEdge(object, generatedTrafficPressureByCell, populationGrid);
            const multiplier = simCong / Math.max(0.05, baseCong);
            const simSpeed = typeof speed === "number" ? speed / multiplier : speed;
            const simTime = typeof ttime === "number" ? ttime * multiplier : ttime;
            return {
              text:
                `${p.name || p.highway || "road"}\n` +
                `hour ${currentHour} base: ${speed?.toFixed?.(1) ?? speed ?? "?"} kph\n` +
                `simulated: ${simSpeed?.toFixed?.(1) ?? simSpeed ?? "?"} kph\n` +
                `travel time: ${simTime?.toFixed?.(1) ?? simTime ?? "?"} s\n` +
                `length: ${p.length_m ?? "?"} m`
            };
          }
          if (layer.id === "uber-cars") {
            return {text: "Uber-like vehicle\nfollowing active trip"};
          }
          return null;
        }}
        style={{width: "100vw", height: "100vh"}}
      >
        <StaticMap
          mapStyle={buildStyleUrl()}
          onLoad={e => {
            const map = e.target;
            hideBaseTransportationLayers(map);
            map.on("styledata", () => hideBaseTransportationLayers(map));
            setTimeout(() => hideBaseTransportationLayers(map), 250);
            setTimeout(() => hideBaseTransportationLayers(map), 1000);
          }}
        />
      </DeckGL>

    </>
  );
}

const root = document.createElement("div");
document.body.style.margin = 0;
document.body.appendChild(root);

createRoot(root).render(<App />);
