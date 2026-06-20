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
const SIM_SPEED_STEPS = [1, 4, 12, 30];
const TIME_PRESETS = [
  {label: "02:00", minute: 2 * 60},
  {label: "08:00", minute: 8 * 60},
  {label: "12:00", minute: 12 * 60},
  {label: "17:00", minute: 17 * 60},
  {label: "21:00", minute: 21 * 60}
];
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
  if (congestion < 1.35) return [45, 220, 90, 240];
  if (congestion < 1.8) return [125, 220, 65, 240];
  if (congestion < 2.4) return [235, 215, 55, 240];
  if (congestion < 3.4) return [255, 155, 45, 240];
  if (congestion < 4.8) return [235, 90, 45, 245];
  return [150, 25, 35, 245];
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

function nextSimSpeed(current) {
  const idx = SIM_SPEED_STEPS.indexOf(current);
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

function App() {
  const [clockMinute, setClockMinute] = useState(7 * 60);
  const [simSpeed, setSimSpeed] = useState(1);
  const [paused, setPaused] = useState(false);
  const [keyMissing, setKeyMissing] = useState(MAPTILER_KEY === "YOUR_MAPTILER_KEY");
  const [network, setNetwork] = useState(null);
  const [populationGrid, setPopulationGrid] = useState(null);
  const [ppoNodes, setPpoNodes] = useState([]);
  const [trips, setTrips] = useState(FALLBACK_TRIPS);
  const [datasetReady, setDatasetReady] = useState(false);
  const [showNodeDensity, setShowNodeDensity] = useState(false);

  // Simulated clock in minutes across a day.
  useEffect(() => {
    const i = setInterval(() => {
      if (paused) return;
      setClockMinute(m => (m + simSpeed) % (24 * 60));
    }, 50);
    return () => clearInterval(i);
  }, [simSpeed, paused]);

  // Load OSMnx-generated artifacts.
  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const [edgesRes, tripsRes, gridRes, nodesRes] = await Promise.all([
          fetch("/data/osmnx_edges.geojson"),
          fetch("/data/sample_trips.json"),
          fetch("/data/population_density_grid.json"),
          fetch("/data/ppo_nodes.json")
        ]);
        if (!edgesRes.ok || !tripsRes.ok) return;

        const [edgeData, tripData] = await Promise.all([edgesRes.json(), tripsRes.json()]);
        if (cancelled) return;
        setNetwork(edgeData);
        setTrips(Array.isArray(tripData) && tripData.length > 0 ? tripData : FALLBACK_TRIPS);

        if (gridRes.ok) {
          const gridData = await gridRes.json();
          if (!cancelled) setPopulationGrid(gridData);
        }
        if (nodesRes.ok) {
          const nodeData = await nodesRes.json();
          if (!cancelled) setPpoNodes(Array.isArray(nodeData?.nodes) ? nodeData.nodes : []);
        }
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

  const currentHour = Math.floor(clockMinute / 60);
  const currentHourFloat = clockMinute / 60;
  const clockLabel = `${String(currentHour).padStart(2, "0")}:${String(
    clockMinute % 60
  ).padStart(2, "0")}`;
  const flowFactor = trafficFlowSpeedFactor(currentHourFloat);
  const tripClockMinute = clockMinute * flowFactor;

  const gridGeoJson = useMemo(
    () => makeGridFeatureCollection(populationGrid),
    [populationGrid]
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
      .map(f => effectiveCongestionForEdge(f, currentHourFloat))
      .filter(v => typeof v === "number" && !Number.isNaN(v))
      .sort((a, b) => a - b);
    if (!vals.length) return {mean: 1, p50: 1, p90: 1, hotShare: 0};
    const q = p => vals[Math.floor((vals.length - 1) * p)];
    const mean = vals.reduce((acc, v) => acc + v, 0) / vals.length;
    const hotShare = vals.filter(v => v >= 2.5).length / vals.length;
    return {mean, p50: q(0.5), p90: q(0.9), hotShare};
  }, [network, currentHourFloat]);

  const edgeLayer = useMemo(() => {
    if (!network) return null;

    return new GeoJsonLayer({
      id: "osmnx-edges",
      data: network,
      stroked: true,
      filled: false,
      lineWidthMinPixels: 1.25,
      lineWidthMaxPixels: 8,
      getLineWidth: f => {
        const congestion = effectiveCongestionForEdge(f, currentHourFloat);
        const t = clamp((congestion - 1.0) / (4.8 - 1.0), 0, 1);
        return 0.9 + t * 5.4;
      },
      getLineColor: f => {
        const congestion = effectiveCongestionForEdge(f, currentHourFloat);
        return trafficColorFromCongestion(congestion);
      },
      updateTriggers: {
        getLineWidth: [currentHourFloat],
        getLineColor: [currentHourFloat]
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
        const simCong = effectiveCongestionForEdge(object, currentHourFloat);
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
  }, [network, currentHour, currentHourFloat, globalCongestionScale]);

  const commuteWaveLayer = useMemo(() => {
    if (!network) return null;

    return new GeoJsonLayer({
      id: "commute-wave",
      data: network,
      stroked: true,
      filled: false,
      lineWidthMinPixels: 0,
      lineWidthMaxPixels: 14,
      getLineWidth: f => {
        const intensity = commuteWaveIntensity(f, currentHourFloat);
        return intensity < 0.08 ? 0 : 2 + intensity * 8;
      },
      getLineColor: f => {
        const intensity = commuteWaveIntensity(f, currentHourFloat);
        if (intensity < 0.08) return [0, 0, 0, 0];
        return [
          255,
          Math.round(120 - 70 * intensity),
          Math.round(35 - 20 * intensity),
          Math.round(80 + 160 * intensity)
        ];
      },
      updateTriggers: {
        getLineWidth: [currentHourFloat],
        getLineColor: [currentHourFloat]
      },
      pickable: false
    });
  }, [network, currentHourFloat]);

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

  const uberLayer = useMemo(
    () =>
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
    [uberCars]
  );

  const layers = [
    edgeLayer,
    commuteWaveLayer,
    nodeDensityLayer,
    new TripsLayer({
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

    new TripsLayer({
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
    uberLayer
  ].filter(Boolean);

  return (
    <>
      <button
        onClick={() => setShowNodeDensity(v => !v)}
        style={{
          position: "absolute",
          zIndex: 10,
          margin: 12,
          padding: "8px 12px",
          background: "black",
          color: "white",
          border: "1px solid #333"
        }}
      >
        Nodes: {showNodeDensity ? "ON" : "OFF"}
      </button>

      <button
        onClick={() => setSimSpeed(s => nextSimSpeed(s))}
        style={{
          position: "absolute",
          zIndex: 10,
          margin: 12,
          marginLeft: 116,
          padding: "8px 12px",
          background: "black",
          color: "white",
          border: "1px solid #333"
        }}
      >
        Fast Forward: x{simSpeed}
      </button>

      <button
        onClick={() => setPaused(v => !v)}
        style={{
          position: "absolute",
          zIndex: 10,
          margin: 12,
          marginLeft: 278,
          padding: "8px 12px",
          background: "black",
          color: "white",
          border: "1px solid #333"
        }}
      >
        {paused ? "Resume" : "Pause"}
      </button>

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

      <div
        style={{
          position: "absolute",
          zIndex: 10,
          margin: 12,
          marginTop: 112,
          padding: "8px 12px",
          background: "rgba(0,0,0,0.78)",
          color: "white",
          border: "1px solid #333",
          fontFamily: "sans-serif",
          fontSize: 13
        }}
      >
        <div>Simulated Time: {clockLabel}</div>
        <div style={{opacity: 0.8}}>Traffic state: {trafficStateLabel(currentHour)}</div>
        <div style={{opacity: 0.75}}>Commute flow: {commuteFlowLabel(currentHour)}</div>
        <div style={{opacity: 0.75}}>Commute wave: {commuteWaveLabel(currentHour)}</div>
        <div style={{opacity: 0.8}}>
          Traffic profile: green (free flow) to red (congested)
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
        <div style={{opacity: 0.65}}>Simulation: {paused ? "Paused" : "Running"}</div>
      </div>

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
            const simCong = effectiveCongestionForEdge(object, currentHourFloat);
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