import React, {useEffect, useMemo, useState} from "react";
import {createRoot} from "react-dom/client";
import DeckGL from "@deck.gl/react";
import {GeoJsonLayer, ScatterplotLayer} from "@deck.gl/layers";
import {StaticMap} from "react-map-gl";
import "mapbox-gl/dist/mapbox-gl.css";

const MAPTILER_KEY = "4WonZ3glTzG3MWfQd6gQ";
const DAY_MINUTES = 24 * 60;
const SPEEDS = [1, 4, 12, 30, 60];
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
  label: "RL Orchestrator",
  route: [168, 85, 247, 245],
  casing: [245, 243, 255, 230],
  reposition: [20, 184, 166, 230],
  active: [168, 85, 247, 245],
  idle: [203, 213, 225, 220],
  panel: "rgba(12,8,22,0.88)",
  accent: "#c4b5fd"
};
const MAX_VISIBLE_ROUTE_SEGMENT_METERS = 650;

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

function scenarioSnapshots(world, scenarioId) {
  if (!world) return [];
  if (scenarioId && world.event_scenarios?.[scenarioId]?.snapshots?.length) {
    return world.event_scenarios[scenarioId].snapshots;
  }
  return world.snapshots ?? [];
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
  const elapsed = Number(car?.route_elapsed ?? job.route_elapsed ?? 0) + elapsedSeconds(snapshot, clockMinute, stepMinutes);
  return clamp(elapsed / routeCost, 0, 1);
}

function animatedCars(snapshot, routeIndex, clockMinute, stepMinutes) {
  const cars = snapshot?.map_dispatch?.cars ?? [];
  const jobByCar = new Map(activeJobs(snapshot).map(job => [job.car_id, job]));
  return cars.map(car => {
    const job = jobByCar.get(car.id);
    const route = resolveRoute(job?.route, routeIndex);
    const coordinates = route?.coordinates ?? [];
    const progress = job && routeCoordinatesAreUsable(coordinates)
      ? jobProgress(job, car, snapshot, clockMinute, stepMinutes)
      : 0;
    return {
      ...car,
      active_job_kind: job?.kind,
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

function makePeopleFeatures(snapshot) {
  const features = [];
  for (const person of snapshot?.map_people ?? []) {
    if (person.pickup_position) {
      features.push({
        type: "Feature",
        geometry: {type: "Point", coordinates: person.pickup_position},
        properties: {kind: "pickup", person_id: person.id, status: person.status}
      });
    }
    if (person.dropoff_position) {
      features.push({
        type: "Feature",
        geometry: {type: "Point", coordinates: person.dropoff_position},
        properties: {kind: "dropoff", person_id: person.id, status: person.status}
      });
    }
  }
  return {type: "FeatureCollection", features};
}

function trafficColor(value) {
  if (value < 1.35) return [52, 168, 83, 185];
  if (value < 1.9) return [251, 188, 4, 190];
  if (value < 2.6) return [251, 140, 0, 205];
  return [234, 67, 53, 220];
}

function metricsFor(snapshot, policyId) {
  const raw = policyId === "greedy" ? snapshot?.map_greedy_stats : snapshot?.map_orchestrator_stats;
  return raw ?? {};
}

function metricDelta(rl, greedy, key, lowerIsBetter = false) {
  const a = Number(rl?.[key] ?? 0);
  const b = Number(greedy?.[key] ?? 0);
  const delta = a - b;
  const good = lowerIsBetter ? delta < 0 : delta > 0;
  return {delta, good};
}

function MapPanel({policy, world, network, snapshot, routeIndex, clockMinute, stepMinutes, viewState, setViewState, single}) {
  const cars = useMemo(
    () => animatedCars(snapshot, routeIndex, clockMinute, stepMinutes),
    [snapshot, routeIndex, clockMinute, stepMinutes]
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
          lineWidthMinPixels: 0.65,
          lineWidthMaxPixels: 4,
          getLineWidth: f => {
            const base = f.properties?.hourly_congestion_factor?.[currentHour] ?? 1;
            return 0.65 + clamp((base - 1) / 3, 0, 1) * 2.8;
          },
          getLineColor: f => trafficColor(f.properties?.hourly_congestion_factor?.[currentHour] ?? 1),
          pickable: false
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
      data: makePeopleFeatures(snapshot),
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
    return [roads, routeCasing, routes, people, carLayer].filter(Boolean);
  }, [network, policy, snapshot, routeIndex, clockMinute, stepMinutes, cars, currentHour]);

  return (
    <section style={{position: "relative", minHeight: single ? "100vh" : "calc(100vh - 82px)", overflow: "hidden"}}>
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
          <Metric label="Revenue" value={formatMoney(metrics.revenue)} />
          <Metric label="Demand" value={formatPct(metrics.demand_served_pct)} />
          <Metric label="Wait" value={`${Number(metrics.wait_time_min ?? 0).toFixed(1)}m`} />
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

function DeltaBar({greedySnapshot, rlSnapshot}) {
  const greedy = metricsFor(greedySnapshot, "greedy");
  const rl = metricsFor(rlSnapshot, "rl");
  const trips = metricDelta(rl, greedy, "completed_trips");
  const revenue = metricDelta(rl, greedy, "revenue");
  const wait = metricDelta(rl, greedy, "wait_time_min", true);
  return (
    <div style={{display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap", fontSize: 13}}>
      <Delta label="Trips" value={trips.delta} good={trips.good} />
      <Delta label="Revenue" value={revenue.delta} good={revenue.good} money />
      <Delta label="Wait" value={wait.delta} good={wait.good} suffix="m" />
    </div>
  );
}

function Delta({label, value, good, money, suffix = ""}) {
  const sign = value > 0 ? "+" : "";
  const shown = money ? `${sign}${formatMoney(value).replace("$-", "-$")}` : `${sign}${value.toFixed(1)}${suffix}`;
  return (
    <span style={{
      display: "inline-flex",
      gap: 6,
      alignItems: "center",
      padding: "6px 8px",
      borderRadius: 999,
      background: good ? "rgba(20,184,166,0.16)" : "rgba(248,113,113,0.15)",
      color: good ? "#99f6e4" : "#fecaca",
      border: good ? "1px solid rgba(45,212,191,0.32)" : "1px solid rgba(248,113,113,0.28)"
    }}>
      <strong>{label}</strong> {shown}
    </span>
  );
}

function useComparisonData() {
  const [state, setState] = useState({status: "loading"});
  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const [greedyWorld, rlWorld, network] = await Promise.all([
          fetchJson(["/data/mobility_world.json"]),
          fetchJson(["/data/mobility_orchestrator_world.json"]),
          fetchJson(["/data/osmnx_edges.geojson", "/dist/data/osmnx_edges.geojson"])
        ]);
        if (!cancelled) setState({status: "ready", greedyWorld, rlWorld, network});
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

function ControlBar({running, onStart, onPause, onReset, speed, setSpeed, clockLabel, greedySnapshot, rlSnapshot, compare}) {
  return (
    <header style={{
      height: compare ? 82 : 76,
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
        <div style={{fontSize: 12, opacity: 0.66}}>{compare ? "Synchronized comparison" : "RL policy page"}</div>
        <div style={{fontSize: 20, fontWeight: 850}}>{compare ? "Greedy vs RL Fleet Dispatch" : "RL Fleet Orchestrator"}</div>
        {compare && <DeltaBar greedySnapshot={greedySnapshot} rlSnapshot={rlSnapshot} />}
      </div>
      <div style={{display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap", justifyContent: "flex-end"}}>
        <a href="/" style={navButtonStyle}>Greedy Page</a>
        <a href="/rl.html" style={navButtonStyle}>RL Page</a>
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

function ComparisonShell({compare}) {
  const data = useComparisonData();
  const [clockMinute, setClockMinute] = useState(0);
  const [running, setRunning] = useState(false);
  const [speed, setSpeed] = useState(12);
  const [viewState, setViewState] = useState(INITIAL_VIEW_STATE);

  const greedySnapshots = useMemo(
    () => scenarioSnapshots(data.greedyWorld, null),
    [data.greedyWorld]
  );
  const rlSnapshots = useMemo(
    () => scenarioSnapshots(data.rlWorld, null),
    [data.rlWorld]
  );
  const greedyStep = scenarioStepMinutes(data.greedyWorld, null);
  const rlStep = scenarioStepMinutes(data.rlWorld, null);
  const endMinute = useMemo(() => {
    const greedyEnd = greedySnapshots.at(-1)?.timestep ?? DAY_MINUTES;
    const rlEnd = rlSnapshots.at(-1)?.timestep ?? DAY_MINUTES;
    return Math.min(greedyEnd, rlEnd) + Math.min(greedyStep, rlStep) - 0.001;
  }, [greedySnapshots, rlSnapshots, greedyStep, rlStep]);

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

  if (data.status !== "ready") {
    return (
      <div style={{height: "100vh", display: "grid", placeItems: "center", background: "#04080e", color: "white", fontFamily: "system-ui, sans-serif"}}>
        {data.status === "error" ? "Could not load comparison data." : "Loading comparison data..."}
      </div>
    );
  }

  const greedySnapshot = snapshotAt(greedySnapshots, clockMinute);
  const rlSnapshot = snapshotAt(rlSnapshots, clockMinute);
  const onStart = () => {
    if (clockMinute >= endMinute - 0.01) setClockMinute(0);
    setRunning(true);
  };
  const onReset = () => {
    setClockMinute(0);
    setRunning(false);
  };

  return (
    <main style={{width: "100vw", height: "100vh", overflow: "hidden", background: "#04080e"}}>
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
      />
      {compare ? (
        <div style={{display: "grid", gridTemplateColumns: "1fr 1fr", height: "calc(100vh - 82px)"}}>
          <MapPanel
            policy={GREEDY}
            world={data.greedyWorld}
            network={data.network}
            snapshot={greedySnapshot}
            routeIndex={data.greedyWorld.routes}
            clockMinute={clockMinute}
            stepMinutes={greedyStep}
            viewState={viewState}
            setViewState={setViewState}
          />
          <MapPanel
            policy={RL}
            world={data.rlWorld}
            network={data.network}
            snapshot={rlSnapshot}
            routeIndex={data.rlWorld.routes}
            clockMinute={clockMinute}
            stepMinutes={rlStep}
            viewState={viewState}
            setViewState={setViewState}
          />
        </div>
      ) : (
        <MapPanel
          policy={RL}
          world={data.rlWorld}
          network={data.network}
          snapshot={rlSnapshot}
          routeIndex={data.rlWorld.routes}
          clockMinute={clockMinute}
          stepMinutes={rlStep}
          viewState={viewState}
          setViewState={setViewState}
          single
        />
      )}
    </main>
  );
}

document.body.style.margin = 0;
const root = document.createElement("div");
document.body.appendChild(root);
createRoot(root).render(<ComparisonShell compare={!window.location.pathname.includes("rl.html")} />);
