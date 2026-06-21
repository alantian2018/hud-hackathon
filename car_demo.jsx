// car_demo.jsx
// Standalone deck.gl studio shot of the procedural 3D car (car_model.js).
// Used to render & verify the model looks like the reference before wiring it
// into the map app. Open /car_demo.html with `npm run dev`.

import React, {useMemo, useState, useEffect} from "react";
import {createRoot} from "react-dom/client";
import DeckGL from "@deck.gl/react";
import {OrbitView, COORDINATE_SYSTEM, LightingEffect, AmbientLight, DirectionalLight} from "@deck.gl/core";
import {SimpleMeshLayer} from "@deck.gl/mesh-layers";
import {buildCarMesh} from "./car_model.js";
import {createCarLayers} from "./car_layers.js";

const lightingEffect = new LightingEffect({
  ambient: new AmbientLight({color: [255, 255, 255], intensity: 0.85}),
  key: new DirectionalLight({color: [255, 255, 255], intensity: 1.5, direction: [-0.8, -0.5, -1.6]}),
  fill: new DirectionalLight({color: [210, 220, 235], intensity: 0.6, direction: [1.2, 0.9, -0.5]})
});

// Flat disc mesh used as a soft contact shadow under the car.
function discMesh(radius = 3.2, seg = 48) {
  const positions = [0, 0, 0];
  const normals = [0, 0, 1];
  const colors = [0.2, 0.2, 0.22];
  const indices = [];
  for (let i = 0; i <= seg; i++) {
    const t = (i / seg) * Math.PI * 2;
    positions.push(Math.cos(t) * radius, Math.sin(t) * radius, 0);
    normals.push(0, 0, 1);
    colors.push(0.2, 0.2, 0.22);
  }
  for (let i = 1; i <= seg; i++) indices.push(0, i, i + 1);
  return {
    attributes: {
      positions: {value: new Float32Array(positions), size: 3},
      normals: {value: new Float32Array(normals), size: 3},
      colors: {value: new Float32Array(colors), size: 3}
    },
    indices: {value: new Uint32Array(indices), size: 1}
  };
}

function App() {
  const car = useMemo(() => buildCarMesh(), []);
  const shadow = useMemo(() => discMesh(), []);
  const q = new URLSearchParams(window.location.search);
  const elev = q.has("elev") ? Number(q.get("elev")) : 28;
  const zoom0 = q.has("zoom") ? Number(q.get("zoom")) : 6.6;
  const [spin, setSpin] = useState(false);
  const [orbit, setOrbit] = useState(q.has("orbit") ? Number(q.get("orbit")) : -52);

  useEffect(() => {
    if (!spin) return;
    let raf;
    const tick = () => {
      setOrbit(o => (o + 0.4) % 360);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [spin]);

  const viewState = {
    target: [0, 0, 0.6],
    rotationX: elev, // elevation (look down)
    rotationOrbit: orbit, // azimuth around +z
    zoom: zoom0,
    minZoom: 3,
    maxZoom: 10
  };

  // Fleet mode (?fleet=1): exercise car_layers.js + heading orientation by
  // placing cars on a grid, each pointing a different direction.
  const fleet = q.get("fleet") === "1";
  if (fleet) {
    const cars = [];
    const colors = [[245, 246, 248], [120, 180, 255], [255, 196, 92]];
    let i = 0;
    for (let gx = -1; gx <= 1; gx++) {
      for (let gy = -1; gy <= 1; gy++) {
        cars.push({position: [gx * 7, gy * 7, 0], yaw: i * 40, color: colors[i % 3]});
        i++;
      }
    }
    const fleetLayers = createCarLayers({
      id: "fleet",
      data: cars,
      getPosition: d => d.position,
      getYaw: d => d.yaw,
      getColor: d => d.color,
      sizeScale: 1,
      coordinateSystem: COORDINATE_SYSTEM.CARTESIAN
    });
    return (
      <div style={{position: "absolute", inset: 0, background: "radial-gradient(circle at 50% 34%, #aeb4c0 0%, #6f7682 85%)"}}>
        <DeckGL
          views={new OrbitView({orbitAxis: "Z", fovy: 40})}
          viewState={{target: [0, 0, 0], rotationX: 62, rotationOrbit: 0, zoom: 4.4}}
          controller
          layers={fleetLayers}
          effects={[lightingEffect]}
          style={{background: "transparent"}}
        />
      </div>
    );
  }

  const layers = [
    new SimpleMeshLayer({
      id: "ground-shadow",
      data: [{position: [0, 0, 0.01]}],
      mesh: shadow,
      coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
      getPosition: d => d.position,
      getColor: [120, 120, 130, 90],
      getOrientation: [0, 0, 0],
      sizeScale: 1,
      material: false,
      opacity: 0.35,
      parameters: {depthTest: false}
    }),
    new SimpleMeshLayer({
      id: "car",
      data: [{position: [0, 0, 0]}],
      mesh: car,
      coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
      getPosition: d => d.position,
      getOrientation: [0, 0, 0],
      getColor: [255, 255, 255],
      sizeScale: 1,
      material: {ambient: 0.55, diffuse: 0.8, shininess: 28, specularColor: [60, 60, 70]}
    })
  ];

  return (
    <div style={{position: "absolute", inset: 0, background: "radial-gradient(circle at 50% 34%, #aeb4c0 0%, #6f7682 85%)"}}>
      <DeckGL
        views={new OrbitView({orbitAxis: "Z", fovy: 38})}
        viewState={viewState}
        onViewStateChange={({viewState: v}) => setOrbit(v.rotationOrbit)}
        controller={{inertia: true}}
        layers={layers}
        effects={[lightingEffect]}
        style={{background: "transparent"}}
      />
      <div style={{position: "absolute", top: 14, left: 16, fontFamily: "system-ui, sans-serif", color: "#333"}}>
        <div style={{fontSize: 18, fontWeight: 700}}>deck.gl 3D car</div>
        <div style={{fontSize: 12, opacity: 0.7}}>drag to orbit · scroll to zoom</div>
        <button
          onClick={() => setSpin(s => !s)}
          style={{marginTop: 8, fontSize: 12, padding: "4px 10px", cursor: "pointer", borderRadius: 6, border: "1px solid #bbb", background: "#fff"}}
        >
          {spin ? "stop" : "auto-spin"}
        </button>
      </div>
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);
