// car_layers.js
// Reusable deck.gl "sprite" for the 3D car (car_model.js): an Uber-style
// vehicle marker you can drop on a map in place of a flat ScatterplotLayer.
//
// Usage:
//   import {createCarLayers} from "./car_layers.js";
//   ...layers = [...createCarLayers({
//     data: cars,                       // [{position:[lng,lat], heading, status}]
//     getPosition: d => d.position,
//     getYaw: d => 90 - (d.bearingDeg ?? 0), // see note below
//     getColor: d => d.status === "idle" ? [120,180,255] : [255,255,255],
//     sizeScale: 6                      // meters-per-unit; bump up so cars read on a city map
//   })]
//
// Heading note: the model's nose points along +x. getOrientation's `yaw`
// rotates the nose CCW in the ground plane (0 deg => nose points +x / east).
// If your data stores a compass bearing (deg clockwise from north / +y), use
// yaw = 90 - bearingDeg.

import {SimpleMeshLayer} from "@deck.gl/mesh-layers";
import {buildCarMesh} from "./car_model.js";

// Build the mesh once and reuse across layers/updates.
let _carMesh = null;
function carMesh() {
  if (!_carMesh) _carMesh = buildCarMesh();
  return _carMesh;
}

export function createCarLayers(opts = {}) {
  const {
    id = "car",
    data = [],
    getPosition = d => d.position,
    getYaw = () => 0,
    getColor = () => [245, 246, 248],
    sizeScale = 6,
    coordinateSystem, // pass COORDINATE_SYSTEM.* if not using default lng/lat
    pickable = true,
    parameters,
    updateTriggers
  } = opts;

  // Snap heading to a fixed set of orientations (default every 15deg) and bucket
  // the cars, so we render one mesh layer per angle with a CONSTANT orientation
  // instead of computing a unique rotation matrix per car.
  const step = opts.angleStep ?? 15;
  const yawOf = d => (typeof getYaw === "function" ? getYaw(d) : getYaw);
  const buckets = new Map(); // snappedYaw -> rows[]
  for (const d of data) {
    const snapped = (((Math.round(yawOf(d) / step) * step) % 360) + 360) % 360;
    let rows = buckets.get(snapped);
    if (!rows) buckets.set(snapped, (rows = []));
    rows.push(d);
  }

  const layers = [];
  for (const [yaw, rows] of buckets) {
    layers.push(
      new SimpleMeshLayer({
        id: `${id}-body-${yaw}`,
        data: rows,
        mesh: carMesh(),
        getPosition,
        getOrientation: [0, yaw, 0], // constant per bucket
        getColor,
        sizeScale,
        pickable,
        material: {ambient: 0.55, diffuse: 0.8, shininess: 28, specularColor: [60, 60, 70]},
        ...(coordinateSystem !== undefined ? {coordinateSystem} : {}),
        ...(parameters ? {parameters} : {}),
        updateTriggers
      })
    );
  }
  return layers;
}
