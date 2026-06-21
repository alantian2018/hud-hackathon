// car_model.js
// Procedural 3D car mesh for deck.gl SimpleMeshLayer.
//
// Builds an "Uber-style" rounded electric hatchback matching the reference
// render: a smooth white pebble-shaped body, a dark glass greenhouse/canopy,
// four black wheels with light hubs, and a glowing front light bar.
//
// Coordinate system (vehicle local, meters):
//   +x = forward (nose)      length
//   +y = left                width
//   +z = up                  height
//
// Returns a mesh in the shape SimpleMeshLayer accepts directly:
//   { attributes: { positions, normals, colors }, indices }
// where colors are float RGB in [0, 1]. Set the layer's getColor to white so
// the per-vertex paint is used unmodified.

// ---------------------------------------------------------------------------
// Small mesh accumulator
// ---------------------------------------------------------------------------
function createMeshBuilder() {
  const positions = [];
  const normals = [];
  const colors = [];
  const indices = [];

  function vertex(p, n, c) {
    const idx = positions.length / 3;
    positions.push(p[0], p[1], p[2]);
    normals.push(n[0], n[1], n[2]);
    colors.push(c[0], c[1], c[2]);
    return idx;
  }

  function pos(i) {
    return [positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]];
  }

  function nrm(i) {
    return [normals[i * 3], normals[i * 3 + 1], normals[i * 3 + 2]];
  }

  // Emit a quad wound so its front face points the same way as the vertex
  // normals (outward). Keeps the whole mesh consistently CCW so default
  // backface culling never hides a panel.
  function quadOriented(a, b, c, d) {
    const pa = pos(a), pb = pos(b), pc = pos(c);
    const e1 = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
    const e2 = [pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]];
    const g = [
      e1[1] * e2[2] - e1[2] * e2[1],
      e1[2] * e2[0] - e1[0] * e2[2],
      e1[0] * e2[1] - e1[1] * e2[0]
    ];
    const n = nrm(a);
    const dot = g[0] * n[0] + g[1] * n[1] + g[2] * n[2];
    if (dot >= 0) indices.push(a, b, c, a, c, d);
    else indices.push(a, d, c, a, c, b);
  }

  function triangleOriented(a, b, c, outward) {
    const pa = pos(a), pb = pos(b), pc = pos(c);
    const e1 = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
    const e2 = [pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]];
    const g = [
      e1[1] * e2[2] - e1[2] * e2[1],
      e1[2] * e2[0] - e1[0] * e2[2],
      e1[0] * e2[1] - e1[1] * e2[0]
    ];
    const dot = g[0] * outward[0] + g[1] * outward[1] + g[2] * outward[2];
    if (dot >= 0) indices.push(a, b, c);
    else indices.push(a, c, b);
  }

  function build() {
    // NOTE: do NOT include a top-level `vertexCount` here. luma.gl's Geometry
    // treats `vertexCount` as the number of elements to draw, which would
    // truncate the index buffer and render only part of the mesh.
    return {
      attributes: {
        positions: {value: new Float32Array(positions), size: 3},
        normals: {value: new Float32Array(normals), size: 3},
        colors: {value: new Float32Array(colors), size: 3}
      },
      indices: {value: new Uint32Array(indices), size: 1}
    };
  }

  return {vertex, quadOriented, triangleOriented, build};
}

// ---------------------------------------------------------------------------
// vector helpers
// ---------------------------------------------------------------------------
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const len3 = (x, y, z) => Math.hypot(x, y, z);

// ---------------------------------------------------------------------------
// Rounded box (signed-distance offset of a box).
//
// Take a subdivided unit cube. For each surface point, scale to the box
// half-extents, clamp into the inner box (h - r), then push back out by the
// corner radius r along the offset direction. This rounds every edge & corner
// smoothly and yields correct analytic normals -> the soft EV look.
// ---------------------------------------------------------------------------
function addRoundedBox(mb, {center, half, radius, color, seg = 18, scaleZTop = 1, rotY = 0}) {
  const [cx, cy, cz] = center;
  const [hx, hy, hz] = half;
  const r = Math.min(radius, hx, hy, hz);
  const a = [Math.max(hx - r, 0), Math.max(hy - r, 0), Math.max(hz - r, 0)];
  const cosY = Math.cos(rotY);
  const sinY = Math.sin(rotY);

  // The six faces of the unit cube, each parameterized by (u, v) in [-1, 1].
  // axis = constant axis, sign = +/-1, then the two varying axes.
  const faces = [
    {fixed: 0, sign: +1, ua: 1, va: 2},
    {fixed: 0, sign: -1, ua: 2, va: 1},
    {fixed: 1, sign: +1, ua: 2, va: 0},
    {fixed: 1, sign: -1, ua: 0, va: 2},
    {fixed: 2, sign: +1, ua: 0, va: 1},
    {fixed: 2, sign: -1, ua: 1, va: 0}
  ];

  for (const face of faces) {
    const grid = [];
    for (let i = 0; i <= seg; i++) {
      const row = [];
      for (let j = 0; j <= seg; j++) {
        const u = (i / seg) * 2 - 1;
        const v = (j / seg) * 2 - 1;
        const pu = [0, 0, 0];
        pu[face.fixed] = face.sign;
        pu[face.ua] = u;
        pu[face.va] = v;

        // Scale unit cube point to box extents.
        const P = [pu[0] * hx, pu[1] * hy, pu[2] * hz];
        // Clamp into inner box.
        const C = [clamp(P[0], -a[0], a[0]), clamp(P[1], -a[1], a[1]), clamp(P[2], -a[2], a[2])];
        // Offset direction.
        let dx = P[0] - C[0], dy = P[1] - C[1], dz = P[2] - C[2];
        let l = len3(dx, dy, dz);
        let n;
        if (l > 1e-6) {
          n = [dx / l, dy / l, dz / l];
        } else {
          n = [0, 0, 0];
          n[face.fixed] = face.sign;
        }
        let sx = C[0] + r * n[0];
        let sy = C[1] + r * n[1];
        let sz = C[2] + r * n[2];

        // Optional taper of the top (z>center) to sculpt a body that is
        // narrower up top, like a real greenhouse/hood blend.
        if (scaleZTop !== 1 && sz > 0) {
          const t = sz / hz; // 0..1
          const k = 1 - (1 - scaleZTop) * t;
          sx *= k;
          sy *= k;
        }

        // Optional rake about the Y axis (windshield / rear-glass slope).
        if (rotY !== 0) {
          const rx = sx * cosY + sz * sinY;
          const rz = -sx * sinY + sz * cosY;
          sx = rx;
          sz = rz;
          const nx = n[0] * cosY + n[2] * sinY;
          const nz = -n[0] * sinY + n[2] * cosY;
          n = [nx, n[1], nz];
        }

        row.push(mb.vertex([cx + sx, cy + sy, cz + sz], n, color));
      }
      grid.push(row);
    }
    for (let i = 0; i < seg; i++) {
      for (let j = 0; j < seg; j++) {
        mb.quadOriented(grid[i][j], grid[i][j + 1], grid[i + 1][j + 1], grid[i + 1][j]);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Cylinder along the y-axis (used for wheels & hubs).
// ---------------------------------------------------------------------------
function addCylinderY(mb, {center, radius, halfLen, color, seg = 24, capColor}) {
  const [cx, cy, cz] = center;
  const top = cy + halfLen;
  const bot = cy - halfLen;
  const ring = i => {
    const t = (i / seg) * Math.PI * 2;
    return [Math.cos(t), Math.sin(t)]; // in x-z plane
  };

  // Side wall.
  const sideTop = [];
  const sideBot = [];
  for (let i = 0; i <= seg; i++) {
    const [c, s] = ring(i);
    const n = [c, 0, s];
    sideTop.push(mb.vertex([cx + c * radius, top, cz + s * radius], n, color));
    sideBot.push(mb.vertex([cx + c * radius, bot, cz + s * radius], n, color));
  }
  for (let i = 0; i < seg; i++) {
    mb.quadOriented(sideBot[i], sideBot[i + 1], sideTop[i + 1], sideTop[i]);
  }

  // End caps (outer = +y face gets hub color for a wheel-rim hint).
  const cc = capColor || color;
  for (const [yPlane, ny, col] of [[top, 1, cc], [bot, -1, color]]) {
    const centerIdx = mb.vertex([cx, yPlane, cz], [0, ny, 0], col);
    const rim = [];
    for (let i = 0; i <= seg; i++) {
      const [c, s] = ring(i);
      rim.push(mb.vertex([cx + c * radius, yPlane, cz + s * radius], [0, ny, 0], col));
    }
    for (let i = 0; i < seg; i++) {
      const a = rim[i], b = rim[i + 1];
      mb.triangleOriented(centerIdx, a, b, [0, ny, 0]);
    }
  }
}

// ---------------------------------------------------------------------------
// Assemble the full car.
// ---------------------------------------------------------------------------
export function buildCarMesh(opts = {}) {
  const paint = opts.paint || [0.94, 0.94, 0.95]; // body white
  const tire = opts.tire || [0.05, 0.05, 0.06]; // black tire
  const hub = opts.hub || [0.55, 0.56, 0.6]; // light rim/hub

  const mb = createMeshBuilder();

  // --- Lower body: a long, low sedan volume (three-box silhouette). Defined,
  // not blobby: modest rounding, low roofline, with a hood ahead of the cabin
  // and a trunk behind it. The fender bulge still sits above the wheel centers
  // so the tires tuck under the arches.
  addRoundedBox(mb, {
    center: [0, 0, 0.62],
    half: [2.2, 0.92, 0.42],
    radius: 0.3,
    color: paint,
    seg: 30,
    scaleZTop: 0.94
  });

  // --- Cabin/greenhouse SHELL (body colour): a shorter box set back so a long
  // hood reads in front and a trunk behind. Its base is sunk deep inside the
  // lower body and its sides nearly match the body width, so the greenhouse
  // grows smoothly out of the body instead of sitting on it as a separate block.
  // (Windows intentionally omitted for now.)
  const cabinZ = 1.16;
  addRoundedBox(mb, {
    center: [-0.12, 0, cabinZ],
    half: [1.02, 0.85, 0.46],
    radius: 0.32,
    color: paint,
    seg: 28,
    scaleZTop: 0.7
  });

  // --- Side mirrors: small housing on a short stalk at the A-pillar base.
  for (const sy of [+1, -1]) {
    addRoundedBox(mb, {
      center: [0.5, sy * 1.0, 0.96],
      half: [0.11, 0.1, 0.07],
      radius: 0.05,
      color: paint,
      seg: 8
    });
  }

  // --- Wheels (4) + hubs.
  // wheelR + wz keeps the tire bottom on the ground (z=0) with the body
  // (bottom ~0.2) clearing it; the tire top tucks under the fender bulge.
  const wheelR = 0.4;
  const wheelW = 0.14;
  const wx = 1.46; // front/rear axle position
  const wyOuter = 0.86; // outer face, inboard of the 0.92 fender so it overhangs
  const wz = 0.4;
  for (const sx of [+1, -1]) {
    for (const sy of [+1, -1]) {
      addCylinderY(mb, {
        center: [sx * wx, sy * (wyOuter - wheelW), wz],
        radius: wheelR,
        halfLen: wheelW,
        color: tire,
        seg: 28,
        capColor: hub
      });
    }
  }

  return mb.build();
}

// Glowing light strips (front + rear), returned as a SEPARATE mesh so they can
// be drawn by an *unlit* layer (material: false) and appear to emit light,
// matching the reference's bright headlight signature.
export function buildCarLights(opts = {}) {
  const head = opts.head || [1.0, 0.98, 0.9]; // warm white headlight
  const tail = opts.tail || [0.95, 0.15, 0.12]; // red tail light
  const mb = createMeshBuilder();

  // Front light bar across the nose.
  addRoundedBox(mb, {
    center: [2.18, 0, 0.6],
    half: [0.05, 0.66, 0.06],
    radius: 0.05,
    color: head,
    seg: 8
  });

  // Rear light bar across the tail.
  addRoundedBox(mb, {
    center: [-2.18, 0, 0.66],
    half: [0.05, 0.64, 0.055],
    radius: 0.045,
    color: tail,
    seg: 8
  });

  return mb.build();
}

// Bounding info, useful for sizing on a map.
export const CAR_LENGTH_M = 4.4;
export const CAR_WIDTH_M = 1.84;
export const CAR_HEIGHT_M = 1.55;
