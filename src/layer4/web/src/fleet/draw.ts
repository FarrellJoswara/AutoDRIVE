import type { FleetCar, FleetTelemetry } from "../api";
import {
  LIDAR_RANGE_MAX_M,
  LIDAR_RANGE_MIN_M,
  beamWorldAngleRad,
  lidarNormToMetres,
} from "./lidarCalibration";
import type { LoadedMap } from "./mapLoader";
import { mapWorldBounds } from "./mapLoader";

/**
 * Axis mapping (documented once):
 *   world x  → canvas +x
 *   world z  → canvas −y  (Unity Y-up; screen Y-down)
 *   screen car angle = −yaw
 */

export interface ViewState {
  showMap: boolean;
  showFleet: boolean;
  showLidar: boolean;
  selectedEnvId: number;
  /** Grow-only pose bounds when no occupancy map */
  fitMinX: number;
  fitMaxX: number;
  fitMinZ: number;
  fitMaxZ: number;
}

export interface DrawFrameOpts {
  map: LoadedMap | null;
  fleet: FleetTelemetry | null;
  stale: boolean;
  view: ViewState;
  cssW: number;
  cssH: number;
  dpr: number;
}

const PAD = 24;
const CAR_R = 5;

function expandFit(view: ViewState, cars: FleetCar[]): void {
  for (const c of cars) {
    if (!c.pose) continue;
    const [x, z] = c.pose;
    view.fitMinX = Math.min(view.fitMinX, x - 2);
    view.fitMaxX = Math.max(view.fitMaxX, x + 2);
    view.fitMinZ = Math.min(view.fitMinZ, z - 2);
    view.fitMaxZ = Math.max(view.fitMaxZ, z + 2);
  }
}

/** World metres → CSS pixel using setTransform-friendly scale/offset. */
export function worldToScreenTransform(
  bounds: { minX: number; maxX: number; minZ: number; maxZ: number },
  cssW: number,
  cssH: number
): { scale: number; ox: number; oy: number } {
  const w = Math.max(1e-3, bounds.maxX - bounds.minX);
  const h = Math.max(1e-3, bounds.maxZ - bounds.minZ);
  const scale = Math.min((cssW - PAD * 2) / w, (cssH - PAD * 2) / h);
  // Center map in canvas; z maps to −y so origin sits accordingly
  const ox = PAD + ((cssW - PAD * 2) - w * scale) / 2 - bounds.minX * scale;
  const oy = PAD + ((cssH - PAD * 2) - h * scale) / 2 + bounds.maxZ * scale;
  return { scale, ox, oy };
}

export function drawStaticMap(
  ctx: CanvasRenderingContext2D,
  map: LoadedMap | null,
  showMap: boolean,
  cssW: number,
  cssH: number,
  dpr: number,
  bounds: { minX: number; maxX: number; minZ: number; maxZ: number }
): void {
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssW, cssH);
  ctx.fillStyle = "#0a100e";
  ctx.fillRect(0, 0, cssW, cssH);

  const { scale, ox, oy } = worldToScreenTransform(bounds, cssW, cssH);

  if (showMap && map) {
    // Draw occupancy: image lower-left = origin in world; flip Y via transform
    const res = map.yaml.resolution;
    const [oxW, ozW] = map.yaml.origin;
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.translate(ox, oy);
    ctx.scale(scale, -scale);
    // ROS: image row 0 = top = origin_z + height*res; flip via negative height
    ctx.drawImage(
      map.image,
      oxW,
      ozW + map.height * res,
      map.width * res,
      -map.height * res
    );
    ctx.restore();
  } else {
    // Grid fallback
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.strokeStyle = "rgba(42, 61, 52, 0.7)";
    ctx.lineWidth = 1;
    const step = niceGridStep(bounds.maxX - bounds.minX, bounds.maxZ - bounds.minZ);
    for (let x = Math.ceil(bounds.minX / step) * step; x <= bounds.maxX; x += step) {
      const sx = ox + x * scale;
      ctx.beginPath();
      ctx.moveTo(sx, 0);
      ctx.lineTo(sx, cssH);
      ctx.stroke();
    }
    for (let z = Math.ceil(bounds.minZ / step) * step; z <= bounds.maxZ; z += step) {
      const sy = oy - z * scale;
      ctx.beginPath();
      ctx.moveTo(0, sy);
      ctx.lineTo(cssW, sy);
      ctx.stroke();
    }
    ctx.restore();
  }

  // Scale bar
  const barM = niceGridStep(bounds.maxX - bounds.minX, bounds.maxZ - bounds.minZ);
  const barPx = barM * scale;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = "rgba(230, 240, 234, 0.85)";
  ctx.font = "11px Cascadia Code, Consolas, monospace";
  const bx = cssW - PAD - barPx;
  const by = cssH - 14;
  ctx.fillRect(bx, by - 3, barPx, 3);
  ctx.fillText(`${barM} m`, bx, by - 8);
}

function niceGridStep(w: number, h: number): number {
  const span = Math.max(w, h, 1);
  const raw = span / 8;
  const pow = Math.pow(10, Math.floor(Math.log10(raw)));
  const n = raw / pow;
  if (n < 1.5) return pow;
  if (n < 3.5) return 2 * pow;
  if (n < 7.5) return 5 * pow;
  return 10 * pow;
}

export function drawDynamic(
  ctx: CanvasRenderingContext2D,
  opts: DrawFrameOpts
): void {
  const { map, fleet, stale, view, cssW, cssH, dpr } = opts;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssW, cssH);

  if (!fleet || !fleet.cars.length) {
    ctx.fillStyle = "rgba(138, 163, 150, 0.8)";
    ctx.font = "13px Segoe UI, sans-serif";
    ctx.fillText("Waiting for fleet telemetry…", 16, 28);
    return;
  }

  if (!map) {
    expandFit(view, fleet.cars);
  }

  const bounds = map
    ? mapWorldBounds(map)
    : {
        minX: view.fitMinX,
        maxX: view.fitMaxX,
        minZ: view.fitMinZ,
        maxZ: view.fitMaxZ,
      };

  const { scale, ox, oy } = worldToScreenTransform(bounds, cssW, cssH);
  const rMin = fleet.lidar_range_min ?? LIDAR_RANGE_MIN_M;
  const rMax = fleet.lidar_range_max ?? LIDAR_RANGE_MAX_M;

  // LiDAR under cars
  if (view.showLidar) {
    const car =
      fleet.cars.find((c) => c.env_id === view.selectedEnvId) ?? fleet.cars[0];
    if (car?.pose && car.lidar && car.lidar.length && car.yaw != null) {
      drawLidarPolygon(ctx, car, scale, ox, oy, rMin, rMax, dpr);
    }
  }

  if (view.showFleet) {
    for (const car of fleet.cars) {
      if (!car.pose) continue;
      drawCar(ctx, car, scale, ox, oy, car.env_id === view.selectedEnvId, dpr);
    }
  }

  if (stale) {
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = "rgba(230, 184, 77, 0.95)";
    ctx.font = "11px Cascadia Code, Consolas, monospace";
    ctx.fillText("stale (PPO gap / no recent sample)", 12, 18);
  }
}

function drawLidarPolygon(
  ctx: CanvasRenderingContext2D,
  car: FleetCar,
  scale: number,
  ox: number,
  oy: number,
  rMin: number,
  rMax: number,
  dpr: number
): void {
  const pose = car.pose!;
  const yaw = car.yaw!;
  const beams = car.lidar!;
  const n = beams.length;
  const [wx, wz] = pose;
  const sx0 = ox + wx * scale;
  const sy0 = oy - wz * scale;

  const path = new Path2D();
  path.moveTo(sx0, sy0);
  for (let i = 0; i < n; i++) {
    const ang = beamWorldAngleRad(yaw, i, n);
    const dist = lidarNormToMetres(beams[i], rMin, rMax);
    const hx = wx + Math.cos(ang) * dist;
    const hz = wz + Math.sin(ang) * dist;
    path.lineTo(ox + hx * scale, oy - hz * scale);
  }
  path.closePath();

  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = "rgba(62, 207, 142, 0.22)";
  ctx.strokeStyle = "rgba(62, 207, 142, 0.45)";
  ctx.lineWidth = 1;
  ctx.fill(path);
  ctx.stroke(path);
}

function drawCar(
  ctx: CanvasRenderingContext2D,
  car: FleetCar,
  scale: number,
  ox: number,
  oy: number,
  selected: boolean,
  dpr: number
): void {
  const [wx, wz] = car.pose!;
  const sx = ox + wx * scale;
  const sy = oy - wz * scale;
  const yaw = car.yaw ?? 0;

  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.translate(sx, sy);
  // screen angle = −yaw
  ctx.rotate(-yaw);

  if (car.collision) {
    ctx.strokeStyle = "#e85d5d";
    ctx.lineWidth = selected ? 2.5 : 2;
    ctx.beginPath();
    ctx.moveTo(-CAR_R, -CAR_R);
    ctx.lineTo(CAR_R, CAR_R);
    ctx.moveTo(CAR_R, -CAR_R);
    ctx.lineTo(-CAR_R, CAR_R);
    ctx.stroke();
  } else {
    ctx.fillStyle = selected ? "#3ecf8e" : "#8aa396";
    ctx.strokeStyle = selected ? "#e6f0ea" : "#2a3d34";
    ctx.lineWidth = 1.5;
    // Chevron pointing +x (forward in car frame after −yaw rotate)
    ctx.beginPath();
    ctx.moveTo(CAR_R + 2, 0);
    ctx.lineTo(-CAR_R, CAR_R * 0.85);
    ctx.lineTo(-CAR_R * 0.4, 0);
    ctx.lineTo(-CAR_R, -CAR_R * 0.85);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
  }

  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = "rgba(230, 240, 234, 0.75)";
  ctx.font = "10px Cascadia Code, Consolas, monospace";
  ctx.fillText(String(car.env_id), sx + CAR_R + 3, sy - CAR_R);
}

export function resolveBounds(
  map: LoadedMap | null,
  view: ViewState,
  fleet: FleetTelemetry | null
): { minX: number; maxX: number; minZ: number; maxZ: number } {
  if (map) return mapWorldBounds(map);
  if (fleet) expandFit(view, fleet.cars);
  if (
    !Number.isFinite(view.fitMinX) ||
    view.fitMaxX <= view.fitMinX ||
    view.fitMaxZ <= view.fitMinZ
  ) {
    return { minX: -15, maxX: 15, minZ: -15, maxZ: 15 };
  }
  return {
    minX: view.fitMinX,
    maxX: view.fitMaxX,
    minZ: view.fitMinZ,
    maxZ: view.fitMaxZ,
  };
}

export function initialViewState(): ViewState {
  return {
    showMap: true,
    showFleet: true,
    showLidar: true,
    selectedEnvId: 0,
    fitMinX: Infinity,
    fitMaxX: -Infinity,
    fitMinZ: Infinity,
    fitMaxZ: -Infinity,
  };
}
