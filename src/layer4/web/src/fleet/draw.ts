import type { EvaluatorCarTelemetry, FleetCar, FleetTelemetry } from "../api";
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
 *
 * Fleet poses are map metres (Bridge→Unity swizzle in layer1 telemetry;
 * hub publishes [x,z]). Canvas does not remap.
 */

export interface ViewState {
  showMap: boolean;
  showFleet: boolean;
  showFrontier: boolean;
  showCurrentProgress: boolean;
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
  evaluatorCar?: EvaluatorCarTelemetry | null;
  staleLabel: string | null;
  ppoProgress?: { label: string; progress: number; detail: string } | null;
  view: ViewState;
  cssW: number;
  cssH: number;
  dpr: number;
}

const PAD = 24;
/** Half-length of the car glyph in metres (F1TENTH / RoboRacer ≈ 0.5 m long). */
const CAR_HALF_LEN_M = 0.22;
const CAR_R_PX_MIN = 5;
const CAR_R_PX_MAX = 10;

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
    // Place occupancy in the same CSS-pixel frame as cars/LiDAR.
    // Avoid drawImage(..., negative height) — ImageBitmap + neg dest is flaky
    // and was shifting the underlay off the fleet glyphs.
    const res = map.yaml.resolution;
    const [oxW, ozW] = map.yaml.origin;
    const worldW = map.width * res;
    const worldH = map.height * res;
    const x0 = ox + oxW * scale;
    const y0 = oy - (ozW + worldH) * scale;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.drawImage(map.image, x0, y0, worldW * scale, worldH * scale);
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
  opts: DrawFrameOpts,
  /** Must match the bounds passed to drawStaticMap in the same frame. */
  bounds: { minX: number; maxX: number; minZ: number; maxZ: number }
): void {
  const { fleet, evaluatorCar, staleLabel, ppoProgress, view, cssW, cssH, dpr } = opts;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssW, cssH);

  if ((!fleet || !fleet.cars.length) && !evaluatorCar) {
    ctx.fillStyle = "rgba(138, 163, 150, 0.8)";
    ctx.font = "13px Segoe UI, sans-serif";
    ctx.fillText("Waiting for fleet telemetry…", 16, 28);
    return;
  }

  const { scale, ox, oy } = worldToScreenTransform(bounds, cssW, cssH);
  const rMin = fleet?.lidar_range_min ?? LIDAR_RANGE_MIN_M;
  const rMax = fleet?.lidar_range_max ?? LIDAR_RANGE_MAX_M;
  const gate = fleet?.lap_gate;
  if (view.showFleet && gate && gate.length === 2) {
    const x1 = ox + gate[0][0] * scale;
    const y1 = oy - gate[0][1] * scale;
    const x2 = ox + gate[1][0] * scale;
    const y2 = oy - gate[1][1] * scale;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.strokeStyle = "rgba(116, 218, 255, 0.95)";
    ctx.lineWidth = 3;
    ctx.setLineDash([5, 3]);
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#74daff";
    for (const [x, y] of [[x1, y1], [x2, y2]]) {
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // The backend owns route projection and frontier geometry; canvas only draws it.
  if (view.showFrontier) {
    for (const car of fleet?.cars ?? []) {
      const line = car.frontier_line;
      if (car.reset || !line || line.length !== 2) continue;
      ctx.globalAlpha = view.selectedEnvId >= 0 && car.env_id !== view.selectedEnvId ? 0.18 : 1;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.strokeStyle = car.env_id === view.selectedEnvId
        ? "rgba(255, 205, 86, 0.95)"
        : "rgba(255, 205, 86, 0.55)";
      ctx.lineWidth = car.env_id === view.selectedEnvId ? 3 : 1.5;
      ctx.setLineDash([7, 4]);
      ctx.beginPath();
      ctx.moveTo(ox + line[0][0] * scale, oy - line[0][1] * scale);
      ctx.lineTo(ox + line[1][0] * scale, oy - line[1][1] * scale);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
    }
  }
  // Current position bar follows the car's signed route position; unlike the
  // best-so-far frontier it moves backward when the car retreats.
  if (view.showCurrentProgress) {
    for (const car of fleet?.cars ?? []) {
      const line = car.current_progress_line;
      if (car.reset || !line || line.length !== 2) continue;
      ctx.globalAlpha = view.selectedEnvId >= 0 && car.env_id !== view.selectedEnvId ? 0.18 : 1;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.strokeStyle = car.env_id === view.selectedEnvId
        ? "rgba(62, 207, 142, 0.98)"
        : "rgba(62, 207, 142, 0.68)";
      ctx.lineWidth = car.env_id === view.selectedEnvId ? 3 : 2;
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.moveTo(ox + line[0][0] * scale, oy - line[0][1] * scale);
      ctx.lineTo(ox + line[1][0] * scale, oy - line[1][1] * scale);
      ctx.stroke();
      ctx.globalAlpha = 1;
    }
  }
  // Draw the ranges published by telemetry directly from the published pose.
  if (view.showLidar) {
    const car = view.selectedEnvId < 0
      ? undefined
      : fleet?.cars.find((c) => c.env_id === view.selectedEnvId);
    if (
      car &&
      !car.reset &&
      car.pose &&
      car.lidar &&
      car.lidar.length &&
      car.yaw != null
    ) {
      drawLidarPolygon(ctx, car, scale, ox, oy, rMin, rMax, dpr);
    }
  }

  if (view.showFleet) {
    for (const car of fleet?.cars ?? []) {
      if (car.reset || !car.pose) continue;
      drawCar(ctx, car, scale, ox, oy, view.selectedEnvId >= 0 && car.env_id !== view.selectedEnvId, dpr);
    }
    if (evaluatorCar) {
      drawEvaluatorCar(ctx, evaluatorCar, scale, ox, oy, dpr);
    }
  }

  if (ppoProgress) {
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const boxWidth = Math.min(280, Math.max(210, cssW - 24));
    const boxHeight = 48;
    ctx.fillStyle = "rgba(7, 15, 12, 0.88)";
    ctx.fillRect(10, 10, boxWidth, boxHeight);
    ctx.strokeStyle = "rgba(81, 118, 99, 0.9)";
    ctx.lineWidth = 1;
    ctx.strokeRect(10.5, 10.5, boxWidth - 1, boxHeight - 1);
    ctx.fillStyle = ppoProgress.label.startsWith("PPO updating")
      ? "rgba(116, 218, 255, 0.98)"
      : "rgba(77, 232, 170, 0.98)";
    ctx.font = "11px Cascadia Code, Consolas, monospace";
    ctx.fillText(ppoProgress.label, 18, 27);
    ctx.fillStyle = "rgba(166, 194, 177, 0.9)";
    ctx.font = "9px Cascadia Code, Consolas, monospace";
    ctx.fillText(ppoProgress.detail, 18, 40);
    const trackX = 18;
    const trackY = 47;
    const trackWidth = boxWidth - 16;
    ctx.fillStyle = "rgba(49, 68, 57, 0.95)";
    ctx.fillRect(trackX, trackY, trackWidth, 4);
    ctx.fillStyle = ppoProgress.label.startsWith("PPO updating")
      ? "rgba(116, 218, 255, 0.98)"
      : "rgba(77, 232, 170, 0.98)";
    ctx.fillRect(trackX, trackY, trackWidth * ppoProgress.progress, 4);
  } else if (staleLabel) {
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = staleLabel.startsWith("PPO updating")
      ? "rgba(116, 218, 255, 0.98)"
      : "rgba(230, 184, 77, 0.95)";
    ctx.font = "11px Cascadia Code, Consolas, monospace";
    ctx.fillText(staleLabel, 12, 18);
  }
}

function carRadiusPx(scale: number): number {
  return Math.min(
    CAR_R_PX_MAX,
    Math.max(CAR_R_PX_MIN, CAR_HALF_LEN_M * scale)
  );
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
  const maxSpan = rMax - rMin;

  // Clip each beam to occupancy walls so rays cannot paint through the underlay.
  const outline = new Path2D();
  outline.moveTo(sx0, sy0);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.lineWidth = 1;
  for (let i = 0; i < n; i++) {
    const ang = beamWorldAngleRad(yaw, i, n);
    const dist = lidarNormToMetres(beams[i], rMin, rMax);
    const hx = wx + Math.sin(ang) * dist;
    const hz = wz + Math.cos(ang) * dist;
    const sx = ox + hx * scale;
    const sy = oy - hz * scale;
    outline.lineTo(sx, sy);
    const hit = maxSpan > 1e-6 ? (dist - rMin) / maxSpan : 1;
    const a = hit < 0.95 ? 0.55 : 0.18;
    ctx.strokeStyle = `rgba(62, 207, 142, ${a})`;
    ctx.beginPath();
    ctx.moveTo(sx0, sy0);
    ctx.lineTo(sx, sy);
    ctx.stroke();
  }
  outline.closePath();
  ctx.fillStyle = "rgba(62, 207, 142, 0.10)";
  ctx.fill(outline);
}

function drawCar(
  ctx: CanvasRenderingContext2D,
  car: FleetCar,
  scale: number,
  ox: number,
  oy: number,
  dimmed: boolean,
  dpr: number
): void {
  const [wx, wz] = car.pose!;
  const sx = ox + wx * scale;
  const sy = oy - wz * scale;
  const yaw = car.yaw ?? 0;
  const r = carRadiusPx(scale);

  ctx.save();
  ctx.globalAlpha = dimmed ? 0.14 : 1;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.translate(sx, sy);
  // Unity yaw=0 → +Z; screen angle of forward = atan2(-cos(yaw), sin(yaw))
  ctx.rotate(Math.atan2(-Math.cos(yaw), Math.sin(yaw)));

  ctx.fillStyle = car.collision ? "#e85d5d" : carColor(car.env_id);
  ctx.strokeStyle = "#0b100e";
  ctx.lineWidth = 1.5;
  // Keep the outline stable as focus/collision state changes. Env IDs own the
  // normal fill color; red fill marks a collision event.
  ctx.beginPath();
  ctx.moveTo(r + 1, 0);
  ctx.lineTo(-r, r * 0.85);
  ctx.lineTo(-r * 0.4, 0);
  ctx.lineTo(-r, -r * 0.85);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();

  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = "rgba(230, 240, 234, 0.75)";
  ctx.font = "10px Cascadia Code, Consolas, monospace";
  const labelY = sy - r - 2;
  ctx.fillText(String(car.env_id), sx + r + 3, labelY);
  if (car.speed != null && Number.isFinite(car.speed)) {
    ctx.fillText(`${car.speed.toFixed(1)}`, sx + r + 3, labelY + 11);
  }

  ctx.restore();
}

function drawEvaluatorCar(
  ctx: CanvasRenderingContext2D,
  car: EvaluatorCarTelemetry,
  scale: number,
  ox: number,
  oy: number,
  dpr: number
): void {
  const [wx, wz] = car.pose;
  const sx = ox + wx * scale;
  const sy = oy - wz * scale;
  const r = carRadiusPx(scale) + 1;
  ctx.save();
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.translate(sx, sy);
  ctx.rotate(Math.atan2(-Math.cos(car.yaw), Math.sin(car.yaw)));
  ctx.fillStyle = "#c684ff";
  ctx.strokeStyle = "#f4eaff";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(r + 2, 0);
  ctx.lineTo(-r, r * 0.9);
  ctx.lineTo(-r * 0.45, 0);
  ctx.lineTo(-r, -r * 0.9);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = "#f4eaff";
  ctx.font = "bold 11px Cascadia Code, Consolas, monospace";
  ctx.fillText("EVAL", sx + r + 4, sy - r - 2);
  if (Number.isFinite(car.speed)) {
    ctx.font = "10px Cascadia Code, Consolas, monospace";
    ctx.fillText(`${car.speed.toFixed(1)} m/s`, sx + r + 4, sy - r + 9);
  }
  ctx.restore();
}

function carColor(envId: number): string {
  const hue = ((envId * 137.508 + 165) % 360 + 360) % 360;
  return `hsl(${hue.toFixed(1)} 58% 55%)`;
}

export function resolveBounds(
  map: LoadedMap | null,
  view: ViewState,
  fleet: FleetTelemetry | null,
  evaluatorPose?: [number, number]
): { minX: number; maxX: number; minZ: number; maxZ: number } {
  const pad = 2;

  if (map) {
    // Keep the camera fixed to the loaded map. Including live car poses here
    // makes the underlay pan and zoom as vehicles move near its edges.
    const mb = mapWorldBounds(map);
    return {
      minX: mb.minX - pad,
      maxX: mb.maxX + pad,
      minZ: mb.minZ - pad,
      maxZ: mb.maxZ + pad,
    };
  }

  if (fleet) expandFit(view, fleet.cars);
  if (evaluatorPose) {
    const [x, z] = evaluatorPose;
    view.fitMinX = Math.min(view.fitMinX, x - 2);
    view.fitMaxX = Math.max(view.fitMaxX, x + 2);
    view.fitMinZ = Math.min(view.fitMinZ, z - 2);
    view.fitMaxZ = Math.max(view.fitMaxZ, z + 2);
  }

  if (
    Number.isFinite(view.fitMinX) &&
    view.fitMaxX > view.fitMinX &&
    view.fitMaxZ > view.fitMinZ
  ) {
    return {
      minX: view.fitMinX - pad,
      maxX: view.fitMaxX + pad,
      minZ: view.fitMinZ - pad,
      maxZ: view.fitMaxZ + pad,
    };
  }

  return { minX: -15, maxX: 15, minZ: -15, maxZ: 15 };
}

export function initialViewState(): ViewState {
  return {
    showMap: true,
    showFleet: true,
    showFrontier: true,
    showCurrentProgress: true,
    showLidar: true,
    selectedEnvId: 0,
    fitMinX: Infinity,
    fitMaxX: -Infinity,
    fitMinZ: Infinity,
    fitMaxZ: -Infinity,
  };
}
