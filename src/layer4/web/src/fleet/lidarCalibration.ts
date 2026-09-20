/**
 * LiDAR beam aiming for the fleet canvas.
 *
 * Layer 1/2 publish distances only (normalised [0,1]). Drawing rays also needs
 * where beam 0 points relative to car yaw, and the angular step around the car.
 *
 * TODO(F7): These defaults are unmeasured for AutoDRIVE RoboRacer. Park beside
 * a wall, tweak OFFSET / SIGN until hit points land on map geometry, then commit.
 *
 * Assumptions until calibrated:
 * - Full 360° FOV over the published beam count (after min-pool, typically 120)
 * - Beam 0 aligns with car forward (+yaw), then ANGLE_SIGN * increment
 * - ANGLE_SIGN = +1 → counterclockwise in the Unity X–Z world frame
 */
export const LIDAR_FOV_RAD = Math.PI * 2;

/** Added to yaw for beam 0 (radians). */
export const LIDAR_ANGLE_OFFSET_RAD = 0;

/** +1 = CCW from beam 0; −1 = CW. */
export const LIDAR_ANGLE_SIGN = 1;

/** Fallback denorm when fleet sample omits range bounds (Layer 1 defaults). */
export const LIDAR_RANGE_MIN_M = 0.05;
export const LIDAR_RANGE_MAX_M = 30.0;

export function beamWorldAngleRad(
  yaw: number,
  beamIndex: number,
  beamCount: number
): number {
  const n = Math.max(1, beamCount);
  const step = LIDAR_FOV_RAD / n;
  // Center of each min-pooled sector
  return yaw + LIDAR_ANGLE_OFFSET_RAD + LIDAR_ANGLE_SIGN * (beamIndex + 0.5) * step;
}

export function lidarNormToMetres(
  norm: number,
  rangeMin = LIDAR_RANGE_MIN_M,
  rangeMax = LIDAR_RANGE_MAX_M
): number {
  return norm * (rangeMax - rangeMin) + rangeMin;
}
