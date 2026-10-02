/**
 * LiDAR beam aiming for the fleet canvas.
 *
 * Layer 1/2 publish distances only (normalised [0,1]). Drawing needs where
 * beam 0 points relative to car yaw, FOV, and range bounds.
 *
 * RoboRacer / AutoDRIVE 2025 guide (1080-beam planar scan):
 *   FOV −135° … +135° (270°), res 0.25°, range 0.06 … 10.0 m
 *   Beam 0 = θ_min = −135° relative to vehicle forward (Unity +Z at yaw=0)
 *   θ increases CCW in the vehicle/world X–Z frame
 *   World hit: (x, z) = pose + (sin(yaw+θ), cos(yaw+θ)) * range
 */

/** Full angular span of the published scan (radians). */
export const LIDAR_FOV_RAD = (270 * Math.PI) / 180;

/** First beam angle relative to yaw (radians). Beam 0 = −135°. */
export const LIDAR_ANGLE_MIN_RAD = (-135 * Math.PI) / 180;

/** +1 = CCW from beam 0 (RoboRacer / ROS LaserScan convention). */
export const LIDAR_ANGLE_SIGN = 1;

/** RoboRacer linear range (metres). */
export const LIDAR_RANGE_MIN_M = 0.06;
export const LIDAR_RANGE_MAX_M = 10.0;

export function beamWorldAngleRad(
  yaw: number,
  beamIndex: number,
  beamCount: number
): number {
  const n = Math.max(1, beamCount);
  const step = LIDAR_FOV_RAD / n;
  // Center of each min-pooled sector within [angle_min, angle_min+fov]
  return (
    yaw +
    LIDAR_ANGLE_MIN_RAD +
    LIDAR_ANGLE_SIGN * (beamIndex + 0.5) * step
  );
}

export function lidarNormToMetres(
  norm: number,
  rangeMin = LIDAR_RANGE_MIN_M,
  rangeMax = LIDAR_RANGE_MAX_M
): number {
  const n = Number.isFinite(norm) ? Math.min(1, Math.max(0, norm)) : 1;
  return n * (rangeMax - rangeMin) + rangeMin;
}
