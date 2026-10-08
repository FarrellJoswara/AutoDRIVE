/** Load ROS occupancy yaml + image for the fleet map underlayer. */

export interface MapYaml {
  image: string;
  resolution: number;
  origin: [number, number, number];
  negate?: number;
  occupied_thresh?: number;
  free_thresh?: number;
}

export interface LoadedMap {
  id: string;
  yaml: MapYaml;
  /** HTMLImageElement or ImageBitmap ready to drawImage */
  image: CanvasImageSource;
  width: number;
  height: number;
  /** Pixel bounds containing occupied cells; avoids fitting large empty margins. */
  contentBounds: { minX: number; maxX: number; minY: number; maxY: number } | null;
}

export interface MapWorldAlignment {
  rotation_rad: number;
  translation_x_m: number;
  translation_z_m: number;
}

/** Catalog id from hub (`none` = grid only). */
export type MapId = string;

/** Minimal YAML subset parser for ROS map_server files (key: value). */
export function parseMapYaml(text: string): MapYaml {
  const out: Record<string, unknown> = {};
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.replace(/#.*$/, "").trim();
    if (!line) continue;
    const idx = line.indexOf(":");
    if (idx < 0) continue;
    const key = line.slice(0, idx).trim();
    let val = line.slice(idx + 1).trim();
    if (val.startsWith("[") && val.endsWith("]")) {
      out[key] = val
        .slice(1, -1)
        .split(",")
        .map((s) => Number(s.trim()));
    } else if (/^-?\d+(\.\d+)?([eE][-+]?\d+)?$/.test(val)) {
      out[key] = Number(val);
    } else {
      out[key] = val;
    }
  }
  const origin = out.origin as number[] | undefined;
  if (!out.image || typeof out.resolution !== "number" || !origin || origin.length < 2) {
    throw new Error("invalid map yaml");
  }
  return {
    image: String(out.image),
    resolution: Number(out.resolution),
    origin: [origin[0], origin[1], origin[2] ?? 0],
    negate: typeof out.negate === "number" ? out.negate : 0,
    occupied_thresh:
      typeof out.occupied_thresh === "number" ? out.occupied_thresh : undefined,
    free_thresh: typeof out.free_thresh === "number" ? out.free_thresh : undefined,
  };
}

async function loadPgmAsImageBitmap(url: string): Promise<{
  bitmap: ImageBitmap;
  width: number;
  height: number;
  contentBounds: LoadedMap["contentBounds"];
}> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`PGM fetch ${res.status}`);
  const buf = await res.arrayBuffer();
  const bytes = new Uint8Array(buf);
  let i = 0;
  const readToken = (): string => {
    while (i < bytes.length) {
      const c = bytes[i];
      if (c === 0x23) {
        while (i < bytes.length && bytes[i] !== 0x0a) i++;
        continue;
      }
      if (c <= 0x20) {
        i++;
        continue;
      }
      break;
    }
    const start = i;
    while (i < bytes.length && bytes[i] > 0x20) i++;
    return String.fromCharCode(...bytes.subarray(start, i));
  };

  const magic = readToken();
  if (magic !== "P5") throw new Error(`unsupported PGM ${magic}`);
  const width = Number(readToken());
  const height = Number(readToken());
  const maxval = Number(readToken());
  if (!width || !height || !maxval) throw new Error("bad PGM header");
  if (i < bytes.length && bytes[i] <= 0x20) i++;
  const pixels = bytes.subarray(i, i + width * height);
  const rgba = new Uint8ClampedArray(width * height * 4);
  let minX = width;
  let maxX = -1;
  let minY = height;
  let maxY = -1;
  // ROS negate:0 → high grey = free, low = occupied.
  for (let p = 0; p < width * height; p++) {
    let g = pixels[p] ?? 0;
    if (maxval !== 255) g = Math.round((g / maxval) * 255);
    if (g < 128) {
      const x = p % width;
      const y = Math.floor(p / width);
      minX = Math.min(minX, x);
      maxX = Math.max(maxX, x);
      minY = Math.min(minY, y);
      maxY = Math.max(maxY, y);
    }
    const v = 255 - g;
    const o = p * 4;
    rgba[o] = Math.round(v * 0.55 + 20);
    rgba[o + 1] = Math.round(v * 0.7 + 28);
    rgba[o + 2] = Math.round(v * 0.6 + 24);
    rgba[o + 3] = 255;
  }
  const imageData = new ImageData(rgba, width, height);
  const bitmap = await createImageBitmap(imageData);
  return {
    bitmap,
    width,
    height,
    contentBounds: maxX >= minX ? { minX, maxX: maxX + 1, minY, maxY: maxY + 1 } : null,
  };
}

function occupiedBounds(source: CanvasImageSource, width: number, height: number): LoadedMap["contentBounds"] {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) return null;
  ctx.drawImage(source, 0, 0, width, height);
  const pixels = ctx.getImageData(0, 0, width, height).data;
  let minX = width;
  let maxX = -1;
  let minY = height;
  let maxY = -1;
  for (let p = 0; p < width * height; p++) {
    const i = p * 4;
    const luminance = (pixels[i] * 299 + pixels[i + 1] * 587 + pixels[i + 2] * 114) / 1000;
    if (luminance < 128) {
      const x = p % width;
      const y = Math.floor(p / width);
      minX = Math.min(minX, x);
      maxX = Math.max(maxX, x);
      minY = Math.min(minY, y);
      maxY = Math.max(maxY, y);
    }
  }
  return maxX >= minX ? { minX, maxX: maxX + 1, minY, maxY: maxY + 1 } : null;
}

async function loadRaster(url: string): Promise<{
  source: CanvasImageSource;
  width: number;
  height: number;
  contentBounds: LoadedMap["contentBounds"];
}> {
  if (url.toLowerCase().endsWith(".pgm")) {
    const { bitmap, width, height, contentBounds } = await loadPgmAsImageBitmap(url);
    return { source: bitmap, width, height, contentBounds };
  }
  const img = new Image();
  img.decoding = "async";
  await new Promise<void>((resolve, reject) => {
    img.onload = () => resolve();
    img.onerror = () => reject(new Error(`image load failed: ${url}`));
    img.src = url;
  });
  return {
    source: img,
    width: img.naturalWidth,
    height: img.naturalHeight,
    contentBounds: occupiedBounds(img, img.naturalWidth, img.naturalHeight),
  };
}

/** Load map by hub catalog yaml URL (e.g. /maps/porto/occupancy/Porto.yaml). */
export async function loadMap(id: string, yamlUrl: string): Promise<LoadedMap> {
  const yamlText = await (await fetch(yamlUrl)).text();
  const yaml = parseMapYaml(yamlText);
  const base = yamlUrl.replace(/[^/]+$/, "");
  const imageUrl = base + yaml.image;
  const { source, width, height, contentBounds } = await loadRaster(imageUrl);
  return { id, yaml, image: source, width, height, contentBounds };
}

/** Bounds of map in world metres (x,z). Fleet poses are already map metres. */
/** Convert an official simulator point into this map asset's display frame. */
export function transformWorldPointToMap(
  point: [number, number],
  alignment?: MapWorldAlignment | null
): [number, number] {
  if (!alignment) return point;
  const c = Math.cos(alignment.rotation_rad);
  const s = Math.sin(alignment.rotation_rad);
  const x = point[0] - alignment.translation_x_m;
  const z = point[1] - alignment.translation_z_m;
  return [
    c * x + s * z,
    -s * x + c * z,
  ];
}

export function mapWorldBounds(map: LoadedMap): {
  minX: number;
  maxX: number;
  minZ: number;
  maxZ: number;
} {
  const res = map.yaml.resolution;
  const [ox, oz] = map.yaml.origin;
  const content = map.contentBounds;
  const minMapX = content ? ox + content.minX * res : ox;
  const maxMapX = content ? ox + content.maxX * res : ox + map.width * res;
  const minMapZ = content
    ? oz + (map.height - content.maxY) * res
    : oz;
  const maxMapZ = content
    ? oz + (map.height - content.minY) * res
    : oz + map.height * res;
  return { minX: minMapX, maxX: maxMapX, minZ: minMapZ, maxZ: maxMapZ };
}
