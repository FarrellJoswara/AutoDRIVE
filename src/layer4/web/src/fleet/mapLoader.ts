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
}

export type MapId = "porto" | "berlin" | "none";

const MAP_MANIFEST: Record<
  Exclude<MapId, "none">,
  { yamlUrl: string; label: string }
> = {
  porto: { yamlUrl: "/maps/porto/Porto.yaml", label: "Porto" },
  berlin: { yamlUrl: "/maps/berlin/Berlin.yaml", label: "Berlin" },
};

export function mapLabel(id: MapId): string {
  if (id === "none") return "Grid only";
  return MAP_MANIFEST[id].label;
}

export function listMaps(): { id: MapId; label: string }[] {
  return [
    { id: "porto", label: "Porto" },
    { id: "berlin", label: "Berlin" },
    { id: "none", label: "Grid only" },
  ];
}

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
        // comment to EOL
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
  // single whitespace after maxval
  if (i < bytes.length && bytes[i] <= 0x20) i++;
  const pixels = bytes.subarray(i, i + width * height);
  const rgba = new Uint8ClampedArray(width * height * 4);
  for (let p = 0; p < width * height; p++) {
    let g = pixels[p] ?? 0;
    if (maxval !== 255) g = Math.round((g / maxval) * 255);
    // Occupancy: free≈white, occupied≈black — invert lightly for dark UI
    const v = 255 - g;
    const o = p * 4;
    rgba[o] = Math.round(v * 0.55 + 20);
    rgba[o + 1] = Math.round(v * 0.7 + 28);
    rgba[o + 2] = Math.round(v * 0.6 + 24);
    rgba[o + 3] = 255;
  }
  const imageData = new ImageData(rgba, width, height);
  const bitmap = await createImageBitmap(imageData);
  return { bitmap, width, height };
}

async function loadRaster(
  url: string
): Promise<{ source: CanvasImageSource; width: number; height: number }> {
  if (url.toLowerCase().endsWith(".pgm")) {
    const { bitmap, width, height } = await loadPgmAsImageBitmap(url);
    return { source: bitmap, width, height };
  }
  const img = new Image();
  img.decoding = "async";
  await new Promise<void>((resolve, reject) => {
    img.onload = () => resolve();
    img.onerror = () => reject(new Error(`image load failed: ${url}`));
    img.src = url;
  });
  return { source: img, width: img.naturalWidth, height: img.naturalHeight };
}

export async function loadMap(id: Exclude<MapId, "none">): Promise<LoadedMap> {
  const { yamlUrl } = MAP_MANIFEST[id];
  const yamlText = await (await fetch(yamlUrl)).text();
  const yaml = parseMapYaml(yamlText);
  const base = yamlUrl.replace(/[^/]+$/, "");
  const imageUrl = base + yaml.image;
  const { source, width, height } = await loadRaster(imageUrl);
  return { id, yaml, image: source, width, height };
}

/**
 * World (Unity X–Z metres) → image pixel.
 * ROS map: origin = lower-left of image in world; row 0 is top of image.
 * We treat yaml origin[1] as Unity Z.
 */
export function worldToMapPixel(
  map: LoadedMap,
  x: number,
  z: number
): { px: number; py: number } {
  const res = map.yaml.resolution;
  const [ox, oz] = map.yaml.origin;
  const px = (x - ox) / res;
  const py = map.height - (z - oz) / res;
  return { px, py };
}

/** Bounds of map in world metres (x,z). */
export function mapWorldBounds(map: LoadedMap): {
  minX: number;
  maxX: number;
  minZ: number;
  maxZ: number;
} {
  const res = map.yaml.resolution;
  const [ox, oz] = map.yaml.origin;
  return {
    minX: ox,
    maxX: ox + map.width * res,
    minZ: oz,
    maxZ: oz + map.height * res,
  };
}
