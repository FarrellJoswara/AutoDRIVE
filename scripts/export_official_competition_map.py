"""Rasterize the official IROS competition wall mesh for the Watch underlay.

Input is UnityPy's read-only OBJ export of the active SRL 2026 IROS Track.
This does not modify any simulator assets. Coordinates are official ROS IPS XY.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw


def export_map(mesh, route, output):
    vertices, faces = [], []
    for line in Path(mesh).read_text().splitlines():
        if line.startswith("v "):
            vertices.append([float(v) for v in line.split()[1:4]])
        elif line.startswith("f "):
            faces.append([int(v.split("/")[0])-1 for v in line.split()[1:]])
    vertices = np.array(vertices)
    # OBJ export reverses Unity X. Track parent is (-10,0,-14), child
    # (4,0.1524,10.525); the IPS frame adds +0.1583 to negative Unity X.
    xy = np.column_stack([vertices[:,2]-3.475, vertices[:,0]+6.1583])
    low, high = xy.min(0)-0.5, xy.max(0)+0.5
    resolution = 0.025
    size = np.ceil((high-low)/resolution).astype(int)
    image = Image.new("L",tuple(size),254)
    draw = ImageDraw.Draw(image)
    def pixel(points):
        px = (points-low)/resolution
        px[:,1] = size[1]-1-px[:,1]
        return [tuple(v) for v in px]
    for face in faces:
        # Project the upper wall surface; ignore ground/bottom triangles.
        if np.min(vertices[face,1]) > 0.01:
            draw.polygon(pixel(xy[face]),fill=0)
    output = Path(output); output.mkdir(parents=True,exist_ok=True)
    image.save(output/"competition.pgm")
    image.save(output/"preview.png")
    (output/"competition.yaml").write_text(
        f"image: competition.pgm\nresolution: {resolution}\norigin: [{low[0]}, {low[1]}, 0.0]\nnegate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n")
    (output/"centerline.csv").write_text(Path(route).read_text())
    (output/"meta.json").write_text(json.dumps({
        "label":"IROS 2026 Competition (official)",
        "source":"https://autodrive-ecosystem.github.io/competitions/roboracer-sim-racing-iros-2026/",
        "centerline":{"status":"ready","file":"centerline.csv","closed":True},
        "official_world_alignment":{"rotation_rad":0.,"translation_x_m":0.,"translation_z_m":0.,"source":"Official scene mesh transformed directly to ROS IPS XY"}
    },indent=2)+"\n")
    return image, low, resolution


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh",type=Path)
    parser.add_argument("route",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    export_map(args.mesh,args.route,args.output)
