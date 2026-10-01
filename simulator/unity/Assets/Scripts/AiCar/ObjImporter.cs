// Minimal Wavefront OBJ → Unity Mesh (triangles only).

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using UnityEngine;

namespace AiCar
{
    public static class ObjImporter
    {
        public static Mesh Import(string path)
        {
            var positions = new List<Vector3>();
            var texcoords = new List<Vector2>();
            var faces = new List<int>();
            var faceUvs = new List<int>();
            bool facesHaveUv = false;

            foreach (var raw in File.ReadLines(path))
            {
                string line = raw.Trim();
                if (line.Length == 0 || line[0] == '#')
                    continue;
                if (line.StartsWith("v ", StringComparison.Ordinal))
                {
                    var p = line.Split((char[])null, StringSplitOptions.RemoveEmptyEntries);
                    if (p.Length < 4)
                        continue;
                    float x = Parse(p[1]);
                    float y = Parse(p[2]);
                    float z = Parse(p[3]);
                    // meshgen emits X,Y,Z with Y up — Unity matches
                    positions.Add(new Vector3(x, y, z));
                }
                else if (line.StartsWith("vt ", StringComparison.Ordinal))
                {
                    var p = line.Split((char[])null, StringSplitOptions.RemoveEmptyEntries);
                    if (p.Length < 3)
                        continue;
                    texcoords.Add(new Vector2(Parse(p[1]), Parse(p[2])));
                }
                else if (line.StartsWith("f ", StringComparison.Ordinal))
                {
                    var p = line.Split((char[])null, StringSplitOptions.RemoveEmptyEntries);
                    if (p.Length < 4)
                        continue;
                    int i0 = FaceIndex(p[1], positions.Count);
                    int t0 = FaceUvIndex(p[1], texcoords.Count);
                    for (int i = 2; i + 1 < p.Length; i++)
                    {
                        int i1 = FaceIndex(p[i], positions.Count);
                        int i2 = FaceIndex(p[i + 1], positions.Count);
                        faces.Add(i0);
                        faces.Add(i1);
                        faces.Add(i2);
                        int t1 = FaceUvIndex(p[i], texcoords.Count);
                        int t2 = FaceUvIndex(p[i + 1], texcoords.Count);
                        if (t0 >= 0 && t1 >= 0 && t2 >= 0)
                        {
                            facesHaveUv = true;
                            faceUvs.Add(t0);
                            faceUvs.Add(t1);
                            faceUvs.Add(t2);
                        }
                        else
                        {
                            faceUvs.Add(-1);
                            faceUvs.Add(-1);
                            faceUvs.Add(-1);
                        }
                    }
                }
            }

            var mesh = new Mesh();
            if (positions.Count > 65535)
                mesh.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;
            mesh.SetVertices(positions);
            mesh.SetTriangles(faces, 0);

            if (facesHaveUv && texcoords.Count > 0)
            {
                // Expand to per-vertex UVs (meshgen uses matching v/vt indices).
                var uvs = new Vector2[positions.Count];
                for (int i = 0; i < faces.Count; i++)
                {
                    int vi = faces[i];
                    int ti = faceUvs[i];
                    if (ti >= 0 && ti < texcoords.Count && vi >= 0 && vi < uvs.Length)
                        uvs[vi] = texcoords[ti];
                }
                mesh.SetUVs(0, uvs);
            }

            mesh.RecalculateNormals();
            mesh.RecalculateBounds();
            mesh.name = Path.GetFileNameWithoutExtension(path);
            return mesh;
        }

        /// <summary>
        /// Planar UVs for extruded wall ribbons so duct fabric / aluminium
        /// materials tile along length (U) and height (V).
        /// </summary>
        public static void EnsureWallUVs(Mesh mesh)
        {
            if (mesh == null)
                return;
            var verts = mesh.vertices;
            if (verts == null || verts.Length == 0)
                return;
            // Skip if OBJ already carried usable UVs.
            var existing = mesh.uv;
            if (existing != null && existing.Length == verts.Length)
            {
                bool any = false;
                for (int i = 0; i < existing.Length; i++)
                {
                    if (existing[i].sqrMagnitude > 1e-8f)
                    {
                        any = true;
                        break;
                    }
                }
                if (any)
                    return;
            }

            var norms = mesh.normals;
            var uvs = new Vector2[verts.Length];
            // Metres → UV: ~1 repeat per metre along wall run; height uses raw Y.
            const float uScale = 1f;
            for (int i = 0; i < verts.Length; i++)
            {
                Vector3 n = (norms != null && norms.Length == verts.Length)
                    ? norms[i]
                    : Vector3.up;
                // Dominant horizontal normal → wall face: U along tangent in XZ.
                float u = (Mathf.Abs(n.x) >= Mathf.Abs(n.z))
                    ? verts[i].z * uScale
                    : verts[i].x * uScale;
                float v = verts[i].y * uScale;
                uvs[i] = new Vector2(u, v);
            }
            mesh.SetUVs(0, uvs);
        }

        /// <summary>
        /// PhysX triangle MeshColliders are one-sided; duplicate flipped tris so
        /// cars cannot phase through walls from the back face / thin edges.
        /// </summary>
        public static Mesh MakeDoubleSided(Mesh source)
        {
            if (source == null)
                return null;
            var verts = source.vertices;
            var tris = source.triangles;
            var doubled = new int[tris.Length * 2];
            Array.Copy(tris, 0, doubled, 0, tris.Length);
            int o = tris.Length;
            for (int i = 0; i < tris.Length; i += 3)
            {
                doubled[o++] = tris[i];
                doubled[o++] = tris[i + 2];
                doubled[o++] = tris[i + 1];
            }
            var mesh = new Mesh();
            if (verts.Length > 65535)
                mesh.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;
            mesh.SetVertices(verts);
            mesh.SetTriangles(doubled, 0);
            var uvs = source.uv;
            if (uvs != null && uvs.Length == verts.Length)
                mesh.SetUVs(0, uvs);
            mesh.RecalculateNormals();
            mesh.RecalculateBounds();
            mesh.name = source.name + "_2S";
            return mesh;
        }

        static float Parse(string s) =>
            float.Parse(s, CultureInfo.InvariantCulture);

        static int FaceIndex(string token, int vertCount)
        {
            // "1", "1/2", "1//3", "1/2/3" — OBJ is 1-based; negative = relative
            string idx = token.Split('/')[0];
            int v = int.Parse(idx, CultureInfo.InvariantCulture);
            if (v < 0)
                v = vertCount + v + 1;
            return v - 1;
        }

        static int FaceUvIndex(string token, int uvCount)
        {
            // "1/2" or "1/2/3" → vt index; "1" or "1//3" → none
            var parts = token.Split('/');
            if (parts.Length < 2 || parts[1].Length == 0 || uvCount <= 0)
                return -1;
            int v = int.Parse(parts[1], CultureInfo.InvariantCulture);
            if (v < 0)
                v = uvCount + v + 1;
            return v - 1;
        }
    }
}
