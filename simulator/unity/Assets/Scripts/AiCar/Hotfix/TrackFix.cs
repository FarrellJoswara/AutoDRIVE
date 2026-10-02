// Hot-patch DLL dropped into AutoDRIVE Simulator_Data/Managed.
// Fixes: destroy baked duct meshes (SetActive was insufficient), solid colliders.
// Compiles without Unity Editor against the player Managed/*.dll refs.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using UnityEngine;

namespace AiCar.Hotfix
{
    public class TrackFixBootstrap
    {
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        static void Boot()
        {
            // Full player builds already include AiCar.TrackLoader — skip this hot-patch.
            if (HasFullTrackLoader())
            {
                Debug.Log("[AiCar.Hotfix] skip TrackFix — AiCar.TrackLoader already in player");
                return;
            }

            ApplyCli();
            var go = new GameObject("AiCarTrackFix");
            UnityEngine.Object.DontDestroyOnLoad(go);
            go.AddComponent<TrackFixRunner>();
        }

        static bool HasFullTrackLoader()
        {
            foreach (var asm in AppDomain.CurrentDomain.GetAssemblies())
            {
                Type t = null;
                try { t = asm.GetType("AiCar.TrackLoader", false); }
                catch { /* ignore */ }
                if (t != null && typeof(MonoBehaviour).IsAssignableFrom(t))
                    return true;
            }
            return false;
        }

        static void ApplyCli()
        {
            string[] args = Environment.GetCommandLineArgs();
            for (int i = 0; i < args.Length - 1; i++)
            {
                if (args[i] == "-map-id" || args[i] == "--map-id")
                {
                    TrackFixRunner.CliMapId = args[i + 1];
                    Debug.Log("[AiCar.Hotfix] CLI -map-id=" + TrackFixRunner.CliMapId);
                    return;
                }
            }
        }
    }

    public class TrackFixRunner : MonoBehaviour
    {
        public static string CliMapId;
        float _suppressUntil;
        bool _done;

        IEnumerator Start()
        {
            // Let the stock TrackLoader Awake/Start run first, then replace it.
            yield return null;
            yield return null;

            string mapId = ResolveMapId();
            if (string.IsNullOrEmpty(mapId))
            {
                Debug.Log("[AiCar.Hotfix] no map id — patch idle");
                yield break;
            }

            // Tear down whatever the stock TrackLoader did.
            foreach (var go in FindNamed("AiCarTrackLoader"))
                UnityEngine.Object.Destroy(go);
            foreach (var go in FindNamed("AiCarTrack"))
                UnityEngine.Object.Destroy(go);

            StripBuiltinTracks(destroy: false);

            string mapsRoot = ResolveMapsRoot();
            string meshDir = Path.Combine(mapsRoot, mapId, "mesh");
            string visualPath = Path.Combine(meshDir, "track.obj");
            string colliderPath = Path.Combine(meshDir, "track_col.obj");
            if (!File.Exists(visualPath) && !File.Exists(colliderPath))
            {
                Debug.LogError("[AiCar.Hotfix] no OBJ under " + meshDir);
                yield break;
            }

            string loadVisual = File.Exists(visualPath) ? visualPath : colliderPath;
            Mesh visual = ObjImport.Import(loadVisual);

            var goTrack = new GameObject("AiCarTrack");
            goTrack.transform.SetParent(transform, false);
            // ROS metres identity XZ; only lift Y to baked Porto Track floor.
            goTrack.transform.localScale = Vector3.one;
            goTrack.transform.position = new Vector3(0f, 0.127f, 0f);
            goTrack.AddComponent<MeshFilter>().sharedMesh = visual;
            goTrack.AddComponent<MeshRenderer>().sharedMaterial = MakeMat();
            var col = goTrack.AddComponent<MeshCollider>();
            col.sharedMesh = ObjImport.MakeDoubleSided(visual);
            col.convex = false;

            _suppressUntil = Time.unscaledTime + 6f;
            _done = true;
            Debug.Log("[AiCar.Hotfix] loaded map=" + mapId + " bounds=" + visual.bounds);
        }

        void Update()
        {
            // One-shot suppress was enough; continuous FindObjects spam is useless
            // once renderers/colliders are disabled.
        }

        static Material MakeMat()
        {
            // Unlit so walls stay visible under F1TENTH's harsh HDRP lighting.
            var shader = Shader.Find("HDRP/Unlit")
                ?? Shader.Find("Unlit/Color")
                ?? Shader.Find("HDRP/Lit")
                ?? Shader.Find("Standard");
            var mat = new Material(shader);
            var gray = new Color(0.55f, 0.58f, 0.62f, 1f);
            if (mat.HasProperty("_UnlitColor")) mat.SetColor("_UnlitColor", gray);
            if (mat.HasProperty("_BaseColor")) mat.SetColor("_BaseColor", gray);
            if (mat.HasProperty("_Color")) mat.SetColor("_Color", gray);
            return mat;
        }

        static string ResolveMapId()
        {
            if (!string.IsNullOrEmpty(CliMapId) &&
                !string.Equals(CliMapId, "none", StringComparison.OrdinalIgnoreCase))
                return CliMapId.Trim();

            string env = Environment.GetEnvironmentVariable("AICAR_MAP_ID");
            if (!string.IsNullOrEmpty(env) &&
                !string.Equals(env, "none", StringComparison.OrdinalIgnoreCase))
                return env.Trim();

            string mapsRoot = ResolveMapsRoot();
            string active = Path.Combine(mapsRoot, ".active_map.json");
            if (!File.Exists(active)) return null;
            try
            {
                string text = File.ReadAllText(active);
                const string key = "\"id\"";
                int k = text.IndexOf(key, StringComparison.Ordinal);
                if (k < 0) return null;
                int colon = text.IndexOf(':', k + key.Length);
                int q1 = text.IndexOf('"', colon + 1);
                int q2 = text.IndexOf('"', q1 + 1);
                if (q1 < 0 || q2 < 0) return null;
                return text.Substring(q1 + 1, q2 - q1 - 1).Trim();
            }
            catch { return null; }
        }

        static string ResolveMapsRoot()
        {
            string env = Environment.GetEnvironmentVariable("AICAR_MAPS_DIR");
            if (!string.IsNullOrEmpty(env) && Directory.Exists(env)) return env;

            string data = Application.dataPath;
            string simRoot = Directory.GetParent(data)?.FullName ?? data;
            string candidate = Path.Combine(simRoot, "maps");
            if (Directory.Exists(candidate)) return candidate;

            string parent = Directory.GetParent(simRoot)?.FullName;
            if (!string.IsNullOrEmpty(parent))
            {
                candidate = Path.Combine(parent, "maps");
                if (Directory.Exists(candidate)) return candidate;
            }
            return Path.Combine(simRoot, "maps");
        }

        static List<GameObject> FindNamed(string name)
        {
            var list = new List<GameObject>();
            foreach (var t in Resources.FindObjectsOfTypeAll<Transform>())
            {
                if (t == null || t.name != name) continue;
                if (!t.gameObject.scene.IsValid()) continue;
                list.Add(t.gameObject);
            }
            return list;
        }

        void StripBuiltinTracks(bool destroy)
        {
            int n = 0;
            foreach (var t in Resources.FindObjectsOfTypeAll<Transform>())
            {
                if (t == null || !InScene(t.gameObject) || IsAiCar(t)) continue;
                string name = t.name ?? "";
                bool trackRoot =
                    name.IndexOf("Porto Track", StringComparison.OrdinalIgnoreCase) >= 0 ||
                    name.IndexOf("Berlin Track", StringComparison.OrdinalIgnoreCase) >= 0 ||
                    (name.EndsWith(" Track", StringComparison.OrdinalIgnoreCase) &&
                     t.parent != null &&
                     string.Equals(t.parent.name, "Infrastructure", StringComparison.OrdinalIgnoreCase));
                if (trackRoot)
                    n += Neutralize(t.gameObject, destroy);
            }

            foreach (var mr in Resources.FindObjectsOfTypeAll<MeshRenderer>())
            {
                if (mr == null || !InScene(mr.gameObject) || IsProtected(mr.transform)) continue;
                if (!LooksLikeDuct(mr)) continue;
                n += Neutralize(mr.gameObject, destroy);
            }

            foreach (var mc in Resources.FindObjectsOfTypeAll<MeshCollider>())
            {
                if (mc == null || !InScene(mc.gameObject) || IsProtected(mc.transform)) continue;
                // Non-floor MeshColliders under Infrastructure are baked track.
                if (mc.transform.root != null &&
                    string.Equals(FindAncestorName(mc.transform, "Infrastructure"), "Infrastructure",
                        StringComparison.OrdinalIgnoreCase))
                {
                    n += Neutralize(mc.gameObject, destroy);
                }
            }

            if (n > 0)
                Debug.Log("[AiCar.Hotfix] stripped pieces=" + n);
        }

        static string FindAncestorName(Transform t, string want)
        {
            for (var p = t; p != null; p = p.parent)
                if (string.Equals(p.name, want, StringComparison.OrdinalIgnoreCase))
                    return p.name;
            return null;
        }

        static bool LooksLikeDuct(MeshRenderer mr)
        {
            var mf = mr.GetComponent<MeshFilter>();
            string meshName = mf != null && mf.sharedMesh != null ? mf.sharedMesh.name : "";
            if (meshName.IndexOf("Porto", StringComparison.OrdinalIgnoreCase) >= 0 ||
                meshName.IndexOf("Berlin", StringComparison.OrdinalIgnoreCase) >= 0)
                return true;
            var mats = mr.sharedMaterials;
            if (mats == null) return false;
            foreach (var mat in mats)
            {
                if (mat == null) continue;
                string mn = mat.name ?? "";
                if (mn.IndexOf("Black Fabric", StringComparison.OrdinalIgnoreCase) >= 0 ||
                    mn.IndexOf("Shiny Aluminium", StringComparison.OrdinalIgnoreCase) >= 0 ||
                    mn.IndexOf("White Fabric", StringComparison.OrdinalIgnoreCase) >= 0 ||
                    mn.IndexOf("Yellow Fabric", StringComparison.OrdinalIgnoreCase) >= 0)
                    return true;
            }
            return false;
        }

        static bool IsProtected(Transform t)
        {
            for (var p = t; p != null; p = p.parent)
            {
                string n = p.name ?? "";
                if (n.StartsWith("AiCar", StringComparison.OrdinalIgnoreCase)) return true;
                if (n.Equals("Floor", StringComparison.OrdinalIgnoreCase)) return true;
                if (n.IndexOf("Checkpoint", StringComparison.OrdinalIgnoreCase) >= 0) return true;
                if (n.IndexOf("Finish Line", StringComparison.OrdinalIgnoreCase) >= 0) return true;
                if (n.IndexOf("Spawn", StringComparison.OrdinalIgnoreCase) >= 0) return true;
                if (p.GetComponent<Rigidbody>() != null) return true;
                if (n.IndexOf("Vehicle", StringComparison.OrdinalIgnoreCase) >= 0 ||
                    n.IndexOf("Wheel", StringComparison.OrdinalIgnoreCase) >= 0 ||
                    n.IndexOf("Nigel", StringComparison.OrdinalIgnoreCase) >= 0)
                    return true;
            }
            return false;
        }

        static bool IsAiCar(Transform t)
        {
            for (var p = t; p != null; p = p.parent)
                if ((p.name ?? "").StartsWith("AiCar", StringComparison.OrdinalIgnoreCase))
                    return true;
            return false;
        }

        static int Neutralize(GameObject go, bool destroy)
        {
            // Unity overloads == for destroyed objects.
            if (go == null || !go) return 0;
            try
            {
                foreach (var r in go.GetComponentsInChildren<Renderer>(true))
                    if (r != null && r) r.enabled = false;
                foreach (var c in go.GetComponentsInChildren<Collider>(true))
                    if (c != null && c) c.enabled = false;
                if (destroy)
                {
                    Debug.Log("[AiCar.Hotfix] destroy " + go.name);
                    UnityEngine.Object.Destroy(go);
                }
                else if (go.activeSelf)
                    go.SetActive(false);
                return 1;
            }
            catch
            {
                return 0;
            }
        }

        static bool InScene(GameObject go) =>
            go != null && go.scene.IsValid() && go.scene.isLoaded;
    }

    static class ObjImport
    {
        public static Mesh Import(string path)
        {
            var positions = new List<Vector3>();
            var faces = new List<int>();
            foreach (var raw in File.ReadLines(path))
            {
                string line = raw.Trim();
                if (line.Length == 0 || line[0] == '#') continue;
                if (line.StartsWith("v ", StringComparison.Ordinal))
                {
                    var p = line.Split((char[])null, StringSplitOptions.RemoveEmptyEntries);
                    if (p.Length < 4) continue;
                    positions.Add(new Vector3(Parse(p[1]), Parse(p[2]), Parse(p[3])));
                }
                else if (line.StartsWith("f ", StringComparison.Ordinal))
                {
                    var p = line.Split((char[])null, StringSplitOptions.RemoveEmptyEntries);
                    if (p.Length < 4) continue;
                    int i0 = FaceIndex(p[1], positions.Count);
                    for (int i = 2; i + 1 < p.Length; i++)
                    {
                        faces.Add(i0);
                        faces.Add(FaceIndex(p[i], positions.Count));
                        faces.Add(FaceIndex(p[i + 1], positions.Count));
                    }
                }
            }
            var mesh = new Mesh();
            if (positions.Count > 65535)
                mesh.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;
            mesh.SetVertices(positions);
            mesh.SetTriangles(faces, 0);
            mesh.RecalculateNormals();
            mesh.RecalculateBounds();
            mesh.name = Path.GetFileNameWithoutExtension(path);
            return mesh;
        }

        public static Mesh MakeDoubleSided(Mesh source)
        {
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
            mesh.RecalculateBounds();
            mesh.name = source.name + "_2S";
            return mesh;
        }

        static float Parse(string s) =>
            float.Parse(s, CultureInfo.InvariantCulture);

        static int FaceIndex(string token, int vertCount)
        {
            string idx = token.Split('/')[0];
            int v = int.Parse(idx, CultureInfo.InvariantCulture);
            if (v < 0) v = vertCount + v + 1;
            return v - 1;
        }
    }
}
