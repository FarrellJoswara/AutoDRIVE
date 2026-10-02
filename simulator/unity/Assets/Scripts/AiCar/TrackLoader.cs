// AiCar TrackLoader — boot-time OBJ track for AutoDRIVE-Simulator.
// Map id: -map-id → AICAR_MAP_ID → maps/.active_map.json
//
// Builtin F1TENTH duct tracks (Porto Track / Berlin Track) are neutralized by
// disabling every Renderer/Collider (incl. inactive children), forceRenderingOff,
// LODGroups off, and SetActive(false) on mesh-bearing children — track ROOT
// GameObjects stay active so LapTimer / checkpoint name refs do not NRE.
// Custom occupancy meshes load in ROS/map metres: identity XZ (scale=1, no
// builtin AABB recenter). Only Y is lifted to duct floor height.

using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

namespace AiCar
{
    public class TrackLoader : MonoBehaviour
    {
        [Tooltip("Override maps root; empty → sibling of player Data folder: ../maps")]
        public string mapsRootOverride = "";

        [Tooltip("If true and no map id is set, leave the scene as authored.")]
        public bool keepBuiltinWhenUnset = true;

        bool _loaded;
        float _suppressUntil;
        // SketchUp / prefab scripts can re-enable renderers for several frames.
        const float SuppressSeconds = 15f;
        int _lastStripLogFrame = -1;

        // Baked Porto/Berlin Track roots sit at y=0.127 in F1TENTH.unity.
        const float BuiltinTrackY = 0.127f;

        void Start()
        {
            TryLoad();
        }

        void OnEnable()
        {
            if (!_loaded)
                TryLoad();
        }

        void Update()
        {
            // AutoDRIVE / SketchUp prefab instances can re-enable renderers after Start.
            if (_loaded && Time.unscaledTime < _suppressUntil)
                StripBuiltinTracks();
        }

        void TryLoad()
        {
            if (_loaded)
                return;

            string mapId = MapConfig.ResolveMapId();
            if (string.IsNullOrEmpty(mapId))
            {
                Debug.Log("[AiCar.TrackLoader] no map id — keeping builtin track");
                SilenceBrokenLapTimers();
                _loaded = true;
                return;
            }

            string mapsRoot = ResolveMapsRoot();
            string meshDir = Path.Combine(mapsRoot, mapId, "mesh");
            string visualPath = Path.Combine(meshDir, "track.obj");
            string colliderPath = Path.Combine(meshDir, "track_col.obj");

            if (!File.Exists(visualPath) && !File.Exists(colliderPath))
            {
                Debug.LogError($"[AiCar.TrackLoader] no OBJ under {meshDir}");
                SilenceBrokenLapTimers();
                _loaded = true;
                return;
            }

            // Capture builtin materials / floor Y BEFORE neutralizing renderers.
            // Do NOT use builtin XZ AABB — ROS mesh verts stay in map metres.
            Bounds? builtinBounds = TryGetBuiltinTrackWorldBounds(mapId, out Transform builtinRoot);
            Material ductMat = TryStealDuctMaterial(builtinRoot);

            StripBuiltinTracks();

            // Prefer full visual mesh for collision — track_col is decimated and leaky.
            string loadVisual = File.Exists(visualPath) ? visualPath : colliderPath;
            Mesh visual = ObjImporter.Import(loadVisual);
            ObjImporter.EnsureWallUVs(visual);
            Mesh colliderMesh = visual;

            var go = new GameObject("AiCarTrack");
            go.transform.SetParent(transform, false);
            var mf = go.AddComponent<MeshFilter>();
            mf.sharedMesh = visual;
            var mr = go.AddComponent<MeshRenderer>();
            mr.sharedMaterial = ductMat != null ? ductMat : CreateTrackMaterial();
            var col = go.AddComponent<MeshCollider>();
            col.sharedMesh = ObjImporter.MakeDoubleSided(colliderMesh);
            col.convex = false;

            AlignTrackTransform(go.transform, visual.bounds, builtinBounds);
            SilenceBrokenLapTimers();
            ForceAutonomousDrivingMode();

            TryApplySpawn(mapsRoot, mapId);
            _loaded = true;
            _suppressUntil = Time.unscaledTime + SuppressSeconds;

            var wb = mr.bounds;
            Debug.Log(
                $"[AiCar.TrackLoader] loaded map={mapId} visual={loadVisual} " +
                $"localBounds={visual.bounds} worldBounds={wb} " +
                $"alignBuiltin=none " +
                $"mat={(mr.sharedMaterial != null ? mr.sharedMaterial.name : "null")}");
        }

        /// <summary>
        /// Place ROS-metre OBJ in Unity with identity XZ (scale=1, tx=tz=0).
        /// Porto/Berlin behave like custom maps (icra): world XZ ≈ ROS yaml metres.
        /// Only Y is lifted so the mesh floor sits near the baked duct height.
        /// </summary>
        static void AlignTrackTransform(Transform t, Bounds localMeshBounds, Bounds? builtinWorld)
        {
            t.localScale = Vector3.one;

            float yFloor = BuiltinTrackY;
            if (builtinWorld.HasValue)
            {
                float y = builtinWorld.Value.min.y;
                if (y >= 0f && y <= 1f)
                    yFloor = y;
            }

            t.position = new Vector3(0f, yFloor - localMeshBounds.min.y, 0f);
            Debug.Log(
                $"[AiCar.TrackLoader] align identity XZ scale=1 pos={t.position} " +
                $"(ROS metres; floorY={yFloor:F3})");
        }

        static Bounds? TryGetBuiltinTrackWorldBounds(string mapId, out Transform builtinRoot)
        {
            builtinRoot = FindBuiltinTrackRoot(mapId);
            if (builtinRoot == null)
                return null;

            bool any = false;
            Bounds b = new Bounds(builtinRoot.position, Vector3.zero);
            foreach (var mf in builtinRoot.GetComponentsInChildren<MeshFilter>(true))
            {
                if (mf == null || mf.sharedMesh == null)
                    continue;
                Bounds wb = TransformBounds(mf.transform, mf.sharedMesh.bounds);
                if (!any)
                {
                    b = wb;
                    any = true;
                }
                else
                {
                    b.Encapsulate(wb);
                }
            }

            if (!any)
            {
                foreach (var r in builtinRoot.GetComponentsInChildren<Renderer>(true))
                {
                    if (r == null)
                        continue;
                    if (!any)
                    {
                        b = r.bounds;
                        any = true;
                    }
                    else
                    {
                        b.Encapsulate(r.bounds);
                    }
                }
            }

            return any ? b : (Bounds?)null;
        }

        static Bounds TransformBounds(Transform t, Bounds local)
        {
            var center = t.TransformPoint(local.center);
            var extents = local.extents;
            var axisX = t.TransformVector(extents.x, 0, 0);
            var axisY = t.TransformVector(0, extents.y, 0);
            var axisZ = t.TransformVector(0, 0, extents.z);
            extents.x = Mathf.Abs(axisX.x) + Mathf.Abs(axisY.x) + Mathf.Abs(axisZ.x);
            extents.y = Mathf.Abs(axisX.y) + Mathf.Abs(axisY.y) + Mathf.Abs(axisZ.y);
            extents.z = Mathf.Abs(axisX.z) + Mathf.Abs(axisY.z) + Mathf.Abs(axisZ.z);
            return new Bounds(center, extents * 2f);
        }

        static Transform FindBuiltinTrackRoot(string mapId)
        {
            // Only Porto/Berlin have matching baked duct geometry. Custom maps
            // (e.g. icra2026_*) must NOT borrow Porto AABB — that mis-aligns the
            // mesh while spawn stays in ROS metres.
            string want = MapIdToTrackName(mapId);
            if (want == null)
                return null;

            foreach (var t in SceneTransforms())
            {
                if (t == null || IsAiCar(t))
                    continue;
                if (string.Equals(t.name, want, StringComparison.OrdinalIgnoreCase))
                    return t;
                if (IsBuiltinTrackRoot(t) &&
                    t.name.IndexOf(want.Replace(" Track", ""), StringComparison.OrdinalIgnoreCase) >= 0)
                    return t;
            }
            return null;
        }

        static string MapIdToTrackName(string mapId)
        {
            if (string.IsNullOrEmpty(mapId))
                return null;
            if (string.Equals(mapId, "porto", StringComparison.OrdinalIgnoreCase))
                return "Porto Track";
            if (string.Equals(mapId, "berlin", StringComparison.OrdinalIgnoreCase))
                return "Berlin Track";
            return null;
        }

        /// <summary>
        /// Reuse AutoDRIVE duct HDRP materials already loaded on the builtin track
        /// (Black Fabric / Shiny Aluminium under F1TENTH Tracks/Materials).
        /// </summary>
        static Material TryStealDuctMaterial(Transform builtinRoot)
        {
            Material shiny = null;
            Material black = null;
            Material anyDuct = null;

            void Consider(Material mat)
            {
                if (mat == null)
                    return;
                string n = mat.name ?? "";
                if (n.IndexOf("Shiny Aluminium", StringComparison.OrdinalIgnoreCase) >= 0)
                    shiny = mat;
                else if (n.IndexOf("Black Fabric", StringComparison.OrdinalIgnoreCase) >= 0)
                    black = mat;
                else if (n.IndexOf("Fabric", StringComparison.OrdinalIgnoreCase) >= 0 ||
                         n.IndexOf("Aluminium", StringComparison.OrdinalIgnoreCase) >= 0 ||
                         n.IndexOf("Aluminum", StringComparison.OrdinalIgnoreCase) >= 0)
                    anyDuct = mat;
            }

            if (builtinRoot != null)
            {
                foreach (var r in builtinRoot.GetComponentsInChildren<Renderer>(true))
                {
                    if (r == null || r.sharedMaterials == null)
                        continue;
                    foreach (var mat in r.sharedMaterials)
                        Consider(mat);
                }
            }

            // Scene-wide fallback (materials stay loaded even after renderer disable).
            foreach (var mat in Resources.FindObjectsOfTypeAll<Material>())
                Consider(mat);

            Material chosen = shiny ?? black ?? anyDuct;
            if (chosen == null)
                return null;

            // Instance so tiling tweaks do not mutate the shared duct asset mid-scene.
            var inst = new Material(chosen);
            inst.name = chosen.name + " (AiCar)";
            // Encourage visible corrugation / fabric repeat along wall length.
            if (inst.HasProperty("_BaseColorMap"))
                inst.SetTextureScale("_BaseColorMap", new Vector2(4f, 1.5f));
            if (inst.HasProperty("_MainTex"))
                inst.SetTextureScale("_MainTex", new Vector2(4f, 1.5f));
            if (inst.HasProperty("_NormalMap"))
                inst.SetTextureScale("_NormalMap", new Vector2(4f, 1.5f));
            Debug.Log($"[AiCar.TrackLoader] duct material={inst.name}");
            return inst;
        }

        static Material CreateTrackMaterial()
        {
            Shader shader =
                Shader.Find("HDRP/Lit")
                ?? Shader.Find("Universal Render Pipeline/Lit")
                ?? Shader.Find("Standard")
                ?? Shader.Find("Diffuse")
                ?? Shader.Find("Unlit/Color");

            var mat = new Material(shader);
            // Cool aluminium fallback when duct mats are unavailable.
            var aluminum = new Color(0.55f, 0.58f, 0.62f, 1f);
            if (mat.HasProperty("_BaseColor"))
                mat.SetColor("_BaseColor", aluminum);
            if (mat.HasProperty("_Color"))
                mat.SetColor("_Color", aluminum);
            if (mat.HasProperty("_UnlitColor"))
                mat.SetColor("_UnlitColor", aluminum);
            if (mat.HasProperty("_Metallic"))
                mat.SetFloat("_Metallic", 0.85f);
            if (mat.HasProperty("_Smoothness"))
                mat.SetFloat("_Smoothness", 0.65f);
            Debug.Log($"[AiCar.TrackLoader] material shader={shader?.name} (procedural fallback)");
            return mat;
        }

        string ResolveMapsRoot()
        {
            if (!string.IsNullOrEmpty(mapsRootOverride) && Directory.Exists(mapsRootOverride))
                return mapsRootOverride;

            string env = Environment.GetEnvironmentVariable("AICAR_MAPS_DIR");
            if (!string.IsNullOrEmpty(env) && Directory.Exists(env))
                return env;

            string data = Application.dataPath;
            string simRoot = Directory.GetParent(data)?.FullName ?? data;
            string candidate = Path.Combine(simRoot, "maps");
            if (Directory.Exists(candidate))
                return candidate;

            string parent = Directory.GetParent(simRoot)?.FullName;
            if (!string.IsNullOrEmpty(parent))
            {
                candidate = Path.Combine(parent, "maps");
                if (Directory.Exists(candidate))
                    return candidate;
            }

            return Path.Combine(simRoot, "maps");
        }

        void StripBuiltinTracks()
        {
            var totals = new NeutralizeStats();
            int roots = 0;
            int rootedRenderers = 0;
            int rootedFilters = 0;

            // 1) Known baked track roots under Infrastructure (and anywhere).
            foreach (var t in SceneTransforms())
            {
                if (t == null || IsAiCar(t))
                    continue;
                if (!IsBuiltinTrackRoot(t))
                    continue;
                rootedRenderers += t.GetComponentsInChildren<Renderer>(true).Length;
                rootedFilters += t.GetComponentsInChildren<MeshFilter>(true).Length;
                totals.Add(Neutralize(t.gameObject, keepSelfActive: true));
                roots++;
            }

            // 2) Any MeshCollider that isn't floor / vehicle / our track → strip.
            foreach (var mc in Resources.FindObjectsOfTypeAll<MeshCollider>())
            {
                if (mc == null || !InLoadedScene(mc.gameObject))
                    continue;
                if (IsProtectedCollider(mc.transform))
                    continue;
                // Already covered if under a neutralized track root.
                if (UnderBuiltinTrackRoot(mc.transform))
                    continue;
                totals.Add(Neutralize(mc.gameObject, keepSelfActive: false));
            }

            // 3) Renderers whose mesh/material smells like F1TENTH air-duct tracks.
            foreach (var r in Resources.FindObjectsOfTypeAll<Renderer>())
            {
                if (r == null || !InLoadedScene(r.gameObject))
                    continue;
                if (IsProtectedCollider(r.transform))
                    continue;
                if (UnderBuiltinTrackRoot(r.transform))
                    continue;
                if (!LooksLikeBuiltinTrackRenderer(r))
                    continue;
                totals.Add(Neutralize(r.gameObject, keepSelfActive: false));
            }

            // Proof for Player.log: anything duct-like still drawing?
            int stillOn = 0;
            foreach (var r in Resources.FindObjectsOfTypeAll<Renderer>())
            {
                if (r == null || !InLoadedScene(r.gameObject) || IsAiCar(r.transform))
                    continue;
                if (!r.enabled && r.forceRenderingOff)
                    continue;
                if (!r.gameObject.activeInHierarchy)
                    continue;
                if (LooksLikeBuiltinTrackRenderer(r) || UnderBuiltinTrackRoot(r.transform))
                    stillOn++;
            }

            // Throttle spam during suppress Update loop — log when work happened or once/sec.
            bool work = totals.renderers > 0 || totals.colliders > 0 || totals.deactivated > 0;
            int frame = Time.frameCount;
            if (work || stillOn > 0 || frame - _lastStripLogFrame >= 60)
            {
                _lastStripLogFrame = frame;
                Debug.Log(
                    $"[AiCar.TrackLoader] strip roots={roots} " +
                    $"rootRenderers={rootedRenderers} rootMeshFilters={rootedFilters} " +
                    $"renderersOff={totals.renderers} collidersOff={totals.colliders} " +
                    $"lodOff={totals.lodGroups} childrenOff={totals.deactivated} " +
                    $"stillDrawingDucts={stillOn} destroy=false");
            }
        }

        static bool IsBuiltinTrackRoot(Transform t)
        {
            string n = t.name ?? "";
            if (n.IndexOf("Porto Track", StringComparison.OrdinalIgnoreCase) >= 0)
                return true;
            if (n.IndexOf("Berlin Track", StringComparison.OrdinalIgnoreCase) >= 0)
                return true;
            // Loose SketchUp export names under Infrastructure.
            if (t.parent != null &&
                string.Equals(t.parent.name, "Infrastructure", StringComparison.OrdinalIgnoreCase))
            {
                if (n.EndsWith(" Track", StringComparison.OrdinalIgnoreCase))
                    return true;
                if (string.Equals(n, "Porto", StringComparison.OrdinalIgnoreCase) ||
                    string.Equals(n, "Berlin", StringComparison.OrdinalIgnoreCase))
                    return true;
            }
            return false;
        }

        static bool UnderBuiltinTrackRoot(Transform t)
        {
            for (var p = t; p != null; p = p.parent)
            {
                if (IsBuiltinTrackRoot(p))
                    return true;
            }
            return false;
        }

        static bool LooksLikeBuiltinTrackRenderer(Renderer r)
        {
            if (r == null)
                return false;

            var mf = r.GetComponent<MeshFilter>();
            string meshName = mf != null && mf.sharedMesh != null ? mf.sharedMesh.name : "";
            if (meshName.IndexOf("Porto", StringComparison.OrdinalIgnoreCase) >= 0 ||
                meshName.IndexOf("Berlin", StringComparison.OrdinalIgnoreCase) >= 0)
                return true;

            var mats = r.sharedMaterials;
            if (mats == null)
                return false;
            foreach (var mat in mats)
            {
                if (mat == null)
                    continue;
                string mn = mat.name ?? "";
                // Strip Unity "(Instance)" suffix noise.
                int paren = mn.IndexOf(" (Instance)", StringComparison.OrdinalIgnoreCase);
                if (paren > 0)
                    mn = mn.Substring(0, paren);
                if (mn.IndexOf("Black Fabric", StringComparison.OrdinalIgnoreCase) >= 0 ||
                    mn.IndexOf("Shiny Aluminium", StringComparison.OrdinalIgnoreCase) >= 0 ||
                    mn.IndexOf("Shiny Aluminum", StringComparison.OrdinalIgnoreCase) >= 0 ||
                    mn.IndexOf("White Fabric", StringComparison.OrdinalIgnoreCase) >= 0 ||
                    mn.IndexOf("Yellow Fabric", StringComparison.OrdinalIgnoreCase) >= 0 ||
                    mn.IndexOf("Aluminium", StringComparison.OrdinalIgnoreCase) >= 0 ||
                    mn.IndexOf("Aluminum", StringComparison.OrdinalIgnoreCase) >= 0)
                    return true;
            }
            return false;
        }

        static bool IsProtectedCollider(Transform t)
        {
            if (t == null)
                return true;
            if (IsAiCar(t))
                return true;
            for (var p = t; p != null; p = p.parent)
            {
                string n = p.name ?? "";
                if (n.Equals("Floor", StringComparison.OrdinalIgnoreCase))
                    return true;
                if (n.IndexOf("Checkpoint", StringComparison.OrdinalIgnoreCase) >= 0)
                    return true;
                if (n.IndexOf("Finish Line", StringComparison.OrdinalIgnoreCase) >= 0)
                    return true;
                if (n.IndexOf("Spawn", StringComparison.OrdinalIgnoreCase) >= 0)
                    return true;
                if (p.GetComponent<Rigidbody>() != null)
                    return true; // vehicle / dynamic
                // Do NOT match bare "F1TENTH" — scene/env parents use that name and
                // would shield Porto Track MeshColliders from path-2/3 stripping.
                if (n.IndexOf("Nigel", StringComparison.OrdinalIgnoreCase) >= 0 ||
                    n.IndexOf("Vehicle", StringComparison.OrdinalIgnoreCase) >= 0 ||
                    n.IndexOf("Wheel", StringComparison.OrdinalIgnoreCase) >= 0 ||
                    n.IndexOf("Ego", StringComparison.OrdinalIgnoreCase) >= 0)
                    return true;
            }
            return false;
        }

        static bool IsAiCar(Transform t)
        {
            for (var p = t; p != null; p = p.parent)
            {
                if ((p.name ?? "").StartsWith("AiCar", StringComparison.OrdinalIgnoreCase))
                    return true;
            }
            return false;
        }

        /// <summary>
        /// F1TENTH.unity wires multiple LapTimers; agents without HUD Text refs
        /// NRE every Update. In batchmode the lap HUD and its checkpoint triggers
        /// are unused, so disable both. Unity still dispatches trigger callbacks
        /// to disabled MonoBehaviours; disabling only the LapTimer is insufficient.
        /// Headed: only disable timers with null HUD refs.
        /// </summary>
        static void SilenceBrokenLapTimers()
        {
            int n = 0;
            bool batch = Application.isBatchMode;
            int checkpointTriggersDisabled = batch ? DisableCheckpointTriggers() : 0;
            foreach (var lt in Resources.FindObjectsOfTypeAll<LapTimer>())
            {
                if (lt == null || !InLoadedScene(lt.gameObject) || !lt.enabled)
                    continue;
                bool broken = lt.txtLapTime == null || lt.txtLastLap == null || lt.txtBestLap == null ||
                              lt.txtLapCount == null || lt.txtCollisionCount == null;
                if (batch || broken)
                {
                    lt.enabled = false;
                    n++;
                    Debug.Log(
                        $"[AiCar.TrackLoader] disabled LapTimer on '{GetPath(lt.transform)}' " +
                        (batch
                            ? "(batchmode; lap HUD unused)"
                            : "(null HUD refs)"));
                }
            }
            if (batch)
                Debug.Log($"[AiCar.TrackLoader] disabled unused checkpoint triggers={checkpointTriggersDisabled}");
            if (n == 0)
                Debug.Log("[AiCar.TrackLoader] LapTimer HUD refs OK (none disabled)");
        }

        static int DisableCheckpointTriggers()
        {
            int disabled = 0;
            foreach (var trigger in Resources.FindObjectsOfTypeAll<Collider>())
            {
                if (trigger == null || !InLoadedScene(trigger.gameObject) ||
                    !trigger.enabled || !trigger.isTrigger ||
                    !trigger.CompareTag("Checkpoint"))
                    continue;

                trigger.enabled = false;
                disabled++;
            }
            return disabled;
        }

        struct NeutralizeStats
        {
            public int renderers;
            public int colliders;
            public int lodGroups;
            public int deactivated;

            public void Add(NeutralizeStats other)
            {
                renderers += other.renderers;
                colliders += other.colliders;
                lodGroups += other.lodGroups;
                deactivated += other.deactivated;
            }
        }

        /// <summary>
        /// Airtight visual/physics strip under <paramref name="go"/>.
        /// Track roots stay active (LapTimer name refs); every child is
        /// deactivated, MeshFilters lose their mesh (post-bounds), and
        /// Renderers get enabled=false + forceRenderingOff.
        /// </summary>
        static NeutralizeStats Neutralize(GameObject go, bool keepSelfActive)
        {
            var stats = new NeutralizeStats();
            if (go == null)
                return stats;

            foreach (var r in go.GetComponentsInChildren<Renderer>(true))
            {
                if (r == null)
                    continue;
                bool changed = false;
                if (r.enabled)
                {
                    r.enabled = false;
                    changed = true;
                }
                if (!r.forceRenderingOff)
                {
                    r.forceRenderingOff = true;
                    changed = true;
                }
                if (changed)
                    stats.renderers++;
            }

            foreach (var lod in go.GetComponentsInChildren<LODGroup>(true))
            {
                if (lod != null && lod.enabled)
                {
                    lod.enabled = false;
                    stats.lodGroups++;
                }
            }

            foreach (var c in go.GetComponentsInChildren<Collider>(true))
            {
                if (c != null && c.enabled)
                {
                    c.enabled = false;
                    stats.colliders++;
                }
            }

            // Detach meshes so a later script re-enabling the Renderer draws nothing.
            // Bounds / duct mats are captured before the first StripBuiltinTracks call.
            foreach (var mf in go.GetComponentsInChildren<MeshFilter>(true))
            {
                if (mf != null && mf.sharedMesh != null)
                    mf.sharedMesh = null;
            }
            foreach (var smr in go.GetComponentsInChildren<SkinnedMeshRenderer>(true))
            {
                if (smr != null && smr.sharedMesh != null)
                    smr.sharedMesh = null;
            }
            foreach (var mc in go.GetComponentsInChildren<MeshCollider>(true))
            {
                if (mc != null && mc.sharedMesh != null)
                    mc.sharedMesh = null;
            }

            // Deactivate every child (and optionally self). Collect first.
            var toDeactivate = new List<GameObject>();
            foreach (Transform child in go.GetComponentsInChildren<Transform>(true))
            {
                if (child == null || child.gameObject == null)
                    continue;
                if (keepSelfActive && child.gameObject == go)
                    continue;
                if (child.gameObject.activeSelf)
                    toDeactivate.Add(child.gameObject);
            }

            var seen = new HashSet<int>();
            foreach (var child in toDeactivate)
            {
                if (child == null)
                    continue;
                int id = child.GetInstanceID();
                if (!seen.Add(id))
                    continue;
                child.SetActive(false);
                stats.deactivated++;
            }

            return stats;
        }

        static string GetPath(Transform t)
        {
            var parts = new List<string>();
            for (var p = t; p != null; p = p.parent)
                parts.Add(p.name);
            parts.Reverse();
            return string.Join("/", parts);
        }

        static IEnumerable<Transform> SceneTransforms()
        {
            foreach (var t in Resources.FindObjectsOfTypeAll<Transform>())
            {
                if (t != null && InLoadedScene(t.gameObject))
                    yield return t;
            }
        }

        static bool InLoadedScene(GameObject go)
        {
            return go != null && go.scene.IsValid() && go.scene.isLoaded;
        }

        /// <summary>
        /// Batchmode has no GUI DrivingMode toggle; its Start() may also NRE on a
        /// null Label and leave VehicleController.DrivingMode=0 (manual), which
        /// makes Socket ignore V1 Throttle. Force autonomous when headless.
        /// </summary>
        static void ForceAutonomousDrivingMode()
        {
            if (!Application.isBatchMode)
                return;
            int n = 0;
            var flags = System.Reflection.BindingFlags.Instance |
                        System.Reflection.BindingFlags.Public |
                        System.Reflection.BindingFlags.NonPublic;
            foreach (var mb in Resources.FindObjectsOfTypeAll<MonoBehaviour>())
            {
                if (mb == null || !InLoadedScene(mb.gameObject)) continue;
                string tn = mb.GetType().Name;
                if (tn != "VehicleController" && tn != "AutomobileController")
                    continue;
                var fi = mb.GetType().GetField("DrivingMode", flags);
                if (fi == null || fi.FieldType != typeof(int)) continue;
                if ((int)fi.GetValue(mb) != 1)
                {
                    fi.SetValue(mb, 1);
                    n++;
                }
            }
            if (n > 0)
                Debug.Log($"[AiCar.TrackLoader] forced DrivingMode=1 on {n} controllers (batchmode)");
        }

        void TryApplySpawn(string mapsRoot, string mapId)
        {
            if (!TryReadSpawn(mapsRoot, mapId, out Vector3 localPos, out float yawRad))
            {
                Debug.LogWarning($"[AiCar.TrackLoader] no spawn for map={mapId} — leaving scene spawn");
                return;
            }

            // Spawn is authored in mesh/ROS metres (same frame as track.obj verts).
            // AiCarTrack is identity XZ — TransformPoint only applies floor Y lift.
            Transform track = transform.Find("AiCarTrack");
            Vector3 worldPos;
            Quaternion worldRot;
            if (track != null)
            {
                worldPos = track.TransformPoint(localPos);
                worldRot = track.rotation * Quaternion.Euler(0f, yawRad * Mathf.Rad2Deg, 0f);
            }
            else
            {
                worldPos = localPos;
                worldRot = Quaternion.Euler(0f, yawRad * Mathf.Rad2Deg, 0f);
            }

            // Keep wheels slightly above the floor / track base.
            if (worldPos.y < BuiltinTrackY)
                worldPos.y = BuiltinTrackY + 0.02f;

            int movedSpawns = MoveSpawnTransforms(worldPos, worldRot);
            int movedVehicles = TeleportVehicles(worldPos, worldRot);

            // Keep re-applying briefly: Bridge V1 Reset copies Spawn transforms, and
            // physics may settle the car for a few FixedUpdates after teleport.
            var keeper = gameObject.GetComponent<SpawnKeeper>();
            if (keeper == null)
                keeper = gameObject.AddComponent<SpawnKeeper>();
            // Short window: only re-teleport if Bridge yanks the car away.
            // Continuous teleport (old 20s) zeroed velocity every FixedUpdate and
            // froze the car on the Fleet canvas after a map change.
            keeper.Arm(worldPos, worldRot, 4f);

            Debug.Log(
                $"[AiCar.TrackLoader] spawn map={mapId} local={localPos} yawRad={yawRad:F3} " +
                $"world={worldPos} spawnsMoved={movedSpawns} vehiclesMoved={movedVehicles}");
        }

        static bool TryReadSpawn(string mapsRoot, string mapId, out Vector3 localPos, out float yawRad)
        {
            localPos = Vector3.zero;
            yawRad = 0f;

            string metaPath = Path.Combine(mapsRoot, mapId, "occupancy", "meta.json");
            if (File.Exists(metaPath))
            {
                try
                {
                    string json = File.ReadAllText(metaPath);
                    if (TryParseSpawnJson(json, out localPos, out yawRad))
                        return true;
                }
                catch (Exception ex)
                {
                    Debug.LogWarning($"[AiCar.TrackLoader] spawn meta parse failed: {ex.Message}");
                }
            }

            // Fallback: first two rows of centerline.csv (x_m,y_m → Unity x,z).
            string clPath = Path.Combine(mapsRoot, mapId, "occupancy", "centerline.csv");
            if (!File.Exists(clPath))
                return false;
            try
            {
                var pts = new List<Vector2>();
                foreach (var raw in File.ReadAllLines(clPath))
                {
                    string line = raw != null ? raw.Trim() : "";
                    if (line.Length == 0 || line.StartsWith("#", StringComparison.Ordinal))
                        continue;
                    var parts = line.Split(',');
                    if (parts.Length < 2) continue;
                    if (!float.TryParse(parts[0], System.Globalization.NumberStyles.Float,
                            System.Globalization.CultureInfo.InvariantCulture, out float x))
                        continue;
                    if (!float.TryParse(parts[1], System.Globalization.NumberStyles.Float,
                            System.Globalization.CultureInfo.InvariantCulture, out float z))
                        continue;
                    pts.Add(new Vector2(x, z));
                    if (pts.Count >= 4) break;
                }
                if (pts.Count == 0) return false;
                int i0 = Math.Min(2, pts.Count - 1);
                int i1 = Math.Min(i0 + 1, pts.Count - 1);
                Vector2 p0 = pts[i0];
                Vector2 p1 = pts[i1];
                localPos = new Vector3(p0.x, 0.05f, p0.y);
                yawRad = Mathf.Atan2(p1.x - p0.x, p1.y - p0.y);
                return true;
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[AiCar.TrackLoader] centerline spawn failed: {ex.Message}");
                return false;
            }
        }

        /// <summary>
        /// Minimal JSON scrape for meta.spawn — avoids a Newtonsoft dependency in player.
        /// Expects: "spawn": { "x": N, "y": N, "z": N, "yaw": N }
        /// </summary>
        static bool TryParseSpawnJson(string json, out Vector3 localPos, out float yawRad)
        {
            localPos = Vector3.zero;
            yawRad = 0f;
            if (string.IsNullOrEmpty(json)) return false;
            int key = json.IndexOf("\"spawn\"", StringComparison.OrdinalIgnoreCase);
            if (key < 0) return false;
            int brace = json.IndexOf('{', key);
            if (brace < 0) return false;
            int depth = 0;
            int end = -1;
            for (int i = brace; i < json.Length; i++)
            {
                char c = json[i];
                if (c == '{') depth++;
                else if (c == '}')
                {
                    depth--;
                    if (depth == 0) { end = i; break; }
                }
            }
            if (end < 0) return false;
            string block = json.Substring(brace, end - brace + 1);
            if (!TryJsonFloat(block, "x", out float x)) return false;
            if (!TryJsonFloat(block, "z", out float z)) return false;
            TryJsonFloat(block, "y", out float y);
            if (y < 0.01f) y = 0.05f;
            TryJsonFloat(block, "yaw", out yawRad);
            localPos = new Vector3(x, y, z);
            return true;
        }

        static bool TryJsonFloat(string block, string key, out float value)
        {
            value = 0f;
            string needle = "\"" + key + "\"";
            int i = block.IndexOf(needle, StringComparison.OrdinalIgnoreCase);
            if (i < 0) return false;
            int colon = block.IndexOf(':', i + needle.Length);
            if (colon < 0) return false;
            int start = colon + 1;
            while (start < block.Length && (block[start] == ' ' || block[start] == '\t'))
                start++;
            int stop = start;
            while (stop < block.Length && "0123456789+-.eE".IndexOf(block[stop]) >= 0)
                stop++;
            if (stop <= start) return false;
            return float.TryParse(
                block.Substring(start, stop - start),
                System.Globalization.NumberStyles.Float,
                System.Globalization.CultureInfo.InvariantCulture,
                out value);
        }

        static int MoveSpawnTransforms(Vector3 worldPos, Quaternion worldRot)
        {
            int n = 0;
            foreach (var t in Resources.FindObjectsOfTypeAll<Transform>())
            {
                if (t == null || !InLoadedScene(t.gameObject)) continue;
                string name = t.name ?? "";
                // Scene has "Spawn 1" / "Spawn 2" under Porto/Berlin Spawn Points.
                if (!name.StartsWith("Spawn", StringComparison.OrdinalIgnoreCase))
                    continue;
                if (name.IndexOf("Points", StringComparison.OrdinalIgnoreCase) >= 0)
                    continue; // parent folder
                t.position = worldPos;
                t.rotation = worldRot;
                n++;
            }
            return n;
        }

        public static int TeleportVehicles(Vector3 worldPos, Quaternion worldRot)
        {
            int n = 0;
            var seen = new HashSet<int>();

            // ResetManager owns Bridge soft-reset targets and caches init pose in
            // Start(). Must patch both live pose AND init_* arrays or V1 Reset
            // yanks the car back to the old F1TENTH grid.
            n += PatchResetManagers(worldPos, worldRot, seen);

            foreach (var mb in Resources.FindObjectsOfTypeAll<MonoBehaviour>())
            {
                if (mb == null || !InLoadedScene(mb.gameObject)) continue;
                if (mb.GetType().Name != "VehicleController") continue;

                GameObject target = null;
                var flags = System.Reflection.BindingFlags.Instance |
                            System.Reflection.BindingFlags.Public |
                            System.Reflection.BindingFlags.NonPublic;
                var fiVeh = mb.GetType().GetField("Vehicle", flags);
                if (fiVeh != null)
                    target = fiVeh.GetValue(mb) as GameObject;
                var fiRb = mb.GetType().GetField("VehicleRigidBody", flags);
                Rigidbody rb = fiRb != null ? fiRb.GetValue(mb) as Rigidbody : null;
                if (rb != null)
                    target = rb.gameObject;
                if (target == null)
                    target = mb.gameObject;
                if (!seen.Add(target.GetInstanceID())) continue;
                TeleportRigidbody(target, worldPos, worldRot);
                n++;
            }

            foreach (var lt in Resources.FindObjectsOfTypeAll<LapTimer>())
            {
                if (lt == null || !InLoadedScene(lt.gameObject)) continue;
                var rb = lt.GetComponent<Rigidbody>();
                if (rb == null) continue; // skip env roots without their own RB
                if (!seen.Add(rb.gameObject.GetInstanceID())) continue;
                TeleportRigidbody(rb.gameObject, worldPos, worldRot);
                n++;
            }
            return n;
        }

        static int PatchResetManagers(Vector3 worldPos, Quaternion worldRot, HashSet<int> seen)
        {
            int n = 0;
            var flags = System.Reflection.BindingFlags.Instance |
                        System.Reflection.BindingFlags.Public |
                        System.Reflection.BindingFlags.NonPublic;
            foreach (var mb in Resources.FindObjectsOfTypeAll<MonoBehaviour>())
            {
                if (mb == null || !InLoadedScene(mb.gameObject)) continue;
                if (mb.GetType().Name != "ResetManager") continue;

                var fiVehicles = mb.GetType().GetField("Vehicles", flags);
                var fiRbs = mb.GetType().GetField("VehicleRigidBodies", flags);
                var fiInitPos = mb.GetType().GetField("init_VehiclePositions", flags);
                var fiInitRot = mb.GetType().GetField("init_VehicleRotations", flags);

                var vehicles = fiVehicles != null ? fiVehicles.GetValue(mb) as Transform[] : null;
                var rbs = fiRbs != null ? fiRbs.GetValue(mb) as Rigidbody[] : null;
                var initPos = fiInitPos != null ? fiInitPos.GetValue(mb) as Vector3[] : null;
                var initRot = fiInitRot != null ? fiInitRot.GetValue(mb) as Quaternion[] : null;

                int count = 0;
                if (vehicles != null) count = vehicles.Length;
                else if (rbs != null) count = rbs.Length;
                if (count == 0) continue;

                if (initPos == null || initPos.Length != count)
                {
                    initPos = new Vector3[count];
                    if (fiInitPos != null) fiInitPos.SetValue(mb, initPos);
                }
                if (initRot == null || initRot.Length != count)
                {
                    initRot = new Quaternion[count];
                    if (fiInitRot != null) fiInitRot.SetValue(mb, initRot);
                }

                for (int i = 0; i < count; i++)
                {
                    Vector3 offset = worldPos;
                    if (count > 1)
                        offset += (worldRot * Vector3.right) * (i * 0.35f);
                    initPos[i] = offset;
                    initRot[i] = worldRot;

                    if (vehicles != null && i < vehicles.Length && vehicles[i] != null)
                    {
                        if (seen.Add(vehicles[i].gameObject.GetInstanceID()))
                        {
                            TeleportRigidbody(vehicles[i].gameObject, offset, worldRot);
                            n++;
                        }
                        else
                            vehicles[i].SetPositionAndRotation(offset, worldRot);
                    }
                    if (rbs != null && i < rbs.Length && rbs[i] != null)
                    {
                        rbs[i].velocity = Vector3.zero;
                        rbs[i].angularVelocity = Vector3.zero;
                        rbs[i].position = offset;
                        rbs[i].rotation = worldRot;
                        if (seen.Add(rbs[i].gameObject.GetInstanceID()))
                            n++;
                    }
                }
                Debug.Log(
                    $"[AiCar.TrackLoader] patched ResetManager '{GetPath(mb.transform)}' vehicles={count}");
            }
            return n;
        }

        static void TeleportRigidbody(GameObject go, Vector3 worldPos, Quaternion worldRot)
        {
            if (go == null) return;
            var rb = go.GetComponent<Rigidbody>();
            if (rb != null)
            {
                rb.velocity = Vector3.zero;
                rb.angularVelocity = Vector3.zero;
                rb.MovePosition(worldPos);
                rb.MoveRotation(worldRot);
                // Also hard-set — MovePosition alone can be ignored if RB is kinematic
                // or sleeping on some AutoDRIVE setups.
                rb.position = worldPos;
                rb.rotation = worldRot;
            }
            go.transform.SetPositionAndRotation(worldPos, worldRot);
        }
    }

    /// <summary>
    /// Keeps Spawn transforms correct briefly after load. Re-teleports vehicles
    /// only if Bridge / ResetManager yanks them far from the intended pose —
    /// never every FixedUpdate (that zeros velocity and freezes the car).
    /// </summary>
    public class SpawnKeeper : MonoBehaviour
    {
        const float YankDistanceM = 1.5f;

        Vector3 _pos;
        Quaternion _rot;
        float _until;
        bool _armed;

        public void Arm(Vector3 worldPos, Quaternion worldRot, float seconds)
        {
            _pos = worldPos;
            _rot = worldRot;
            _until = Time.unscaledTime + Mathf.Max(0.5f, seconds);
            _armed = true;
            enabled = true;
        }

        void FixedUpdate()
        {
            if (!_armed) return;
            if (Time.unscaledTime > _until)
            {
                _armed = false;
                enabled = false;
                return;
            }

            // Spawn markers must stay correct so V1 Reset uses the new map pose.
            foreach (var t in Resources.FindObjectsOfTypeAll<Transform>())
            {
                if (t == null || !t.gameObject.scene.IsValid()) continue;
                string name = t.name ?? "";
                if (!name.StartsWith("Spawn", StringComparison.OrdinalIgnoreCase)) continue;
                if (name.IndexOf("Points", StringComparison.OrdinalIgnoreCase) >= 0) continue;
                t.SetPositionAndRotation(_pos, _rot);
            }

            // Only yank-correct: if the car is still near spawn, let physics drive.
            if (VehicleFarFromSpawn(_pos, YankDistanceM))
                TrackLoader.TeleportVehicles(_pos, _rot);
        }

        static bool VehicleFarFromSpawn(Vector3 spawn, float thresholdM)
        {
            float threshSq = thresholdM * thresholdM;
            bool any = false;
            foreach (var mb in Resources.FindObjectsOfTypeAll<MonoBehaviour>())
            {
                if (mb == null || mb.GetType().Name != "VehicleController") continue;
                if (!mb.gameObject.scene.IsValid() || !mb.gameObject.scene.isLoaded) continue;
                Vector3 p = mb.transform.position;
                var flags = System.Reflection.BindingFlags.Instance |
                            System.Reflection.BindingFlags.Public |
                            System.Reflection.BindingFlags.NonPublic;
                var fiRb = mb.GetType().GetField("VehicleRigidBody", flags);
                if (fiRb != null)
                {
                    var rb = fiRb.GetValue(mb) as Rigidbody;
                    if (rb != null) p = rb.position;
                }
                any = true;
                float dx = p.x - spawn.x;
                float dz = p.z - spawn.z;
                if (dx * dx + dz * dz > threshSq)
                    return true;
            }
            // No vehicle found yet — allow one corrective teleport via caller.
            return !any;
        }
    }
}
