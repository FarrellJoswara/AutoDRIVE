// Auto-connect AutoDRIVE Socket.IO when -ip/-port (or batchmode) is present.
// F1TENTH.unity has SocketConnection (UI Connect button) but NO CLIManager, so
// stock -ip/-port only work if something activates Socket + SocketIO GameObjects.
// SocketIOComponent.OnEnable reads InputFields then Connect() when autoConnect.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Reflection;
using UnityEngine;
using UnityEngine.UI;

namespace AiCar
{
    public class ForceConnect : MonoBehaviour
    {
        const int MaxAttempts = 90; // ~1.5s at 60fps, or ~3s with yields
        const float RetryInterval = 0.25f;

        string _cliIp;
        string _cliPort;
        bool _wantConnect;
        bool _done;
        int _attempts;

        void Awake()
        {
            ParseCli();
            _wantConnect = ShouldForceConnect();
            if (!_wantConnect)
            {
                Debug.Log("[AiCar.ForceConnect] idle (no -ip/-port and not batchmode)");
                enabled = false;
                return;
            }

            Debug.Log(
                $"[AiCar.ForceConnect] armed ip={_cliIp ?? "(default)"} port={_cliPort ?? "(default)"} " +
                $"batchMode={Application.isBatchMode}");
        }

        IEnumerator Start()
        {
            if (!_wantConnect)
                yield break;

            // Let scene Awakes finish; Socket / SocketIO start inactive.
            yield return null;
            yield return null;

            while (!_done && _attempts < MaxAttempts)
            {
                _attempts++;
                // Don't let Layer 1 start stepping until TrackLoader has applied
                // the selected map spawn and populated Socket.ResetManagers. A
                // reset sent before that setup is silently ignored by AutoDRIVE.
                if (!TrackLoader.IsReady)
                {
                    yield return new WaitForSecondsRealtime(RetryInterval);
                    continue;
                }
                if (TryForceConnect())
                {
                    _done = true;
                    yield break;
                }
                yield return new WaitForSecondsRealtime(RetryInterval);
            }

            if (!_done)
            {
                if (!TrackLoader.IsReady)
                {
                    Debug.LogError(
                        $"[AiCar.ForceConnect] TrackLoader map/reset initialization did not finish " +
                        $"after {_attempts} attempts; Bridge connection was withheld");
                    yield break;
                }
                Debug.LogError(
                    $"[AiCar.ForceConnect] gave up after {_attempts} attempts — " +
                    "SocketIOComponent / Socket GameObjects not found or Connect failed");
            }
        }

        void ParseCli()
        {
            string[] args = Environment.GetCommandLineArgs();
            for (int i = 0; i < args.Length - 1; i++)
            {
                string a = args[i];
                if (string.Equals(a, "-ip", StringComparison.OrdinalIgnoreCase) ||
                    string.Equals(a, "--ip", StringComparison.OrdinalIgnoreCase))
                    _cliIp = args[i + 1];
                else if (string.Equals(a, "-port", StringComparison.OrdinalIgnoreCase) ||
                         string.Equals(a, "--port", StringComparison.OrdinalIgnoreCase))
                    _cliPort = args[i + 1];
            }
        }

        bool ShouldForceConnect()
        {
            if (!string.IsNullOrEmpty(_cliIp) || !string.IsNullOrEmpty(_cliPort))
                return true;
            return Application.isBatchMode;
        }

        bool TryForceConnect()
        {
            // Prefer typed lookup; fall back to reflection for renamed/obfuscated builds.
            var socketIos = FindBehavioursByTypeName("SocketIOComponent");
            var cliManagers = FindBehavioursByTypeName("CLIManager");
            var socketConnections = FindBehavioursByTypeName("SocketConnection");

            ApplyIpPortToInputFields(cliManagers);
            ApplyIpPortToInputFields(socketIos);
            ApplyIpPortToStringFields(socketIos, "IPAddress", "PortNumber");

            if (AnySocketIoConnected(socketIos))
            {
                Debug.Log("[AiCar.ForceConnect] already connected");
                return true;
            }

            // Stock CLIManager.Start activates Socket/SocketIO only in batchmode —
            // F1TENTH often has no CLIManager at all, so do it ourselves.
            bool activated = ActivateSocketObjects(cliManagers, socketConnections, socketIos);

            if (!activated && socketIos.Count == 0)
            {
                if (_attempts == 1 || _attempts % 8 == 0)
                    Debug.Log("[AiCar.ForceConnect] waiting for SocketIOComponent...");
                return false;
            }

            // One connect path only: bounce so OnEnable rebuilds ws URL + autoConnect.
            // Do NOT also InvokeConnect — that races WebSocketSharp ("already established").
            bool bounced = false;
            foreach (var sio in socketIos)
            {
                if (sio == null) continue;
                var go = sio.gameObject;
                if (go == null) continue;
                EnsureActiveHierarchy(go);
                if (!go.activeInHierarchy) continue;
                go.SetActive(false);
                go.SetActive(true);
                bounced = true;
                Debug.Log($"[AiCar.ForceConnect] bounced SocketIO GameObject '{GetPath(go.transform)}'");
            }

            SuppressSocketErrorFormatting(socketIos);

            if (!bounced)
            {
                // Fallback when bounce isn't possible (already active / no GO).
                InvokeConnect(socketIos);
                InvokeToggleConnect(socketConnections);
            }

            string ip = _cliIp ?? "(component default)";
            string port = _cliPort ?? "(component default)";
            Debug.Log(
                $"[AiCar.ForceConnect] connect attempted → {ip}:{port} " +
                $"(SocketIO={socketIos.Count} CLIManager={cliManagers.Count} " +
                $"SocketConnection={socketConnections.Count} bounced={bounced})");

            // Treat as success once SocketIO exists — autoConnect may finish async.
            // Caller stops retrying; SocketIO handles its own reconnect.
            return socketIos.Count > 0 || activated;
        }

        static void SuppressSocketErrorFormatting(List<MonoBehaviour> socketIos)
        {
            // WebSocketSharp formats connection-refused errors on its worker
            // thread. Unity's Linux player crashes in Mono's timezone formatter
            // while doing that work, so keep the network library's logger quiet.
            // Connection state and bridge failures remain observable through
            // the existing ForceConnect and Python bridge status logs.
            var flags = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;
            foreach (var sio in socketIos)
            {
                if (sio == null) continue;
                try
                {
                    var socketProperty = sio.GetType().GetProperty("socket", flags);
                    var socket = socketProperty?.GetValue(sio, null) as WebSocketSharp.WebSocket;
                    if (socket?.Log != null)
                        socket.Log.Output = (data, path) => { };
                }
                catch (Exception ex)
                {
                    Debug.LogWarning("[AiCar.ForceConnect] could not configure Socket.IO logging: " + ex.Message);
                }
            }
        }

        static bool AnySocketIoConnected(List<MonoBehaviour> socketIos)
        {
            foreach (var sio in socketIos)
            {
                if (sio == null) continue;
                var flags = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;
                foreach (var name in new[] { "IsConnected", "connected", "autoConnect" })
                {
                    var p = sio.GetType().GetProperty(name, flags);
                    if (p != null && p.PropertyType == typeof(bool) && name != "autoConnect")
                    {
                        try
                        {
                            if ((bool)p.GetValue(sio, null)) return true;
                        }
                        catch { /* ignore */ }
                    }
                    var f = sio.GetType().GetField(name, flags);
                    if (f != null && f.FieldType == typeof(bool) && name != "autoConnect")
                    {
                        try
                        {
                            if ((bool)f.GetValue(sio)) return true;
                        }
                        catch { /* ignore */ }
                    }
                }
                // websocket-sharp WebSocket often exposed as `ws` / `socket`
                foreach (var fname in new[] { "ws", "socket", "webSocket" })
                {
                    var f = sio.GetType().GetField(fname, flags);
                    if (f == null) continue;
                    var ws = f.GetValue(sio);
                    if (ws == null) continue;
                    var rp = ws.GetType().GetProperty("IsAlive", flags) ??
                             ws.GetType().GetProperty("IsConnected", flags);
                    if (rp != null && rp.PropertyType == typeof(bool))
                    {
                        try
                        {
                            if ((bool)rp.GetValue(ws, null)) return true;
                        }
                        catch { /* ignore */ }
                    }
                }
            }
            return false;
        }

        void ApplyIpPortToInputFields(List<MonoBehaviour> behaviours)
        {
            foreach (var b in behaviours)
            {
                if (b == null) continue;
                SetInputField(b, "IP", _cliIp);
                SetInputField(b, "Port", _cliPort);
            }
        }

        void ApplyIpPortToStringFields(List<MonoBehaviour> behaviours, string ipField, string portField)
        {
            foreach (var b in behaviours)
            {
                if (b == null) continue;
                if (!string.IsNullOrEmpty(_cliIp))
                    SetStringField(b, ipField, _cliIp);
                if (!string.IsNullOrEmpty(_cliPort))
                    SetStringField(b, portField, _cliPort);
            }
        }

        static void SetInputField(object target, string fieldName, string value)
        {
            if (string.IsNullOrEmpty(value)) return;
            var flags = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;
            var fi = target.GetType().GetField(fieldName, flags);
            if (fi == null) return;
            var obj = fi.GetValue(target);
            if (obj == null) return;
            if (obj is InputField input)
            {
                input.text = value;
                return;
            }
            // Reflection fallback if UI assembly types differ
            var textProp = obj.GetType().GetProperty("text", flags);
            if (textProp != null && textProp.CanWrite)
                textProp.SetValue(obj, value, null);
        }

        static void SetStringField(object target, string fieldName, string value)
        {
            var flags = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;
            var fi = target.GetType().GetField(fieldName, flags);
            if (fi != null && fi.FieldType == typeof(string))
                fi.SetValue(target, value);
        }

        bool ActivateSocketObjects(
            List<MonoBehaviour> cliManagers,
            List<MonoBehaviour> socketConnections,
            List<MonoBehaviour> socketIos)
        {
            bool any = false;

            foreach (var list in new[] { cliManagers, socketConnections })
            {
                foreach (var b in list)
                {
                    if (b == null) continue;
                    any |= ActivateGoField(b, "Socket");
                    any |= ActivateGoField(b, "SocketIO");
                }
            }

            foreach (var sio in socketIos)
            {
                if (sio == null) continue;
                any |= EnsureActiveHierarchy(sio.gameObject);
            }

            // Named fallbacks used by AutoDRIVE scenes
            foreach (var name in new[] { "Socket", "SocketIO" })
            {
                var go = GameObject.Find(name);
                if (go == null)
                {
                    // Find even inactive
                    foreach (var t in Resources.FindObjectsOfTypeAll<Transform>())
                    {
                        if (t == null || t.name != name) continue;
                        if (!t.gameObject.scene.IsValid() || !t.gameObject.scene.isLoaded) continue;
                        go = t.gameObject;
                        break;
                    }
                }
                if (go != null)
                    any |= EnsureActiveHierarchy(go);
            }

            return any;
        }

        static bool ActivateGoField(object target, string fieldName)
        {
            var flags = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;
            var fi = target.GetType().GetField(fieldName, flags);
            if (fi == null) return false;
            var go = fi.GetValue(target) as GameObject;
            if (go == null) return false;
            return EnsureActiveHierarchy(go);
        }

        static bool EnsureActiveHierarchy(GameObject go)
        {
            if (go == null) return false;
            bool changed = false;
            // Activate parents first so child OnEnable runs with hierarchy active.
            var stack = new List<Transform>();
            for (var t = go.transform; t != null; t = t.parent)
                stack.Add(t);
            for (int i = stack.Count - 1; i >= 0; i--)
            {
                var g = stack[i].gameObject;
                if (!g.activeSelf)
                {
                    g.SetActive(true);
                    changed = true;
                    Debug.Log($"[AiCar.ForceConnect] SetActive(true) '{GetPath(stack[i])}'");
                }
            }
            return changed;
        }

        static bool InvokeConnect(List<MonoBehaviour> socketIos)
        {
            bool any = false;
            string[] names = { "Connect", "OnConnect", "ConnectToServer" };
            foreach (var sio in socketIos)
            {
                if (sio == null) continue;
                foreach (var name in names)
                {
                    if (InvokeNoArg(sio, name))
                    {
                        Debug.Log($"[AiCar.ForceConnect] invoked {sio.GetType().Name}.{name}()");
                        any = true;
                        break;
                    }
                }
            }
            return any;
        }

        static bool InvokeToggleConnect(List<MonoBehaviour> socketConnections)
        {
            bool any = false;
            foreach (var sc in socketConnections)
            {
                if (sc == null) continue;
                // Only toggle if SocketIO still inactive — ToggleSocketConnection flips state.
                var flags = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;
                var fio = sc.GetType().GetField("SocketIO", flags);
                var socketIoGo = fio != null ? fio.GetValue(sc) as GameObject : null;
                if (socketIoGo != null && socketIoGo.activeInHierarchy)
                    continue;
                if (InvokeNoArg(sc, "ToggleSocketConnection") ||
                    InvokeNoArg(sc, "Connect") ||
                    InvokeNoArg(sc, "OnConnect") ||
                    InvokeNoArg(sc, "ConnectToServer"))
                {
                    Debug.Log($"[AiCar.ForceConnect] invoked {sc.GetType().Name} connect/toggle");
                    any = true;
                }
            }
            return any;
        }

        static bool InvokeNoArg(object target, string methodName)
        {
            var flags = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;
            var mi = target.GetType().GetMethod(methodName, flags, null, Type.EmptyTypes, null);
            if (mi == null) return false;
            try
            {
                mi.Invoke(target, null);
                return true;
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[AiCar.ForceConnect] {methodName} threw: {ex.InnerException?.Message ?? ex.Message}");
                return false;
            }
        }

        static List<MonoBehaviour> FindBehavioursByTypeName(string typeName)
        {
            var list = new List<MonoBehaviour>();
            foreach (var mb in Resources.FindObjectsOfTypeAll<MonoBehaviour>())
            {
                if (mb == null) continue;
                if (!mb.gameObject.scene.IsValid() || !mb.gameObject.scene.isLoaded)
                    continue;
                if (mb.GetType().Name == typeName)
                    list.Add(mb);
            }
            return list;
        }

        static string GetPath(Transform t)
        {
            if (t == null) return "(null)";
            var parts = new List<string>();
            for (var p = t; p != null; p = p.parent)
                parts.Add(p.name);
            parts.Reverse();
            return string.Join("/", parts);
        }
    }
}
