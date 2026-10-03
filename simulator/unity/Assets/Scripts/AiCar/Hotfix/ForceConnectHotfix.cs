// Standalone Mono hot-patch (Windows player only): drop compiled DLL into
// AutoDRIVE Simulator_Data/Managed and register in ScriptingAssemblies.json +
// RuntimeInitializeOnLoads.json. Prefer a full player rebuild that includes
// AiCar.ForceConnect instead — see simulator/unity/README.md.
//
// Compiles without Unity Editor against player Managed/*.dll refs:
//   csc /target:library /out:AiCar.ForceConnect.dll ForceConnectHotfix.cs
//     /r:UnityEngine.CoreModule.dll /r:UnityEngine.dll /r:UnityEngine.UI.dll
//     /r:UnityEngine.IMGUIModule.dll (paths under *_Data/Managed)

using System;
using System.Collections;
using System.Collections.Generic;
using System.Reflection;
using UnityEngine;

namespace AiCar.Hotfix
{
    public class ForceConnectHotfixBootstrap
    {
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        static void Boot()
        {
            // Full player builds already compile AiCar.ForceConnect into Assembly-CSharp.
            // Skip this hot-fix path so we don't double-Connect and crash headless.
            if (HasFullForceConnect())
            {
                Debug.Log("[AiCar.Hotfix.ForceConnect] skip — AiCar.ForceConnect already in player");
                return;
            }

            var go = new GameObject("AiCarForceConnectHotfix");
            UnityEngine.Object.DontDestroyOnLoad(go);
            go.AddComponent<ForceConnectHotfixRunner>();
            Debug.Log("[AiCar.Hotfix.ForceConnect] boot");
        }

        static bool HasFullForceConnect()
        {
            foreach (var asm in AppDomain.CurrentDomain.GetAssemblies())
            {
                Type t = null;
                try { t = asm.GetType("AiCar.ForceConnect", false); }
                catch { /* ignore */ }
                if (t != null && typeof(MonoBehaviour).IsAssignableFrom(t))
                    return true;
            }
            return false;
        }
    }

    public class ForceConnectHotfixRunner : MonoBehaviour
    {
        string _cliIp;
        string _cliPort;
        bool _want;
        bool _done;
        int _attempts;

        IEnumerator Start()
        {
            ParseCli();
            _want = !string.IsNullOrEmpty(_cliIp) || !string.IsNullOrEmpty(_cliPort) ||
                    Application.isBatchMode;
            if (!_want)
            {
                Debug.Log("[AiCar.Hotfix.ForceConnect] idle");
                yield break;
            }

            Debug.Log("[AiCar.Hotfix.ForceConnect] armed ip=" + (_cliIp ?? "?") +
                      " port=" + (_cliPort ?? "?"));
            SilenceLapTimersIfBatch();
            yield return null;
            yield return null;
            SilenceLapTimersIfBatch();

            while (!_done && _attempts < 40)
            {
                _attempts++;
                if (TryConnect())
                {
                    _done = true;
                    SilenceLapTimersIfBatch();
                    yield break;
                }
                yield return new WaitForSecondsRealtime(0.25f);
            }
            if (!_done)
                Debug.LogError("[AiCar.Hotfix.ForceConnect] gave up");
        }

        static void SilenceLapTimersIfBatch()
        {
            if (!Application.isBatchMode) return;
            int n = 0;
            foreach (var mb in Resources.FindObjectsOfTypeAll<MonoBehaviour>())
            {
                if (mb == null || mb.GetType().Name != "LapTimer") continue;
                if (!mb.gameObject.scene.IsValid() || !mb.enabled) continue;
                mb.enabled = false;
                n++;
            }
            if (n > 0)
                Debug.Log("[AiCar.Hotfix.ForceConnect] disabled LapTimers=" + n + " (batchmode)");
        }

        void ParseCli()
        {
            string[] args = Environment.GetCommandLineArgs();
            for (int i = 0; i < args.Length - 1; i++)
            {
                string a = args[i];
                if (string.Equals(a, "-ip", StringComparison.OrdinalIgnoreCase))
                    _cliIp = args[i + 1];
                else if (string.Equals(a, "-port", StringComparison.OrdinalIgnoreCase))
                    _cliPort = args[i + 1];
            }
        }

        bool TryConnect()
        {
            var sios = FindByName("SocketIOComponent");
            var scs = FindByName("SocketConnection");
            foreach (var b in sios)
            {
                SetInput(b, "IP", _cliIp);
                SetInput(b, "Port", _cliPort);
                SetString(b, "IPAddress", _cliIp);
                SetString(b, "PortNumber", _cliPort);
            }

            foreach (var name in new[] { "Socket", "SocketIO" })
            {
                foreach (var t in Resources.FindObjectsOfTypeAll<Transform>())
                {
                    if (t == null || t.name != name) continue;
                    if (!t.gameObject.scene.IsValid()) continue;
                    ActivateChain(t.gameObject);
                }
            }

            foreach (var sc in scs)
            {
                var fio = sc.GetType().GetField("SocketIO",
                    BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
                var go = fio != null ? fio.GetValue(sc) as GameObject : null;
                if (go != null && !go.activeInHierarchy)
                    Invoke(sc, "ToggleSocketConnection");
            }

            foreach (var sio in sios)
            {
                if (sio == null) continue;
                ActivateChain(sio.gameObject);
                sio.gameObject.SetActive(false);
                sio.gameObject.SetActive(true);
                Invoke(sio, "Connect");
            }

            if (sios.Count == 0) return false;
            Debug.Log("[AiCar.Hotfix.ForceConnect] connect attempted SocketIO=" + sios.Count);
            return true;
        }

        static void ActivateChain(GameObject go)
        {
            var stack = new List<Transform>();
            for (var t = go.transform; t != null; t = t.parent) stack.Add(t);
            for (int i = stack.Count - 1; i >= 0; i--)
                if (!stack[i].gameObject.activeSelf)
                    stack[i].gameObject.SetActive(true);
        }

        static void SetInput(object target, string field, string value)
        {
            if (string.IsNullOrEmpty(value) || target == null) return;
            var fi = target.GetType().GetField(field,
                BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
            if (fi == null) return;
            var obj = fi.GetValue(target);
            if (obj == null) return;
            var tp = obj.GetType().GetProperty("text");
            if (tp != null && tp.CanWrite) tp.SetValue(obj, value, null);
        }

        static void SetString(object target, string field, string value)
        {
            if (string.IsNullOrEmpty(value) || target == null) return;
            var fi = target.GetType().GetField(field,
                BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
            if (fi != null && fi.FieldType == typeof(string))
                fi.SetValue(target, value);
        }

        static void Invoke(object target, string method)
        {
            if (target == null) return;
            var mi = target.GetType().GetMethod(method,
                BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic,
                null, Type.EmptyTypes, null);
            if (mi == null) return;
            try { mi.Invoke(target, null); }
            catch (Exception ex)
            {
                Debug.LogWarning("[AiCar.Hotfix.ForceConnect] " + method + ": " +
                                 (ex.InnerException != null ? ex.InnerException.Message : ex.Message));
            }
        }

        static List<MonoBehaviour> FindByName(string typeName)
        {
            var list = new List<MonoBehaviour>();
            foreach (var mb in Resources.FindObjectsOfTypeAll<MonoBehaviour>())
            {
                if (mb == null || !mb.gameObject.scene.IsValid()) continue;
                if (mb.GetType().Name == typeName) list.Add(mb);
            }
            return list;
        }
    }
}
