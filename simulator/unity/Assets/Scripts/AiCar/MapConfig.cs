// Shared map-id resolution for TrackLoader / CLIManager patch.
// Call MapConfig.ApplyCommandLine() early; TrackLoader uses ResolveMapId().

using System;
using System.IO;
using UnityEngine;

namespace AiCar
{
    public static class MapConfig
    {
        static string _cliMapId;

        /// <summary>Call once from CLIManager after parsing argv.</summary>
        public static void ApplyCommandLine()
        {
            string[] args = Environment.GetCommandLineArgs();
            for (int i = 0; i < args.Length - 1; i++)
            {
                if (args[i] == "-map-id" || args[i] == "--map-id")
                {
                    _cliMapId = args[i + 1];
                    Debug.Log($"[AiCar.MapConfig] CLI -map-id={_cliMapId}");
                    return;
                }
            }
        }

        public static void SetCliMapId(string id) => _cliMapId = id;

        public static string ResolveMapId()
        {
            if (!string.IsNullOrEmpty(_cliMapId) &&
                !string.Equals(_cliMapId, "none", StringComparison.OrdinalIgnoreCase))
                return _cliMapId.Trim();

            string env = Environment.GetEnvironmentVariable("AICAR_MAP_ID");
            if (!string.IsNullOrEmpty(env) &&
                !string.Equals(env, "none", StringComparison.OrdinalIgnoreCase))
                return env.Trim();

            string mapsRoot = Environment.GetEnvironmentVariable("AICAR_MAPS_DIR");
            if (string.IsNullOrEmpty(mapsRoot))
            {
                string data = Application.dataPath;
                string simRoot = Directory.GetParent(data)?.FullName ?? data;
                mapsRoot = Path.Combine(simRoot, "maps");
            }

            string active = Path.Combine(mapsRoot, ".active_map.json");
            if (!File.Exists(active))
                return null;

            try
            {
                string text = File.ReadAllText(active);
                // Prefer "id": "porto". Unquoted JSON null must not fall into
                // the next key ("activated_at") — that wrongly yields map id
                // "activated_at" and breaks mesh load / boot.
                const string key = "\"id\"";
                int k = text.IndexOf(key, StringComparison.Ordinal);
                if (k < 0)
                    return null;
                int colon = text.IndexOf(':', k + key.Length);
                if (colon < 0)
                    return null;
                int i = colon + 1;
                while (i < text.Length && char.IsWhiteSpace(text[i]))
                    i++;
                if (i >= text.Length || text[i] != '"')
                    return null;
                int q1 = i;
                int q2 = text.IndexOf('"', q1 + 1);
                if (q2 < 0)
                    return null;
                string id = text.Substring(q1 + 1, q2 - q1 - 1).Trim();
                if (string.IsNullOrEmpty(id) ||
                    string.Equals(id, "none", StringComparison.OrdinalIgnoreCase) ||
                    string.Equals(id, "null", StringComparison.OrdinalIgnoreCase))
                    return null;
                return id;
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[AiCar.MapConfig] failed reading {active}: {ex.Message}");
                return null;
            }
        }
    }
}
