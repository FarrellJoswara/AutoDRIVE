// Creates TrackLoader after the scene loads so baked tracks (Porto Track, etc.)
// exist when DisableDefaultTrack runs. BeforeSceneLoad only applies CLI map id.

using UnityEngine;

namespace AiCar
{
    public static class TrackLoaderBootstrap
    {
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.BeforeSceneLoad)]
        static void ApplyCliEarly()
        {
            MapConfig.ApplyCommandLine();
        }

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        static void Boot()
        {
            MapConfig.ApplyCommandLine();

            // Always ensure ForceConnect exists (even if TrackLoader already present).
            if (Object.FindObjectOfType<ForceConnect>() == null)
            {
                var goFc = new GameObject("AiCarForceConnect");
                Object.DontDestroyOnLoad(goFc);
                goFc.AddComponent<ForceConnect>();
            }

            if (Object.FindObjectOfType<AiCarSimulationGate>() == null)
            {
                var goGate = new GameObject("AiCarSimulationGate");
                Object.DontDestroyOnLoad(goGate);
                goGate.AddComponent<AiCarSimulationGate>();
            }

            if (Object.FindObjectOfType<TrackLoader>() != null)
                return;

            var go = new GameObject("AiCarTrackLoader");
            Object.DontDestroyOnLoad(go);
            go.AddComponent<TrackLoader>();
        }
    }
}
