// Batch build entry points for TrackLoader-enabled AutoDRIVE player.
// Unity.exe -batchmode -quit -projectPath ... -executeMethod AiCar.Editor.AiCarPlayerBuild.BuildAll

#if UNITY_EDITOR
using System;
using System.IO;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEngine;

namespace AiCar.Editor
{
    public static class AiCarPlayerBuild
    {
        const string ScenePath = "Assets/Scenes/F1TENTH.unity";

        static string SimulatorRoot
        {
            get
            {
                var project = Directory.GetParent(Application.dataPath)?.FullName; // AutoDRIVE
                var unityDir = Directory.GetParent(project)?.FullName; // unity
                var simulator = Directory.GetParent(unityDir)?.FullName; // simulator
                return simulator ?? project;
            }
        }

        public static void BuildAll()
        {
            BuildWindows();
            BuildLinux();
            WriteTrackloaderMarker();
            Debug.Log("[AiCar] BuildAll finished");
        }

        public static void BuildLinuxOnly()
        {
            BuildLinux();
            WriteTrackloaderMarker();
            Debug.Log("[AiCar] BuildLinuxOnly finished");
        }

        public static void BuildLinuxStagingOnly()
        {
            BuildLinux(publish: false, stagingName: "linux-cpu-experiment");
            Debug.Log("[AiCar] Linux player staged without replacing the active simulator");
        }

        public static void BuildLinuxFixedStepStagingOnly()
        {
            var staging = Path.Combine(SimulatorRoot, "_build", "linux-fixed-step-experiment");
            BuildLinux(publish: false, stagingName: "linux-fixed-step-experiment");
            SetJobWorkerCount(staging, 1);
            Debug.Log("[AiCar] Linux fixed-step player staged without replacing the active simulator");
        }

        public static void BuildLinuxFixedStepLidarStagingOnly()
        {
            var staging = Path.Combine(SimulatorRoot, "_build", "linux-fixed-step-lidar-experiment");
            BuildLinux(publish: false, stagingName: "linux-fixed-step-lidar-experiment");
            SetJobWorkerCount(staging, 1);
            Debug.Log("[AiCar] Linux fixed-step LiDAR candidate staged without replacing the active simulator");
        }

        public static void BuildLinuxFixedStepAutoSyncOffStagingOnly()
        {
            var project = Directory.GetParent(Application.dataPath)?.FullName;
            var settingsPath = Path.Combine(project ?? throw new InvalidOperationException(
                "Could not locate the Unity project root"), "ProjectSettings", "DynamicsManager.asset");
            var originalSettings = File.ReadAllText(settingsPath);
            const string enabled = "m_AutoSyncTransforms: 1";
            const string disabled = "m_AutoSyncTransforms: 0";
            if (originalSettings.IndexOf(enabled, StringComparison.Ordinal) < 0 ||
                originalSettings.IndexOf(enabled, StringComparison.Ordinal) !=
                originalSettings.LastIndexOf(enabled, StringComparison.Ordinal))
                throw new InvalidOperationException(
                    "Expected exactly one enabled m_AutoSyncTransforms setting in " + settingsPath);

            try
            {
                File.WriteAllText(settingsPath,
                    originalSettings.Replace(enabled, disabled));
                var staging = Path.Combine(SimulatorRoot, "_build", "linux-fixed-step-autosync-off-experiment");
                BuildLinux(publish: false, stagingName: "linux-fixed-step-autosync-off-experiment");
                SetJobWorkerCount(staging, 1);
                Debug.Log("[AiCar] Linux fixed-step Auto Sync Transforms-off candidate staged without replacing the active simulator");
            }
            finally
            {
                File.WriteAllText(settingsPath, originalSettings);
            }
        }

        public static void BuildLinuxFixedStepExplicitSyncStagingOnly()
        {
            var project = Directory.GetParent(Application.dataPath)?.FullName;
            var settingsPath = Path.Combine(project ?? throw new InvalidOperationException(
                "Could not locate the Unity project root"), "ProjectSettings", "DynamicsManager.asset");
            var originalSettings = File.ReadAllText(settingsPath);
            const string enabled = "m_AutoSyncTransforms: 1";
            const string disabled = "m_AutoSyncTransforms: 0";
            if (originalSettings.IndexOf(enabled, StringComparison.Ordinal) < 0 ||
                originalSettings.IndexOf(enabled, StringComparison.Ordinal) !=
                originalSettings.LastIndexOf(enabled, StringComparison.Ordinal))
                throw new InvalidOperationException(
                    "Expected exactly one enabled m_AutoSyncTransforms setting in " + settingsPath);

            try
            {
                File.WriteAllText(settingsPath, originalSettings.Replace(enabled, disabled));
                var staging = Path.Combine(SimulatorRoot, "_build", "linux-fixed-step-explicit-sync-experiment");
                BuildLinux(publish: false, stagingName: "linux-fixed-step-explicit-sync-experiment");
                SetJobWorkerCount(staging, 1);
                Debug.Log("[AiCar] Linux fixed-step explicit-sync candidate staged without replacing the active simulator");
            }
            finally
            {
                File.WriteAllText(settingsPath, originalSettings);
            }
        }

        public static void BuildLinuxFixedStepLidarBatchStagingOnly()
        {
            var staging = Path.Combine(SimulatorRoot, "_build", "linux-fixed-step-lidar-batch-experiment");
            BuildLinux(publish: false, stagingName: "linux-fixed-step-lidar-batch-experiment");
            SetJobWorkerCount(staging, 1);
            Debug.Log("[AiCar] Linux fixed-step LiDAR batch candidate staged without replacing the active simulator");
        }

        public static void BuildLinuxFixedStepProfilerStagingOnly()
        {
            EnsureSceneInBuild();
            PlayerSettings.SetScriptingBackend(BuildTargetGroup.Standalone, ScriptingImplementation.IL2CPP);
            var staging = Path.Combine(SimulatorRoot, "_build", "linux-fixed-step-profiler-experiment");
            WipeDir(staging);
            Directory.CreateDirectory(staging);
            var options = new BuildPlayerOptions
            {
                scenes = new[] { ScenePath },
                locationPathName = Path.Combine(staging, "AutoDRIVE Simulator.x86_64"),
                target = BuildTarget.StandaloneLinux64,
                options = BuildOptions.Development,
            };
            var report = BuildPipeline.BuildPlayer(options);
            FailIfBad(report, "Linux profiler Development");
            SetJobWorkerCount(staging, 1);
            Debug.Log("[AiCar] Linux fixed-step profiler Development player staged without replacing the active simulator");
        }

        public static void BuildLinuxFixedStepServerStagingOnly()
        {
            var staging = Path.Combine(SimulatorRoot, "_build", "linux-fixed-step-server-experiment");
            BuildLinux(publish: false, stagingName: "linux-fixed-step-server-experiment", dedicatedServer: true);
            SetJobWorkerCount(staging, 1);
            Debug.Log("[AiCar] Linux fixed-step Dedicated Server player staged without replacing the active simulator");
        }

        static void SetJobWorkerCount(string staging, int workerCount)
        {
            var bootConfig = Path.Combine(staging, "AutoDRIVE Simulator_Data", "boot.config");
            if (!File.Exists(bootConfig))
                throw new FileNotFoundException("Linux player boot.config not found", bootConfig);

            var lines = new System.Collections.Generic.List<string>(File.ReadAllLines(bootConfig));
            var setting = "job-worker-count=" + workerCount;
            var found = false;
            for (var i = 0; i < lines.Count; i++)
            {
                if (!lines[i].StartsWith("job-worker-count=", StringComparison.Ordinal)) continue;
                lines[i] = setting;
                found = true;
            }
            if (!found) lines.Add(setting);
            File.WriteAllLines(bootConfig, lines);
            Debug.Log("[AiCar] Set fixed-step Unity job workers to " + workerCount);
        }

        public static void PublishStagedLinuxOnly()
        {
            var staging = Path.Combine(SimulatorRoot, "_build", "linux");
            var executable = Path.Combine(staging, "AutoDRIVE Simulator.x86_64");
            if (!File.Exists(executable))
                throw new FileNotFoundException("staged Linux player not found", executable);

            PublishLinux(staging);
            WriteTrackloaderMarker();
            Debug.Log("[AiCar] staged Linux player published");
        }

        public static void BuildWindowsOnly()
        {
            BuildWindows();
            WriteTrackloaderMarker();
            Debug.Log("[AiCar] BuildWindowsOnly finished");
        }

        public static void BuildWindows()
        {
            EnsureSceneInBuild();
            // Mono avoids requiring Visual Studio C++ / Windows SDK for IL2CPP.
            PlayerSettings.SetScriptingBackend(BuildTargetGroup.Standalone, ScriptingImplementation.Mono2x);
            // Must be an empty dir — Unity refuses overwrite across scripting backends.
            var staging = Path.Combine(SimulatorRoot, "_build", "windows");
            WipeDir(staging);
            Directory.CreateDirectory(staging);
            var exe = Path.Combine(staging, "AutoDRIVE Simulator.exe");
            var opts = new BuildPlayerOptions
            {
                scenes = new[] { ScenePath },
                locationPathName = exe,
                target = BuildTarget.StandaloneWindows64,
                options = BuildOptions.None,
            };
            var report = BuildPipeline.BuildPlayer(opts);
            FailIfBad(report, "Windows");
            PublishWindows(staging);
        }

        public static void BuildLinux()
        {
            BuildLinux(publish: true);
        }

        static void BuildLinux(bool publish, string stagingName = "linux", bool dedicatedServer = false)
        {
            EnsureSceneInBuild();
            // Linux Mono nondevelopment player is often missing on Windows Editor installs;
            // go straight to IL2CPP. If postprocess NREs (incomplete sysroot), retry Development.
            var scriptingTarget = dedicatedServer
                ? UnityEditor.Build.NamedBuildTarget.Server
                : UnityEditor.Build.NamedBuildTarget.Standalone;
            PlayerSettings.SetScriptingBackend(scriptingTarget, ScriptingImplementation.IL2CPP);
            var staging = Path.Combine(SimulatorRoot, "_build", stagingName);
            WipeDir(staging);
            Directory.CreateDirectory(staging);
            var exe = Path.Combine(staging, "AutoDRIVE Simulator.x86_64");
            var opts = new BuildPlayerOptions
            {
                scenes = new[] { ScenePath },
                locationPathName = exe,
                target = BuildTarget.StandaloneLinux64,
                subtarget = (int)(dedicatedServer ? StandaloneBuildSubtarget.Server : StandaloneBuildSubtarget.Player),
                options = BuildOptions.None,
            };
            var report = BuildPipeline.BuildPlayer(opts);
            if (report.summary.result != BuildResult.Succeeded)
            {
                Debug.LogWarning("[AiCar] Linux IL2CPP release failed — retrying Development");
                opts.options = BuildOptions.Development;
                WipeDir(staging);
                Directory.CreateDirectory(staging);
                report = BuildPipeline.BuildPlayer(opts);
            }
            FailIfBad(report, "Linux");
            if (publish)
                PublishLinux(staging);
        }

        static void PublishWindows(string staging)
        {
            var dest = Path.Combine(SimulatorRoot, "windows");
            Directory.CreateDirectory(dest);
            // Replace player bits only; keep any non-Unity junk if present.
            foreach (var name in new[]
                     {
                         "AutoDRIVE Simulator.exe",
                         "AutoDRIVE Simulator_Data",
                         "UnityCrashHandler64.exe",
                         "UnityPlayer.dll",
                         "GameAssembly.dll",
                         "baselib.dll",
                     })
            {
                var from = Path.Combine(staging, name);
                var to = Path.Combine(dest, name);
                if (Directory.Exists(from))
                {
                    if (Directory.Exists(to))
                        Directory.Delete(to, true);
                    CopyDirectory(from, to);
                }
                else if (File.Exists(from))
                {
                    File.Copy(from, to, true);
                }
            }
            // Some Unity versions name Data folder differently
            foreach (var dir in Directory.GetDirectories(staging))
            {
                var name = Path.GetFileName(dir);
                if (name.EndsWith("_Data", StringComparison.OrdinalIgnoreCase) ||
                    name.Equals("Data", StringComparison.OrdinalIgnoreCase))
                {
                    var to = Path.Combine(dest, name);
                    if (Directory.Exists(to))
                        Directory.Delete(to, true);
                    CopyDirectory(dir, to);
                }
            }
            Debug.Log("[AiCar] published Windows player → " + dest);
        }

        static void PublishLinux(string staging)
        {
            // Linux player lives at simulator/ root beside maps/
            foreach (var name in new[]
                     {
                         "AutoDRIVE Simulator.x86_64",
                         "UnityPlayer.so",
                         "GameAssembly.so",
                         "baselib.so",
                     })
            {
                var from = Path.Combine(staging, name);
                var to = Path.Combine(SimulatorRoot, name);
                if (File.Exists(from))
                    CopyFileReplace(from, to);
            }
            foreach (var dir in Directory.GetDirectories(staging))
            {
                var name = Path.GetFileName(dir);
                if (name.EndsWith("_Data", StringComparison.OrdinalIgnoreCase) ||
                    name.Equals("Data", StringComparison.OrdinalIgnoreCase))
                {
                    // Prefer canonical Data/ for docker entrypoint layout
                    var to = Path.Combine(SimulatorRoot, "Data");
                    ReplaceDirectory(dir, to);
                }
            }
            Debug.Log("[AiCar] published Linux player → " + SimulatorRoot);
        }

        static void CopyFileReplace(string from, string to)
        {
            try
            {
                File.Copy(from, to, true);
                return;
            }
            catch (IOException)
            {
                // Destination often locked by Docker/WSL — stage beside then swap.
                var tmp = to + ".new";
                File.Copy(from, tmp, true);
                if (File.Exists(to))
                {
                    var bak = to + ".old";
                    if (File.Exists(bak)) File.Delete(bak);
                    File.Move(to, bak);
                }
                File.Move(tmp, to);
            }
        }

        static void ReplaceDirectory(string from, string to)
        {
            if (!Directory.Exists(to))
            {
                CopyDirectory(from, to);
                return;
            }
            var bak = to + ".bak";
            if (Directory.Exists(bak))
                Directory.Delete(bak, true);
            try
            {
                Directory.Move(to, bak);
            }
            catch (IOException)
            {
                Directory.Delete(to, true);
            }
            CopyDirectory(from, to);
        }

        static void EnsureSceneInBuild()
        {
            if (!File.Exists(ScenePath))
                throw new FileNotFoundException("missing scene: " + ScenePath);

            EditorBuildSettings.scenes = new[]
            {
                new EditorBuildSettingsScene(ScenePath, true),
            };
        }

        static void WriteTrackloaderMarker()
        {
            var marker = Path.Combine(SimulatorRoot, ".aicar_trackloader");
            // Consumers use this as a capability flag (presence only), not as a
            // build timestamp. Rewriting it can fail with Win32 error 1224 when
            // a running simulator/container has the file mapped. Create it only
            // when absent so a successful player publish is never undone by a
            // metadata refresh.
            if (File.Exists(marker))
            {
                Debug.Log("[AiCar] TrackLoader marker already exists: " + marker);
                return;
            }

            try
            {
                using (var stream = new FileStream(marker, FileMode.CreateNew, FileAccess.Write, FileShare.Read))
                using (var writer = new StreamWriter(stream))
                {
                    writer.WriteLine("TrackLoader-enabled player built " + DateTime.UtcNow.ToString("o"));
                }
                Debug.Log("[AiCar] created " + marker);
            }
            catch (IOException) when (File.Exists(marker))
            {
                // Another build created the shared capability marker concurrently.
                Debug.Log("[AiCar] TrackLoader marker was created concurrently: " + marker);
            }
            catch (Exception exception)
            {
                // The player has already been published; marker metadata must
                // not turn that successful build into a failed Unity command.
                Debug.LogWarning("[AiCar] player published, but could not create TrackLoader marker: " + exception.Message);
            }
        }

        static void WipeDir(string path)
        {
            if (Directory.Exists(path))
                Directory.Delete(path, true);
        }

        static void CopyDirectory(string source, string dest)
        {
            Directory.CreateDirectory(dest);
            foreach (var file in Directory.GetFiles(source))
                File.Copy(file, Path.Combine(dest, Path.GetFileName(file)), true);
            foreach (var dir in Directory.GetDirectories(source))
                CopyDirectory(dir, Path.Combine(dest, Path.GetFileName(dir)));
        }

        static void FailIfBad(BuildReport report, string label)
        {
            if (report.summary.result == BuildResult.Succeeded)
            {
                Debug.Log($"[AiCar] {label} build OK → {report.summary.outputPath}");
                return;
            }
            throw new Exception($"[AiCar] {label} build failed: {report.summary.result}");
        }
    }
}
#endif
