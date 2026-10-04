#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.Profiling;
using UnityEditorInternal;
using UnityEngine;

namespace AiCar.Editor
{
    public static class AiCarProfilerExport
    {
        [Serializable]
        sealed class MarkerRow
        {
            public string thread;
            public string sample;
            public double selfMilliseconds;
            public double totalMilliseconds;
            public double calls;
        }

        [Serializable]
        sealed class ProfileSummary
        {
            public int firstFrame;
            public int lastFrame;
            public int validFrameThreadPairs;
            public List<MarkerRow> samples = new List<MarkerRow>();
        }

        sealed class MarkerAggregate
        {
            public double selfMilliseconds;
            public double totalMilliseconds;
            public double calls;
        }

        public static void ExportLoadedProfile()
        {
            var input = Environment.GetEnvironmentVariable("AICAR_PROFILER_RAW");
            var output = Environment.GetEnvironmentVariable("AICAR_PROFILER_JSON");
            if (string.IsNullOrWhiteSpace(input) || string.IsNullOrWhiteSpace(output))
                throw new InvalidOperationException("Set AICAR_PROFILER_RAW and AICAR_PROFILER_JSON.");

            Action onLoaded = null;
            onLoaded = () =>
            {
                ProfilerDriver.profileLoaded -= onLoaded;
                Export(input, output);
            };
            ProfilerDriver.profileLoaded += onLoaded;
            if (!ProfilerDriver.LoadProfile(input, false))
            {
                ProfilerDriver.profileLoaded -= onLoaded;
                throw new InvalidOperationException("Unity could not load profiler capture: " + input);
            }
        }

        static void Export(string input, string output)
        {
            var first = ProfilerDriver.firstFrameIndex;
            var last = ProfilerDriver.lastFrameIndex;
            var aggregates = new Dictionary<string, MarkerAggregate>();
            var validPairs = 0;

            for (var frame = first; frame <= last; frame++)
            {
                for (var thread = 0; thread < 128; thread++)
                {
                    using (var view = ProfilerDriver.GetHierarchyFrameDataView(
                               frame, thread, HierarchyFrameDataView.ViewModes.Default,
                               HierarchyFrameDataView.columnSelfTime, false))
                    {
                        if (!view.valid) continue;
                        validPairs++;
                        var threadName = view.threadName ?? ("thread-" + thread);
                        var pending = new Stack<int>();
                        var children = new List<int>();
                        var root = view.GetRootItemID();
                        view.GetItemChildren(root, children);
                        for (var i = 0; i < children.Count; i++) pending.Push(children[i]);

                        while (pending.Count > 0)
                        {
                            var id = pending.Pop();
                            var path = view.GetItemPath(id) ?? view.GetItemName(id) ?? "<unknown>";
                            var key = threadName + "\t" + path;
                            if (!aggregates.TryGetValue(key, out var aggregate))
                            {
                                aggregate = new MarkerAggregate();
                                aggregates.Add(key, aggregate);
                            }
                            aggregate.selfMilliseconds += view.GetItemColumnDataAsDouble(
                                id, HierarchyFrameDataView.columnSelfTime);
                            aggregate.totalMilliseconds += view.GetItemColumnDataAsDouble(
                                id, HierarchyFrameDataView.columnTotalTime);
                            aggregate.calls += view.GetItemColumnDataAsDouble(
                                id, HierarchyFrameDataView.columnCalls);
                            view.GetItemChildren(id, children);
                            for (var i = 0; i < children.Count; i++) pending.Push(children[i]);
                        }
                    }
                }
            }

            var summary = new ProfileSummary
            {
                firstFrame = first,
                lastFrame = last,
                validFrameThreadPairs = validPairs,
                samples = aggregates.Select(pair => new MarkerRow
                {
                    thread = pair.Key.Substring(0, pair.Key.IndexOf('\t')),
                    sample = pair.Key.Substring(pair.Key.IndexOf('\t') + 1),
                    selfMilliseconds = pair.Value.selfMilliseconds,
                    totalMilliseconds = pair.Value.totalMilliseconds,
                    calls = pair.Value.calls,
                }).OrderByDescending(row => row.selfMilliseconds).ToList(),
            };
            Directory.CreateDirectory(Path.GetDirectoryName(output));
            File.WriteAllText(output, JsonUtility.ToJson(summary, true));
            UnityEngine.Debug.Log("[AiCar] Profiler hierarchy exported: " + output +
                                  " (frames " + first + ".." + last + ") from " + input);
            EditorApplication.Exit(0);
        }
    }
}
#endif
