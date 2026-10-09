"""Apply tracked action-step Unity sources to the ignored AutoDRIVE vendor project.

The Unity AutoDRIVE checkout is intentionally git-ignored. Keep this patch helper
in the tracked tree so the staged experiment can be reproduced against a fresh
vendor project without committing third-party sources.
"""
from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PROJECT = ROOT / "simulator" / "unity" / "AutoDRIVE"


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"expected patch anchor missing in {path}: {old[:100]!r}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        stream.write(text.replace(old, new, 1))


def patch_frame_grabber(path: Path) -> None:
    """Provide headless cameras a target and render it before CPU readback."""
    existing = (
        "        camera.Render(); // Force camera update before reading its RenderTexture\n"
        "        RenderTexture targetTexture = camera.targetTexture;\n"
    )
    patched = (
        "        RenderTexture targetTexture = camera.targetTexture;\n"
        "        if (targetTexture == null && Application.isBatchMode)\n"
        "        {\n"
        "            targetTexture = new RenderTexture(1280, 720, 16, RenderTextureFormat.ARGB32);\n"
        "            targetTexture.Create();\n"
        "            camera.targetTexture = targetTexture;\n"
        "        }\n"
        "        if (targetTexture == null)\n"
        "        {\n"
        "            Debug.LogError(\"[AiCar.Camera] missing target texture; camera frame skipped\");\n"
        "            return new byte[0];\n"
        "        }\n"
        "        camera.Render(); // Force camera update before reading its RenderTexture\n"
    )
    text = path.read_text(encoding="utf-8")
    if patched in text:
        return
    replace_once(path, existing if existing in text else "        RenderTexture targetTexture = camera.targetTexture;\n", patched)


def patch_socket_telemetry_dispatch(path: Path) -> None:
    """Emit Bridge data directly from Unity's main-thread Socket.IO callbacks.

    SocketIOComponent pumps callbacks in Update/FixedUpdate, so EmitTelemetry is
    already on Unity's main thread. The vendor scene has no
    UnityMainThreadDispatcher object; routing through it silently drops the
    first telemetry packet because SocketIOComponent swallows callback errors.
    """
    text = path.read_text(encoding="utf-8")
    wrapped = (
        "    void EmitTelemetry(SocketIOEvent obj, int stepId = 0, int physicsTicks = 0, string stepError = null)\n"
        "    {\n"
        "        UnityMainThreadDispatcher.Instance().Enqueue(() =>\n"
        "        {\n"
    )
    if wrapped not in text:
        if "UnityMainThreadDispatcher.Instance().Enqueue" not in text and 'socket.Emit("Bridge", new JSONObject(data));' in text:
            return
        raise RuntimeError(f"expected telemetry-dispatch wrapper missing in {path}")

    start = text.index(wrapped) + len(wrapped)
    close = "        });\n    }\n"
    end = text.find(close, start)
    if end < 0:
        raise RuntimeError(f"expected telemetry-dispatch closing anchor missing in {path}")
    body = text[start:end]
    replacement = (
        "    void EmitTelemetry(SocketIOEvent obj, int stepId = 0, int physicsTicks = 0, string stepError = null)\n"
        "    {\n" + body + "    }\n"
    )
    text = text[:text.index(wrapped)] + replacement + text[end + len(close):]
    with path.open("w", encoding="utf-8", newline="") as stream:
        stream.write(text)


def main() -> None:
    if not (PROJECT / "ProjectSettings" / "ProjectVersion.txt").is_file():
        raise SystemExit(f"Unity project not found: {PROJECT}")

    # In batch/offscreen runs Unity does not necessarily render every camera
    # automatically. The recorder already forces a render before ReadPixels;
    # the Bridge's FrameGrabber must do the same or it can publish stale/black
    # camera frames even while the RenderTexture is configured correctly.
    frame_grabber = PROJECT / "Assets" / "Scripts" / "FrameGrabber.cs"
    patch_frame_grabber(frame_grabber)

    gate_source = ROOT / "simulator" / "unity" / "Assets" / "Scripts" / "AiCar" / "AiCarSimulationGate.cs"
    gate_target = PROJECT / "Assets" / "Scripts" / "AiCar" / "AiCarSimulationGate.cs"
    shutil.copy2(gate_source, gate_target)
    shutil.copy2(gate_source.with_suffix(gate_source.suffix + ".meta"), gate_target.with_suffix(gate_target.suffix + ".meta"))

    force_connect_source = ROOT / "simulator" / "unity" / "Assets" / "Scripts" / "AiCar" / "ForceConnect.cs"
    force_connect_target = PROJECT / "Assets" / "Scripts" / "AiCar" / "ForceConnect.cs"
    shutil.copy2(force_connect_source, force_connect_target)
    shutil.copy2(force_connect_source.with_suffix(force_connect_source.suffix + ".meta"), force_connect_target.with_suffix(force_connect_target.suffix + ".meta"))

    build_source = ROOT / "simulator" / "unity" / "Assets" / "Scripts" / "AiCar" / "Editor" / "AiCarPlayerBuild.cs"
    build_target = PROJECT / "Assets" / "Scripts" / "AiCar" / "Editor" / "AiCarPlayerBuild.cs"
    shutil.copy2(build_source, build_target)
    shutil.copy2(build_source.with_suffix(build_source.suffix + ".meta"), build_target.with_suffix(build_target.suffix + ".meta"))

    socketio = PROJECT / "Assets" / "Plugins" / "SocketIO" / "Scripts" / "SocketIO" / "SocketIOComponent.cs"
    plugin_source = ROOT / "simulator" / "unity" / "Assets" / "Plugins" / "SocketIO" / "Scripts" / "SocketIO" / "SocketIOComponent.cs"
    shutil.copy2(plugin_source, socketio)

    socket = PROJECT / "Assets" / "Scripts" / "Socket.cs"
    socket_text = socket.read_text(encoding="utf-8")
    if "int explicitStepId = -1;" not in socket_text:
        replace_once(
            socket,
            "        JSONObject jsonObject = obj.data; // Read incoming data and store it in a `JSONObject`\n        // Debug.Log(obj.data);\n",
            "        JSONObject jsonObject = obj.data; // Read incoming data and store it in a `JSONObject`\n"
            "        int explicitStepId = -1;\n"
            "        if (AiCar.AiCarSimulationGate.ActionStepMode)\n"
            "        {\n"
            "            var gate = AiCar.AiCarSimulationGate.Instance;\n"
            "            string validationError = null;\n"
            "            if (!jsonObject.HasField(\"AICAR Step ID\") || !int.TryParse(jsonObject.GetField(\"AICAR Step ID\").str, NumberStyles.Integer, CultureInfo.InvariantCulture, out explicitStepId))\n"
            "                validationError = \"missing or invalid AICAR Step ID\";\n"
            "            else if (gate == null) validationError = \"action-step gate is unavailable\";\n"
            "            else gate.ValidateStepId(explicitStepId, out validationError);\n"
            "            if (validationError != null)\n"
            "            {\n"
            "                EmitTelemetry(obj, explicitStepId, 0, validationError);\n"
            "                return;\n"
            "            }\n"
            "        }\n"
            "        // Debug.Log(obj.data);\n",
        )
    replace_once(
        socket,
        "using System;\nusing SocketIO;",
        "using System;\nusing System.Globalization;\nusing SocketIO;",
    )
    replace_once(
        socket,
        "    void OnConnect(SocketIOEvent obj)\n    {\n        //Debug.Log(\"Connected!\");\n        ConnectionButton.interactable = false;\n        ConnectionLabel.text = \"Connected\";\n        EmitTelemetry(obj);\n    }",
        "    void OnConnect(SocketIOEvent obj)\n    {\n"
        "        Debug.Log(\"[AiCar.Socket] Socket.IO namespace connected; sending initial Bridge telemetry\");\n"
        "        if (ConnectionButton != null) ConnectionButton.interactable = false;\n"
        "        if (ConnectionLabel != null) ConnectionLabel.text = \"Connected\";\n"
        "        try { EmitTelemetry(obj); }\n"
        "        catch (Exception ex) { Debug.LogError(\"[AiCar.Socket] initial Bridge telemetry failed: \" + ex); }\n"
        "    }",
    )
    replace_once(socket, "        ConnectionButton.interactable = true;", "        if (ConnectionButton != null) ConnectionButton.interactable = true;")
    replace_once(socket, "        ConnectionLabel.text = \"Disconnected\";", "        if (ConnectionLabel != null) ConnectionLabel.text = \"Disconnected\";")
    socket_text = socket.read_text(encoding="utf-8")
    if "private bool disableCameraStream;" not in socket_text:
        replace_once(socket, "    private SocketIOComponent socket; // Socket.IO instance\n",
                     "    private SocketIOComponent socket; // Socket.IO instance\n    private bool disableCameraStream;\n")
    if "disableCameraStream = Application.isBatchMode" not in socket.read_text(encoding="utf-8"):
        replace_once(
            socket,
            "    void Start()\n    {\n",
            "    void Start()\n    {\n"
            "        // Camera readback is disabled only with explicit fixed-step training.\n"
            "        disableCameraStream = Application.isBatchMode && AiCar.AiCarSimulationGate.ActionStepMode &&\n"
            "            string.Equals(Environment.GetEnvironmentVariable(\"AICAR_DISABLE_CAMERA_STREAM\"), \"1\", StringComparison.Ordinal);\n",
        )
    replace_once(
        socket,
        "disableCameraStream = Application.isBatchMode &&\n            string.Equals(Environment.GetEnvironmentVariable(\"AICAR_DISABLE_CAMERA_STREAM\"), \"1\", StringComparison.Ordinal);",
        "disableCameraStream = Application.isBatchMode && AiCar.AiCarSimulationGate.ActionStepMode &&\n            string.Equals(Environment.GetEnvironmentVariable(\"AICAR_DISABLE_CAMERA_STREAM\"), \"1\", StringComparison.Ordinal);",
    )
    socket_text = socket.read_text(encoding="utf-8")
    if "if(!disableCameraStream && FrontCameras.Length != 0)" not in socket_text:
        marker = "                    if(FrontCameras.Length != 0)\n"
        if marker not in socket_text:
            raise RuntimeError("camera telemetry capture block anchor missing")
        socket_text = socket_text.replace(marker, "                    if(!disableCameraStream && FrontCameras.Length != 0)\n")
        with socket.open("w", encoding="utf-8", newline="") as stream:
            stream.write(socket_text)
    if "if(!disableCameraStream && RearCameras.Length != 0)" not in socket_text:
        marker = "                    if(RearCameras.Length != 0)\n"
        if marker not in socket_text:
            raise RuntimeError("rear camera telemetry capture block anchor missing")
        socket_text = socket_text.replace(marker, "                    if(!disableCameraStream && RearCameras.Length != 0)\n")
        with socket.open("w", encoding="utf-8", newline="") as stream:
            stream.write(socket_text)
    socket_text = socket.read_text(encoding="utf-8")
    previous_explicit_branch = (
        "        // Explicit RL mode returns only after the requested fixed-tick batch.\n"
        "        var simulationGate = AiCar.AiCarSimulationGate.Instance;\n"
        "        if (AiCar.AiCarSimulationGate.ActionStepMode && jsonObject.HasField(\"AICAR Step ID\"))\n"
        "        {\n"
        "            int stepId;\n"
        "            if (!int.TryParse(jsonObject.GetField(\"AICAR Step ID\").str, NumberStyles.Integer, CultureInfo.InvariantCulture, out stepId))\n"
        "            {\n"
        "                EmitTelemetry(obj, 0, 0, \"invalid AICAR Step ID\");\n"
        "                return;\n"
        "            }\n"
        "            bool reset = jsonObject.HasField(\"V1 Reset\") && string.Equals(jsonObject.GetField(\"V1 Reset\").str, \"True\", StringComparison.OrdinalIgnoreCase);\n"
        "            simulationGate.BeginActionStep(stepId, reset, (id, simTime, ticks, error) => EmitTelemetry(obj, id, ticks, error));\n"
        "            return;\n"
        "        }\n"
    )
    if previous_explicit_branch in socket_text:
        replace_once(
            socket,
            previous_explicit_branch,
            "        // Explicit RL mode returns only after the requested fixed-tick batch.\n"
            "        var simulationGate = AiCar.AiCarSimulationGate.Instance;\n"
            "        if (AiCar.AiCarSimulationGate.ActionStepMode)\n"
            "        {\n"
            "            bool reset = jsonObject.HasField(\"V1 Reset\") && string.Equals(jsonObject.GetField(\"V1 Reset\").str, \"True\", StringComparison.OrdinalIgnoreCase);\n"
            "            simulationGate.BeginActionStep(explicitStepId, reset, (id, simTime, ticks, error) => EmitTelemetry(obj, id, ticks, error));\n"
            "            return;\n"
            "        }\n",
        )
    replace_once(
        socket,
        "        // Emit telemetry data\n        EmitTelemetry(obj);\n    }\n\n    void EmitTelemetry(SocketIOEvent obj)\n",
        "        // Explicit RL mode returns only after the requested fixed-tick batch.\n"
        "        var simulationGate = AiCar.AiCarSimulationGate.Instance;\n"
        "        if (AiCar.AiCarSimulationGate.ActionStepMode)\n"
        "        {\n"
        "            bool reset = jsonObject.HasField(\"V1 Reset\") && string.Equals(jsonObject.GetField(\"V1 Reset\").str, \"True\", StringComparison.OrdinalIgnoreCase);\n"
        "            simulationGate.BeginActionStep(explicitStepId, reset, (id, simTime, ticks, error) => EmitTelemetry(obj, id, ticks, error));\n"
        "            return;\n"
        "        }\n"
        "        EmitTelemetry(obj);\n    }\n\n    void EmitTelemetry(SocketIOEvent obj, int stepId = 0, int physicsTicks = 0, string stepError = null)\n",
    )
    replace_once(
        socket,
        "            Dictionary<string, string> data = new Dictionary<string, string>(); // Create new `data` dictionary\n",
        "            Dictionary<string, string> data = new Dictionary<string, string>(); // Create new `data` dictionary\n"
        "            if (AiCar.AiCarSimulationGate.ActionStepMode)\n"
        "            {\n"
        "                var gate = AiCar.AiCarSimulationGate.Instance;\n"
        "                data[\"AICAR Step Protocol\"] = \"1\";\n"
        "                data[\"AICAR Action Interval\"] = gate.ActionIntervalSeconds.ToString(\"R\", CultureInfo.InvariantCulture);\n"
        "                data[\"AICAR Sim Time\"] = gate.SimulatedSeconds.ToString(\"R\", CultureInfo.InvariantCulture);\n"
        "                data[\"AICAR Step ID\"] = stepId.ToString(CultureInfo.InvariantCulture);\n"
        "                data[\"AICAR Physics Ticks\"] = physicsTicks.ToString(CultureInfo.InvariantCulture);\n"
        "                data[\"AICAR Step Error\"] = stepError ?? gate.ConfigurationError ?? \"\";\n"
        "            }\n",
    )
    patch_socket_telemetry_dispatch(socket)
    print("Prepared opt-in fixed-tick action mode in", PROJECT)


if __name__ == "__main__":
    main()
