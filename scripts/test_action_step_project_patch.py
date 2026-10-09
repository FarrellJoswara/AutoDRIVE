"""Check camera/Bridge patches on a fresh vendor-shaped source and on reruns."""
import importlib.util
from pathlib import Path


def test_socketio_v4_waits_for_namespace_connect_before_bridge_callbacks():
    source = Path(__file__).resolve().parents[1] / (
        "simulator/unity/Assets/Plugins/SocketIO/Scripts/SocketIO/SocketIOComponent.cs"
    )
    text = source.read_text()
    assert 'EmitPacket(new Packet(EnginePacketType.MESSAGE, SocketPacketType.CONNECT' in text
    assert 'eventQueue.Enqueue(new SocketIOEvent("connect"))' in text
    assert 'if(packet.socketPacketType == SocketPacketType.CONNECT)' in text
    # Opening the websocket is transport-only; Bridge's connect callback must
    # wait until the Engine.IO v4 namespace handshake is accepted.
    transport_transition = text.split("if(wsConnected != ws.IsConnected){", 1)[1].split("\n\t\t\t}", 1)[0]
    assert 'EmitEvent("open")' in transport_transition
    assert 'EmitEvent("connect")' not in transport_transition


def test_project_patch_is_reproducible_and_idempotent(tmp_path):
    source_path = Path(__file__).resolve().parents[1] / "simulator/unity/scripts/prepare_action_step_project.py"
    spec = importlib.util.spec_from_file_location("action_patch", source_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = tmp_path
    project = tmp_path / "simulator/unity/AutoDRIVE"
    module.PROJECT = project
    (project / "ProjectSettings").mkdir(parents=True)
    (project / "ProjectSettings/ProjectVersion.txt").write_text("test")
    for relative in ["Scripts/AiCar/AiCarSimulationGate.cs", "Scripts/AiCar/ForceConnect.cs", "Scripts/AiCar/Editor/AiCarPlayerBuild.cs",
                     "Plugins/SocketIO/Scripts/SocketIO/SocketIOComponent.cs"]:
        source = tmp_path / "simulator/unity/Assets" / relative
        target = project / "Assets" / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("// tracked source")
        source.with_suffix(".cs.meta").write_text("guid: test")
    force_connect_fixture = tmp_path / "simulator/unity/Assets/Scripts/AiCar/ForceConnect.cs"
    repo_force_connect = Path(__file__).resolve().parents[1] / "simulator/unity/Assets/Scripts/AiCar/ForceConnect.cs"
    force_connect_fixture.write_text(repo_force_connect.read_text())
    # Both controller families have separate camera blocks in the real source.
    cameras = ("                    if(FrontCameras.Length != 0)\n                    { /* capture */ }\n"
               "                    if(RearCameras.Length != 0)\n                    { /* capture */ }\n") * 2
    vendor = (
        "using System;\nusing SocketIO;\nclass Socket {\n"
        "    private SocketIOComponent socket; // Socket.IO instance\n"
        "    void Start()\n    {\n    }\n"
        "    void OnConnect(SocketIOEvent obj)\n    {\n"
        "        //Debug.Log(\"Connected!\");\n"
        "        ConnectionButton.interactable = false;\n"
        "        ConnectionLabel.text = \"Connected\";\n"
        "        EmitTelemetry(obj);\n    }\n"
        "    void OnDisconnect(SocketIOEvent obj)\n    {\n"
        "        ConnectionButton.interactable = true;\n"
        "        ConnectionLabel.text = \"Disconnected\";\n    }\n"
        "    void OnBridge(SocketIOEvent obj)\n    {\n"
        "        JSONObject jsonObject = obj.data; // Read incoming data and store it in a `JSONObject`\n"
        "        // Debug.Log(obj.data);\n"
        "        // Existing unrelated collision behavior must be retained.\n"
        "        // Emit telemetry data\n        EmitTelemetry(obj);\n    }\n\n"
        "    void EmitTelemetry(SocketIOEvent obj)\n    {\n"
        "        UnityMainThreadDispatcher.Instance().Enqueue(() =>\n        {\n"
        "            Dictionary<string, string> data = new Dictionary<string, string>(); // Create new `data` dictionary\n"
        + cameras
        + "            socket.Emit(\"Bridge\", new JSONObject(data)); // Write data to server\n"
        + "        });\n    }\n}\n"
    )
    socket = project / "Assets/Scripts/Socket.cs"
    socket.write_text(vendor)
    frame_grabber = project / "Assets/Scripts/FrameGrabber.cs"
    frame_grabber.write_text(
        "public static class FrameGrabber {\n"
        "    public static byte[] CaptureFrame(Camera camera) {\n"
        "        RenderTexture targetTexture = camera.targetTexture;\n"
        "    }\n}\n"
    )
    module.main()
    force_connect = project / "Assets/Scripts/AiCar/ForceConnect.cs"
    assert "if (!activated)" in force_connect.read_text()
    assert "starts a second WebSocket" in force_connect.read_text()
    patched = socket.read_text()
    assert patched.count("if(!disableCameraStream && FrontCameras.Length != 0)") == 2
    assert patched.count("if(!disableCameraStream && RearCameras.Length != 0)") == 2
    assert "Existing unrelated collision behavior must be retained" in patched
    assert "ValidateStepId" in patched
    assert patched.index("ValidateStepId") < patched.index("// Existing unrelated collision")
    assert "AICAR Step Protocol" in patched
    assert "ConnectionButton != null" in patched
    assert "initial Bridge telemetry failed" in patched
    assert "UnityMainThreadDispatcher.Instance().Enqueue" not in patched
    assert 'socket.Emit("Bridge", new JSONObject(data));' in patched
    module.main()
    assert socket.read_text() == patched
