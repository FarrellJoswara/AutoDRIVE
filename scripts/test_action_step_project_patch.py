"""Check camera/Bridge patches on a fresh vendor-shaped source and on reruns."""
import importlib.util
from pathlib import Path


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
    for relative in ["Scripts/AiCar/AiCarSimulationGate.cs", "Scripts/AiCar/Editor/AiCarPlayerBuild.cs",
                     "Plugins/SocketIO/Scripts/SocketIO/SocketIOComponent.cs"]:
        source = tmp_path / "simulator/unity/Assets" / relative
        target = project / "Assets" / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("// tracked source")
        source.with_suffix(".cs.meta").write_text("guid: test")
    # Both controller families have separate camera blocks in the real source.
    cameras = ("                    if(FrontCameras.Length != 0)\n                    { /* capture */ }\n"
               "                    if(RearCameras.Length != 0)\n                    { /* capture */ }\n") * 2
    vendor = (
        "using System;\nusing SocketIO;\nclass Socket {\n"
        "    private SocketIOComponent socket; // Socket.IO instance\n"
        "    void Start()\n    {\n    }\n"
        "    void OnBridge(SocketIOEvent obj)\n    {\n"
        "        JSONObject jsonObject = obj.data; // Read incoming data and store it in a `JSONObject`\n"
        "        // Debug.Log(obj.data);\n"
        "        // Existing unrelated collision behavior must be retained.\n"
        "        // Emit telemetry data\n        EmitTelemetry(obj);\n    }\n\n"
        "    void EmitTelemetry(SocketIOEvent obj)\n    {\n"
        "            Dictionary<string, string> data = new Dictionary<string, string>(); // Create new `data` dictionary\n"
        + cameras + "    }\n}\n"
    )
    socket = project / "Assets/Scripts/Socket.cs"
    socket.write_text(vendor)
    module.main()
    patched = socket.read_text()
    assert patched.count("if(!disableCameraStream && FrontCameras.Length != 0)") == 2
    assert patched.count("if(!disableCameraStream && RearCameras.Length != 0)") == 2
    assert "Existing unrelated collision behavior must be retained" in patched
    assert "ValidateStepId" in patched
    assert patched.index("ValidateStepId") < patched.index("// Existing unrelated collision")
    assert "AICAR Step Protocol" in patched
    module.main()
    assert socket.read_text() == patched
