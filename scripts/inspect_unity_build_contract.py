"""Compare retained generated Unity code and relate it to the active Linux build.

Reads existing artifacts only; writes JSON only when --output is supplied.
Generated code is evidence from retained build folders, not an attestation of
every original source file or reproducible build configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


BACKUP = "AutoDRIVE Simulator_BackUpThisFolder_ButDontShipItWithYourGame"
FUNCTION = re.compile(
    r"^IL2CPP_EXTERN_C IL2CPP_METHOD_ATTR [^\n]+ \n\{.*?^\}", re.M | re.S
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def functions(path: Path) -> dict[str, str]:
    result = {}
    for match in FUNCTION.finditer(path.read_text(encoding="utf-8")):
        function = match.group()
        name = re.search(r"(\w+) \(", function.splitlines()[0]).group(1)
        if name in result:
            raise ValueError(f"duplicate method definition: {name}")
        result[name] = function
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulator-root", type=Path, default=Path("simulator"))
    parser.add_argument("--baseline-build", default="linux")
    parser.add_argument("--experiment-build", default="linux-cpu-experiment")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.simulator_root.resolve()
    baseline = root / "_build" / args.baseline_build
    experiment = root / "_build" / args.experiment_build
    runtime = []
    pairs = [
        ("GameAssembly.so", "GameAssembly.so"),
        ("UnityPlayer.so", "UnityPlayer.so"),
        ("AutoDRIVE Simulator.x86_64", "AutoDRIVE Simulator.x86_64"),
        ("Data/globalgamemanagers", "AutoDRIVE Simulator_Data/globalgamemanagers"),
        ("Data/il2cpp_data/Metadata/global-metadata.dat",
         "AutoDRIVE Simulator_Data/il2cpp_data/Metadata/global-metadata.dat"),
    ]
    for active_name, baseline_name in pairs:
        active_path, baseline_path = root / active_name, baseline / baseline_name
        active_hash, baseline_hash = digest(active_path), digest(baseline_path)
        runtime.append({
            "active_path": str(active_path), "baseline_path": str(baseline_path),
            "active_sha256": active_hash, "baseline_sha256": baseline_hash,
            "identical": active_hash == baseline_hash,
        })
    generated = []
    relevant_functions = []
    baseline_code, experiment_code = (
        build / BACKUP / "il2cppOutput" for build in (baseline, experiment)
    )
    for baseline_path in sorted(baseline_code.glob("Assembly-CSharp*.cpp")):
        experiment_path = experiment_code / baseline_path.name
        base_hash, exp_hash = digest(baseline_path), digest(experiment_path)
        entry = {
            "name": baseline_path.name, "baseline_path": str(baseline_path),
            "experiment_path": str(experiment_path),
            "baseline_sha256": base_hash, "experiment_sha256": exp_hash,
            "identical": base_hash == exp_hash,
        }
        base_functions, exp_functions = functions(baseline_path), functions(experiment_path)
        for name in sorted(base_functions.keys() & exp_functions.keys()):
            if not any(token in name for token in (
                "Socket_OnBridge", "FrameGrabber_CaptureFrame",
                "VehicleController_OnCollisionEnter", "VehicleController_FixedUpdate",
                "AutomobileController_FixedUpdate", "SocketIOComponent_FixedUpdate",
                "UnityMainThreadDispatcher_Update",
            )):
                continue
            relevant_functions.append({
                "name": name,
                "baseline_path": str(baseline_path),
                "experiment_path": str(experiment_path),
                "baseline_function_sha256": hashlib.sha256(base_functions[name].encode()).hexdigest(),
                "experiment_function_sha256": hashlib.sha256(exp_functions[name].encode()).hexdigest(),
                "identical": base_functions[name] == exp_functions[name],
            })
        if base_hash != exp_hash:
            common = base_functions.keys() & exp_functions.keys()
            entry.update({
                "baseline_function_count": len(base_functions),
                "experiment_function_count": len(exp_functions),
                "changed_shared_functions": sorted(
                    name for name in common if base_functions[name] != exp_functions[name]
                ),
                "removed_functions": sorted(base_functions.keys() - exp_functions.keys()),
                "added_functions": sorted(exp_functions.keys() - base_functions.keys()),
                "unchanged_relevant_functions": sorted(
                    name for name in common
                    if base_functions[name] == exp_functions[name]
                    and any(token in name for token in ("Socket_OnBridge", "FrameGrabber_CaptureFrame"))
                ),
            })
        generated.append(entry)
    if not generated:
        raise ValueError("no retained generated Assembly-CSharp sources found")
    result = {
        "method": "SHA256 runtime artifact comparison and exact retained generated C++ comparison",
        "active_matches_retained_baseline_artifacts": all(item["identical"] for item in runtime),
        "runtime_artifacts": runtime,
        "generated_assembly_sources": generated,
        "relevant_function_evidence": relevant_functions,
        "limitations": [
            "Matching selected runtime artifacts ties this baseline folder to the active executable/code/settings; other assets are not exhaustively compared.",
            "Retained generated C++ has no signed source manifest; this does not prove original source-file provenance.",
            "Upstream Git differences describe local customization, not differences against the active player.",
            "Corrects the earlier upstream-diff warning: collision telemetry edits already appear in the retained active baseline; VehicleController generated code is byte-identical to the camera experiment.",
            "Exact generated-code equality does not prove identical action timing when camera work is skipped.",
        ],
    }
    output = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
