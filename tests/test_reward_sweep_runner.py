from scripts.run_reward_sweep import probe_summary, set_base_config, comparable_config, ROOT


def test_sweep_base_disables_incompatible_auto_probe_mode(tmp_path):
    cfg = set_base_config({"automatic_evaluation_mode": "training_probe"}, tmp_path / "model.zip")
    assert cfg["automatic_evaluation_enabled"] is False
    assert cfg["automatic_evaluation_mode"] == "competitive"
    assert cfg["n_envs"] == 2
    assert cfg["collision_penalty_magnitude"] == cfg["failed_episode_penalty"] == 250
    assert cfg["centerline_shaping_strength"] == 0


def test_porto_clean_gate_requires_all_nine_laps_and_readiness():
    attempts = [{"race_lap_times_s": [40, 30 + i, 31 + i], "collision_count": 0,
                 "stop_reason": "race_complete"} for i in range(3)]
    row = {"result": {"runs": attempts, "readiness_passed": True}}
    summary = probe_summary(row)
    assert summary["clean_gate"]
    assert summary["median_flying_lap_s"] == 31.5
    attempts[0]["collision_count"] = 1
    assert not probe_summary(row)["clean_gate"]
    attempts[0]["collision_count"] = 0
    attempts[1]["race_lap_times_s"].pop()
    assert not probe_summary(row)["clean_gate"]


def test_resume_reconciliation_normalizes_container_paths():
    assert comparable_config({"resume": "/runs/rl/example/model.zip"}) == {
        "resume": str(ROOT / "logs/rl/example/model.zip")}
