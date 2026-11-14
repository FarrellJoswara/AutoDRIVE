import json

from scripts.run_speed_campaign import CANDIDATES, comparable_config, summarize, training_evidence, is_zero_update_sensor_startup_failure, production_gate_action, ROOT


def test_candidate_settings_match_frontier_target_schema():
    assert len(CANDIDATES) == 4
    assert all("frontier_pace_target_mps" in row for row in CANDIDATES)
    assert all("frontier_target_speed_mps" not in row for row in CANDIDATES)


def test_resume_aliases_reconcile_to_local_checkpoint():
    expected = str(ROOT / "logs/rl/example/model.zip")
    assert comparable_config({"resume": "/runs/rl/example/model.zip"})["resume"] == expected


def test_porto_gate_requires_all_nine_collision_free_laps():
    attempts = [{"race_lap_times_s": [40.0, 30.0, 29.0], "collision_count": 0,
                 "race_collisions": 0, "raw_collision_count_at_finish": 0,
                 "stop_reason": "race_complete"} for _ in range(3)]
    row = {"result": {"runs": attempts, "readiness_passed": True}}
    assert summarize(row)["clean_gate"] is True
    attempts[0]["raw_collision_count_at_finish"] = 1
    assert summarize(row)["clean_gate"] is False


def test_failed_production_gate_continues_only_with_explicit_override():
    assert production_gate_action(False, False) == "halt"
    assert production_gate_action(False, True) == "continue"
    assert production_gate_action(True, False) == "continue"


def test_clean_ten_lap_training_evidence_requires_episode_gate(tmp_path):
    path = tmp_path / "race_metrics.jsonl"
    row = {"event_type": "episode", "clean_10_lap_episode": True,
           "race_laps_completed": 10, "race_lap_timing_valid": True,
           "race_lap_times_s": [30.0] * 10, "race_collisions": 0, "raw_collision_count": 0}
    path.write_text(json.dumps(row) + "\n")
    assert training_evidence(tmp_path)["clean_ten_lap_episode"] is True
    row["clean_10_lap_episode"] = False
    path.write_text(json.dumps(row) + "\n")
    assert training_evidence(tmp_path)["clean_ten_lap_episode"] is False


def test_bounded_retry_matches_only_zero_update_sensor_startup_timeout(tmp_path):
    root=tmp_path/"failed"
    (root/"tensorboard").mkdir(parents=True)
    (root/"hub_train.log").write_text(
        "Official PPO configuration verified before rollout.\n"
        "OfficialRosTransportError: Timed out after 180.0s waiting for a fresh official ROS sensor frame\n"
    )
    (root/"tensorboard"/"progress.csv").write_text("")
    row={"state":"failed","error":"official API or simulator container exited (trainer exit code 1)","output_dir":str(root)}
    assert is_zero_update_sensor_startup_failure(row)
    (root/"tensorboard"/"progress.csv").write_text("total_timesteps\n4096\n")
    assert not is_zero_update_sensor_startup_failure(row)
    (root/"tensorboard"/"progress.csv").write_text("")
    (root/"ppo_optimizer_audit.jsonl").write_text("{}\n")
    assert not is_zero_update_sensor_startup_failure(row)
