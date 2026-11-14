import json
from types import SimpleNamespace
from datetime import datetime, timezone

from scripts import supervise_speed_campaign as supervisor


def test_restart_gate_requires_active_phase_and_blocks_exhausted_retry():
    assert supervisor.restart_block_reason({"phase": "screen", "runs": {}}) is None
    assert supervisor.restart_block_reason({"phase": "halted_production_startup", "runs": {}}) == (
        "phase_not_active:halted_production_startup"
    )
    assert supervisor.restart_block_reason({
        "phase": "production", "runs": {"production:0": {"retry_exhausted": True}}
    }) == "retry_exhausted:production:0"


def test_failed_run_blocks_restart_unless_runner_persisted_retry_pending():
    state = {"phase": "production", "runs": {"production:0": {"run_id": "abc"}}}
    inventory = {"abc": {"run_id": "abc", "state": "failed"}}
    assert supervisor.terminal_failure_reason(state, inventory) == "terminal_failed_run:production:0:abc"
    state["runs"]["production:0"]["retry_pending"] = True
    assert supervisor.terminal_failure_reason(state, inventory) is None


def test_supervisor_waits_for_inventory_then_launches_runner_and_stops_on_phase_change(tmp_path):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    state_path = campaign / "state.json"
    state_path.write_text(json.dumps({"phase": "screen", "runs": {}}))
    args = SimpleNamespace(
        campaign_dir=campaign, log=tmp_path / "supervisor.log", hub_url="http://hub",
        adaptation_run_id="adapt", base_config=tmp_path / "base.json", python="python-test",
        runner_poll_seconds=5, interval=4, backoff_initial=2, backoff_max=8,
    )
    calls = []

    class Child:
        def poll(self):
            state_path.write_text(json.dumps({"phase": "halted_test", "runs": {}}))
            return 7

    def popen(command, **kwargs):
        calls.append((command, kwargs))
        return Child()

    sleeps = []
    result = supervisor.run_supervisor(
        args, popen=popen, sleep=sleeps.append, healthy=lambda url: True,
        inventory_reader=lambda url: {},
    )
    assert result == 0
    assert len(calls) == 1
    command = calls[0][0]
    assert command[0] == "python-test"
    assert command[command.index("--adaptation-run-id") + 1] == "adapt"
    assert command[-1] == "--execute"
    assert calls[0][1]["start_new_session"] is True
    assert sleeps == []
    events = [json.loads(line)["event"] for line in args.log.read_text().splitlines()]
    assert events == ["runner_starting", "runner_exited", "supervision_stopped"]


def test_supervisor_propagates_explicit_failed_evaluation_override(tmp_path):
    args = SimpleNamespace(
        campaign_dir=tmp_path, adaptation_run_id="adapt", base_config=tmp_path / "base.json",
        python="native-python", hub_url="http://hub", runner_poll_seconds=5,
        continue_after_failed_production_eval=True,
    )
    assert supervisor.runner_command(args)[-1] == "--continue-after-failed-production-eval"


def test_supervisor_refuses_current_halted_campaign_without_launch(tmp_path):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    (campaign / "state.json").write_text(json.dumps({
        "phase": "halted_production_startup",
        "runs": {"production:0": {"retry_exhausted": True}},
    }))
    args = SimpleNamespace(
        campaign_dir=campaign, log=tmp_path / "supervisor.log", hub_url="http://hub",
        adaptation_run_id="adapt", base_config=tmp_path / "base.json", python="python",
        runner_poll_seconds=5, interval=4, backoff_initial=2, backoff_max=8,
    )
    launches = []
    assert supervisor.run_supervisor(args, popen=lambda *a, **k: launches.append(a),
                                     sleep=lambda _: None, healthy=lambda _: True,
                                     inventory_reader=lambda _: {}) == 0
    assert launches == []
    row = json.loads(args.log.read_text().splitlines()[-1])
    assert row["event"] == "supervision_stopped"


def test_supervisor_applies_bounded_backoff_after_unexpected_runner_exit(tmp_path):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    state_path = campaign / "state.json"
    state_path.write_text(json.dumps({"phase": "screen", "runs": {}}))
    args = SimpleNamespace(
        campaign_dir=campaign, log=tmp_path / "supervisor.log", hub_url="http://hub",
        adaptation_run_id="adapt", base_config=tmp_path / "base.json", python="python",
        runner_poll_seconds=5, interval=4, backoff_initial=2, backoff_max=3,
    )
    launches = []

    class Child:
        def poll(self):
            if len(launches) == 2:
                state_path.write_text(json.dumps({"phase": "halted_test", "runs": {}}))
            return 9

    def popen(command, **kwargs):
        launches.append(command)
        return Child()

    sleeps = []
    supervisor.run_supervisor(args, popen=popen, sleep=sleeps.append,
                              healthy=lambda _: True, inventory_reader=lambda _: {})
    assert len(launches) == 2
    assert sleeps == [2]


def test_monitor_stall_alert_does_not_stop_child_or_launch_duplicate(tmp_path):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    state_path = campaign / "state.json"
    state_path.write_text(json.dumps({"phase": "screen", "runs": {}}))
    checked = datetime.now(timezone.utc).isoformat()
    (campaign / "monitor_status.json").write_text(json.dumps({
        "checked_utc": checked, "alerts": ["audit_progress_stalled:screen:control"]
    }))
    args = SimpleNamespace(
        campaign_dir=campaign, log=tmp_path / "supervisor.log", hub_url="http://hub",
        adaptation_run_id="adapt", base_config=tmp_path / "base.json", python="python",
        runner_poll_seconds=5, interval=1, backoff_initial=2, backoff_max=8,
    )
    starts = []

    class Child:
        polls = 0

        def poll(self):
            self.polls += 1
            if self.polls == 1:
                return None
            state_path.write_text(json.dumps({"phase": "halted_test", "runs": {}}))
            return 9

        def terminate(self):
            raise AssertionError("supervisor must not stop a stalled trainer")

    def popen(*_args, **_kwargs):
        starts.append(1)
        return Child()

    supervisor.run_supervisor(args, popen=popen, sleep=lambda _: None,
                              healthy=lambda _: True, inventory_reader=lambda _: {})
    rows = [json.loads(line) for line in args.log.read_text().splitlines()]
    assert len(starts) == 1
    assert any(row["event"] == "trainer_stall_alert" for row in rows)
    assert all(row.get("action") != "terminate" for row in rows)
