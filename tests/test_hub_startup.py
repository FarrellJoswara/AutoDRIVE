from src.layer4.hub import app as hub_app


def test_main_passes_the_existing_app_object_to_uvicorn(monkeypatch):
    import uvicorn

    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda target, **kwargs: calls.append((target, kwargs)))

    hub_app.main()

    assert len(calls) == 1
    assert calls[0][0] is hub_app.app
    assert calls[0][1]["reload"] is False


def test_health_exposes_the_startup_captured_hub_identity():
    import asyncio

    from src.layer4.hub.campaign_deadline import _validate_hub_identity

    identity = hub_app.HUB_RUNTIME_IDENTITY["hub_identity"]
    _validate_hub_identity(identity, "hub_identity")
    health = asyncio.run(hub_app.health())

    assert health["hub_identity"] == identity
    assert identity["pid"] == hub_app.HUB_RUNTIME_IDENTITY["hub_pid"]
    assert identity["process_start_identity"] == hub_app.HUB_RUNTIME_IDENTITY["hub_start_identity"]
    assert identity["source_sha256"] == hub_app.HUB_RUNTIME_IDENTITY["source_fingerprint"]
    assert isinstance(identity["argv"], list) and identity["argv"]
    sources = hub_app.HUB_RUNTIME_IDENTITY["source_files_sha256"]
    assert "src.layer4.hub.app" in sources
    assert "src.layer4.hub.campaign_deadline" in sources


def test_training_route_passes_startup_identity_and_normalized_resume_to_manager(monkeypatch):
    import asyncio

    captured = {}

    class Request:
        async def json(self):
            return {"config": {"resume": "logs/rl/official_run_123456abcdef/checkpoints/post.zip"}}

    class Runs:
        def create_run(self, config, display_name, *, maintenance_resume):
            captured["config"] = config
            captured["display_name"] = display_name
            captured["maintenance_resume"] = maintenance_resume
            return {"run_id": "123456abcdef"}

    admissions = []
    monkeypatch.setattr(hub_app, "official_runs", Runs())
    from src.layer4.official_settings import OfficialTrainSettings
    monkeypatch.setattr(
        hub_app,
        "normalize_official_config",
        lambda raw, **_kwargs: {**OfficialTrainSettings().model_dump(mode="python"), **raw},
    )
    monkeypatch.setattr(hub_app, "assert_campaign_allows", lambda kind, **kw: admissions.append((kind, kw)))
    monkeypatch.setattr(hub_app, "assert_campaign_training_config", lambda _config: None)
    monkeypatch.setattr(hub_app, "save_official_train_settings", lambda _config: None)

    result = asyncio.run(hub_app.create_official_run(Request()))

    assert result == {"run_id": "123456abcdef"}
    assert len(admissions) == 1 and admissions[0][0] == "training"
    context = admissions[0][1]["maintenance_resume"]
    assert context == captured["maintenance_resume"]
    assert context["new_hub_identity"] == hub_app.HUB_RUNTIME_IDENTITY["hub_identity"]
    assert context["normalized_config"] == captured["config"]
    assert context["normalized_config"]["resume"].endswith("post.zip")


def test_training_route_skips_baseline_plan_only_for_verified_maintenance_resume(monkeypatch):
    import asyncio

    from src.layer4.official_settings import OfficialTrainSettings

    calibrated_time_cost = 3.0 / 110.255550384521
    captured = {}

    class Request:
        async def json(self):
            return {"config": {"time_cost_per_simulated_second": calibrated_time_cost}}

    class Runs:
        def create_run(self, config, _display_name, *, maintenance_resume):
            captured["config"] = config
            captured["maintenance_resume"] = maintenance_resume
            return {"run_id": "abc123abc123"}

    identity = hub_app.HUB_RUNTIME_IDENTITY["hub_identity"]
    monkeypatch.setattr(hub_app, "official_runs", Runs())
    monkeypatch.setattr(
        hub_app,
        "normalize_official_config",
        lambda raw, **_kwargs: {**OfficialTrainSettings().model_dump(mode="python"), **raw},
    )
    monkeypatch.setattr(hub_app, "assert_campaign_allows", lambda *_a, **_k: None)
    monkeypatch.setattr(
        hub_app,
        "is_exact_campaign_maintenance_resume",
        lambda config, new_identity: config["time_cost_per_simulated_second"] == calibrated_time_cost
        and new_identity == identity,
    )
    monkeypatch.setattr(
        hub_app,
        "assert_campaign_training_config",
        lambda _config: (_ for _ in ()).throw(AssertionError("frozen baseline assertion ran")),
    )
    monkeypatch.setattr(hub_app, "save_official_train_settings", lambda _config: None)

    result = asyncio.run(hub_app.create_official_run(Request()))

    assert result == {"run_id": "abc123abc123"}
    assert captured["config"]["time_cost_per_simulated_second"] == calibrated_time_cost
    assert captured["maintenance_resume"]["new_hub_identity"] == identity
