import asyncio
from types import SimpleNamespace

from src.layer1.official_bridge_relay import NEUTRAL_COMMAND, OfficialBridgeRelay


def test_shutdown_sends_neutral_to_each_connected_simulator_before_cleanup(monkeypatch, capsys):
    events = []

    class Router:
        def add_get(self, *_args, **_kwargs):
            pass

    class App:
        def __init__(self, **_kwargs):
            self.router = Router()

    class Runner:
        def __init__(self, _app):
            pass

        async def setup(self):
            events.append("runner_setup")

        async def cleanup(self):
            events.append("runner_cleanup")

    class Site:
        def __init__(self, _runner, *_args):
            pass

        async def start(self):
            events.append("site_start")

    class ImmediateEvent:
        def __init__(self):
            pass

        def set(self):
            pass

        async def wait(self):
            return

    class SocketServer:
        def attach(self, *_args, **_kwargs):
            pass

        async def emit(self, event, *, data, to):
            events.append(("emit", event, data, to))

    class Upstream:
        async def connect(self, *_args, **_kwargs):
            events.append("upstream_connect")

        async def disconnect(self):
            events.append("upstream_disconnect")

    web = SimpleNamespace(
        middleware=lambda fn: fn,
        Application=App,
        AppRunner=Runner,
        TCPSite=Site,
        json_response=lambda *args, **kwargs: None,
    )
    monkeypatch.setattr("aiohttp.web.middleware", web.middleware)
    monkeypatch.setattr("aiohttp.web.Application", web.Application)
    monkeypatch.setattr("aiohttp.web.AppRunner", web.AppRunner)
    monkeypatch.setattr("aiohttp.web.TCPSite", web.TCPSite)
    monkeypatch.setattr("aiohttp.web.json_response", web.json_response)
    monkeypatch.setattr(asyncio, "Event", ImmediateEvent)

    relay = OfficialBridgeRelay.__new__(OfficialBridgeRelay)
    relay.connect_timeout_s = 1.0
    relay.response_timeout_s = 1.0
    relay.shutdown_neutral_timeout_s = 0.5
    relay._connected_simulators = {"sim-a", "sim-b"}
    relay._forwarded_frames = 0
    relay._camera_frames = 0
    relay._incomplete_frames = 0
    relay._response_timeouts = 0
    relay.sio = SocketServer()
    relay.upstream = Upstream()

    asyncio.run(relay.serve("127.0.0.1", 4568, "http://127.0.0.1:4567"))

    emits = [event for event in events if isinstance(event, tuple) and event[0] == "emit"]
    assert len(emits) == 2
    assert {event[3] for event in emits} == {"sim-a", "sim-b"}
    assert all(event[1] == "Bridge" and event[2] == NEUTRAL_COMMAND for event in emits)
    first_emit = min(events.index(event) for event in emits)
    assert first_emit < events.index("upstream_disconnect") < events.index("runner_cleanup")

    output = capsys.readouterr().out
    assert output.count('"event": "shutdown_neutral_command_sent"') == 2
    assert '"event": "shutdown_neutral_release_complete"' in output
    assert '"transport_acknowledged": false' in output
