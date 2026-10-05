"""Focused regression test for latest-state WebSocket fleet delivery."""

from __future__ import annotations

import asyncio
import json
import unittest

from src.layer4.hub.telemetry import WSClient


class RecordingWebSocket:
    def __init__(self) -> None:
        self.messages: list[dict] = []

    async def send_text(self, message: str) -> None:
        self.messages.append(json.loads(message))

    async def close(self) -> None:
        return None


class WSClientFleetCoalescingTests(unittest.IsolatedAsyncioTestCase):
    async def test_fleet_snapshots_coalesce_and_history_events_remain_ordered(self) -> None:
        ws = RecordingWebSocket()
        client = WSClient(ws)  # type: ignore[arg-type]
        client.enqueue({"type": "telemetry", "payload": {"kind": "fleet", "step": 10}})
        client.enqueue({"type": "telemetry", "payload": {"kind": "metrics", "step": 10}})
        client.enqueue({"type": "telemetry", "payload": {"kind": "fleet", "step": 20}})
        client.enqueue({"type": "telemetry", "payload": {"kind": "metrics", "step": 20}})
        client.enqueue({"type": "telemetry", "payload": {"kind": "fleet", "step": 30}})
        client.enqueue({"type": "replay_telemetry", "payload": {"kind": "fleet", "step": 7}})

        sender = asyncio.create_task(client.sender())
        for _ in range(10):
            if len(ws.messages) == 4:
                break
            await asyncio.sleep(0)
        client.close()
        await sender

        metrics_steps = [
            message["payload"]["step"]
            for message in ws.messages
            if message["payload"].get("kind") == "metrics"
        ]
        fleet_steps = {
            message["type"]: message["payload"]["step"]
            for message in ws.messages
            if message["payload"].get("kind") == "fleet"
        }
        self.assertEqual(metrics_steps, [10, 20])
        self.assertEqual(fleet_steps, {"telemetry": 30, "replay_telemetry": 7})
        self.assertEqual(client.fleet_frames_coalesced, 2)


if __name__ == "__main__":
    unittest.main()
