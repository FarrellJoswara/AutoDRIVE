"""Race-KPI accumulation + map/run health checks for the Watch overlay.

Viewer-side only — never imported by the train loop.

KPI math mirrors ``metrics_io.make_metrics``
(``adjusted_time = mean_lap_time + 10 * collisions``) so Watch speaks the same
language as the leaderboard. Watch numbers are **unofficial**: twins spawn
jittered and act stochastically. Promote from ``eval_cli`` / ``compare_models``,
never from what looks fast here.
"""

from __future__ import annotations

import json
import re
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .contracts import CONTRACTS_VERSION
from .eval_protocol import protocol_timeout_s

LAMBDA_COLLISION = 10.0

# Short tags from ``RacingEnv`` info["crash_tag"], in overlay order.
CRASH_TAGS = ("wall", "stall", "ttc", "timeout")


def adjusted_time(mean_lap_time: float | None, collisions: int) -> float | None:
    """Race score, or ``None`` when no lap was completed (no lap → no score)."""
    if mean_lap_time is None:
        return None
    return float(mean_lap_time) + LAMBDA_COLLISION * int(collisions)


def adjusted_or_dq(mean_lap_time: float | None, collisions: int) -> float:
    """Race score with the timeout DQ proxy used by the official protocol."""
    if mean_lap_time is None:
        return float(protocol_timeout_s()) + LAMBDA_COLLISION * int(collisions)
    return float(mean_lap_time) + LAMBDA_COLLISION * int(collisions)


def _fmt(value: float | None, digits: int = 1) -> str:
    return "--" if value is None else f"{float(value):.{digits}f}"


@dataclass
class KpiTally:
    """Lap / collision / crash counters for one episode or a whole session."""

    n_agents: int = 1
    lap_times: list[float] = field(default_factory=list)
    collisions: int = 0
    crash_tags: dict[str, int] = field(default_factory=dict)
    finished: int = 0
    episodes: int = 0

    @property
    def mean_lap_time(self) -> float | None:
        if not self.lap_times:
            return None
        return float(sum(self.lap_times)) / len(self.lap_times)

    @property
    def adjusted_time(self) -> float | None:
        return adjusted_time(self.mean_lap_time, self.collisions)

    @property
    def adjusted_or_dq(self) -> float:
        return adjusted_or_dq(self.mean_lap_time, self.collisions)

    @property
    def dq(self) -> bool:
        """True while no lap has been scored — the number shown is a proxy."""
        return not self.lap_times

    def crash_summary(self) -> str:
        bits = [f"{tag} {self.crash_tags[tag]}" for tag in CRASH_TAGS if self.crash_tags.get(tag)]
        return " ".join(bits) if bits else "none"

    def merge(self, other: "KpiTally") -> None:
        self.lap_times.extend(other.lap_times)
        self.collisions += other.collisions
        self.finished += other.finished
        for tag, n in other.crash_tags.items():
            self.crash_tags[tag] = self.crash_tags.get(tag, 0) + n
        self.episodes += 1


class FleetKpi:
    """Live race KPIs for one fleet (the twins, or the single FTG ghost).

    History is a rolling window: a Watch session can run for hours, and a
    running total of collisions would inflate the score forever instead of
    describing how the current weights drive.
    """

    def __init__(self, n_agents: int, label: str = "PPO", window: int = 10):
        self.label = str(label)
        self.n_agents = max(1, int(n_agents))
        self.ep = KpiTally(n_agents=self.n_agents)
        self.history: deque[KpiTally] = deque(maxlen=max(1, int(window)))

    @property
    def session(self) -> KpiTally:
        """Aggregate over the rolling window of finished episodes."""
        agg = KpiTally(n_agents=self.n_agents)
        for tally in self.history:
            agg.merge(tally)
        return agg

    def start_episode(self, n_agents: int | None = None) -> None:
        if n_agents is not None:
            self.n_agents = max(1, int(n_agents))
        self.ep = KpiTally(n_agents=self.n_agents)

    def record(self, info: dict[str, Any] | None) -> str | None:
        """Fold one ``env.step`` info dict in; returns its crash tag if any."""
        if not info:
            return None
        if info.get("collision"):
            self.ep.collisions += 1
        lap = info.get("lap_time")
        if lap is not None:
            self.ep.lap_times.append(float(lap))
        tag = info.get("crash_tag")
        if tag:
            tag = str(tag)
            self.ep.crash_tags[tag] = self.ep.crash_tags.get(tag, 0) + 1
            return tag
        return None

    def note_finished(self) -> None:
        self.ep.finished += 1

    def end_episode(self) -> None:
        self.history.append(self.ep)

    def shown_tally(self) -> KpiTally:
        """The tally this fleet's overlay row displays (window once it exists)."""
        return self.session if self.history else self.ep

    def line(self, *, suffix: str = "") -> str:
        """One overlay row, e.g. ``PPO  adj 41.3 | lap 31.3 | col 1 | laps 3/8``."""
        ep = self.ep
        adj = f"DQ~{_fmt(ep.adjusted_or_dq)}" if ep.dq else _fmt(ep.adjusted_time)
        row = (
            f"{self.label:<4} adj {adj} | lap {_fmt(ep.mean_lap_time)} "
            f"| col {ep.collisions} | laps {len(ep.lap_times)}/{self.n_agents}"
        )
        return f"{row} | {suffix}" if suffix else row

    def session_line(self) -> str:
        s = self.session
        if s.episodes <= 0:
            return f"{self.label:<4} last eps: (first episode running)"
        adj = f"DQ~{_fmt(s.adjusted_or_dq)}" if s.dq else _fmt(s.adjusted_time)
        return (
            f"{self.label:<4} last {s.episodes} ep: adj {adj} "
            f"| col {s.collisions} | crash {s.crash_summary()}"
        )


def beat_line(ppo: FleetKpi, ghost: FleetKpi) -> str:
    """Δ between the two race scores the overlay is actually showing."""
    a, b = ppo.ep, ghost.shown_tally()
    if a.dq and b.dq:
        return "delta: no lap on either side yet"
    delta = a.adjusted_or_dq - b.adjusted_or_dq
    verdict = "PPO ahead" if delta < 0 else "FTG ahead"
    proxy = " (DQ proxy)" if (a.dq or b.dq) else ""
    return f"delta {delta:+.1f}s vs FTG ghost -> {verdict}{proxy}"


def train_status_line(status: dict[str, Any] | None) -> str | None:
    """Train-side truth from ``live_status.json`` — race counters first, reward last."""
    if not status:
        return None
    bits: list[str] = []
    ts = status.get("timesteps")
    if ts is not None:
        bits.append(f"ts {int(ts)}")
    sps = status.get("steps_per_sec")
    if sps is not None:
        bits.append(f"{float(sps):.0f} steps/s")
    cols = status.get("collisions_estimate")
    if cols is not None:
        bits.append(f"crash eps {int(cols)}")
    rate = status.get("crash_rate_estimate")
    if rate is not None:
        bits.append(f"rate {100.0 * float(rate):.0f}%")
    rew = status.get("ep_rew_mean")
    if rew is not None:
        bits.append(f"rew {float(rew):.1f}")
    if not bits:
        return None
    return "train " + " | ".join(bits)


def read_run_config(models_root: Path, run_id: str | None) -> dict[str, Any] | None:
    """Read ``models/<run_id>/config.json`` (training fingerprint) if present."""
    if not run_id:
        return None
    path = Path(models_root) / str(run_id) / "config.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _map_ids(config: dict[str, Any]) -> list[str]:
    raw = config.get("maps") or ([config["map"]] if config.get("map") else [])
    return [Path(str(p)).stem for p in raw if p]


def map_ids_in_run_id(run_id: str | None) -> list[str]:
    """Best-effort map ids baked into a run_id (``..._ppo_gym_map3_hard``).

    Only ``map<N>`` tokens count: run ids carry free-form suffixes, and a false
    mismatch banner is worse than a missing one.
    """
    if not run_id:
        return []
    return [tok for tok in str(run_id).split("_") if re.fullmatch(r"map\d+", tok)]


def map_mismatch_warning(
    map_yaml: Path,
    config: dict[str, Any] | None,
    *,
    run_id: str | None = None,
) -> str | None:
    """Loud banner text when Watch is not showing the map the run trained on.

    Catches "wrong map picked" and "same name, regenerated content". Without a
    ``config.json`` it falls back to ``map<N>`` tokens in the run id.
    """
    watched = Path(map_yaml).stem
    if not config:
        from_id = map_ids_in_run_id(run_id)
        if from_id and watched not in from_id:
            return f"MAP MISMATCH: watching '{watched}' but run id says {', '.join(from_id)}"
        return None
    trained = _map_ids(config)
    if trained and watched not in trained:
        return f"MAP MISMATCH: watching '{watched}' but run trained on {', '.join(trained)}"
    want = (config.get("map_hashes") or {}).get(watched)
    if want:
        try:
            from .eval_protocol import map_file_hash

            got = map_file_hash(Path(map_yaml))
        except Exception:
            return None
        if got != want:
            return (
                f"MAP CHANGED: '{watched}' content differs from the trained map "
                f"({got[:8]} != {str(want)[:8]}) - Stop and retrain"
            )
    return None


def contracts_warning(config: dict[str, Any] | None) -> str | None:
    """Banner when the run's config predates the frozen contracts."""
    if not config:
        return None
    ver = str(config.get("contracts_version") or "")
    if ver and ver != CONTRACTS_VERSION:
        return f"CONTRACTS MISMATCH: run is v{ver}, this build is v{CONTRACTS_VERSION}"
    return None
