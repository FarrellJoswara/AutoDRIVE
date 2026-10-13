"""Pure Watch snapshot helpers kept separate from hub startup side effects."""

from datetime import datetime, timezone
from typing import Any


def watch_snapshot_age_ms(payload: Any) -> int | None:
    """Age of the source frame, not how recently the hub served it."""
    if not isinstance(payload, dict):
        return None
    timestamp = payload.get("ts")
    if not isinstance(timestamp, str):
        return None
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return max(0, int((datetime.now(timezone.utc) - parsed).total_seconds() * 1000))
    except (TypeError, ValueError, OverflowError):
        return None
