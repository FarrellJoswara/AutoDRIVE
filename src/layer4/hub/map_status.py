"""Map selection vs active physics mismatch helpers (Phase 4)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from src.layer4.hub.map_activate import read_active_map, trackloader_available


def map_status_payload(
    maps_root: Optional[Path],
    *,
    selected_id: Optional[str] = None,
    repo_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Return active map + mismatch flags for Mission Control.

    Sims read `.active_map.json` only at boot (entrypoint). After Activate + restart,
    running compose sims should match `active.id`. Local headed sims need a manual
    relaunch. We cannot reliably read in-process map id from Docker Env (entrypoint
    resolves it at start), so mismatch vs sims is inferred: selection ≠ active, or
    TrackLoader missing.
    """
    active = read_active_map(maps_root)
    active_id = active.get("id")  # None = builtin
    sel = (selected_id or "").strip() or None
    if sel == "none":
        sel = None

    selection_mismatch = sel is not None and sel != active_id
    # Also flag when UI selected "none"/builtin but a custom map is active
    if selected_id is not None and selected_id.strip() == "none" and active_id is not None:
        selection_mismatch = True

    loader = trackloader_available(repo_root)
    warnings: List[str] = []
    if selection_mismatch:
        warnings.append(
            f"Fleet selection ({sel or 'builtin'}) ≠ active physics map "
            f"({active_id or 'builtin'}) — Activate to switch sims"
        )
    if active_id is not None and not loader:
        warnings.append(
            "Active map is set but TrackLoader player is not marked installed "
            "(stock binary ignores -map-id)"
        )

    return {
        "active": active,
        "selected_id": sel,
        "selection_mismatch": selection_mismatch,
        "trackloader": loader,
        "warnings": warnings,
    }
