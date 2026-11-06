import { useEffect, useState } from "react";

const STORAGE_KEY = "aicar.camera-feed.enabled";
let enabledInMemory = true;
let preferenceInitialized = false;
const listeners = new Set<() => void>();

function initializePreference(): boolean {
  if (preferenceInitialized) return enabledInMemory;
  preferenceInitialized = true;
  try {
    const stored = window.localStorage.getItem(STORAGE_KEY);
    if (stored === "true" || stored === "false") enabledInMemory = stored === "true";
  } catch {
    // Keep the in-memory preference when storage is disabled or unavailable.
  }
  return enabledInMemory;
}

function publish(enabled: boolean) {
  enabledInMemory = enabled;
  for (const listener of listeners) listener();
}

export function useCameraFeedEnabled(): [boolean, (enabled: boolean) => void] {
  const [enabled, setEnabled] = useState(initializePreference);

  useEffect(() => {
    const update = () => setEnabled(enabledInMemory);
    const onStorage = (event: StorageEvent) => {
      if (event.key !== STORAGE_KEY && event.key !== null) return;
      publish(event.newValue === "false" ? false : true);
    };
    listeners.add(update);
    window.addEventListener("storage", onStorage);
    update();
    return () => {
      listeners.delete(update);
      window.removeEventListener("storage", onStorage);
    };
  }, []);

  return [enabled, (next) => {
    try {
      window.localStorage.setItem(STORAGE_KEY, String(next));
    } catch {
      // The in-memory value remains authoritative for this page.
    }
    publish(next);
  }];
}

export function CameraFeedControl() {
  const [enabled, setEnabled] = useCameraFeedEnabled();
  return <button
    type="button"
    className="btn camera-feed-button"
    aria-pressed={enabled}
    aria-label={`Camera feed ${enabled ? "on" : "off"}`}
    onClick={() => setEnabled(!enabled)}
    style={{ padding: "0.3rem 0.6rem", fontSize: "0.8rem" }}
  >
    Camera feed: {enabled ? "On" : "Off"}
  </button>;
}

export function CameraFeedPlaceholder() {
  const [, setEnabled] = useCameraFeedEnabled();
  return <div className="meta" style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: "0.65rem", padding: "0.45rem 0.6rem" }}>
    <span>Camera feed is off.</span>
    <button type="button" className="btn camera-feed-button" aria-label="Enable camera feed" onClick={() => setEnabled(true)} style={{ padding: "0.25rem 0.5rem", fontSize: "0.78rem" }}>
      Enable
    </button>
  </div>;
}

