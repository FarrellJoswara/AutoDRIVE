# Camera toggle browser verification

Run on 2026-11-06 with headless Google Chrome and Playwright. The harness starts a temporary Vite server and mock API on dynamically allocated localhost ports. It does not connect to, stop, or restart the live hub or training process.

## Result

All 20 browser assertions passed. The fixture used two active `official_sensors_camera` Watch runs and an active `training_probe` Replay with environments 0 and 1.

- Camera feed defaults on and issues requests for the selected run and environment.
- Turning it off stores `false` under `aicar.camera-feed.enabled` and removes every camera image, including Replay's hidden probe image.
- Watch made zero camera requests for 2.5 seconds with the feed off while telemetry continued. Replay also made zero requests for 2.5 seconds while diagnostic status continued polling.
- Off state survived changing Watch environment and run, navigating between Watch and Replay, and a full browser refresh. Following refresh, both pages remained off with no camera images or new camera requests.
- Turning the feed on requested the current Watch run/environment and the Replay diagnostic run/selected environment. Request URLs included cachebusters.
- With localStorage reads and writes forced to throw, the control still switched off without a browser error.
- No uncaught page errors were recorded.

Evidence: [assertions and request log](../outputs/camera-toggle-qa/results.json). Styled UI screenshots are in [outputs/camera-toggle-qa](../outputs/camera-toggle-qa/): Watch on/off, Watch reenabled on the current run, Watch off after refresh, Replay on/off, and storage-denied fallback. The camera responses are synthetic fixture images.

## Reproduce

```bash
node --check scripts/verify_camera_toggle.mjs
node scripts/verify_camera_toggle.mjs
```

The script expects Playwright at `/tmp/live-lap-browser-qa/node_modules/playwright` and Chrome at `/usr/bin/google-chrome`; `PLAYWRIGHT_MODULE` and `CHROME_BIN` can override those paths. To refresh screenshots without rerunning the functional assertions after a CSS-only change:

```bash
CAMERA_TOGGLE_SCREENSHOTS_ONLY=1 node scripts/verify_camera_toggle.mjs
```
