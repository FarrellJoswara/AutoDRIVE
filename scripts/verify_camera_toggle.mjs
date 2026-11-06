#!/usr/bin/env node
// Isolated browser QA for the Watch/Replay camera feed preference.
// Uses only a temporary Vite server and a synthetic API on ephemeral ports.
// Requires Playwright from /tmp/live-lap-browser-qa and /usr/bin/google-chrome.
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { createServer } from "node:net";
import { createServer as createHttpServer } from "node:http";
import { mkdtemp, rm, writeFile, mkdir } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL, fileURLToPath } from "node:url";
import { createRequire } from "node:module";

const root = resolve(fileURLToPath(new URL("..", import.meta.url)));
const webRoot = join(root, "src/layer4/web");
const outputDir = resolve(process.env.CAMERA_TOGGLE_OUTPUT_DIR ?? join(root, "outputs/camera-toggle-qa"));
const require = createRequire(import.meta.url);
const { chromium } = require(process.env.PLAYWRIGHT_MODULE ?? "/tmp/live-lap-browser-qa/node_modules/playwright");
const bindPort = async () => new Promise((ok, fail) => {
  const server = createServer();
  server.once("error", fail);
  server.listen(0, "127.0.0.1", () => {
    const port = server.address().port;
    server.close((error) => error ? fail(error) : ok(port));
  });
});
const apiPort = await bindPort();
const uiPort = await bindPort();
const tempDir = await mkdtemp(join(tmpdir(), "camera-toggle-qa-"));
const configPath = join(tempDir, "vite.config.mjs");
const viteEntry = pathToFileURL(join(webRoot, "node_modules/vite/dist/node/index.js")).href;
const reactEntry = pathToFileURL(join(webRoot, "node_modules/@vitejs/plugin-react/dist/index.mjs")).href;
await writeFile(configPath, `import { defineConfig } from ${JSON.stringify(viteEntry)};\nimport react from ${JSON.stringify(reactEntry)};\nexport default defineConfig({ root: ${JSON.stringify(webRoot)}, plugins:[react()], server:{host:"127.0.0.1",port:${uiPort},strictPort:true,proxy:{"/health":"http://127.0.0.1:${apiPort}","/api":"http://127.0.0.1:${apiPort}","/train":"http://127.0.0.1:${apiPort}","/settings":"http://127.0.0.1:${apiPort}","/telemetry":"http://127.0.0.1:${apiPort}","/maps":"http://127.0.0.1:${apiPort}"}}});\n`);

const fleet = (runId, step) => ({
  kind: "fleet", runtime: "official", step, episode: 3, run_id: runId,
  ts: new Date().toISOString(), cars: [0, 1].map((env_id) => ({
    env_id, pose: [step / 10 + env_id, env_id], yaw: 0.2, collision: false,
    collision_count: 0, speed: 3.1 + env_id, speed_source: "simulator",
    episode_return: null, lap_count: 2, lap_elapsed_s: 8.5 + env_id,
    last_lap_time_s: 43.2 + env_id,
    lap_session: { completed_laps: 2, latest_lap_time_s: 43.2 + env_id,
      best_lap_time_s: 43.2 + env_id, last_lap_delta_s: null,
      current_lap_number: 3, current_lap_elapsed_s: 8.5 + env_id },
  })),
});
const runs = [
  { run_id: "qa-camera-one", state: "running", display_name: "Camera QA one", map_id: "none", config: { simulator_backend: "windows_external", observation_profile: "official_sensors_camera" } },
  { run_id: "qa-camera-two", state: "running", display_name: "Camera QA two", map_id: "none", config: { simulator_backend: "windows_external", observation_profile: "official_sensors_camera" } },
];
let telemetryRequests = 0;
let replayPolls = 0;
let cameraRequests = [];
let fleetStep = 10;
const replayFleet = () => fleet("replay_qa-camera-diagnostic", fleetStep);
const replayStatus = () => ({ state: "running", run_id: "replay_qa-camera-diagnostic", model: "qa-checkpoint", validation_mode: "training_probe", attempts_completed: 0, attempts_required: 1, last_fleet: replayFleet() });
const png = Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/f7sAAAAASUVORK5CYII=", "base64");
const response = (res, status, body, type = "application/json") => {
  res.writeHead(status, { "content-type": type, "access-control-allow-origin": "*", "cache-control": "no-store" });
  res.end(type === "application/json" ? JSON.stringify(body) : body);
};
const api = createHttpServer((req, res) => {
  const url = new URL(req.url, `http://127.0.0.1:${apiPort}`);
  if (url.pathname === "/health") return response(res, 200, { ok: true });
  if (url.pathname === "/api/maps") return response(res, 200, []);
  if (url.pathname === "/train/runs") return response(res, 200, { runs, max_concurrent_runs: 2 });
  if (url.pathname === "/api/replay/status") { replayPolls += 1; return response(res, 200, replayStatus()); }
  if (url.pathname === "/api/replay/models") return response(res, 200, []);
  if (url.pathname === "/api/evaluations/best") return response(res, 200, { best: null });
  if (url.pathname === "/api/evaluations/history") return response(res, 200, []);
  if (url.pathname === "/train/official-settings") return response(res, 200, { settings: {} });
  if (/^\/train\/runs\/[^/]+\/(evaluations|ppo-diagnostics)$/.test(url.pathname)) return response(res, 200, []);
  const telemetryMatch = url.pathname.match(/^\/train\/runs\/([^/]+)\/telemetry$/);
  if (telemetryMatch) {
    telemetryRequests += 1;
    fleetStep += 1;
    const runId = decodeURIComponent(telemetryMatch[1]);
    return response(res, 200, { run_id: runId, state: "running", map_id: "none", updated_at: new Date().toISOString(), last_fleet: fleet(runId, fleetStep) });
  }
  const cameraMatch = url.pathname.match(/^\/train\/runs\/([^/]+)\/camera\/(\d+)$/);
  if (cameraMatch) {
    cameraRequests.push({ run_id: decodeURIComponent(cameraMatch[1]), env_id: Number(cameraMatch[2]), cachebuster: url.searchParams.get("v"), at: new Date().toISOString() });
    return response(res, 200, png, "image/png");
  }
  console.error("Unmocked API route:", req.method, url.pathname, url.search);
  return response(res, 404, { detail: `No synthetic route for ${url.pathname}` });
});

const children = [];
let browser;
try {
  await mkdir(outputDir, { recursive: true });
  await new Promise((ok, fail) => api.once("error", fail).listen(apiPort, "127.0.0.1", ok));
  const vite = spawn(process.env.NODE ?? "/usr/bin/node", [join(webRoot, "node_modules/vite/bin/vite.js"), "--config", configPath], { cwd: root, stdio: ["ignore", "pipe", "pipe"] });
  children.push(vite);
  let viteLog = "";
  vite.stdout.on("data", (chunk) => { viteLog += chunk.toString(); });
  vite.stderr.on("data", (chunk) => { viteLog += chunk.toString(); });
  const readyDeadline = Date.now() + 30000;
  let ready = false;
  while (Date.now() < readyDeadline) {
    if (vite.exitCode != null) throw new Error(`Vite exited early: ${viteLog}`);
    try { const response = await fetch(`http://127.0.0.1:${uiPort}`); if (response.ok) { ready = true; break; } } catch { /* wait for isolated Vite */ }
    await new Promise((r) => setTimeout(r, 200));
  }
  assert.ok(ready, `Vite did not become ready: ${viteLog}`);
  browser = await chromium.launch({ headless: true, executablePath: process.env.CHROME_BIN ?? "/usr/bin/google-chrome", args: ["--no-sandbox"] });
  const context = await browser.newContext({ viewport: { width: 1440, height: 1100 }, deviceScaleFactor: 1 });
  const page = await context.newPage();
  page.setDefaultTimeout(8000);
  const pageErrors = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  page.on("console", (message) => { if (message.type() === "error") pageErrors.push(`console: ${message.text()}`); });
  const assertions = [];
  const check = (label, value) => { assert.ok(value, label); assertions.push(label); };
  const wait = (ms) => page.waitForTimeout(ms);
  const cameraControl = () => page.getByRole("button", { name: /^Camera feed (on|off)$/i });
  const clickCamera = async (enabled) => {
    const control = cameraControl();
    await control.waitFor({ state: "visible" });
    const pressed = (await control.getAttribute("aria-pressed")) === "true";
    if (pressed !== enabled) await control.click();
    await page.waitForFunction((expected) => document.querySelector('button[aria-label^="Camera feed "]')?.getAttribute("aria-pressed") === String(expected), enabled);
    return control;
  };
  const cameraImages = () => page.locator('img[src*="/camera/"]');
  const waitForCamera = async (predicate, description) => page.waitForFunction(({ start, runId, envId }) => {
    return window.__cameraRequests?.slice(start).some((item) => item.run_id === runId && item.env_id === envId);
  }, { start: cameraRequests.length, runId: predicate.runId, envId: predicate.envId }, { timeout: 7000 }).catch(() => { throw new Error(`Timed out waiting for camera request ${description}`); });
  await page.route("**/train/runs/*/camera/**", async (route) => {
    const url = new URL(route.request().url());
    const match = url.pathname.match(/\/train\/runs\/([^/]+)\/camera\/(\d+)$/);
    if (match) await page.evaluate((item) => { window.__cameraRequests ??= []; window.__cameraRequests.push(item); }, { run_id: decodeURIComponent(match[1]), env_id: Number(match[2]), cachebuster: url.searchParams.get("v") });
    await route.continue();
  });

  if (process.env.CAMERA_TOGGLE_SCREENSHOTS_ONLY === "1") {
    await page.goto(`http://127.0.0.1:${uiPort}`, { waitUntil: "networkidle" });
    await page.getByRole("button", { name: "Watch", exact: true }).click();
    await page.getByRole("button", { name: "Focus run", exact: true }).click();
    await page.locator(".watch-run-select select").selectOption("qa-camera-one");
    await page.getByRole("region", { name: "Front camera" }).waitFor();
    await cameraImages().first().waitFor({ state: "attached" });
    await page.screenshot({ path: join(outputDir, "watch-camera-on.png"), fullPage: true });
    await clickCamera(false);
    await page.waitForFunction(() => document.querySelectorAll('img[src*="/camera/"]').length === 0);
    await page.screenshot({ path: join(outputDir, "watch-camera-off.png"), fullPage: true });
    await page.getByRole("button", { name: "Replay", exact: true }).click();
    await page.getByRole("region", { name: "Training probe telemetry" }).waitFor();
    await page.screenshot({ path: join(outputDir, "replay-camera-off.png"), fullPage: true });
    await clickCamera(true);
    await cameraImages().first().waitFor({ state: "attached" });
    await page.screenshot({ path: join(outputDir, "replay-camera-on.png"), fullPage: true });
    console.log("Updated styled Watch/Replay screenshots only; full functional assertions were not rerun.");
  } else {
  await page.goto(`http://127.0.0.1:${uiPort}`, { waitUntil: "networkidle" });
  await page.getByRole("button", { name: "Watch", exact: true }).click();
  await page.getByRole("heading", { name: "Watch" }).waitFor();
  await page.getByRole("button", { name: "Focus run", exact: true }).click();
  await page.locator(".watch-run-select select").selectOption("qa-camera-one");
  await page.getByRole("region", { name: "Front camera" }).waitFor();
  const watchCamera = page.getByRole("region", { name: "Front camera" });
  await waitForCamera({ runId: "qa-camera-one", envId: 0 }, "Watch default Env 0");
  check("camera preference defaults on and Watch requests the selected run / Env 0", (await cameraControl().getAttribute("aria-pressed")) === "true" && cameraRequests.some((x) => x.run_id === "qa-camera-one" && x.env_id === 0));
  await page.screenshot({ path: join(outputDir, "watch-camera-on.png"), fullPage: true });

  await clickCamera(false);
  await page.waitForFunction(() => localStorage.getItem("aicar.camera-feed.enabled") === "false");
  await page.waitForFunction(() => document.querySelectorAll('img[src*="/camera/"]').length === 0);
  check("Watch off state persists false and unmounts camera images", (await cameraControl().getAttribute("aria-pressed")) === "false" && await cameraImages().count() === 0 && (await page.evaluate(() => localStorage.getItem("aicar.camera-feed.enabled"))) === "false");
  const offCameraCount = cameraRequests.length;
  const offTelemetryCount = telemetryRequests;
  const watchRefreshPeriod = 1000;
  await wait(2 * watchRefreshPeriod + 500);
  check("Watch off produced zero camera requests for more than two refresh intervals", cameraRequests.length === offCameraCount);
  check("Watch telemetry continued updating while camera feed was off", telemetryRequests > offTelemetryCount);
  await page.screenshot({ path: join(outputDir, "watch-camera-off.png"), fullPage: true });

  await page.locator(".fleet-toolbar select").selectOption("1");
  await page.waitForFunction(() => document.querySelector('[aria-label="Front camera"] h3')?.textContent?.includes("Env 1"));
  await page.locator(".watch-run-select select").selectOption("qa-camera-two");
  await wait(400);
  check("off preference remains in force after Watch environment and run changes", await cameraImages().count() === 0 && cameraRequests.length === offCameraCount);
  const beforeReenable = cameraRequests.length;
  await clickCamera(true);
  await page.waitForFunction(() => localStorage.getItem("aicar.camera-feed.enabled") === "true");
  await page.waitForFunction(() => document.querySelectorAll('img[src*="/camera/"]').length > 0);
  await page.waitForFunction(({ before }) => window.__cameraRequests?.slice(before).some((item) => item.run_id === "qa-camera-two" && item.env_id === 1), { before: beforeReenable });
  const reenabledRequest = cameraRequests.slice(beforeReenable).find((x) => x.run_id === "qa-camera-two" && x.env_id === 1);
  check("reenabling Watch immediately requests current run / Env 1 with a fresh cachebuster", Boolean(reenabledRequest && reenabledRequest.cachebuster));
  check("Watch environment selector remains on Env 1 after reenable", (await page.locator(".fleet-toolbar select").inputValue()) === "1");
  await page.screenshot({ path: join(outputDir, "watch-camera-reenabled-current-run.png"), fullPage: true });

  await page.reload({ waitUntil: "networkidle" });
  await page.getByRole("button", { name: "Watch", exact: true }).click();
  await page.getByRole("button", { name: "Focus run", exact: true }).click();
  await page.locator(".watch-run-select select").selectOption("qa-camera-two");
  await page.waitForFunction(() => document.querySelector('[aria-label="Front camera"] h3')?.textContent?.includes("Env 0"));
  check("refresh restores camera feed preference from storage", (await cameraControl().getAttribute("aria-pressed")) === "true" && (await page.evaluate(() => localStorage.getItem("aicar.camera-feed.enabled"))) === "true");

  await clickCamera(false);
  await page.waitForFunction(() => localStorage.getItem("aicar.camera-feed.enabled") === "false");
  await page.getByRole("button", { name: "Replay", exact: true }).click();
  await page.getByRole("heading", { name: "Official evaluation / replay" }).waitFor();
  await page.getByRole("region", { name: "Training probe telemetry" }).waitFor();
  check("Replay inherits the off preference and mounts no visible or hidden probe camera images", (await cameraControl().getAttribute("aria-pressed")) === "false" && await cameraImages().count() === 0);
  const replayOffRequests = cameraRequests.length;
  const replayTelemetryBefore = replayPolls;
  await wait(2500);
  check("Replay off produced zero camera requests for over two refresh intervals", cameraRequests.length === replayOffRequests);
  check("Replay diagnostic telemetry continued polling while camera feed was off", replayPolls > replayTelemetryBefore);
  await page.screenshot({ path: join(outputDir, "replay-camera-off.png"), fullPage: true });

  await page.getByLabel("Focus environment").selectOption("1");
  const replayBeforeOn = cameraRequests.length;
  await clickCamera(true);
  await page.waitForFunction(({ before }) => window.__cameraRequests?.slice(before).some((item) => item.run_id === "replay_qa-camera-diagnostic" && item.env_id === 1), { before: replayBeforeOn });
  check("Replay reenable immediately requests the active diagnostic run / selected Env 1", cameraRequests.slice(replayBeforeOn).some((x) => x.run_id === "replay_qa-camera-diagnostic" && x.env_id === 1));
  check("Replay probe image markup includes visible and hidden feed while enabled", await cameraImages().count() >= 2);
  await page.screenshot({ path: join(outputDir, "replay-camera-on.png"), fullPage: true });

  await clickCamera(false);
  await page.waitForFunction(() => document.querySelectorAll('img[src*="/camera/"]').length === 0);
  await page.reload({ waitUntil: "networkidle" });
  const cameraCountBeforeOffReload = cameraRequests.length;
  await page.getByRole("button", { name: "Replay", exact: true }).click();
  await page.getByRole("region", { name: "Training probe telemetry" }).waitFor();
  check("reload while off restores persisted false in Replay", (await cameraControl().getAttribute("aria-pressed")) === "false" && await cameraImages().count() === 0 && (await page.evaluate(() => localStorage.getItem("aicar.camera-feed.enabled"))) === "false");
  const replayPollsBeforeOffReloadWait = replayPolls;
  await wait(2500);
  check("after off-state reload, Replay still makes zero requests for more than two refresh intervals", cameraRequests.length === cameraCountBeforeOffReload);
  check("after off-state reload, Replay diagnostic telemetry continues polling", replayPolls > replayPollsBeforeOffReloadWait);
  await page.getByRole("button", { name: "Watch", exact: true }).click();
  await page.getByRole("button", { name: "Focus run", exact: true }).click();
  await page.locator(".watch-run-select select").selectOption("qa-camera-one");
  check("off preference after reload also carries into Watch with no camera image", (await cameraControl().getAttribute("aria-pressed")) === "false" && await cameraImages().count() === 0 && cameraRequests.length === cameraCountBeforeOffReload);
  await page.screenshot({ path: join(outputDir, "watch-camera-off-after-refresh.png"), fullPage: true });

  const switchRunOffCount = cameraRequests.length;
  await page.getByRole("button", { name: "Watch", exact: true }).click();
  await page.getByRole("button", { name: "Focus run", exact: true }).click();
  await page.locator(".watch-run-select select").selectOption("qa-camera-one");
  check("off preference follows page navigation back to Watch", (await cameraControl().getAttribute("aria-pressed")) === "false" && await cameraImages().count() === 0 && cameraRequests.length === switchRunOffCount);

  const deniedContext = await browser.newContext({ viewport: { width: 1200, height: 900 } });
  await deniedContext.addInitScript(() => {
    const original = Storage.prototype.setItem;
    Storage.prototype.setItem = function(key, value) {
      if (key === "aicar.camera-feed.enabled") throw new DOMException("Storage disabled", "QuotaExceededError");
      return original.call(this, key, value);
    };
    Storage.prototype.getItem = function(key) { if (key === "aicar.camera-feed.enabled") throw new DOMException("Storage disabled", "SecurityError"); return null; };
  });
  const deniedPage = await deniedContext.newPage();
  deniedPage.setDefaultTimeout(8000);
  deniedPage.on("pageerror", (error) => pageErrors.push(`storage-denied: ${error.message}`));
  await deniedPage.goto(`http://127.0.0.1:${uiPort}`, { waitUntil: "networkidle" });
  await deniedPage.getByRole("button", { name: "Watch", exact: true }).click();
  await deniedPage.getByRole("button", { name: "Focus run", exact: true }).click();
  await deniedPage.locator(".watch-run-select select").selectOption("qa-camera-one");
  const deniedControl = deniedPage.getByRole("button", { name: /^Camera feed on$/i });
  await deniedControl.waitFor();
  await deniedControl.click();
  await deniedPage.getByRole("button", { name: /^Camera feed off$/i }).waitFor();
  await deniedPage.waitForFunction(() => document.querySelectorAll('img[src*="/camera/"]').length === 0);
  check("camera setting still switches off cleanly when localStorage reads and writes throw", await deniedPage.getByRole("button", { name: /^Camera feed off$/i }).getAttribute("aria-pressed") === "false" && await deniedPage.locator('img[src*="/camera/"]').count() === 0);
  await deniedPage.screenshot({ path: join(outputDir, "storage-denied-camera-off.png"), fullPage: true });
  await deniedContext.close();

  check("no uncaught browser page errors", pageErrors.length === 0);
  const evidence = {
    completed_at_utc: new Date().toISOString(),
    method: "Headless Google Chrome via Playwright; isolated Vite UI and synthetic API on dynamically allocated localhost ports.",
    fixture: { camera_runs: runs.map((run) => ({ run_id: run.run_id, observation_profile: run.config.observation_profile })), replay_run_id: "replay_qa-camera-diagnostic", replay_validation_mode: "training_probe", environments: [0, 1] },
    assertions,
    camera_request_count: cameraRequests.length,
    camera_requests: cameraRequests,
    telemetry_requests: telemetryRequests,
    replay_status_polls: replayPolls,
    persisted_preference_key: "aicar.camera-feed.enabled",
    screenshots: ["watch-camera-on.png", "watch-camera-off.png", "watch-camera-reenabled-current-run.png", "watch-camera-off-after-refresh.png", "replay-camera-off.png", "replay-camera-on.png", "storage-denied-camera-off.png"],
    page_errors: pageErrors,
  };
  await writeFile(join(outputDir, "results.json"), `${JSON.stringify(evidence, null, 2)}\n`);
  console.log(JSON.stringify(evidence, null, 2));
  }
  await context.close();
} finally {
  if (browser) await browser.close();
  for (const child of children) child.kill("SIGTERM");
  await new Promise((done) => api.close(() => done()));
  await rm(tempDir, { recursive: true, force: true });
}
