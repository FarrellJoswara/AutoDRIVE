#!/usr/bin/env node
// Isolated browser QA for the Watch live lap panel. Runs Vite and a synthetic
// API on dynamically allocated localhost ports; never contacts the live hub.
// Requires Playwright (install with `npm install --prefix /tmp/live-lap-browser-qa --package-lock=false playwright`; PLAYWRIGHT_MODULE may override its package entry).
import assert from "node:assert/strict";
import { spawn, execFileSync } from "node:child_process";
import { createServer } from "node:net";
import { createServer as createHttpServer } from "node:http";
import { mkdtemp, rm, writeFile, mkdir } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL, fileURLToPath } from "node:url";
import { createRequire } from "node:module";

const root = resolve(fileURLToPath(new URL("..", import.meta.url)));
const webRoot = join(root, "src/layer4/web");
const outputDir = resolve(process.env.LAP_UI_OUTPUT_DIR ?? join(root, "outputs/lap-ui-browser-qa"));
const require = createRequire(import.meta.url);
const playwrightEntry = process.env.PLAYWRIGHT_MODULE ?? "/tmp/live-lap-browser-qa/node_modules/playwright";
const { chromium } = require(playwrightEntry);
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
const tempDir = await mkdtemp(join(tmpdir(), "lap-ui-qa-"));
const configPath = join(tempDir, "vite.config.mjs");
const viteEntry = pathToFileURL(join(webRoot, "node_modules/vite/dist/node/index.js")).href;
const reactEntry = pathToFileURL(join(webRoot, "node_modules/@vitejs/plugin-react/dist/index.mjs")).href;
await writeFile(configPath, `import { defineConfig } from ${JSON.stringify(viteEntry)};\nimport react from ${JSON.stringify(reactEntry)};\nexport default defineConfig({ root: ${JSON.stringify(webRoot)}, plugins:[react()], server:{host:"127.0.0.1",port:${uiPort},strictPort:true,proxy:{"/health":"http://127.0.0.1:${apiPort}","/api":"http://127.0.0.1:${apiPort}","/train":"http://127.0.0.1:${apiPort}","/settings":"http://127.0.0.1:${apiPort}","/telemetry":"http://127.0.0.1:${apiPort}","/maps":"http://127.0.0.1:${apiPort}"}}});\n`);

const fleetCar = (env_id, values) => ({
  env_id, pose: [env_id * 2, 0], yaw: 0, collision: false, collision_count: 0,
  speed: 2.4, speed_source: "simulator", episode_return: null, lap_count: values.completed_laps,
  lap_elapsed_s: values.current_lap_elapsed_s,
  last_lap_time_s: values.latest_lap_time_s,
  lap_session: values,
});
let telemetryRequests = 0;
let disconnected = false;
let phase = 0;
let staleTelemetry = false;
let historyRequestedUrl = null;
let historyFailuresRemaining = 1;
let historyRequestCount = 0;
const historyPayload = JSON.parse(execFileSync("python", ["-c", String.raw`
import importlib.util, json, tempfile
from pathlib import Path
spec = importlib.util.spec_from_file_location("lap_session", "src/layer4/hub/lap_session.py")
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
with tempfile.TemporaryDirectory() as tmp:
    store = module.LapSessionStore(Path(tmp))
    for run_id, track, layout, direction, times in [
        ("qa-live", "porto", "main", "forward", [42.5, 41.9]),
        ("qa-prior-match", "porto", "main", "forward", [44.0]),
        ("qa-prior-wrong-track", "other", "main", "forward", [30.0]),
        ("replay_qa-session", "porto", "main", "forward", [39.9]),
    ]:
        store.observe({"run_id":run_id,"episode":2,"map_id":track,"layout_id":layout,"direction":direction,"ts":"2026-11-06T13:00:00Z","cars":[{"env_id":0,"episode":2,"lap_times_s":times,"lap_elapsed_s":3.2}]})
    print(json.dumps({"watch":store.history("qa-live", "0"), "replay":store.history("replay_qa-session", "0")}))
`], { cwd: root, encoding: "utf8" }));
const historyPayloads = historyPayload;
const watchHistoryPayload = historyPayloads.watch;
const replayFleet = {
  kind: "fleet", runtime: "official", step: 90, episode: 2, run_id: "replay_qa-session", ts: new Date().toISOString(),
  cars: [fleetCar(0, { completed_laps: 1, latest_lap_time_s: 39.9, best_lap_time_s: 39.9,
    last_lap_delta_s: -0.5, current_lap_number: 2, current_lap_elapsed_s: 4.4 })],
};

const runs = [{
  run_id: "qa-live", state: "running", display_name: "Synthetic lap QA", map_id: "porto",
  config: { simulator_backend: "linux_external", observation_profile: "official_sensors" },
}];
const response = (res, status, body) => {
  res.writeHead(status, { "content-type": "application/json", "access-control-allow-origin": "*" });
  res.end(JSON.stringify(body));
};
const api = createHttpServer((req, res) => {
  const url = new URL(req.url, `http://127.0.0.1:${apiPort}`);
  if (url.pathname === "/health") return response(res, 200, { ok: true });
  if (url.pathname === "/api/maps") return response(res, 200, []);
  if (url.pathname === "/api/replay/status") return response(res, 200, { state: "running", run_id: "replay_qa-session", model: "qa-checkpoint", last_fleet: { ...replayFleet, ts: new Date().toISOString() } });
  if (url.pathname === "/api/replay/models") return response(res, 200, []);
  if (url.pathname === "/api/evaluations/best") return response(res, 200, { best: null });
  if (url.pathname === "/api/evaluations/history") return response(res, 200, []);
  if (url.pathname === "/train/official-settings") return response(res, 200, { settings: {} });
  if (/^\/train\/runs\/[^/]+\/(evaluations|ppo-diagnostics)$/.test(url.pathname)) return response(res, 200, url.pathname.endsWith("/evaluations") ? [] : { run_id: "qa-live", updates: [] });
  if (url.pathname === "/train/runs") return response(res, 200, { runs, max_concurrent_runs: 2 });
  if (url.pathname === "/train/runs/qa-live/telemetry") {
    telemetryRequests += 1;
    if (disconnected) return response(res, 503, { detail: "synthetic disconnect" });
    phase += 1;
    const newBest = phase >= 40;
    const stamp = staleTelemetry ? new Date(Date.now() - 15000).toISOString() : new Date().toISOString();
    const cars = [
      fleetCar(0, { completed_laps: newBest ? 2 : 1, latest_lap_time_s: newBest ? 41.9 : 42.5,
        best_lap_time_s: newBest ? 41.9 : 42.5, last_lap_delta_s: newBest ? -0.6 : null,
        current_lap_number: newBest ? 3 : 2, current_lap_elapsed_s: 3.2 }),
      fleetCar(1, { completed_laps: 1, latest_lap_time_s: 46.125, best_lap_time_s: 46.125,
        last_lap_delta_s: null, current_lap_number: 2, current_lap_elapsed_s: 5.75 }),
    ];
    return response(res, 200, { run_id: "qa-live", state: "running", map_id: "porto", updated_at: stamp,
      last_fleet: { kind: "fleet", runtime: "official", step: phase, episode: 2, run_id: "qa-live", ts: stamp, cars } });
  }
  if (/^\/train\/runs\/[^/]+\/lap-history$/.test(url.pathname)) {
    historyRequestedUrl = `${url.pathname}${url.search}`;
    historyRequestCount += 1;
    // Match the actual handler's query parameter and fail closed on the UI's
    // former /cars/{envId} path form. Payload was emitted by LapSessionStore.
    if (url.searchParams.get("env_id") !== "0") return response(res, 422, { detail: "env_id query parameter required" });
    if (historyFailuresRemaining > 0) {
      historyFailuresRemaining -= 1;
      return response(res, 503, { detail: "synthetic history transient failure" });
    }
    return response(res, 200, historyPayloads[url.pathname.includes("replay_qa-session") ? "replay" : "watch"]);
  }
  console.error("unmocked API route:", req.method, url.pathname, url.search);
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
  while (Date.now() < readyDeadline) {
    if (vite.exitCode != null) throw new Error(`Vite exited early: ${viteLog}`);
    try { await fetch(`http://127.0.0.1:${uiPort}`); break; } catch { await new Promise((r) => setTimeout(r, 200)); }
  }
  browser = await chromium.launch({ headless: true, executablePath: process.env.CHROME_BIN ?? "/usr/bin/google-chrome", args: ["--no-sandbox"] });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1100 }, deviceScaleFactor: 1 });
  page.setDefaultTimeout(7000);
  await page.route("**/favicon.ico", (route) => route.fulfill({ status: 204, body: "" }));
  const pageErrors = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  page.on("console", (message) => { if (message.type() === "error") console.error("browser console:", message.text()); });
  const assertions = [];
  const check = (label, condition) => { assert.ok(condition, label); assertions.push(label); };
  await page.goto(`http://127.0.0.1:${uiPort}`, { waitUntil: "networkidle" });
  await page.getByRole("button", { name: "Watch", exact: true }).click();
  await page.getByRole("heading", { name: "Watch" }).waitFor({ state: "visible" });
  await page.getByRole("button", { name: "Focus run", exact: true }).click();
  await page.locator(".watch-run-select select").waitFor({ state: "visible" });
  await page.locator(".watch-run-select select").selectOption("qa-live");
  const panel = page.getByRole("region", { name: "Live lap timing" });
  await panel.getByText("Latest lap", { exact: true }).waitFor();
  await panel.locator(".live-lap-primary strong").first().filter({ hasText: "0:42.500" }).waitFor();
  check("first completed lap has no previous-best delta", (await panel.locator(".live-lap-secondary span").nth(2).locator("strong").innerText()) === "—");
  const frozenElapsed = await panel.locator(".live-lap-secondary span").nth(1).locator("strong").innerText();
  await page.waitForTimeout(500);
  check("paused current-lap timer holds the last reported elapsed value", frozenElapsed === "0:03.200" && (await panel.locator(".live-lap-secondary span").nth(1).locator("strong").innerText()) === frozenElapsed);
  await panel.getByRole("status").filter({ hasText: "New session best" }).waitFor({ timeout: 7000 });
  check("latest lap displays formatted 0:41.900", (await panel.innerText()).includes("0:41.900"));
  check("session best displays formatted 0:41.900", (await panel.innerText()).includes("Session best\n0:41.900"));
  check("new-best indicator appears after a faster completed lap", await panel.getByRole("status").filter({ hasText: "New session best" }).count() === 1);
  check("new-best visual pulse is exposed on panel", await panel.getAttribute("data-new-best") === "true");
  check("delta against previous best is shown with faster styling", (await panel.locator(".lap-delta-faster").innerText()) === "-0.600 s");
  await page.screenshot({ path: join(outputDir, "live-lap-new-best.png"), fullPage: true });
  await panel.getByRole("status").filter({ hasText: "New session best" }).waitFor({ state: "hidden", timeout: 4000 });
  check("new-best indicator clears at its 2.2 second boundary", await panel.getAttribute("data-new-best") === "false");

  await page.locator(".fleet-toolbar select").selectOption("1");
  await page.getByRole("heading", { name: "Lap timing · Env 1" }).waitFor();
  check("per-car selection changes lap panel to Env 1", (await panel.innerText()).includes("0:46.125"));
  check("selected car current elapsed time shown", (await panel.innerText()).includes("0:05.750"));

  await page.locator(".fleet-toolbar select").selectOption("0");
  await panel.getByRole("button", { name: /Session details/ }).click();
  await panel.getByRole("alert").filter({ hasText: "Session details unavailable" }).waitFor();
  check("history failure offers a retry action", await panel.getByRole("button", { name: "Retry session details" }).count() === 1);
  await panel.getByRole("button", { name: "Retry session details" }).click();
  await panel.getByText("Current session", { exact: true }).waitFor();
  check("session details show actual current session laps", (await panel.innerText()).includes("Lap 2 · episode 2"));
  check("session details show backend-compatible previous session", (await panel.innerText()).includes("qa-prior-match"));
  check("wrong-track session is excluded by actual backend result", !watchHistoryPayload.previous_sessions.some((row) => row.run_id === "qa-prior-wrong-track"));
  check("UI requested actual backend route/query contract", historyRequestedUrl === "/train/runs/qa-live/lap-history?env_id=0");
  check("history retries after the synthetic transient backend failure", historyRequestCount >= 2);
  await page.screenshot({ path: join(outputDir, "lap-session-history.png"), fullPage: true });

  disconnected = true;
  await page.waitForFunction(() => document.querySelector('[aria-label="Live lap timing"]')?.textContent?.includes("Disconnected · showing last received values"), null, { timeout: 5000 });
  check("disconnect state retains last completed and current lap values", (await panel.innerText()).includes("Disconnected · showing last received values") && (await panel.innerText()).includes("0:41.900") && (await panel.innerText()).includes("0:03.200"));
  await page.screenshot({ path: join(outputDir, "lap-ui-disconnected.png"), fullPage: true });

  disconnected = false;
  staleTelemetry = true;
  await page.waitForFunction(() => document.querySelector('[aria-label="Live lap timing"]')?.textContent?.includes("Stale · showing last received values"), null, { timeout: 5000 });
  check("stale telemetry is labeled and last received timing remains visible", (await panel.innerText()).includes("Stale · showing last received values") && (await panel.innerText()).includes("0:41.900"));
  const requestsBeforeRefresh = telemetryRequests;
  await page.reload({ waitUntil: "networkidle" });
  await page.getByRole("button", { name: "Watch", exact: true }).click();
  await page.getByRole("button", { name: "Focus run", exact: true }).click();
  await page.locator(".watch-run-select select").selectOption("qa-live");
  await page.waitForTimeout(450);
  check("refresh reloads Watch and requests new telemetry", telemetryRequests > requestsBeforeRefresh);
  check("browser reports no uncaught page errors", pageErrors.length === 0);
  await page.screenshot({ path: join(outputDir, "lap-ui-stale-after-refresh.png"), fullPage: true });

  await page.getByRole("button", { name: "Replay", exact: true }).click();
  await page.getByRole("heading", { name: "Official evaluation / replay" }).waitFor();
  const replayPanel = page.getByRole("region", { name: "Live lap timing" });
  await replayPanel.getByText("Latest lap", { exact: true }).waitFor();
  check("Replay renders shared lap panel from its current replay fleet", (await replayPanel.innerText()).includes("Lap timing · Env 0") && (await replayPanel.innerText()).includes("0:39.900"));
  check("Replay session identity uses replay run_id, not checkpoint model id", (await page.locator("body").innerText()).includes("replay_qa-session") && historyPayloads.replay.run_id === "replay_qa-session");
  check("Replay panel shows its live elapsed current lap", (await replayPanel.innerText()).includes("0:04.400"));
  await replayPanel.getByRole("button", { name: /Session details/ }).click();
  await replayPanel.getByText("Current session", { exact: true }).waitFor();
  check("Replay lap history uses replay run path and actual backend query", historyRequestedUrl === "/train/runs/replay_qa-session/lap-history?env_id=0");
  check("Replay history displays compatible prior session", (await replayPanel.innerText()).includes("qa-prior-match"));
  await page.screenshot({ path: join(outputDir, "replay-live-lap-history.png"), fullPage: true });
  const evidence = {
    completed_at_utc: new Date().toISOString(),
    method: "Headless Google Chrome via Playwright; isolated Vite UI and synthetic API on dynamically allocated localhost ports.",
    backend_contract: "History fixture generated directly by src/layer4/hub/lap_session.py LapSessionStore.history(); HTTP path/query checked against GET /train/runs/{run_id}/lap-history?env_id=.",
    assertions,
    telemetryRequests,
    historyRequestedUrl,
    historyPayloadShape: Object.keys(watchHistoryPayload),
    backendGeneratedCompatibleSessions: watchHistoryPayload.previous_sessions.map((row) => row.run_id),
    screenshots: ["live-lap-new-best.png", "lap-session-history.png", "lap-ui-disconnected.png", "lap-ui-stale-after-refresh.png", "replay-live-lap-history.png"],
    pageErrors,
  };
  await writeFile(join(outputDir, "results.json"), `${JSON.stringify(evidence, null, 2)}\n`);
  console.log(JSON.stringify(evidence, null, 2));
} finally {
  if (browser) await browser.close();
  for (const child of children) { child.kill("SIGTERM"); }
  await new Promise((resolveDone) => api.close(() => resolveDone()));
  await rm(tempDir, { recursive: true, force: true });
}
