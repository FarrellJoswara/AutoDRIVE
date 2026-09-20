import { useEffect, useRef } from "react";
import { useHubStore } from "../store";

function Sparkline({
  steps,
  rewards,
  len,
}: {
  steps: Float32Array;
  rewards: Float32Array;
  len: number;
}) {
  const ref = useRef<SVGSVGElement>(null);

  useEffect(() => {
    const svg = ref.current;
    if (!svg || len < 2) return;
    const w = 800;
    const h = 72;
    const n = Math.min(len, steps.length);
    // Ring buffer: if full, oldest is at index len % CAP — for simplicity plot [0..n)
    let minR = Infinity;
    let maxR = -Infinity;
    for (let i = 0; i < n; i++) {
      minR = Math.min(minR, rewards[i]);
      maxR = Math.max(maxR, rewards[i]);
    }
    if (!Number.isFinite(minR) || !Number.isFinite(maxR) || minR === maxR) {
      maxR = minR + 1;
    }
    const pts: string[] = [];
    for (let i = 0; i < n; i++) {
      const x = (i / (n - 1)) * (w - 4) + 2;
      const y = h - 4 - ((rewards[i] - minR) / (maxR - minR)) * (h - 8);
      pts.push(`${x.toFixed(1)},${y.toFixed(1)}`);
    }
    const poly = svg.querySelector("polyline");
    if (poly) poly.setAttribute("points", pts.join(" "));
  }, [steps, rewards, len]);

  return (
    <svg ref={ref} className="spark" viewBox="0 0 800 72" preserveAspectRatio="none">
      <polyline fill="none" stroke="#3ecf8e" strokeWidth="2" points="" />
    </svg>
  );
}

export function LivePage() {
  const { metrics, steps, rewards, metricsLen } = useHubStore();

  return (
    <section className="panel">
      <h2>Live</h2>
      <p className="lede">Train metrics from hub WebSocket (best-effort).</p>

      <table className="live-table">
        <thead>
          <tr>
            <th>Field</th>
            <th>Value</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>step</td>
            <td>{metrics?.step ?? "—"}</td>
          </tr>
          <tr>
            <td>reward (rollout mean)</td>
            <td>{metrics?.reward != null ? metrics.reward.toFixed(4) : "—"}</td>
          </tr>
          <tr>
            <td>episode</td>
            <td>{metrics?.episode ?? "—"}</td>
          </tr>
          <tr>
            <td>loss</td>
            <td>{metrics?.loss != null ? metrics.loss.toFixed(6) : "—"}</td>
          </tr>
          <tr>
            <td>run_id</td>
            <td>{metrics?.run_id ?? "—"}</td>
          </tr>
          <tr>
            <td>checkpoint</td>
            <td>{metrics?.checkpoint ?? "—"}</td>
          </tr>
          <tr>
            <td>ts</td>
            <td>{metrics?.ts ?? "—"}</td>
          </tr>
        </tbody>
      </table>

      <Sparkline steps={steps} rewards={rewards} len={metricsLen} />
    </section>
  );
}
