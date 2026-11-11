# live view in the browser: camera with masks, tracked object poses and trial status
import re
import json
import time
import threading
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import cv2
import numpy as np
from kinematics import fk
from config import object_dim

# mask colors (RGB) by object index
COLORS = [(255, 153, 0), (0, 200, 255), (180, 0, 255), (0, 90, 255), (120, 255, 120)]
METHODS = {0: "Baseline DP", 1: "InneR DP", 2: "Fisher trace DP", 3: "Fisher nuclear DP"}
LOG = deque(maxlen=12)


def draw_masks(frame, masks, step=2):
    if masks is not None:
        for k, m in enumerate(masks[:, ::step, ::step]):
            frame[m] = (0.5 * frame[m] + 0.5 * np.array(COLORS[k % len(COLORS)])).astype(np.uint8)
    return frame


def model_info(path, policy):
    """task, set, seed and method labels for the viewer, parsed from the model path"""
    stats = policy.stats
    m = re.search(r"([^/]+)/seed(\d+)/\w+?_weights(\d+)$", path.rstrip("/"))
    info = {"task": path, "set": "", "seed": "", "method": policy.kind}
    if m:
        info.update(set=m.group(1), task=m.group(1).split("_")[0], seed=m.group(2),
                    method=METHODS.get(int(m.group(3)), f"model {m.group(3)}"))
    info.update(path=path, spaces=f"{stats.get('state_space', 'joint')} state, {stats.get('action_space', 'joint')} action")
    return info


class Viewer:

    def __init__(self, objects, info, port, fps=15, open_browser=True):
        self.objects, self.info, self.period = objects, info, 1.0 / fps
        self.cam = objects.cameras[0]
        c = objects.calib[self.cam]
        # half resolution image
        self.K = c["K"].astype(np.float64).copy()
        self.K[:2] *= 0.5
        self.T_cam_base = np.linalg.inv(c["T_base_cam"].astype(np.float64))
        # push object xy went through xy_correction, undo it to land on the image
        corr = getattr(objects, "correction", None)
        self.uncorrect = None
        if corr is not None:
            A, b = corr[:, :2].astype(np.float64), corr[:, 2].astype(np.float64)
            self.uncorrect = (np.linalg.inv(A), b)
        self.status, self.lock = {}, threading.Lock()
        self.jpeg, self.last_client = None, 0.0
        self.on_key = None
        threading.Thread(target=self._render_loop, daemon=True).start()

        viewer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                try:
                    viewer._handle(self)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{port}"
        print(f"[Viewer] {self.url}", flush=True)
        if open_browser:
            webbrowser.open(self.url)

    def update(self, **kw):
        with self.lock:
            self.status.update(kw)

    def project(self, p, table=False):
        """project base frame points (..., 3) to half resolution pixels, None behind the camera"""
        p = np.array(p, dtype=np.float64)
        if self.uncorrect is not None:
            Ainv, b = self.uncorrect
            p[..., :2] = (p[..., :2] - b) @ Ainv.T
        if table:
            p[..., 2] = 0.0
        cam = p @ self.T_cam_base[:3, :3].T + self.T_cam_base[:3, 3]
        if (cam[..., 2] <= 0.05).any():
            return None
        uv = cam @ self.K.T
        return (uv[..., :2] / uv[..., 2:]).round().astype(int)

    def project_tcp(self, p):
        """project the tcp (3,) to half resolution pixels, using the calibration's draw model if there is one"""
        d = getattr(self.objects, "draw", None)
        if d is None:
            return self.project(p, table=self.uncorrect is not None)
        cam = np.asarray(p, dtype=np.float64) @ self.T_cam_base[:3, :3].T + self.T_cam_base[:3, 3]
        if cam[2] <= 0.05:
            return None
        u, v = d["fx"] * cam[0] / cam[2] + d["cx"], d["fy"] * cam[1] / cam[2] + d["cy"]
        if d.get("mirrored"):
            u = d["width"] - 1 - u
        return (0.5 * np.array([u, v])).round().astype(int)

    def render(self):
        frames = self.objects.sub.get_last_frames()
        if frames is None:
            return None
        frame = draw_masks(frames[f"{self.cam}_img_rgb"][::2, ::2].copy(), self.objects.masks)
        state = self.objects.state().reshape(-1, object_dim)
        for k, (label, s) in enumerate(zip(self.objects.labels, state)):
            if np.isnan(s).any():
                continue
            d = 0.05 * np.array([np.cos(s[3]), np.sin(s[3]), 0.0])
            pts = self.project(np.stack([s[:3], s[:3] - d, s[:3] + d]))
            if pts is None:
                continue
            color = COLORS[k % len(COLORS)]
            cv2.line(frame, tuple(pts[1]), tuple(pts[2]), (255, 255, 255), 4, cv2.LINE_AA)
            cv2.line(frame, tuple(pts[1]), tuple(pts[2]), color, 2, cv2.LINE_AA)
            cv2.circle(frame, tuple(pts[0]), 7, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, tuple(pts[0]), 5, color, -1, cv2.LINE_AA)
            cv2.putText(frame, label, tuple(pts[0] + [10, -10]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, label, tuple(pts[0] + [10, -10]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        with self.lock:
            q = self.status.get("q")
        if q is not None:
            tcp = self.project_tcp(fk(np.asarray(q))[:3, 3])
            if tcp is not None:
                cv2.drawMarker(frame, tuple(tcp), (0, 0, 0), cv2.MARKER_CROSS, 22, 5, cv2.LINE_AA)
                cv2.drawMarker(frame, tuple(tcp), (255, 255, 255), cv2.MARKER_CROSS, 20, 2, cv2.LINE_AA)
                cv2.putText(frame, "tcp", tuple(tcp + [12, 18]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(frame, "tcp", tuple(tcp + [12, 18]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
        return jpg.tobytes() if ok else None

    def _render_loop(self):
        while True:
            start = time.time()
            if start - self.last_client < 2.0:
                try:
                    self.jpeg = self.render()
                except Exception as e:
                    print(f"[Viewer] render failed: {e}", flush=True)
            time.sleep(max(0.0, self.period - (time.time() - start)))

    def status_json(self):
        with self.lock:
            st = dict(self.status)
        st.pop("q", None)
        objects = []
        for label, s in zip(self.objects.labels, self.objects.state().reshape(-1, object_dim)):
            seen = not np.isnan(s).any()
            objects.append({"label": label, "seen": seen,
                            "xyz": [round(float(v), 3) for v in s[:3]] if seen else None,
                            "yaw": round(float(np.degrees(s[3])), 1) if seen else None})
        times = list(getattr(self.objects, "update_times", []))
        tracker_hz = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 and times[-1] > times[0] else None
        return json.dumps({**self.info, **st, "objects": objects, "tracker_hz": tracker_hz, "log": list(LOG)})

    def _handle(self, h):
        self.last_client = time.time()
        path = h.path.split("?")[0]
        if path == "/":
            body = PAGE.encode()
            h.send_response(200)
            h.send_header("Content-Type", "text/html; charset=utf-8")
            h.send_header("Content-Length", str(len(body)))
            h.end_headers()
            h.wfile.write(body)
        elif path == "/status":
            body = self.status_json().encode()
            h.send_response(200)
            h.send_header("Content-Type", "application/json")
            h.send_header("Cache-Control", "no-store")
            h.send_header("Content-Length", str(len(body)))
            h.end_headers()
            h.wfile.write(body)
        elif path == "/key":
            from urllib.parse import parse_qs, urlparse
            k = parse_qs(urlparse(h.path).query, keep_blank_values=True).get("k", [""])[0]
            if self.on_key is not None and len(k) == 1 and k in "gs 10rhpq":
                self.on_key(k)
            h.send_response(204)
            h.end_headers()
        elif path == "/stream":
            h.send_response(200)
            h.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            h.send_header("Cache-Control", "no-store")
            h.end_headers()
            last = None
            while True:
                self.last_client = time.time()
                jpg = self.jpeg
                if jpg is not None and jpg is not last:
                    h.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                  + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                    last = jpg
                time.sleep(self.period / 2)
        else:
            h.send_response(404)
            h.end_headers()


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Deploy</title>
<style>
:root { --bg: #16161a; --panel: #212127; --ink: #ececef; --muted: #9a9aa6; --line: #34343c;
        --accent: #ff9900; --good: #3fb871; --bad: #e5534b; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink); font: 16px/1.4 "Palatino Linotype", Palatino, "Book Antiqua", serif; }
main { display: grid; grid-template-columns: minmax(0, 1fr) 360px; gap: 16px; padding: 16px; }
@media (max-width: 1100px) { main { grid-template-columns: 1fr; } }
#cam { width: 100%; border-radius: 6px; background: #000; display: block; aspect-ratio: 16 / 9; }
.panel { background: var(--panel); border-radius: 6px; padding: 14px 16px; margin-bottom: 12px; }
h2 { margin: 0 0 8px; font-size: 13px; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); font-weight: normal; }
.big { font-size: 40px; line-height: 1.1; font-variant-numeric: tabular-nums; }
.row { display: flex; justify-content: space-between; align-items: center; gap: 12px; padding: 2px 0; }
.row span:first-child { color: var(--muted); }
.row span:last-child { text-align: right; font-variant-numeric: tabular-nums; }
#state { display: inline-block; padding: 2px 10px; border-radius: 4px; background: var(--line); }
#state.running { background: var(--accent); color: #000; }
#state.success { background: var(--good); color: #000; }
#state.failure { background: var(--bad); color: #000; }
#chips { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 8px; }
.chip { width: 26px; height: 26px; border-radius: 4px; display: grid; place-items: center; font-size: 12px; color: #000; }
.chip.s { background: var(--good); } .chip.f { background: var(--bad); }
table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
td { padding: 2px 4px; } td:first-child { padding-left: 0; }
.dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; }
.miss { color: var(--bad); }
#log { font: 13px/1.35 ui-monospace, monospace; color: var(--muted); white-space: pre-wrap; margin: 0; }
.keys { color: var(--muted); font-size: 14px; }
kbd { font: 13px ui-monospace, monospace; background: var(--line); color: var(--ink); padding: 0 5px; border-radius: 3px; }
#warn { display: none; background: var(--accent); color: #000; padding: 6px 12px; border-radius: 4px; margin-bottom: 12px; }
</style></head><body><main>
<section>
  <img id="cam" src="/stream" alt="camera">
  <div class="panel" style="margin-top:12px"><h2>Log</h2><pre id="log"></pre></div>
</section>
<aside>
  <div id="warn">DRY RUN: the arm does not move</div>
  <div class="panel"><h2>Model</h2>
    <div class="row"><span>Task</span><span id="task"></span></div>
    <div class="row"><span>Method</span><span id="method"></span></div>
    <div class="row"><span>Set / seed</span><span id="setseed"></span></div>
    <div class="row"><span>Spaces</span><span id="spaces"></span></div>
    <div class="row"><span>Controller</span><span id="controller"></span></div>
    <div class="row"><span>z floor</span><span id="floor"></span></div>
  </div>
  <div class="panel"><h2>Trial</h2>
    <div class="row"><span><span class="big" id="trial"></span></span><span id="state"></span></div>
    <div class="row"><span>Step</span><span id="step"></span></div>
    <div class="row"><span>Control / tracker</span><span id="hz"></span></div>
  </div>
  <div class="panel"><h2>Success</h2>
    <div class="big" id="rate">-</div><div id="count" class="keys"></div>
    <div id="chips"></div>
  </div>
  <div class="panel"><h2>Objects</h2><table id="objects"></table></div>
  <div class="panel keys">
    <kbd>g</kbd> go <kbd>s</kbd> stop <kbd>1</kbd>/<kbd>0</kbd> success/failure <kbd>r</kbd> next trial
    <kbd>h</kbd> home <kbd>p</kbd> print <kbd>q</kbd> quit (on this page or the terminal)
  </div>
</aside>
</main>
<script>
const COLORS = ["rgb(255,153,0)", "rgb(0,200,255)", "rgb(180,0,255)", "rgb(0,90,255)"];
const $ = id => document.getElementById(id);
function fmt(x, d) { return x == null ? "-" : x.toFixed(d); }
async function poll() {
  try {
    const s = await (await fetch("/status", {cache: "no-store"})).json();
    $("task").textContent = s.task; $("method").textContent = s.method;
    $("setseed").textContent = (s.set || "-") + " / " + (s.seed === "" ? "-" : s.seed);
    $("spaces").textContent = s.spaces; $("controller").textContent = s.controller || "-";
    $("floor").textContent = s.z_floor == null ? "off" : s.z_floor.toFixed(3) + " m";
    $("warn").style.display = s.dry_run ? "block" : "none";
    $("trial").textContent = "Trial " + (s.trial ?? "-") + (s.max_trials ? " of " + s.max_trials : "");
    const st = s.state || "-";
    $("state").textContent = st; $("state").className = st;
    $("step").textContent = (s.step ?? 0) + " / " + (s.max_steps ?? "-");
    $("hz").textContent = fmt(s.control_hz, 1) + " / " + fmt(s.tracker_hz, 1) + " Hz";
    const r = s.results || [];
    const k = r.filter(x => x[1]).length;
    $("rate").textContent = r.length ? Math.round(100 * k / r.length) + "%" : "-";
    $("count").textContent = r.length ? k + " of " + r.length + " labeled trials" : "no labeled trials yet";
    $("chips").innerHTML = r.map(x => `<div class="chip ${x[1] ? "s" : "f"}" title="trial ${x[0]}">${x[0]}</div>`).join("");
    $("objects").innerHTML = s.objects.map((o, i) => o.seen
      ? `<tr><td><span class="dot" style="background:${COLORS[i % 4]}"></span>${o.label}</td><td>${o.xyz.map(v => v.toFixed(3)).join(", ")}</td><td>${o.yaw.toFixed(0)}&deg;</td></tr>`
      : `<tr><td><span class="dot" style="background:${COLORS[i % 4]}"></span>${o.label}</td><td class="miss" colspan="2">not seen</td></tr>`).join("");
    $("log").textContent = s.log.join("\\n");
  } catch (e) {
    $("state").textContent = "disconnected"; $("state").className = "";
  }
  setTimeout(poll, 300);
}
poll();
document.addEventListener("keydown", e => {
  if (e.ctrlKey || e.metaKey || e.altKey || e.key.length !== 1) return;
  const k = e.key.toLowerCase();
  if ("gs 10rhpq".includes(k)) { e.preventDefault(); fetch("/key?k=" + encodeURIComponent(k)); }
});
</script></body></html>
"""
