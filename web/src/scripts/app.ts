import { HOLES, PHYSICS, playback, shotState, type Playback, type Shot, type Vec2 } from "../lib/golf";
import * as fhe from "../lib/fhe/client";
import { CourseView, type Aim } from "../lib/render/CourseView";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

const view = new CourseView($("course"));
const status = $("status");
const shootButton = $<HTMLButtonElement>("shoot");
const nextButton = $<HTMLButtonElement>("next");
const powerBar = $("power-bar");
const powerValue = $("power-value");
const scorecard = $<HTMLTableElement>("scorecard");

// --- Round ------------------------------------------------------------------------

type Phase = "aiming" | "rolling" | "holed" | "finished";

interface Round {
  /** Strokes per hole, null if not played yet. */
  scores: (number | null)[];
  hole: number;
  strokes: number;
  ball: Vec2;
}

const STORAGE_KEY = "mini-golfhe:round";

function newRound(): Round {
  return { scores: HOLES.map(() => null), hole: 0, strokes: 0, ball: HOLES[0].spec.tee };
}

function loadRound(): Round {
  try {
    const r = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "null") as Round | null;
    if (r && r.scores.length === HOLES.length && r.hole >= 0 && r.hole < HOLES.length) return r;
  } catch {}
  return newRound();
}

function saveRound() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(round));
  } catch {}
}

let round = loadRound();
let phase: Phase = round.scores.every((s) => s !== null) ? "finished" : "aiming";
let message = "";
let lastShot: { aim: Aim; shot: Shot; pb: Playback; playbackMs: number } | null = null;
let lastStats: fhe.ShotStats | null = null;
/** Waiting for the client to encrypt, have the server simulate, and decrypt. */
let requesting = false;
let requestStart = 0;
let requestTimer: number | undefined;
let lastProgress: fhe.Progress | null = null;
let progressPending = false;
const progress = $("progress");
const progressBar = $("progress-bar");

function setRequesting(on: boolean) {
  requesting = on;
  clearInterval(requestTimer);
  lastProgress = null;
  progress.hidden = !on;
  setProgress(0);
  if (on) {
    requestStart = performance.now();
    requestTimer = window.setInterval(pollProgress, 250);
  }
}

function setProgress(fraction: number) {
  const percent = Math.round(Math.min(1, Math.max(0, fraction)) * 100);
  progressBar.style.width = `${percent}%`;
  progress.setAttribute("aria-valuenow", String(percent));
}

/** Follows the encrypted simulation: client steps around the server's contacts. */
async function pollProgress() {
  refreshStatusLine();
  if (progressPending) return;
  progressPending = true;
  try {
    const p = await fhe.fetchProgress();
    if (!requesting) return;
    lastProgress = p;
    const evaluated = p.server?.fraction ?? (p.step === "decrypting" ? 1 : 0);
    setProgress(p.step === "idle" ? 0 : 0.02 + 0.96 * evaluated + (p.step === "decrypting" ? 0.02 : 0));
  } catch {
  } finally {
    progressPending = false;
  }
}

const STEP_LABELS = {
  gaps: "gaps to every face",
  bootstrapping: "bootstrapping",
  contact: "first contact",
  output: "packing contacts",
} as const;

function progressLabel(): string {
  const p = lastProgress;
  if (!p || p.step === "idle") return "encrypting…";
  if (p.step !== "evaluating" || !p.server) return `${p.step}…`;
  const { contact, contacts, step } = p.server;
  return `contact ${Math.min(contact + 1, contacts)}/${contacts}, ${STEP_LABELS[step]}`;
}

const parTotal = HOLES.reduce((s, h) => s + h.spec.par, 0);

function relativeName(strokes: number, par: number): string {
  if (strokes === 1) return "hole in one!";
  const d = strokes - par;
  const names: Record<number, string> = { [-3]: "albatross", [-2]: "eagle", [-1]: "birdie", 0: "par", 1: "bogey", 2: "double bogey" };
  return names[d] ?? `+${d}`;
}

/** Power of the last shot, reused for the next one. */
let lastPower = 0.35;

/** Towards the cup, at the last shot's power. */
function defaultAim(): Aim {
  const [bx, by] = round.ball;
  const [cx, cy] = HOLES[round.hole].spec.cup;
  return { angle: Math.atan2(cy - by, cx - bx), power: lastPower };
}

function startHole(index: number) {
  round.hole = index;
  round.strokes = 0;
  round.scores[index] = null;
  round.ball = HOLES[index].spec.tee;
  phase = "aiming";
  message = "";
  showHole();
// The keys take a while to generate and upload: follow them.
refreshClientStatus();
  saveRound();
}

function showHole() {
  const hole = HOLES[round.hole];
  view.setHole(hole, round.ball);
  view.setAim(phase === "aiming" ? defaultAim() : null);
  view.setAimEnabled(phase === "aiming");
  refresh();
}

async function shoot(aim: Aim) {
  if (phase !== "aiming") return;
  const hole = HOLES[round.hole];
  phase = "rolling";
  message = "";
  view.setAimEnabled(false);
  const from = round.ball;
  round.strokes++;
  lastPower = aim.power;

  setRequesting(true);
  refresh();
  let result: fhe.ShotResult;
  try {
    result = await fhe.shoot(hole.index, shotState(from, aim.angle, aim.power));
  } catch (e) {
    // The shot never happened.
    setRequesting(false);
    round.strokes--;
    phase = "aiming";
    message = `error: ${e instanceof Error ? e.message : String(e)}`;
    view.setAimEnabled(true);
    refresh();
    refreshClientStatus();
    return;
  }
  setRequesting(false);
  lastStats = result.stats;
  const t0 = performance.now();
  const pb = playback(hole, result.shot);
  lastShot = { aim, shot: result.shot, pb, playbackMs: performance.now() - t0 };
  refresh();

  await view.play(pb);

  const par = hole.spec.par;
  if (pb.outcome === "cup") {
    round.scores[round.hole] = round.strokes;
    phase = "holed";
    message = `${relativeName(round.strokes, par)} (${round.strokes} stroke${round.strokes > 1 ? "s" : ""})`;
  } else {
    if (pb.outcome === "water") {
      round.strokes++;
      round.ball = from;
      message = "splash! one penalty stroke, back to where you played from";
    } else {
      round.ball = pb.final;
    }
    if (round.strokes >= PHYSICS.maxStrokes) {
      round.scores[round.hole] = PHYSICS.maxStrokes;
      phase = "holed";
      message = `picked up after ${PHYSICS.maxStrokes} strokes`;
    } else {
      phase = "aiming";
    }
  }
  if (phase === "holed" && round.scores.every((s) => s !== null)) {
    phase = "finished";
    const total = round.scores.reduce<number>((s, v) => s + (v ?? 0), 0);
    const d = total - parTotal;
    message = `round complete: ${total} strokes (${d === 0 ? "even par" : d > 0 ? `+${d}` : d})`;
  }
  saveRound();
  view.setBall(round.ball, phase === "aiming");
  view.setAim(phase === "aiming" ? defaultAim() : null);
  view.setAimEnabled(phase === "aiming");
  refresh();
}

function nextHole() {
  if (phase !== "holed") return;
  const next = round.scores.findIndex((s, i) => s === null && i > round.hole);
  const first = next >= 0 ? next : round.scores.findIndex((s) => s === null);
  if (first >= 0) startHole(first);
}

view.addEventListener("aim", () => refreshPower());
view.addEventListener("shoot", (e) => void shoot((e as CustomEvent<Aim>).detail));
shootButton.addEventListener("click", () => {
  const aim = view.getAim();
  if (aim) void shoot(aim);
});
nextButton.addEventListener("click", nextHole);
$("new-round").addEventListener("click", () => {
  const played = round.scores.some((s) => s !== null) || round.strokes > 0;
  if (played && phase !== "finished" && !confirm("Start a new round? This one's scores will be lost.")) return;
  round = newRound();
  startHole(0);
});

// --- Keyboard ---------------------------------------------------------------------

/** W A S D move the view across the course. */
const PAN_KEYS: Record<string, [number, number]> = { w: [0, 1], s: [0, -1], a: [-1, 0], d: [1, 0] };

window.addEventListener("keydown", (e) => {
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  if (e.target instanceof HTMLElement && e.target.closest("input, textarea, [contenteditable]")) return;
  const aim = view.getAim();
  const turn = (e.shiftKey ? 0.25 : 2) * (Math.PI / 180);
  if (phase === "aiming" && aim && (e.key === "ArrowLeft" || e.key === "ArrowRight")) {
    view.setAim({ ...aim, angle: aim.angle + (e.key === "ArrowLeft" ? turn : -turn) });
  } else if (phase === "aiming" && aim && (e.key === "ArrowUp" || e.key === "ArrowDown")) {
    const step = e.shiftKey ? 0.005 : 0.02;
    view.setAim({ ...aim, power: Math.min(1, Math.max(0, aim.power + (e.key === "ArrowUp" ? step : -step))) });
  } else if (PAN_KEYS[e.key.toLowerCase()]) {
    const [dx, dy] = PAN_KEYS[e.key.toLowerCase()];
    view.pan(dx * 0.25, dy * 0.25);
  } else if (e.key === " " && !e.repeat && phase === "aiming" && aim) {
    void shoot(aim);
  } else if (e.key === "Enter" && !e.repeat && phase === "holed") {
    nextHole();
  } else if (e.key === "`" && !e.repeat) {
    setPanelOpen(!panelOpen);
  } else if (e.key === "Escape" && panelOpen) {
    setPanelOpen(false);
  } else {
    return;
  }
  e.preventDefault();
  refreshPower();
});

// --- Theme ----------------------------------------------------------------------

$("theme-toggle").addEventListener("click", () => {
  const dark = document.documentElement.classList.toggle("dark");
  try {
    localStorage.setItem("theme", dark ? "dark" : "light");
  } catch {}
});

matchMedia("(prefers-color-scheme: dark)").addEventListener("change", (e) => {
  let stored = null;
  try {
    stored = localStorage.getItem("theme");
  } catch {}
  if (!stored) document.documentElement.classList.toggle("dark", e.matches);
});

// --- Status and stats for nerds ------------------------------------------------

const panel = $("nerd-panel");
const toggle = $("nerd-toggle");
let panelOpen = false;
let statsTimer: number | undefined;

function setPanelOpen(open: boolean) {
  panelOpen = open;
  panel.toggleAttribute("data-open", open);
  panel.inert = !open;
  toggle.setAttribute("aria-pressed", String(open));
  // On wide screens, make room for the panel instead of covering the course.
  document.body.classList.toggle("lg:pr-96", open);
  clearInterval(statsTimer);
  if (open) {
    refresh();
    refreshClientStatus();
    statsTimer = window.setInterval(refreshRenderStats, 250);
  }
}

toggle.addEventListener("click", () => setPanelOpen(!panelOpen));
$("nerd-close").addEventListener("click", () => setPanelOpen(false));

const stat = (name: string) => panel.querySelector<HTMLElement>(`[data-stat="${name}"]`)!;
const fmt = (v: number, d = 3) => v.toFixed(d);
function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} KiB`;
  if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)} MiB`;
  return `${(n / 1024 ** 3).toFixed(2)} GiB`;
}

function formatMs(ms: number): string {
  if (ms < 1) return `${(ms * 1000).toFixed(0)} µs`;
  if (ms < 1000) return `${ms.toFixed(1)} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}

const SETUP_LABELS: Record<fhe.SetupPhase["phase"], string> = {
  preparing: "preparing",
  generating_keys: "generating keys",
  uploading: "uploading keys",
  ready: "ready",
  waiting_for_server: "waiting for the server",
  failed: "setup failed",
};

let statusTimer: number | undefined;
let clientSetup: fhe.SetupPhase | null = null;

async function refreshClientStatus() {
  let s: fhe.Status | null = null;
  try {
    s = await fhe.fetchStatus();
  } catch {}
  clearTimeout(statusTimer);
  if (!s) {
    clientSetup = null;
    for (const name of ["fheScheme", "fheBackend", "fheClientBackend", "fheServer", "fheCircuit", "fheKey", "fheKeyTimes"])
      stat(name).textContent = "–";
    stat("fheClient").textContent = "not running";
    return;
  }
  const { client, server } = s;
  const setup = (clientSetup = client.setup);
  stat("fheScheme").textContent = client.params ? `${client.scheme}, ${client.params}` : client.scheme;
  // The server evaluates, possibly on a GPU; the client generates keys and encrypts on the CPU.
  stat("fheBackend").textContent = server.info?.backend ?? "–";
  stat("fheClientBackend").textContent = client.backend;
  let clientText = `v${client.version}`;
  if (setup) {
    clientText += `, ${SETUP_LABELS[setup.phase]}`;
    if (setup.phase === "uploading") clientText += ` (${formatBytes(setup.sent_bytes)})`;
  }
  stat("fheClient").textContent = clientText;
  stat("fheServer").textContent = server.reachable
    ? `${server.addr}, ${server.info?.key_id ? "keys installed" : "no keys"}${server.error ? `, ${server.error}` : ""}`
    : `unreachable at ${server.addr}`;
  const info = server.info;
  stat("fheCircuit").textContent = info
    ? `${info.max_contacts} contacts × ${info.steps} steps of ${(info.dt * 1000).toFixed(0)} ms × 32 faces`
    : "–";
  stat("fheKey").textContent = setup?.phase === "ready" ? formatBytes(setup.key_bytes) : "–";
  stat("fheKeyTimes").textContent =
    setup?.phase === "ready" && client.keygen_ms
      ? `${formatMs(client.keygen_ms)} / ${formatMs(setup.server_prepare_ms)}`
      : "–";
  refresh();
  // Follow the key setup while it runs.
  if (setup && setup.phase !== "ready" && setup.phase !== "failed") {
    statusTimer = window.setTimeout(refreshClientStatus, 1000);
  }
}

function refreshLastShot() {
  const s = lastStats;
  if (!s) return;
  const ckks = s.input_bytes !== undefined;
  stat("fheCiphertexts").textContent = ckks
    ? `${formatBytes(s.input_bytes!)} / ${formatBytes(s.output_bytes!)}`
    : `${formatBytes(s.upload_bytes)} / ${formatBytes(s.download_bytes)} (plaintext)`;
  stat("fheEncrypt").textContent = ckks ? formatMs(s.encrypt_ms!) : "–";
  stat("fheSimulate").textContent = formatMs(s.simulate_ms);
  stat("fheRefresh").textContent = ckks ? String(s.refreshes) : "–";
  stat("fheDecrypt").textContent = ckks ? formatMs(s.decrypt_ms!) : "–";
  stat("fheTotal").textContent = `${formatMs(s.network_ms)} / ${formatMs(s.total_ms)}`;
  stat("fheContacts").textContent = `${s.contacts} / ${s.slots}`;
  stat("fheConfidence").textContent = s.confidence.map((c) => c.toFixed(4)).join(" ");
}

function refreshRenderStats() {
  const s = view.stats();
  stat("fps").textContent = s.fps.toFixed(0);
  stat("drawCalls").textContent = String(s.drawCalls);
  stat("triangles").textContent = s.triangles.toLocaleString("en");
  stat("pixelRatio").textContent = String(s.pixelRatio);
}

function refreshPower() {
  const aim = phase === "aiming" ? view.getAim() : null;
  const p = aim?.power ?? 0;
  powerBar.style.width = `${(p * 100).toFixed(1)}%`;
  powerBar.style.backgroundColor = `hsl(${((1 - p) * 120).toFixed(0)} 75% 45%)`;
  powerValue.textContent = aim ? `${Math.round(p * 100)}%` : "–";
}

function refreshScorecard() {
  const cell = (text: string, cls = "") => {
    const td = document.createElement("td");
    td.textContent = text;
    td.className = `border border-zinc-200 px-1.5 py-0.5 text-center dark:border-zinc-800 ${cls}`;
    return td;
  };
  const rows: HTMLTableRowElement[] = [];
  for (const [label, values, total] of [
    ["hole", HOLES.map((_, i) => String(i + 1)), "tot"],
    ["par", HOLES.map((h) => String(h.spec.par)), String(parTotal)],
    [
      "score",
      round.scores.map((s, i) => (s !== null ? String(s) : i === round.hole && round.strokes > 0 ? `(${round.strokes})` : "")),
      String(round.scores.reduce<number>((s, v) => s + (v ?? 0), 0) || ""),
    ],
  ] as const) {
    const tr = document.createElement("tr");
    tr.append(cell(label, "text-left text-zinc-500"));
    values.forEach((v, i) => {
      const s = round.scores[i];
      const par = HOLES[i].spec.par;
      let cls = i === round.hole ? "bg-zinc-100 dark:bg-zinc-900 font-semibold" : "";
      if (label === "score" && s !== null) cls += s < par ? " text-emerald-600" : s > par ? " text-rose-600" : "";
      const td = cell(v, cls);
      if (label === "hole") {
        td.classList.add("cursor-pointer", "hover:bg-zinc-200", "dark:hover:bg-zinc-800");
        td.title = `play hole ${i + 1}: ${HOLES[i].spec.name}`;
        td.addEventListener("click", () => {
          if (phase !== "rolling") startHole(i);
        });
      }
      tr.append(td);
    });
    tr.append(cell(total, "text-zinc-500"));
    rows.push(tr);
  }
  scorecard.replaceChildren(...rows);
}

function refreshStatusLine() {
  if (requesting)
    status.textContent = `${progressLabel()} ${((performance.now() - requestStart) / 1000).toFixed(1)} s`;
  else if (phase === "rolling") status.textContent = "rolling…";
  else if (message) status.textContent = message;
  else if (clientSetup && clientSetup.phase !== "ready")
    status.textContent = `encryption keys: ${SETUP_LABELS[clientSetup.phase]}${
      clientSetup.phase === "uploading" ? ` (${formatBytes(clientSetup.sent_bytes)})` : ""
    }…`;
  else status.textContent = "";
}

function refresh() {
  const hole = HOLES[round.hole];
  $("hole-title").textContent = `Hole ${round.hole + 1}`;
  $("hole-name").textContent = `· ${hole.spec.name} · par ${hole.spec.par}`;
  $("hole-strokes").textContent = `stroke ${Math.min(round.strokes + (phase === "aiming" ? 1 : 0), PHYSICS.maxStrokes)}`;
  shootButton.hidden = phase === "holed" || phase === "finished";
  shootButton.disabled = phase !== "aiming";
  nextButton.hidden = phase !== "holed";
  refreshStatusLine();
  refreshPower();
  refreshScorecard();

  if (!panelOpen) return;
  refreshRenderStats();
  const counts = { rail: 0, block: 0, bumper: 0 };
  let faces = 0;
  for (const o of hole.obstacles) {
    counts[o.kind === "block" && o.style === "bumper" ? "bumper" : o.kind]++;
    faces += o.faces.length;
  }
  const steps = Math.round(PHYSICS.horizon / PHYSICS.dt);
  stat("hole").textContent = `${round.hole + 1} / ${hole.spec.par}`;
  stat("obstacles").textContent = `${counts.rail} / ${counts.block} / ${counts.bumper}`;
  stat("faces").textContent = String(faces);
  stat("ponds").textContent = String(hole.spec.water?.length ?? 0);
  const t = hole.terrain;
  stat("terrain").textContent = `c = (${t.center.join(", ")})  H = [${t.hessian[0]} ${t.hessian[1]}; ${t.hessian[1]} ${t.hessian[2]}]  s = (${t.slope.join(", ")})`;
  stat("drag").textContent = `${PHYSICS.drag} /s / ${fmt(PHYSICS.rollFactor, 3)}`;
  stat("grid").textContent = `${PHYSICS.dt * 1000} ms × ${steps}`;
  stat("maxContacts").textContent = String(PHYSICS.maxContacts);
  stat("tests").textContent = (steps * faces).toLocaleString("en");

  refreshLastShot();
  if (lastShot) {
    const { aim, shot, pb } = lastShot;
    stat("shotAim").textContent = `${fmt((aim.angle * 180) / Math.PI, 1)}° / ${Math.round(aim.power * 100)}%`;
    stat("shotPlayback").textContent = formatMs(lastShot.playbackMs);
    stat("shotContacts").textContent = `${shot.segments.length - 1} / ${shot.exhausted}`;
    stat("shotDuration").textContent = `${fmt(pb.samples[pb.samples.length - 1].t, 2)} s / ${pb.samples.length}`;
    stat("shotOutcome").textContent = pb.outcome;
    stat("segments").textContent = shot.segments
      .map((s) => `${fmt(s.t, 2)}: ${s.state.map((v) => fmt(v)).join(" ")}`)
      .join("\n");
    stat("segments").style.whiteSpace = "pre";
  }
}

showHole();
// The keys take a while to generate and upload: follow them.
refreshClientStatus();
