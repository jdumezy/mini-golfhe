import { HOLES, PHYSICS, playback, shotState, simulate, type Playback, type Shot, type Vec2 } from "../lib/golf";
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
let lastShot: { aim: Aim; shot: Shot; pb: Playback; simulateMs: number; playbackMs: number } | null = null;

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

  const t0 = performance.now();
  const shot = simulate(hole, shotState(from, aim.angle, aim.power));
  const t1 = performance.now();
  const pb = playback(hole, shot);
  const t2 = performance.now();
  lastShot = { aim, shot, pb, simulateMs: t1 - t0, playbackMs: t2 - t1 };
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
    statsTimer = window.setInterval(refreshRenderStats, 250);
  }
}

toggle.addEventListener("click", () => setPanelOpen(!panelOpen));
$("nerd-close").addEventListener("click", () => setPanelOpen(false));

const stat = (name: string) => panel.querySelector<HTMLElement>(`[data-stat="${name}"]`)!;
const fmt = (v: number, d = 3) => v.toFixed(d);
const formatMs = (ms: number) => (ms < 10 ? `${ms.toFixed(2)} ms` : `${ms.toFixed(0)} ms`);

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

function refresh() {
  const hole = HOLES[round.hole];
  $("hole-title").textContent = `Hole ${round.hole + 1}`;
  $("hole-name").textContent = `· ${hole.spec.name} · par ${hole.spec.par}`;
  $("hole-strokes").textContent = `stroke ${Math.min(round.strokes + (phase === "aiming" ? 1 : 0), PHYSICS.maxStrokes)}`;
  shootButton.hidden = phase === "holed" || phase === "finished";
  shootButton.disabled = phase !== "aiming";
  nextButton.hidden = phase !== "holed";
  status.textContent = phase === "rolling" ? "rolling…" : message;
  refreshPower();
  refreshScorecard();

  if (!panelOpen) return;
  refreshRenderStats();
  const counts = { rail: 0, block: 0, bumper: 0 };
  let faces = 0;
  for (const o of hole.obstacles) {
    counts[o.kind]++;
    faces += o.kind === "bumper" ? 1 : o.faces.length;
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

  if (lastShot) {
    const { aim, shot, pb } = lastShot;
    stat("shotAim").textContent = `${fmt((aim.angle * 180) / Math.PI, 1)}° / ${Math.round(aim.power * 100)}%`;
    stat("shotTimes").textContent = `${formatMs(lastShot.simulateMs)} / ${formatMs(lastShot.playbackMs)}`;
    stat("shotContacts").textContent = `${shot.contacts.length} / ${shot.exhausted}`;
    stat("shotDuration").textContent = `${fmt(pb.samples[pb.samples.length - 1].t, 2)} s / ${pb.samples.length}`;
    stat("shotOutcome").textContent = pb.outcome;
    stat("segments").textContent = shot.segments
      .map((s) => `${fmt(s.t, 2)}: ${s.state.map((v) => fmt(v)).join(" ")}`)
      .join("\n");
    stat("segments").style.whiteSpace = "pre";
  }
}

showHole();
