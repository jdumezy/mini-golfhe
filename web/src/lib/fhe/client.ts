/**
 * HTTP API of the local mini-golfhe client (fhe/client), proxied under /api by
 * the dev server. The browser only ever talks to the client, which holds the
 * secret key; the client talks to the server.
 */
import type { Shot, State } from "../golf";

/** The client's CKKS setup: keys are generated, then uploaded to the server. */
export type SetupPhase =
  | { phase: "preparing" }
  | { phase: "generating_keys" }
  | { phase: "uploading"; sent_bytes: number }
  | { phase: "ready"; key_id: number; key_bytes: number; upload_ms: number; server_prepare_ms: number }
  | { phase: "waiting_for_server"; error: string }
  | { phase: "failed"; error: string };

export interface Status {
  client: {
    version: string;
    scheme: string;
    backend: string;
    params: string | null;
    setup: SetupPhase | null;
    keygen_ms: number | null;
  };
  server: {
    addr: string;
    reachable: boolean;
    info: {
      backend: string;
      ckks_params: string;
      key_id: number | null;
      max_contacts: number;
      steps: number;
      dt: number;
    } | null;
    error: string | null;
  };
}

export interface ShotStats {
  scheme: string;
  slots: number;
  contacts: number;
  /** Each contact's decrypted weight: near 1 or 0. */
  confidence: number[];
  upload_bytes: number;
  download_bytes: number;
  simulate_ms: number;
  network_ms: number;
  total_ms: number;
  // CKKS only.
  input_bytes?: number;
  output_bytes?: number;
  encrypt_ms?: number;
  decrypt_ms?: number;
  refreshes?: number;
}

export interface Progress {
  step: "idle" | "encrypting" | "evaluating" | "decrypting";
  server: {
    contact: number;
    contacts: number;
    step: "gaps" | "bootstrapping" | "contact" | "output";
    fraction: number;
    elapsed_ms: number;
  } | null;
}

export interface ShotResult {
  shot: Shot;
  stats: ShotStats;
}

/** The client process is not running (or not reachable through the proxy). */
export class ClientUnavailable extends Error {
  constructor() {
    super("the mini-golfhe client is not running (start everything with `npm run dev`)");
  }
}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  let body: unknown;
  try {
    res = await fetch(path, init);
    body = await res.json();
  } catch {
    throw new ClientUnavailable();
  }
  if (!res.ok) {
    const message = (body as { error?: string } | null)?.error;
    if (!message) throw new ClientUnavailable();
    throw new Error(message);
  }
  return body as T;
}

export function fetchStatus(): Promise<Status> {
  return call<Status>("/api/status");
}

export function fetchProgress(): Promise<Progress> {
  return call<Progress>("/api/progress");
}

/** Has the shot `state` on hole `hole` (index from 0) simulated by the server. */
export async function shoot(hole: number, state: State): Promise<ShotResult> {
  const body = await call<Shot & { stats: ShotStats }>("/api/shot", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ hole, shot: state }),
  });
  return { shot: { segments: body.segments, exhausted: body.exhausted }, stats: body.stats };
}
