/**
 * HTTP API of the local mini-golfhe client (fhe/client), proxied under /api by
 * the dev server. The browser only ever talks to the client, which holds the
 * secret key; the client talks to the server.
 */
import type { Shot, State } from "../golf";

export interface Status {
  client: {
    version: string;
    scheme: string;
    keygen_ms: number;
    eval_key_bytes: number;
  };
  server: {
    addr: string;
    reachable: boolean;
    max_contacts: number | null;
    steps: number | null;
    dt: number | null;
    error: string | null;
  };
}

export interface ShotStats {
  scheme: string;
  slots: number;
  contacts: number;
  eval_key_bytes: number;
  shot_bytes: number;
  trajectory_bytes: number;
  upload_bytes: number;
  download_bytes: number;
  encrypt_ms: number;
  simulate_ms: number;
  network_ms: number;
  decrypt_ms: number;
  total_ms: number;
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

/** Has the shot `state` on hole `hole` (index from 0) simulated by the server. */
export async function shoot(hole: number, state: State): Promise<ShotResult> {
  const body = await call<Shot & { stats: ShotStats }>("/api/shot", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ hole, shot: state }),
  });
  return { shot: { segments: body.segments, exhausted: body.exhausted }, stats: body.stats };
}
