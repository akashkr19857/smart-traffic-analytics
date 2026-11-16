export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8400";
export const WS_URL = API_URL.replace(/^http/, "ws");

export type CamStats = {
  status: string; fps?: number; video_t?: number; counts?: Record<string, number>; passed?: number; active_tracks?: number;
  avg_speed_kmh?: number | null; events?: Record<string, number>; timings_ms?: Record<string, number>; resolution?: number[];
};
export type Camera = { id: string; name: string; description: string; anpr: boolean; speed_limit_kmh: number | null; features: string[]; stats: CamStats };
export type TrafficEvent = {
  id: number; camera_id: string; type: string; severity: string; track_id: number; cls: string; plate: string; plate_conf: number;
  speed_kmh: number | null; message: string; snapshot: string; plate_crop: string; video_t: number; details: Record<string, unknown>; created_at: string;
};
export type VehicleRow = {
  id: number; camera_id: string; track_id: number; cls: string; video_t: number; direction: string; avg_speed_kmh: number | null;
  max_speed_kmh: number | null; plate: string; plate_conf: number; plate_reads: number; counted: boolean; first_seen: string;
};

export async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  if (init.body && !(init.body instanceof FormData)) headers.set("Content-Type", "application/json");
  const res = await fetch(`${API_URL}${path}`, { ...init, headers });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      /* not json */
    }
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json() as Promise<T>;
}

export const snapshotUrl = (name: string) => `${API_URL}/api/snapshots/${encodeURIComponent(name)}`;
export const streamUrl = (cam: string) => `${API_URL}/api/cameras/${cam}/stream.mjpg`;

export function eventsSocket(onEvent: (ev: Record<string, unknown>) => void): () => void {
  let ws: WebSocket | null = null;
  let closed = false;
  let retry: ReturnType<typeof setTimeout> | undefined;
  const connect = () => {
    ws = new WebSocket(`${WS_URL}/ws/events`);
    ws.onmessage = (m) => {
      try {
        onEvent(JSON.parse(m.data));
      } catch {
        /* ignore */
      }
    };
    ws.onclose = () => {
      if (!closed) retry = setTimeout(connect, 2000);
    };
  };
  connect();
  return () => {
    closed = true;
    if (retry) clearTimeout(retry);
    ws?.close();
  };
}

export const EVENT_META: Record<string, { label: string; variant: "destructive" | "warning" | "default" | "secondary" | "success" }> = {
  wrong_way: { label: "Wrong way", variant: "destructive" },
  illegal_parking: { label: "Illegal parking", variant: "warning" },
  speeding: { label: "Speeding", variant: "warning" },
  plate: { label: "Plate read", variant: "default" },
  watchlist: { label: "Watchlist hit", variant: "destructive" },
};
