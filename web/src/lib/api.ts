export type Driver = {
  driver_id: string;
  events: number;
  avg_speed: number;
  p95_speed: number;
  max_speed: number;
  hard_brake_rate: number;
  phone_use_rate: number;
  night_share: number;
  highway_share: number;
  city_share: number;
  urban_share: number;
  rain_share: number;
  prob_incident: number;
  risk_score: number;
  premium: number;
};

export type Fleet = {
  drivers: number;
  events: number;
  start: string | null;
  end: string | null;
  base_premium: number;
  avg_risk_score: number;
  avg_premium: number;
  score_histogram: { bucket: string; drivers: number }[];
  road_mix: Record<string, number>;
};

export type Point = { driver_id: string; lat: number; lon: number; speed_kph: number; hard_brake: number };
export type DriverEvent = { ts: string; speed_kph: number; hard_brake: number; phone_use: number; road_type: string; weather_code: string };

async function get<T>(path: string): Promise<T> {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`Request failed (${r.status}) for ${path}`);
  return r.json();
}

export const fetchFleet = () => get<{ drivers: Driver[]; fleet: Fleet }>('/fleet?base_premium=1200');
export const fetchPoints = () => get<{ count: number; points: Point[] }>('/fleet/points?limit=5000');
export const fetchDriverEvents = (id: string) =>
  get<{ count: number; events: DriverEvent[] }>(`/driver/${encodeURIComponent(id)}/events`);

/* Pricing, mirrored from src/pricing.py so the base-premium slider can reprice instantly. */
export const relativity = (score: number) => Math.max(0.7, Math.min(1.5, 1 + 0.35 * ((score - 50) / 50)));
export const premiumFor = (base: number, score: number) => Math.round(base * relativity(score) * 100) / 100;

export const usd = (n: number, digits = 0) =>
  n.toLocaleString('en-US', { style: 'currency', currency: 'USD', minimumFractionDigits: digits, maximumFractionDigits: digits });
export const pct = (n: number) => `${Math.round(n * 100)}%`;

export type Band = { label: string; color: string; icon: string };
/** Risk bands for quick scanning; always shown with a label, never color alone. */
export const band = (score: number): Band =>
  score >= 70 ? { label: 'High risk', color: 'var(--critical)', icon: '▲' }
  : score >= 40 ? { label: 'Moderate', color: 'var(--warning)', icon: '◆' }
  : { label: 'Low risk', color: 'var(--good)', icon: '●' };
