import { useEffect, useMemo, useState } from 'react';
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import type { Driver, DriverEvent } from '../lib/api';
import { band, fetchDriverEvents, pct, premiumFor, usd } from '../lib/api';

type Key = 'driver_id' | 'risk_score' | 'premium' | 'avg_speed' | 'hard_brake_rate' | 'phone_use_rate' | 'events';
const COLS: { key: Key; label: string; right?: boolean }[] = [
  { key: 'driver_id', label: 'Driver' },
  { key: 'risk_score', label: 'Risk', right: true },
  { key: 'premium', label: 'Premium', right: true },
  { key: 'avg_speed', label: 'Avg speed', right: true },
  { key: 'hard_brake_rate', label: 'Hard brakes', right: true },
  { key: 'phone_use_rate', label: 'Phone use', right: true },
  { key: 'events', label: 'Events', right: true },
];

export function BandChip({ score }: { score: number }) {
  const b = band(score);
  return (
    <span className="inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11.5px] font-bold text-ink"
      style={{ background: `color-mix(in srgb, ${b.color} 16%, transparent)`, border: `1px solid color-mix(in srgb, ${b.color} 50%, transparent)` }}>
      <span aria-hidden="true" style={{ color: b.color }}>{b.icon}</span>{b.label}
    </span>
  );
}

/** Every driver, sortable; click a row to open it in the detail panel. */
export function DriverTable({ drivers, base, selected, onSelect }: { drivers: Driver[]; base: number; selected: string | null; onSelect: (id: string) => void }) {
  const [sort, setSort] = useState<{ key: Key; dir: 1 | -1 }>({ key: 'risk_score', dir: -1 });
  const [q, setQ] = useState('');
  const [page, setPage] = useState(0);
  const PAGE = 12;
  const rows = useMemo(() => {
    const needle = q.trim().toLowerCase();
    const val = (d: Driver) => (sort.key === 'premium' ? premiumFor(base, d.risk_score) : d[sort.key]);
    return drivers
      .filter((d) => !needle || d.driver_id.toLowerCase().includes(needle))
      .sort((a, b) => (val(a) > val(b) ? 1 : val(a) < val(b) ? -1 : 0) * sort.dir);
  }, [drivers, sort, q, base]);
  const pages = Math.max(1, Math.ceil(rows.length / PAGE));
  const cur = Math.min(page, pages - 1);
  const shown = rows.slice(cur * PAGE, cur * PAGE + PAGE);

  return (
    <section className="card min-w-0 p-5" aria-labelledby="drv-h">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h2 id="drv-h" className="m-0 text-lg font-bold">Drivers</h2>
        <input id="driver-search" type="search" value={q} onChange={(e) => { setQ(e.target.value); setPage(0); }} placeholder="Search driver ID (e.g. D_004)"
          className="w-56 rounded-lg border border-line bg-surface-2 px-3 py-2 text-sm text-ink placeholder:text-faint" />
      </div>
      <p className="m-0 mt-1 text-[13px] text-faint">Click a driver to see their score, premium and driving.</p>
      <div className="mt-3 overflow-x-auto">
        <table className="w-full min-w-[620px] border-collapse text-sm">
          <thead className="border-b border-line text-[12.5px]">
            <tr>
              {COLS.map((c) => {
                const active = sort.key === c.key;
                return (
                  <th key={c.key} scope="col" aria-sort={active ? (sort.dir === 1 ? 'ascending' : 'descending') : 'none'} className={`px-2.5 py-2.5 ${c.right ? 'text-right' : 'text-left'}`}>
                    <button type="button" className="cursor-pointer border-0 bg-transparent p-0 font-semibold text-muted hover:text-ink"
                      onClick={() => setSort({ key: c.key, dir: active ? (sort.dir === 1 ? -1 : 1) : c.key === 'driver_id' ? 1 : -1 })}>
                      {c.label} {active ? (sort.dir === 1 ? '↑' : '↓') : ''}
                    </button>
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody>
            {shown.map((d) => {
              const on = d.driver_id === selected;
              return (
                <tr key={d.driver_id} onClick={() => onSelect(d.driver_id)} aria-selected={on}
                  className={`cursor-pointer border-b border-line last:border-0 ${on ? 'bg-brand-soft' : 'hover:bg-surface-2'}`}>
                  <td className="px-2.5 py-2">
                    <button type="button" className="cursor-pointer border-0 bg-transparent p-0 font-mono text-[13px] font-semibold text-ink" onClick={() => onSelect(d.driver_id)}>{d.driver_id}</button>
                  </td>
                  <td className="px-2.5 py-2 text-right"><span className="num mr-2 font-bold text-ink">{d.risk_score}</span><BandChip score={d.risk_score} /></td>
                  <td className="num px-2.5 py-2 text-right font-semibold text-ink">{usd(premiumFor(base, d.risk_score))}</td>
                  <td className="num px-2.5 py-2 text-right text-muted">{d.avg_speed} km/h</td>
                  <td className="num px-2.5 py-2 text-right text-muted">{pct(d.hard_brake_rate)}</td>
                  <td className="num px-2.5 py-2 text-right text-muted">{pct(d.phone_use_rate)}</td>
                  <td className="num px-2.5 py-2 text-right text-muted">{d.events}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="mt-3 flex items-center justify-between text-sm text-muted">
        <span className="num">{rows.length} drivers</span>
        <div className="flex items-center gap-2">
          <button type="button" className="btn !px-3 !py-1.5" disabled={cur === 0} onClick={() => setPage(cur - 1)}>← Prev</button>
          <span className="num">{cur + 1} / {pages}</span>
          <button type="button" className="btn !px-3 !py-1.5" disabled={cur >= pages - 1} onClick={() => setPage(cur + 1)}>Next →</button>
        </div>
      </div>
    </section>
  );
}

/** Semicircle gauge for a 0–100 score. */
function Gauge({ score }: { score: number }) {
  const a = Math.PI * (1 - score / 100);
  const x = 60 + 48 * Math.cos(a), y = 60 - 48 * Math.sin(a);
  const b = band(score);
  return (
    <svg viewBox="0 0 120 72" className="h-auto w-[150px]" role="img" aria-label={`Risk score ${score} of 100, ${b.label}`}>
      <path d="M12 60 A48 48 0 0 1 108 60" fill="none" stroke="var(--line)" strokeWidth="10" strokeLinecap="round" />
      <path d={`M12 60 A48 48 0 0 1 ${x.toFixed(2)} ${y.toFixed(2)}`} fill="none" stroke={b.color} strokeWidth="10" strokeLinecap="round" />
      <text x="60" y="56" textAnchor="middle" fill="var(--ink)" fontSize="24" fontWeight="800" fontFamily="var(--f-display)">{score}</text>
      <text x="60" y="70" textAnchor="middle" fill="var(--faint)" fontSize="8.5" fontFamily="var(--f-mono)">OF 100</text>
    </svg>
  );
}

const Stat = ({ label, value, hint }: { label: string; value: string; hint?: string }) => (
  <div className="rounded-xl border border-line bg-surface-2 px-3 py-2.5">
    <span className="block text-[12px] font-semibold text-muted">{label}</span>
    <span className="num block font-display text-lg font-bold text-ink">{value}</span>
    {hint && <span className="block text-[11.5px] text-faint">{hint}</span>}
  </div>
);

/** Everything about one driver: score, premium, behavior, road mix, and speed over time. */
export function DriverDetail({ d, base }: { d: Driver; base: number }) {
  const [events, setEvents] = useState<DriverEvent[] | null>(null);
  useEffect(() => {
    let alive = true;
    setEvents(null);
    fetchDriverEvents(d.driver_id).then((r) => { if (alive) setEvents(r.events); }).catch(() => { if (alive) setEvents([]); });
    return () => { alive = false; };
  }, [d.driver_id]);

  const premium = premiumFor(base, d.risk_score);
  const delta = premium - base;
  const series = (events ?? []).map((e) => ({
    t: new Date(e.ts).toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' }),
    speed: Math.round(e.speed_kph * 10) / 10,
    brake: e.hard_brake,
  }));
  const road = [
    { k: 'Highway', v: d.highway_share, c: 'var(--s1)' },
    { k: 'City', v: d.city_share, c: 'var(--serious)' },
    { k: 'Urban', v: d.urban_share, c: 'var(--faint)' },
  ];

  return (
    <section className="card min-w-0 p-5 lg:sticky lg:top-20" aria-labelledby="det-h">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="eyebrow m-0">Selected driver</p>
          <h2 id="det-h" className="m-0 mt-1 font-mono text-2xl font-bold">{d.driver_id}</h2>
          <div className="mt-2"><BandChip score={d.risk_score} /></div>
        </div>
        <Gauge score={d.risk_score} />
      </div>

      <div className="mt-4 grid grid-cols-2 gap-2.5">
        <div className="col-span-2 rounded-xl border border-line bg-brand-soft px-4 py-3">
          <span className="block text-[12px] font-semibold text-muted">Annual premium</span>
          <span className="num font-display text-3xl font-extrabold text-ink">{usd(premium)}</span>
          <span className={`num ml-2 text-sm font-bold ${delta > 0 ? 'text-critical' : 'text-ink'}`}>
            {delta > 0 ? `▲ +${usd(delta)} over base` : delta < 0 ? `▼ ${usd(-delta)} discount` : 'same as base'}
          </span>
          <span className="mt-1 block text-[12px] text-faint">
            Incident probability {(d.prob_incident * 100).toFixed(1)}% → score {d.risk_score} → {(premium / base).toFixed(2)}× base
          </span>
        </div>
        <Stat label="Average speed" value={`${d.avg_speed} km/h`} hint={`95th pct ${d.p95_speed} · max ${d.max_speed}`} />
        <Stat label="Hard-brake events" value={pct(d.hard_brake_rate)} hint="share of this driver's events" />
        <Stat label="Phone in use" value={pct(d.phone_use_rate)} hint="share of events" />
        <Stat label="Night driving" value={pct(d.night_share)} hint="events between 10pm and 6am" />
      </div>

      <div className="mt-4">
        <span className="text-[12px] font-semibold text-muted">Road mix</span>
        <div className="mt-1.5 flex h-2.5 overflow-hidden rounded-full bg-line" aria-hidden="true">
          {road.map((r) => <span key={r.k} style={{ width: `${r.v * 100}%`, background: r.c }} />)}
        </div>
        <ul className="m-0 mt-1.5 flex list-none flex-wrap gap-x-4 p-0 text-[12.5px] text-muted">
          {road.map((r) => <li key={r.k} className="flex items-center gap-1.5"><i className="inline-block size-2.5 rounded-sm" style={{ background: r.c }} />{r.k} <b className="num text-ink">{pct(r.v)}</b></li>)}
          <li className="text-faint">· rain {pct(d.rain_share)}</li>
        </ul>
      </div>

      <div className="mt-4">
        <span className="text-[12px] font-semibold text-muted">Speed over the day (km/h)</span>
        <div className="mt-1 h-[150px]">
          {events === null ? (
            <div className="h-full animate-pulse rounded-lg bg-surface-2" aria-busy="true" />
          ) : (
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={series} margin={{ top: 6, right: 8, left: -18, bottom: 0 }}>
                <CartesianGrid stroke="var(--line)" vertical={false} />
                <XAxis dataKey="t" tick={{ fill: 'var(--faint)', fontSize: 10.5, fontFamily: 'var(--f-mono)' }} axisLine={false} tickLine={false} minTickGap={36} />
                <YAxis tick={{ fill: 'var(--faint)', fontSize: 11, fontFamily: 'var(--f-mono)' }} axisLine={false} tickLine={false} width={44} />
                <Tooltip content={({ active, payload, label }) => active && payload?.length ? (
                  <div className="rounded-lg border border-line bg-surface px-3 py-2 text-[13px] text-ink shadow-lg">
                    <b>{label}</b><br /><span className="num">{payload[0].value} km/h</span>
                    {Number((payload[0].payload as { brake: number }).brake) > 0 && <span className="block text-[12px] text-critical">▲ hard brake</span>}
                  </div>
                ) : null} />
                <Line dataKey="speed" stroke="var(--s1)" strokeWidth={2} isAnimationActive={false}
                  dot={(p: { cx?: number; cy?: number; payload?: { brake: number }; index?: number }) =>
                    p.payload?.brake ? <circle key={p.index} cx={p.cx} cy={p.cy} r={4} fill="var(--critical)" stroke="var(--chart-surface)" strokeWidth={2} /> : <g key={p.index} />} />
              </LineChart>
            </ResponsiveContainer>
          )}
        </div>
        <p className="m-0 mt-1 text-[11.5px] text-faint"><span className="text-critical">●</span> marks a hard-brake event.</p>
      </div>
    </section>
  );
}
