import { useEffect, useState } from 'react';
import type { Driver, Fleet } from '../lib/api';
import { premiumFor, usd } from '../lib/api';

const REPO = 'https://github.com/Shreyas9265/Telematics-UBI';

function readTheme(): 'light' | 'dark' {
  try {
    const t = localStorage.getItem('ubi-theme');
    if (t === 'light' || t === 'dark') return t;
  } catch { /* storage blocked */ }
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

export function Header() {
  const [theme, setTheme] = useState(readTheme);
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try { localStorage.setItem('ubi-theme', theme); } catch { /* storage blocked */ }
  }, [theme]);
  return (
    <header className="sticky top-0 z-20 border-b border-line bg-bg/85 backdrop-blur">
      <div className="wrap flex items-center justify-between gap-4 py-3">
        <a href="#top" className="flex items-center gap-2.5 no-underline">
          <span className="grid size-8 place-items-center rounded-lg bg-brand text-brand-ink" aria-hidden="true">
            <svg viewBox="0 0 24 24" className="size-5" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round">
              <path d="M4 16a8 8 0 0 1 16 0" /><path d="M12 16l4.5-5" />
            </svg>
          </span>
          <span className="font-display font-bold tracking-tight text-ink">Telematics UBI</span>
        </a>
        <nav className="flex items-center gap-1.5 text-sm sm:gap-2">
          <a className="btn hidden !py-2 sm:inline-flex" href="#how">How it works</a>
          <a className="btn !py-2" href="/docs" target="_blank" rel="noopener">API docs</a>
          <a className="btn hidden !py-2 md:inline-flex" href={REPO} target="_blank" rel="noopener">GitHub ↗</a>
          <button type="button" className="btn !px-2.5 !py-2" onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
            aria-label={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'} title="Toggle theme">
            {theme === 'dark' ? '☀' : '☾'}
          </button>
        </nav>
      </div>
    </header>
  );
}

const fmtWindow = (start: string | null, end: string | null) => {
  if (!start || !end) return '';
  const o: Intl.DateTimeFormatOptions = { month: 'short', day: 'numeric', year: 'numeric' };
  const a = new Date(start).toLocaleDateString('en-US', o), b = new Date(end).toLocaleDateString('en-US', o);
  return a === b ? a : `${a} – ${b}`;
};

/** Hero: what this is, the dataset, and the base premium every price is relative to. */
export function Hero({ fleet, base, setBase }: { fleet: Fleet | null; base: number; setBase: (n: number) => void }) {
  return (
    <section id="top" className="grid gap-6 pb-6 pt-8 lg:grid-cols-[1.3fr_1fr] lg:items-end lg:pt-12">
      <div className="min-w-0">
        <p className="eyebrow m-0">Insurtech · FastAPI + React</p>
        <h1 className="m-0 mt-2 text-[clamp(30px,4.4vw,46px)] font-extrabold leading-[1.05] tracking-tight [text-wrap:balance]">
          Pay-how-you-drive pricing from driving data.
        </h1>
        <p className="m-0 mt-3 max-w-[58ch] text-[15.5px] leading-relaxed text-muted">
          Telematics events (speed, braking, phone use, road and weather) roll up into a risk score per driver, and the score sets each driver's premium relative to a base price.
        </p>
      </div>
      <div className="card grid gap-3 p-4">
        <div className="flex items-center justify-between gap-3">
          <span className="eyebrow">Dataset</span>
          <span className="chip">Simulated</span>
        </div>
        <p className="m-0 text-sm text-ink">
          {fleet ? (
            <><b className="num">{fleet.events.toLocaleString()}</b> driving events from <b className="num">{fleet.drivers}</b> drivers · {fmtWindow(fleet.start, fleet.end)} · Buffalo, NY area</>
          ) : 'Loading the dataset…'}
        </p>
        <label htmlFor="base" className="grid gap-1.5">
          <span className="flex items-baseline justify-between text-sm">
            <span className="font-semibold text-muted">Base annual premium</span>
            <b className="num text-lg text-ink">{usd(base)}</b>
          </span>
          <input id="base" type="range" min={600} max={3000} step={50} value={base} onChange={(e) => setBase(Number(e.target.value))} />
          <span className="text-xs text-faint">Every driver's premium is priced relative to this, from 0.70× (safest) to 1.35× (riskiest).</span>
        </label>
      </div>
    </section>
  );
}

/** Four headline numbers. */
export function Kpis({ drivers, fleet, base }: { drivers: Driver[]; fleet: Fleet; base: number }) {
  const premiums = drivers.map((d) => premiumFor(base, d.risk_score));
  const avgPremium = premiums.reduce((a, b) => a + b, 0) / Math.max(1, premiums.length);
  const high = drivers.filter((d) => d.risk_score >= 70).length;
  const discounted = drivers.filter((d) => d.risk_score < 50).length;
  const tiles = [
    { label: 'Drivers scored', value: String(fleet.drivers), note: `${fleet.events.toLocaleString()} events` },
    { label: 'Average risk score', value: `${fleet.avg_risk_score}`, note: 'out of 100 (higher = riskier)' },
    { label: 'Average premium', value: usd(avgPremium), note: `${avgPremium >= base ? '+' : ''}${(((avgPremium / base) - 1) * 100).toFixed(1)}% vs ${usd(base)} base` },
    { label: 'High-risk drivers', value: String(high), note: `score 70+ · ${discounted} drivers get a discount`, alert: high > 0 },
  ];
  return (
    <section aria-label="Summary" className="grid grid-cols-2 gap-3 lg:grid-cols-4">
      {tiles.map((t) => (
        <div key={t.label} className="card grid gap-1 p-4">
          <span className="text-[13px] font-semibold text-muted">{t.label}</span>
          <span className={`num font-display text-[clamp(24px,3vw,32px)] font-bold leading-tight tracking-tight ${t.alert ? 'text-critical' : 'text-ink'}`}>
            {t.alert && <span aria-hidden="true" className="mr-1.5 text-[0.65em]">▲</span>}{t.value}
          </span>
          <span className="text-[12.5px] text-faint">{t.note}</span>
        </div>
      ))}
    </section>
  );
}
