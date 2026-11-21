import { Bar, BarChart, CartesianGrid, Line, LineChart, ReferenceDot, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import type { Driver, Fleet } from '../lib/api';
import { premiumFor, relativity, usd } from '../lib/api';

const axis = { fill: 'var(--faint)', fontSize: 12, fontFamily: 'var(--f-mono)' };

function Box({ children }: { children: React.ReactNode }) {
  return <div className="rounded-lg border border-line bg-surface px-3 py-2 text-[13px] text-ink shadow-lg">{children}</div>;
}

/** How many drivers fall in each 10-point score band. */
export function ScoreHistogram({ fleet }: { fleet: Fleet }) {
  return (
    <section className="card min-w-0 p-5" aria-labelledby="hist-h">
      <h2 id="hist-h" className="m-0 text-lg font-bold">Risk score distribution</h2>
      <p className="m-0 mt-1 text-[13px] text-faint">Drivers per 10-point band. Higher scores mean higher expected incident risk.</p>
      <div className="mt-4 h-[240px]">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={fleet.score_histogram} margin={{ top: 8, right: 4, left: -8, bottom: 0 }} barCategoryGap="14%">
            <CartesianGrid vertical={false} stroke="var(--line)" />
            <XAxis dataKey="bucket" tick={axis} axisLine={false} tickLine={false} interval={0} fontSize={11} />
            <YAxis tick={axis} axisLine={false} tickLine={false} allowDecimals={false} width={40} />
            <Tooltip cursor={{ fill: 'var(--surface-2)' }}
              content={({ active, payload, label }) => active && payload?.length ? (
                <Box><b>Score {label}</b><br /><span className="num">{payload[0].value} drivers</span></Box>
              ) : null} />
            <Bar dataKey="drivers" fill="var(--s1)" radius={[4, 4, 0, 0]} isAnimationActive={false} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}

/** The pricing rule as a curve: premium vs score, with the selected driver marked. */
export function PricingCurve({ base, selected }: { base: number; selected: Driver | null }) {
  const data = Array.from({ length: 101 }, (_, s) => ({ score: s, premium: premiumFor(base, s) }));
  return (
    <section className="card min-w-0 p-5" aria-labelledby="curve-h">
      <h2 id="curve-h" className="m-0 text-lg font-bold">How the score sets the price</h2>
      <p className="m-0 mt-1 text-[13px] text-faint">
        premium = base × (1 + 0.35 × (score − 50) ÷ 50), clamped to 0.70×–1.50×. Score 50 pays exactly the base.
      </p>
      <div className="mt-4 h-[240px]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 8, right: 16, left: 4, bottom: 0 }}>
            <CartesianGrid stroke="var(--line)" vertical={false} />
            <XAxis dataKey="score" type="number" domain={[0, 100]} ticks={[0, 25, 50, 75, 100]} tick={axis} axisLine={false} tickLine={false} />
            <YAxis tick={axis} axisLine={false} tickLine={false} width={56} tickFormatter={(v) => usd(v)} domain={['dataMin - 50', 'dataMax + 50']} />
            <ReferenceLine y={base} stroke="var(--faint)" strokeDasharray="4 4" />
            <Tooltip content={({ active, payload, label }) => active && payload?.length ? (
              <Box><b>Score {label}</b><br /><span className="num">{usd(Number(payload[0].value))} · {relativity(Number(label)).toFixed(2)}×</span></Box>
            ) : null} />
            <Line dataKey="premium" stroke="var(--s1)" strokeWidth={2} dot={false} isAnimationActive={false} />
            {selected && (
              <ReferenceDot x={selected.risk_score} y={premiumFor(base, selected.risk_score)} r={6} fill="var(--brand)" stroke="var(--chart-surface)" strokeWidth={2}
                label={{ value: selected.driver_id, position: 'top', fill: 'var(--ink)', fontSize: 12, fontWeight: 700 }} />
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}
