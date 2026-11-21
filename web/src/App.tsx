import { useEffect, useState } from 'react';
import { Header, Hero, Kpis } from './components/Top';
import { PricingCurve, ScoreHistogram } from './components/FleetCharts';
import { DriverDetail, DriverTable } from './components/Drivers';
import { fetchFleet, type Driver, type Fleet } from './lib/api';

export default function App() {
  const [drivers, setDrivers] = useState<Driver[]>([]);
  const [fleet, setFleet] = useState<Fleet | null>(null);
  const [base, setBase] = useState(1200);
  const [selected, setSelected] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchFleet()
      .then((r) => { setDrivers(r.drivers); setFleet(r.fleet); setSelected(r.drivers[0]?.driver_id ?? null); })
      .catch((e) => setError(e instanceof Error ? e.message : 'Could not load the fleet.'));
  }, []);

  const current = drivers.find((d) => d.driver_id === selected) ?? null;

  return (
    <>
      <Header />
      <main className="wrap pb-16">
        <Hero fleet={fleet} base={base} setBase={setBase} />
        {error && <p role="alert" className="card p-4 text-sm">⚠ {error}</p>}
        {fleet && (
          <div className="grid gap-4">
            <Kpis drivers={drivers} fleet={fleet} base={base} />
            <div className="grid gap-4 lg:grid-cols-2">
              <ScoreHistogram fleet={fleet} />
              <PricingCurve base={base} selected={current} />
            </div>
            <div className="grid items-start gap-4 lg:grid-cols-[1.45fr_1fr]">
              <DriverTable drivers={drivers} base={base} selected={selected} onSelect={setSelected} />
              {current && <DriverDetail d={current} base={base} />}
            </div>
          </div>
        )}
      </main>
    </>
  );
}
