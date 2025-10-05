# data/simulate_telematics.py
import json, random, math
from pathlib import Path
from datetime import datetime, timedelta

def simulate_events(n_drivers=100, hours=6, seed=42, out_path="data/events.ndjson"):
    random.seed(seed)
    start = datetime.utcnow()
    road_types = ["city","highway","rural"]
    weather = ["CLR","RAN","SNW"]
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for d in range(n_drivers):
            driver = f"D_{d:03d}"
            ts = start
            lat, lon = 42.95 + random.random()/100, -78.82 - random.random()/100
            odo = random.uniform(1000, 50000)
            for _ in range(hours*60):  # per-minute events
                rt = random.choices(road_types, weights=[0.5,0.35,0.15])[0]
                wx = random.choices(weather, weights=[0.7,0.25,0.05])[0]
                base = {"city": 40, "highway": 95, "rural": 65}[rt]
                speed = max(0, random.gauss(base, 10))
                accel = random.gauss(0.2, 1.5)
                brake = random.gauss(-0.4, 1.8)
                phone = 1 if random.random() < 0.05 else 0
                heading = random.uniform(0, 360)
                dist_km = speed/60.0
                odo += dist_km
                lat += math.cos(math.radians(heading))*dist_km/110.574
                lon += math.sin(math.radians(heading))*dist_km/(111.320*math.cos(math.radians(lat)))
                ev = {
                    "driver_id": driver,
                    "ts": ts.isoformat() + "Z",
                    "lat": round(lat,6), "lon": round(lon,6),
                    "speed_kph": round(speed,2),
                    "accel_mps2": round(accel,2),
                    "brake_mps2": round(brake,2),
                    "heading_deg": round(heading,1),
                    "odometer_km": round(odo,2),
                    "phone_use": phone,
                    "road_type": rt,
                    "weather_code": wx
                }
                f.write(json.dumps(ev) + "\n")
                ts += timedelta(minutes=1)

if __name__ == "__main__":
    from pathlib import Path
    Path("data/events.ndjson").unlink(missing_ok=True)
    simulate_events(n_drivers=100, hours=6)   # was 10; bump to 100
    print("Wrote data/events.ndjson")
