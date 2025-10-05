from pydantic import BaseModel

class Event(BaseModel):
    driver_id: str
    ts: str
    lat: float
    lon: float
    speed_kph: float
    accel_mps2: float
    brake_mps2: float
    heading_deg: float
    odometer_km: float
    phone_use: int
    road_type: str
    weather_code: str

class PriceReq(BaseModel):
    driver_id: str
    base_premium: float
