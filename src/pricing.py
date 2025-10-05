def risk_score(prob: float) -> int:
    p = max(0.0, min(1.0, float(prob)))
    s = round(100 * (p - 0.02) / (0.25 - 0.02))
    return int(max(0, min(100, s)))

def premium_from_score(base_premium: float, score: int) -> float:
    alpha = 0.35
    rel = 1.0 + alpha * ((score - 50)/50.0)
    rel = max(0.7, min(1.5, rel))
    return round(base_premium * rel, 2)
