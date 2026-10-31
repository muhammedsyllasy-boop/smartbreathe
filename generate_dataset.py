from pathlib import Path
import numpy as np
import pandas as pd

# Synthetic demonstration data only. These are not real monitoring observations.
RNG = np.random.default_rng(42)
N = 4000

def clipped_normal(mean, sd, low, high, n=N):
    return np.clip(RNG.normal(mean, sd, n), low, high)

# Create a range of plausible educational examples, deliberately covering
# clean through highly polluted conditions. Units are shown in the dashboard.
pm25 = np.clip(RNG.gamma(shape=2.1, scale=28, size=N), 2, 300)
pm10 = np.clip(pm25 * RNG.uniform(1.2, 2.8, N) + RNG.normal(8, 15, N), 5, 500)
no2 = np.clip(RNG.gamma(2.0, 18, N), 2, 250)
so2 = np.clip(RNG.gamma(1.7, 8, N), 1, 180)
co = np.clip(RNG.gamma(1.8, 1.2, N), 0.1, 40)  # mg/m3
o3 = np.clip(RNG.gamma(2.0, 22, N), 2, 300)
temperature = clipped_normal(29, 7, 5, 48)
humidity = clipped_normal(58, 20, 10, 100)

# Educational proxy: assign severity using approximate concentration bands.
# This is not a full official AQI calculation, which requires pollutant-specific
# averaging periods, breakpoints and sufficient valid monitoring data.
def pm25_band(x):
    if x <= 30: return 0
    if x <= 60: return 1
    if x <= 90: return 2
    if x <= 120: return 3
    if x <= 250: return 4
    return 5

def pm10_band(x):
    if x <= 50: return 0
    if x <= 100: return 1
    if x <= 250: return 2
    if x <= 350: return 3
    if x <= 430: return 4
    return 5

def no2_band(x):
    if x <= 40: return 0
    if x <= 80: return 1
    if x <= 180: return 2
    if x <= 280: return 3
    if x <= 400: return 4
    return 5

severity = np.array([max(pm25_band(a), pm10_band(b), no2_band(c)) for a,b,c in zip(pm25, pm10, no2)])
# Small amount of realistic label ambiguity for demonstrating model evaluation.
# It is still synthetic and must not be presented as measured real-world accuracy.
noise_mask = RNG.random(N) < 0.035
severity[noise_mask] = np.clip(severity[noise_mask] + RNG.choice([-1, 1], noise_mask.sum()), 0, 5)

categories = np.array(["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"])
risk_map = np.array(["Low", "Low", "Moderate", "High", "Very High", "Very High"])
df = pd.DataFrame({
    "PM2.5": np.round(pm25, 2),
    "PM10": np.round(pm10, 2),
    "NO2": np.round(no2, 2),
    "SO2": np.round(so2, 2),
    "CO": np.round(co, 2),
    "O3": np.round(o3, 2),
    "temperature": np.round(temperature, 1),
    "humidity": np.round(humidity, 1),
    "AQI_Category": categories[severity],
    "Health_Risk": risk_map[severity],
})
out = Path(__file__).resolve().parent / "data" / "breathe_smart_sample.csv"
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
print(f"Saved {len(df)} synthetic demonstration rows to {out}")
print(df.head())
