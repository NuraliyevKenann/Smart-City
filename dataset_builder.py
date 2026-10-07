"""
Baku traffic dataset builder.

Goal:
Create a training dataset for a traffic congestion ML project in Baku.

What this script does:
1. Generates many road-time observations for Baku.
2. Optionally enriches them with real historical weather from Open-Meteo.
3. Optionally adds a news/event signal from GDELT.
4. Saves a CSV dataset that can be used later for ML training.

The congestion label is synthetic for now, because we do not yet have real
traffic labels from cameras, GPS traces, or road sensors. Later, you can replace
the `estimate_congestion_level` function with real labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import ssl
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.error import URLError
from urllib.request import urlopen


BAKU_LATITUDE = 40.4093
BAKU_LONGITUDE = 49.8671
BAKU_TIMEZONE = "Asia/Baku"

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


@dataclass(frozen=True)
class RoadSegment:
    district: str
    road_name: str
    base_congestion: float
    has_metro_nearby: int
    bus_lane_available: int
    parking_pressure: int


ROAD_SEGMENTS = [
    RoadSegment("Yasamal", "Inshaatchilar Ave", 0.70, 1, 0, 1),
    RoadSegment("Yasamal", "Tbilisi Ave", 0.78, 1, 0, 1),
    RoadSegment("Nasimi", "28 May St", 0.82, 1, 0, 1),
    RoadSegment("Nasimi", "Azadliq Ave", 0.74, 1, 1, 1),
    RoadSegment("Sabail", "Neftchilar Ave", 0.76, 1, 0, 1),
    RoadSegment("Sabail", "Istiglaliyyat St", 0.66, 0, 0, 1),
    RoadSegment("Khatai", "Babek Ave", 0.72, 1, 1, 0),
    RoadSegment("Khatai", "Nobel Ave", 0.67, 0, 0, 0),
    RoadSegment("Narimanov", "Heydar Aliyev Ave", 0.80, 1, 1, 0),
    RoadSegment("Narimanov", "Ataturk Ave", 0.69, 1, 0, 1),
    RoadSegment("Binagadi", "Ziya Bunyadov Ave", 0.77, 0, 1, 0),
    RoadSegment("Nizami", "Qara Qarayev Ave", 0.71, 1, 1, 1),
    RoadSegment("Surakhani", "Airport Highway", 0.60, 0, 0, 0),
    RoadSegment("Garadagh", "Baku-Salyan Highway", 0.48, 0, 0, 0),
]

EVENT_KEYWORDS = (
    "traffic OR concert OR football OR accident OR road OR protest OR festival "
    "OR Formula 1 OR Baku"
)


def ensure_directories() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


ALLOW_INSECURE_SSL = False


def fetch_json(url: str, retries: int = 3, sleep_seconds: float = 1.0) -> dict[str, Any]:
    context = build_ssl_context(allow_insecure=ALLOW_INSECURE_SSL)

    for attempt in range(1, retries + 1):
        try:
            with urlopen(url, timeout=30, context=context) as response:
                return json.loads(response.read().decode("utf-8"))
        except URLError as error:
            if "CERTIFICATE_VERIFY_FAILED" in str(error) and not ALLOW_INSECURE_SSL:
                raise RuntimeError(
                    "SSL certificate verification failed. Try installing certifi "
                    "or run this educational script with --allow-insecure-ssl."
                ) from error
            if attempt == retries:
                raise
            time.sleep(sleep_seconds)
        except Exception:
            if attempt == retries:
                raise
            time.sleep(sleep_seconds)
    return {}


def build_ssl_context(allow_insecure: bool = False) -> ssl.SSLContext:
    if allow_insecure:
        return ssl._create_unverified_context()

    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def fetch_open_meteo_weather(start_date: date, end_date: date) -> dict[str, dict[str, Any]]:
    """Return hourly weather indexed by ISO datetime string."""
    params = {
        "latitude": BAKU_LATITUDE,
        "longitude": BAKU_LONGITUDE,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": ",".join(
            [
                "temperature_2m",
                "precipitation",
                "rain",
                "weather_code",
                "wind_speed_10m",
                "relative_humidity_2m",
            ]
        ),
        "timezone": BAKU_TIMEZONE,
    }
    url = "https://archive-api.open-meteo.com/v1/archive?" + urlencode(params)
    payload = fetch_json(url)
    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])

    weather_by_time: dict[str, dict[str, Any]] = {}
    for index, iso_time in enumerate(times):
        weather_by_time[iso_time] = {
            "temperature": safe_list_get(hourly.get("temperature_2m", []), index, 18.0),
            "rain_mm": safe_list_get(hourly.get("rain", []), index, 0.0),
            "precipitation_mm": safe_list_get(hourly.get("precipitation", []), index, 0.0),
            "weather_code": safe_list_get(hourly.get("weather_code", []), index, 0),
            "wind_speed_kmh": safe_list_get(hourly.get("wind_speed_10m", []), index, 7.0),
            "humidity": safe_list_get(hourly.get("relative_humidity_2m", []), index, 65),
        }
    return weather_by_time


def fetch_gdelt_event_counts(start_date: date, end_date: date) -> dict[str, int]:
    """
    Return approximate Baku-related news/event article counts by date.

    GDELT DOC 2.0 is a news search API. We use article counts as a weak feature:
    if the city has more event/traffic-related news on a date, the chance of
    disruption may be higher.
    """
    counts: dict[str, int] = {}
    current = start_date

    while current <= end_date:
        next_day = current + timedelta(days=1)
        params = {
            "query": f'({EVENT_KEYWORDS}) sourcecountry:AZ',
            "mode": "artlist",
            "format": "json",
            "maxrecords": 250,
            "startdatetime": current.strftime("%Y%m%d000000"),
            "enddatetime": next_day.strftime("%Y%m%d000000"),
        }
        url = "https://api.gdeltproject.org/api/v2/doc/doc?" + urlencode(params)

        try:
            payload = fetch_json(url, retries=2)
            counts[current.isoformat()] = len(payload.get("articles", []))
        except Exception:
            counts[current.isoformat()] = 0

        current = next_day
        time.sleep(0.2)

    return counts


def safe_list_get(values: list[Any], index: int, default: Any) -> Any:
    if index >= len(values) or values[index] is None:
        return default
    return values[index]


def weather_name(weather_code: int, rain_mm: float) -> str:
    if rain_mm >= 1.0:
        return "rain"
    if weather_code in {45, 48}:
        return "fog"
    if weather_code in {71, 73, 75, 77, 85, 86}:
        return "snow"
    if weather_code in {1, 2, 3}:
        return "cloudy"
    return "clear"


def synthetic_weather(observation_time: datetime) -> dict[str, Any]:
    month = observation_time.month
    seasonal_temp = 16 + 11 * math.sin((month - 3) / 12 * 2 * math.pi)
    temperature = round(random.gauss(seasonal_temp, 4.0), 1)
    rain_chance = 0.18 if month in {10, 11, 12, 1, 2, 3, 4} else 0.08
    rain_mm = round(max(0.0, random.gauss(3.0, 2.0)), 1) if random.random() < rain_chance else 0.0
    code = 61 if rain_mm > 0 else random.choice([0, 1, 2, 3])
    return {
        "temperature": temperature,
        "rain_mm": rain_mm,
        "precipitation_mm": rain_mm,
        "weather_code": code,
        "wind_speed_kmh": round(max(0.0, random.gauss(14.0, 6.0)), 1),
        "humidity": random.randint(45, 90),
    }


def is_holiday_like_day(day: date) -> int:
    fixed_holidays = {
        (1, 1),
        (1, 20),
        (3, 8),
        (3, 20),
        (3, 21),
        (3, 22),
        (5, 9),
        (5, 28),
        (6, 15),
        (6, 26),
        (11, 8),
        (11, 9),
        (12, 31),
    }
    return int((day.month, day.day) in fixed_holidays)


def estimate_congestion_level(
    road: RoadSegment,
    observation_time: datetime,
    weather: dict[str, Any],
    event_count: int,
) -> tuple[str, float, int, int]:
    """
    Build a realistic synthetic target.

    This is not the final ML logic. It only creates labels for practice while
    real traffic labels are unavailable.
    """
    hour = observation_time.hour
    weekday = observation_time.weekday()
    is_weekend = weekday >= 5

    morning_peak = 7 <= hour <= 10
    evening_peak = 17 <= hour <= 20
    late_night = hour <= 5 or hour >= 23

    score = road.base_congestion
    score += 0.42 if morning_peak and not is_weekend else 0.0
    score += 0.48 if evening_peak and not is_weekend else 0.0
    score += 0.18 if morning_peak and is_weekend else 0.0
    score += 0.12 if evening_peak and is_weekend else 0.0
    score -= 0.30 if late_night else 0.0
    score += min(float(weather["rain_mm"]) * 0.055, 0.35)
    score += min(float(weather["wind_speed_kmh"]) * 0.006, 0.12)
    score += 0.16 if event_count >= 15 else 0.0
    score += 0.10 if is_holiday_like_day(observation_time.date()) else 0.0
    score += 0.08 if road.parking_pressure else 0.0
    score -= 0.07 if road.bus_lane_available else 0.0
    score += random.gauss(0, 0.12)

    score = max(0.0, min(score, 2.2))

    if score < 0.65:
        level = "low"
    elif score < 1.05:
        level = "medium"
    elif score < 1.45:
        level = "high"
    else:
        level = "severe"

    avg_speed = max(6, round(58 - score * 24 + random.gauss(0, 4), 1))
    delay_minutes = max(0, round(score * 18 + random.gauss(0, 4)))
    return level, avg_speed, delay_minutes, round(score, 3)


def build_dataset(
    start_date: date,
    end_date: date,
    output_path: Path,
    use_api: bool,
) -> None:
    ensure_directories()

    weather_by_time: dict[str, dict[str, Any]] = {}
    event_counts: dict[str, int] = {}

    if use_api:
        print("Fetching historical weather from Open-Meteo...")
        weather_by_time = fetch_open_meteo_weather(start_date, end_date)
        print("Fetching event/news signals from GDELT...")
        event_counts = fetch_gdelt_event_counts(start_date, end_date)

    fieldnames = [
        "date",
        "hour",
        "day_of_week",
        "is_weekend",
        "is_holiday",
        "district",
        "road_name",
        "has_metro_nearby",
        "bus_lane_available",
        "parking_pressure",
        "weather",
        "temperature",
        "rain_mm",
        "precipitation_mm",
        "wind_speed_kmh",
        "humidity",
        "event_news_count",
        "event_nearby",
        "avg_speed_kmh",
        "delay_minutes",
        "congestion_score",
        "congestion_level",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        current = datetime.combine(start_date, datetime.min.time())
        finish = datetime.combine(end_date, datetime.max.time()).replace(minute=0, second=0, microsecond=0)
        rows = 0

        while current <= finish:
            iso_hour = current.strftime("%Y-%m-%dT%H:00")
            weather = weather_by_time.get(iso_hour, synthetic_weather(current))
            event_count = event_counts.get(current.date().isoformat(), random.randint(0, 18))

            for road in ROAD_SEGMENTS:
                congestion_level, avg_speed, delay_minutes, score = estimate_congestion_level(
                    road=road,
                    observation_time=current,
                    weather=weather,
                    event_count=event_count,
                )

                writer.writerow(
                    {
                        "date": current.date().isoformat(),
                        "hour": current.hour,
                        "day_of_week": current.strftime("%A"),
                        "is_weekend": int(current.weekday() >= 5),
                        "is_holiday": is_holiday_like_day(current.date()),
                        "district": road.district,
                        "road_name": road.road_name,
                        "has_metro_nearby": road.has_metro_nearby,
                        "bus_lane_available": road.bus_lane_available,
                        "parking_pressure": road.parking_pressure,
                        "weather": weather_name(int(weather["weather_code"]), float(weather["rain_mm"])),
                        "temperature": weather["temperature"],
                        "rain_mm": weather["rain_mm"],
                        "precipitation_mm": weather["precipitation_mm"],
                        "wind_speed_kmh": weather["wind_speed_kmh"],
                        "humidity": weather["humidity"],
                        "event_news_count": event_count,
                        "event_nearby": int(event_count >= 10),
                        "avg_speed_kmh": avg_speed,
                        "delay_minutes": delay_minutes,
                        "congestion_score": score,
                        "congestion_level": congestion_level,
                    }
                )
                rows += 1

            current += timedelta(hours=1)

    print(f"Saved {rows:,} rows to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Baku traffic ML dataset.")
    parser.add_argument("--start-date", default="2025-01-01", help="YYYY-MM-DD")
    parser.add_argument("--end-date", default="2025-12-31", help="YYYY-MM-DD")
    parser.add_argument(
        "--output",
        default=str(PROCESSED_DIR / "baku_traffic_dataset.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--use-api",
        action="store_true",
        help="Fetch real weather and news/event signals. Slower, needs internet.",
    )
    parser.add_argument(
        "--allow-insecure-ssl",
        action="store_true",
        help="Disable SSL verification if your local Python has broken certificates.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    global ALLOW_INSECURE_SSL

    args = parse_args()
    random.seed(args.seed)
    ALLOW_INSECURE_SSL = args.allow_insecure_ssl

    start = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end = datetime.strptime(args.end_date, "%Y-%m-%d").date()

    if end < start:
        raise ValueError("end-date must be after start-date")

    build_dataset(
        start_date=start,
        end_date=end,
        output_path=Path(args.output),
        use_api=args.use_api,
    )


if __name__ == "__main__":
    main()
