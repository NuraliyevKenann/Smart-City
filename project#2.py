import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "data/processed/baku_traffic_dataset.csv"


df = pd.read_csv(DATA_PATH)

print("First 5 rows:")
print(df.head())

print("\nDataset info:")
df.info()

X = df.drop(columns=["congestion_level"])
y = df["congestion_level"]

X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=42)

print(X_test.shape)
print(X_train.shape)
print(y_test.shape)
print(y_train.shape)

categorical_features = [
    "day_of_week",
    "district",
    "road_name",
    "weather",
]

numeric_features = [
    "hour",
    "is_weekend",
    "is_holiday",
    "has_metro_nearby",
    "bus_lane_available",
    "parking_pressure",
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
]



