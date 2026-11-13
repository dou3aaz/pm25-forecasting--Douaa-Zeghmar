import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

TARGET = "pm25"
DROP_COLS = ["date", "datetime_utc", "id", "latitude", "longitude", "sensor_name",
             "year", "month", TARGET]

def get_features(df):
    return [c for c in df.columns if c not in DROP_COLS]

def make_pipeline(model, scale=False):
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)
