"""A research script that leaks the future, used as the reference fixture.

Every construct marked with a rule code below is a real look-ahead bias or
target leakage idiom that has shipped to production in somebody's backtest. The
corrected version of the same script is clean_research.py.

Nothing here is meant to be executed; it is parsed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_prices(rows: int = 500, seed: int = 7) -> pd.DataFrame:
    """Generate a synthetic daily close series."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2020-01-01", periods=rows, freq="D")
    close = 100.0 + np.cumsum(rng.normal(0.0, 1.0, rows))
    return pd.DataFrame({"close": close, "volume": rng.integers(1_000, 9_000, rows)}, index=index)


def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Build the feature matrix, leaking future information five different ways."""
    features = pd.DataFrame(index=prices.index)
    features["ret"] = prices["close"].pct_change()
    features["target"] = prices["close"].pct_change().shift(-1)  # LA001
    features["trend"] = prices["close"].rolling(20, center=True).mean()  # LA003
    features["ret"] = features["ret"].bfill()  # LA002
    features["volume"] = prices["volume"].fillna(method="backfill")  # LA002
    features["spread"] = features["ret"].interpolate(limit_direction="both")  # LA002
    return features


def attach_events(features: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Join scheduled events onto the feature frame."""
    return pd.merge_asof(features, events, on="timestamp", direction="forward")  # LA007


def realized_next_return(prices: pd.DataFrame) -> list[float]:
    """Compute the return realized on the bar after each observation."""
    close = prices["close"].to_numpy()
    returns: list[float] = []
    for i in range(len(close) - 1):
        returns.append(close[i + 1] / close[i] - 1.0)  # LA006
    return returns


def fit_model(features: pd.DataFrame) -> Ridge:
    """Standardize, split, and fit, in exactly the wrong order."""
    frame = features.dropna()
    labels = frame.pop("target")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(frame)  # LA004
    x_train, x_test, y_train, y_test = train_test_split(scaled, labels, test_size=0.2)  # LA005
    model = Ridge().fit(x_train, y_train)
    print(f"held-out r2: {model.score(x_test, y_test):.4f}")
    return model
