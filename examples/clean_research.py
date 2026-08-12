"""The corrected twin of leaky_research.py. It reports zero findings.

The only forward-looking line left is the label, which is legitimate and is
declared as such with an inline suppression so that a reviewer sees the intent
instead of a silenced warning.
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
    """Build the feature matrix from trailing information only."""
    features = pd.DataFrame(index=prices.index)
    features["ret"] = prices["close"].pct_change()
    # The label is the one place a forward shift belongs. Declared, not hidden.
    target = prices["close"].pct_change().shift(-1)  # lookahead-lint: ignore[LA001] label
    features["target"] = target
    features["trend"] = prices["close"].rolling(20).mean()
    features["ret"] = features["ret"].ffill()
    features["volume"] = prices["volume"].fillna(method="ffill")
    features["spread"] = features["ret"].interpolate(limit_direction="forward")
    return features


def attach_events(features: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Join the most recent already-published event onto each row."""
    return pd.merge_asof(features, events, on="timestamp", direction="backward")


def realized_prior_return(prices: pd.DataFrame) -> list[float]:
    """Compute the return realized on the bar before each observation."""
    close = prices["close"].to_numpy()
    returns: list[float] = []
    for i in range(1, len(close)):
        returns.append(close[i] / close[i - 1] - 1.0)
    return returns


def fit_model(features: pd.DataFrame) -> Ridge:
    """Split on time order first, then learn every statistic on the training rows."""
    frame = features.dropna()
    labels = frame.pop("target")
    x_train, x_test, y_train, y_test = train_test_split(frame, labels, test_size=0.2, shuffle=False)
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(x_train)
    scaled_test = scaler.transform(x_test)
    model = Ridge().fit(scaled_train, y_train)
    print(f"held-out r2: {model.score(scaled_test, y_test):.4f}")
    return model
