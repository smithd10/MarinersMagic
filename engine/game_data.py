"""Game database: CSV loading, feature computation, normalization."""

import pandas as pd
import numpy as np
from datetime import datetime

PREMIUM_OPPONENTS = {"Yankees", "Dodgers", "Red Sox", "Cubs", "Mets", "Giants", "Phillies", "Braves", "Padres", "Blue Jays"}
MID_OPPONENTS = {"Astros", "Rays", "Orioles", "Twins", "Royals", "Diamondbacks", "Reds", "Rangers"}
LOW_OPPONENTS = {"Athletics", "White Sox", "Angels", "Tigers", "Guardians", "Marlins", "Rockies", "Pirates", "Nationals"}

OPPONENT_ABBREVS = {
    "Yankees": "NYY", "Guardians": "CLE", "Astros": "HOU", "Rangers": "TEX",
    "Athletics": "OAK", "Royals": "KC", "Braves": "ATL", "Padres": "SD",
    "White Sox": "CWS", "Diamondbacks": "ARI", "Mets": "NYM", "Orioles": "BAL",
    "Red Sox": "BOS", "Angels": "LAA", "Blue Jays": "TOR", "Giants": "SF",
    "Reds": "CIN", "Twins": "MIN", "Tigers": "DET", "Rays": "TB",
    "Cubs": "CHC", "Phillies": "PHI", "Dodgers": "LAD", "Marlins": "MIA",
    "Rockies": "COL", "Pirates": "PIT", "Nationals": "WSH", "Brewers": "MIL",
    "Cardinals": "STL",
}

JEWISH_HOLIDAYS_2026 = {
    "2026-09-25": "Sukkot",
    "2026-09-26": "Sukkot",
    "2026-09-27": "Sukkot",
}

FACE_VALUE_BY_TIER = {
    "Value": 22,
    "Standard": 30,
    "Premium": 40,
    "Elite": 55,
}

PROMO_ICONS = {
    "bobblehead": "B",
    "fireworks": "F",
    "giveaway": "G",
}


def opponent_tier(opponent: str) -> str:
    if opponent in PREMIUM_OPPONENTS:
        return "premium"
    if opponent in MID_OPPONENTS:
        return "mid"
    return "low"


def month_bucket(month: int) -> str:
    if month <= 5:
        return "spring"
    if month <= 8:
        return "summer"
    return "fall"


def default_preference(row: pd.Series) -> int:
    tier = row["opponent_tier"]
    base = {"premium": 7, "mid": 5, "low": 3}[tier]
    day = row["day_of_week"]
    day_bonus = 2 if day == "Saturday" else (1 if day in ("Friday", "Sunday") else 0)
    summer_bonus = 1 if row["month_bucket"] == "summer" else 0
    promo_bonus = 1 if row.get("promo", "") else 0
    return max(1, min(10, base + day_bonus + summer_bonus + promo_bonus))


def load_games(csv_path: str, face_value_override: float = None) -> pd.DataFrame:
    """Load game data from CSV and compute all derived fields."""
    df = pd.read_csv(csv_path)

    col_map = {}
    for col in df.columns:
        lc = col.lower().strip()
        if "date" in lc:
            col_map["date"] = col
        elif "opponent" in lc:
            col_map["opponent"] = col
        elif "time" in lc:
            col_map["time"] = col
        elif "day" in lc and "week" not in lc:
            col_map["day"] = col
        elif "seatgeek" in lc or "from_price" in lc or "from price" in lc:
            col_map["resale"] = col
        elif "tier" in lc:
            col_map["tier"] = col

    games = pd.DataFrame()
    games["date"] = pd.to_datetime(df[col_map["date"]])
    games["date_str"] = games["date"].dt.strftime("%Y-%m-%d")
    games["day_of_week"] = df.get(col_map.get("day"), games["date"].dt.day_name())
    games["opponent"] = df[col_map["opponent"]].str.strip()
    games["opponent_abbr"] = games["opponent"].map(OPPONENT_ABBREVS).fillna(games["opponent"].str[:3].str.upper())
    games["time"] = df[col_map["time"]].astype(str).str.strip()
    games["resale_value"] = pd.to_numeric(df[col_map["resale"]], errors="coerce").fillna(0)

    if "tier" in col_map:
        games["pricing_tier"] = df[col_map["tier"]].str.strip()
    else:
        games["pricing_tier"] = "Standard"

    if face_value_override is not None:
        games["face_value"] = face_value_override
    else:
        games["face_value"] = games["pricing_tier"].map(FACE_VALUE_BY_TIER).fillna(30)

    games["resale_profit"] = games["resale_value"] - games["face_value"]
    games["opponent_tier"] = games["opponent"].apply(opponent_tier)
    games["is_weekend"] = games["day_of_week"].isin(["Friday", "Saturday", "Sunday"])
    games["month"] = games["date"].dt.month
    games["month_bucket"] = games["month"].apply(month_bucket)
    games["jewish_holiday"] = games["date_str"].map(JEWISH_HOLIDAYS_2026).fillna("")
    games["is_jewish_holiday"] = games["jewish_holiday"] != ""
    games["promo"] = ""
    games["promo_icon"] = ""

    games["preference"] = games.apply(default_preference, axis=1)
    games["available"] = True
    games["must_have"] = False

    games["id"] = range(len(games))
    games = games.reset_index(drop=True)

    games = normalize_games(games)
    return games


def normalize_games(games: pd.DataFrame) -> pd.DataFrame:
    """Compute normalized 0-1 values for preferences and resale profit."""
    pmin, pmax = games["preference"].min(), games["preference"].max()
    if pmax > pmin:
        games["pref_norm"] = (games["preference"] - pmin) / (pmax - pmin)
    else:
        games["pref_norm"] = 0.5

    rmin, rmax = games["resale_profit"].min(), games["resale_profit"].max()
    if rmax > rmin:
        games["resale_profit_norm"] = (games["resale_profit"] - rmin) / (rmax - rmin)
    else:
        games["resale_profit_norm"] = 0.5

    rvmin, rvmax = games["resale_value"].min(), games["resale_value"].max()
    if rvmax > rvmin:
        games["resale_value_norm"] = (games["resale_value"] - rvmin) / (rvmax - rvmin)
    else:
        games["resale_value_norm"] = 0.5

    return games


def build_feature_matrix(games: pd.DataFrame) -> np.ndarray:
    """Build (n_games, n_features) matrix for opponent utility computation.

    Features: weekend, premium_opp, mid_opp, promo, summer, september,
              resale_tier_high, resale_tier_mid, jewish_holiday
    """
    n = len(games)
    features = np.zeros((n, 9), dtype=np.float32)
    features[:, 0] = games["is_weekend"].astype(float).values
    features[:, 1] = (games["opponent_tier"] == "premium").astype(float).values
    features[:, 2] = (games["opponent_tier"] == "mid").astype(float).values
    features[:, 3] = (games["promo"] != "").astype(float).values
    features[:, 4] = (games["month_bucket"] == "summer").astype(float).values
    features[:, 5] = (games["month"] == 9).astype(float).values
    resale_pctile = games["resale_profit_norm"].values
    features[:, 6] = (resale_pctile >= 0.7).astype(float)
    features[:, 7] = ((resale_pctile >= 0.4) & (resale_pctile < 0.7)).astype(float)
    features[:, 8] = games["is_jewish_holiday"].astype(float).values
    return features


FEATURE_NAMES = [
    "weekend", "premium_opp", "mid_opp", "promo", "summer",
    "september", "resale_high", "resale_mid", "jewish_holiday",
]
