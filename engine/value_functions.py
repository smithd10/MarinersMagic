"""Value functions for player and opponent utility computation."""

import numpy as np
import pandas as pd


DMU_BASE = 0.85
DMU_CATEGORIES = ["day_type", "month_bucket", "opponent_tier"]


def get_game_categories(games: pd.DataFrame, game_id: int) -> dict:
    """Get the DMU category values for a game."""
    row = games.iloc[game_id]
    return {
        "day_type": "weekend" if row["is_weekend"] else "weekday",
        "month_bucket": row["month_bucket"],
        "opponent_tier": row["opponent_tier"],
    }


def compute_dmu_decay(games: pd.DataFrame, game_id: int,
                      my_picks: list[int], base: float = DMU_BASE) -> float:
    """Compute diminishing marginal utility decay for a candidate game."""
    if not my_picks:
        return 1.0
    cat = get_game_categories(games, game_id)
    count = 0
    for pid in my_picks:
        pcat = get_game_categories(games, pid)
        for k in DMU_CATEGORIES:
            if pcat[k] == cat[k]:
                count += 1
                break
    return base ** count


def compute_my_value(games: pd.DataFrame, game_id: int, my_picks: list[int],
                     w_att: float = 0.70, w_profit: float = 0.30,
                     hassle_discount: float = 0.85) -> float:
    """Compute personal value for a single game per the plan's My_Value function."""
    row = games.iloc[game_id]
    pref_norm = row["pref_norm"]
    resale_norm = row["resale_profit_norm"]
    available = row["available"]
    must_have = row.get("must_have", False)

    if must_have:
        pref_norm = 1.1

    dmu = compute_dmu_decay(games, game_id, my_picks)

    if available:
        return w_att * pref_norm * dmu + w_profit * resale_norm
    else:
        return resale_norm * hassle_discount


def compute_my_values_array(games: pd.DataFrame, my_picks: list[int],
                            w_att: float = 0.70, w_profit: float = 0.30,
                            hassle_discount: float = 0.85) -> np.ndarray:
    """Compute My_Value for all games. Returns (n_games,) array."""
    n = len(games)
    values = np.zeros(n, dtype=np.float64)
    for i in range(n):
        values[i] = compute_my_value(games, i, my_picks, w_att, w_profit, hassle_discount)
    return values


def compute_opponent_utilities(
    feature_matrix: np.ndarray,
    resale_norm: np.ndarray,
    v_att: np.ndarray,
    sigma: np.ndarray,
    dmu_factors: np.ndarray,
    jewish_holiday_mask: np.ndarray,
    is_observant: bool = False,
) -> np.ndarray:
    """Compute opponent utility for all games across all simulations.

    feature_matrix: (n_games, n_features)
    resale_norm: (n_games,)
    v_att: (n_games,) attendance values from Dirichlet model
    sigma: (n_sims,) sampled sigma values
    dmu_factors: (n_games,) decay factors
    jewish_holiday_mask: (n_games,) bool
    is_observant: whether opponent observes Jewish holidays

    Returns: (n_sims, n_games) utility matrix
    """
    n_sims = len(sigma)
    perceived_resale = sigma[:, np.newaxis] * resale_norm[np.newaxis, :]
    v_att_broadcast = v_att[np.newaxis, :]

    if is_observant:
        holiday_penalty = np.where(jewish_holiday_mask, 0.0, 1.0)
        v_att_broadcast = v_att_broadcast * holiday_penalty[np.newaxis, :]

    u = np.maximum(v_att_broadcast, perceived_resale)
    u = u * dmu_factors[np.newaxis, :]
    return u
