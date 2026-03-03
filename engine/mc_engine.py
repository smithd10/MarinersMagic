"""Monte Carlo simulation engine for draft optimization."""

import numpy as np
import pandas as pd
from engine.personas import OpponentModel, PERSONAS
from engine.value_functions import (
    compute_my_values_array, compute_opponent_utilities, get_game_categories, DMU_BASE,
)
from engine.game_data import build_feature_matrix
from engine.snake_draft import is_pair_pick


def temperature_for_round(round_num: int) -> float:
    return 1.0 * (0.85 ** (round_num - 1))


def softmax_sample_vectorized(utilities: np.ndarray, available_mask: np.ndarray,
                              temperature: float) -> np.ndarray:
    """Sample picks from softmax distribution across simulations.

    utilities: (n_sims, n_games)
    available_mask: (n_sims, n_games) bool
    temperature: scalar

    Returns: (n_sims,) indices of picked games
    """
    masked = np.where(available_mask, utilities, -1e9)
    scaled = masked / max(temperature, 0.01)
    scaled -= scaled.max(axis=1, keepdims=True)
    exp_u = np.exp(scaled)
    exp_u = np.where(available_mask, exp_u, 0.0)
    row_sums = exp_u.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-12)
    probs = exp_u / row_sums
    cum = np.cumsum(probs, axis=1)
    r = np.random.random((utilities.shape[0], 1))
    picks = np.argmax(cum >= r, axis=1)
    return picks


def run_mc_simulation(
    games: pd.DataFrame,
    feature_matrix: np.ndarray,
    snake_order: list[tuple[int, int]],
    current_pick_idx: int,
    my_slot: int,
    my_picks: list[int],
    opponent_models: dict[int, OpponentModel],
    taken_games: set[int],
    candidate_game: int,
    w_att: float = 0.70,
    w_profit: float = 0.30,
    hassle_discount: float = 0.85,
    n_sims: int = 500,
    max_future_picks: int = 30,
) -> float:
    """Run MC simulations for a single candidate pick. Returns expected portfolio value."""
    n_games = len(games)
    resale_norm = games["resale_profit_norm"].values.astype(np.float64)
    jewish_mask = games["is_jewish_holiday"].values

    available_init = np.ones((n_sims, n_games), dtype=bool)
    for gid in taken_games:
        available_init[:, gid] = False
    available_init[:, candidate_game] = False

    my_picked = np.zeros((n_sims, n_games), dtype=bool)
    for gid in my_picks:
        my_picked[:, gid] = True
    my_picked[:, candidate_game] = True

    sampled_sigmas = {}
    for slot, model in opponent_models.items():
        mu = model.sigma
        std = model.sigma_std
        s = np.clip(np.random.normal(mu, std, size=n_sims), 0.0, 1.0).astype(np.float64)
        sampled_sigmas[slot] = s

    opp_v_att_cache = {}
    opp_dmu_base_cache = {}
    for slot, model in opponent_models.items():
        opp_v_att_cache[slot] = model.compute_v_att(feature_matrix).astype(np.float64)
        dmu = np.ones(n_games, dtype=np.float64)
        for gid in range(n_games):
            cats = get_game_categories(games, gid)
            dmu[gid] = model.get_dmu_decay(cats)
        opp_dmu_base_cache[slot] = dmu

    remaining_order = snake_order[current_pick_idx + 1:]
    if max_future_picks:
        remaining_order = remaining_order[:max_future_picks]

    sim_opp_dmu_counts = {slot: dict(m.dmu_counts) for slot, m in opponent_models.items()}

    for pick_slot, round_num in remaining_order:
        if pick_slot == my_slot:
            my_values = compute_my_values_array(games, my_picks, w_att, w_profit, hassle_discount)
            vals = np.where(available_init, my_values[np.newaxis, :], -1e9)
            best = np.argmax(vals, axis=1)
            idx = np.arange(n_sims)
            available_init[idx, best] = False
            my_picked[idx, best] = True
        else:
            if pick_slot not in opponent_models:
                continue
            model = opponent_models[pick_slot]
            v_att = opp_v_att_cache[pick_slot]
            dmu = opp_dmu_base_cache[pick_slot]
            sigma = sampled_sigmas.get(pick_slot, np.full(n_sims, 0.5))

            u = compute_opponent_utilities(
                feature_matrix, resale_norm, v_att, sigma, dmu,
                jewish_mask, model.is_observant,
            )

            T = temperature_for_round(round_num) * model.temperature_mult
            picks = softmax_sample_vectorized(u, available_init, T)
            idx = np.arange(n_sims)
            available_init[idx, picks] = False

    candidate_value = compute_my_values_array(
        games, my_picks, w_att, w_profit, hassle_discount
    )[candidate_game]

    future_values = np.zeros(n_sims, dtype=np.float64)
    my_values_final = compute_my_values_array(games, my_picks, w_att, w_profit, hassle_discount)
    picked_vals = np.where(my_picked, my_values_final[np.newaxis, :], 0.0)
    future_values = picked_vals.sum(axis=1)

    return float(future_values.mean())


def recommend_picks(
    games: pd.DataFrame,
    snake_order: list[tuple[int, int]],
    current_pick_idx: int,
    my_slot: int,
    my_picks: list[int],
    opponent_models: dict[int, OpponentModel],
    taken_games: set[int],
    w_att: float = 0.70,
    w_profit: float = 0.30,
    hassle_discount: float = 0.85,
    n_sims: int = 500,
    top_k: int = 8,
) -> list[dict]:
    """Get ranked recommendations for the current pick.

    Returns list of dicts with game info, expected value, survival probability, EVONA.
    """
    feature_matrix = build_feature_matrix(games)
    available_ids = [
        i for i in range(len(games))
        if i not in taken_games and i not in my_picks
    ]

    if not available_ids:
        return []

    my_values = compute_my_values_array(games, my_picks, w_att, w_profit, hassle_discount)
    sorted_by_raw = sorted(available_ids, key=lambda i: -my_values[i])
    candidates = sorted_by_raw[:min(20, len(sorted_by_raw))]

    candidate_ev = {}
    for gid in candidates:
        ev = run_mc_simulation(
            games, feature_matrix, snake_order, current_pick_idx,
            my_slot, my_picks, opponent_models, taken_games, gid,
            w_att, w_profit, hassle_discount, n_sims=n_sims,
        )
        candidate_ev[gid] = ev

    survival_probs = estimate_survival(
        games, feature_matrix, snake_order, current_pick_idx,
        my_slot, opponent_models, taken_games, available_ids, n_sims=min(n_sims, 300),
    )

    if candidate_ev:
        max_ev = max(candidate_ev.values())
        avg_ev = np.mean(list(candidate_ev.values()))
    else:
        max_ev = avg_ev = 0

    results = []
    for gid in candidates:
        ev = candidate_ev.get(gid, 0)
        surv = survival_probs.get(gid, 0.5)
        evona = my_values[gid] * (1 - surv)
        reason = generate_reason(games, gid, surv, my_picks, snake_order,
                                 current_pick_idx, my_slot, opponent_models)
        row = games.iloc[gid]
        results.append({
            "game_id": gid,
            "date": row["date_str"],
            "day": row["day_of_week"][:3],
            "opponent": row["opponent_abbr"],
            "promo_icon": row.get("promo_icon", ""),
            "preference": int(row["preference"]),
            "resale_profit": float(row["resale_profit"]),
            "survival_pct": float(surv * 100),
            "evona": float(evona),
            "ev": float(ev),
            "reason": reason,
            "must_have": bool(row.get("must_have", False)),
        })

    results.sort(key=lambda x: (-x["must_have"], -x["ev"]))
    return results[:top_k]


def recommend_pairs(
    games: pd.DataFrame,
    snake_order: list[tuple[int, int]],
    current_pick_idx: int,
    my_slot: int,
    my_picks: list[int],
    opponent_models: dict[int, OpponentModel],
    taken_games: set[int],
    w_att: float = 0.70,
    w_profit: float = 0.30,
    hassle_discount: float = 0.85,
    n_sims: int = 300,
    top_k: int = 5,
) -> list[dict]:
    """Recommend pairs for consecutive snake turn picks."""
    feature_matrix = build_feature_matrix(games)
    available_ids = [
        i for i in range(len(games))
        if i not in taken_games and i not in my_picks
    ]
    my_values = compute_my_values_array(games, my_picks, w_att, w_profit, hassle_discount)
    sorted_by_raw = sorted(available_ids, key=lambda i: -my_values[i])
    top_candidates = sorted_by_raw[:min(15, len(sorted_by_raw))]

    pairs = []
    for i, a in enumerate(top_candidates):
        for b in top_candidates[i + 1:]:
            combined_raw = my_values[a] + my_values[b]
            pairs.append((a, b, combined_raw))

    pairs.sort(key=lambda x: -x[2])
    pairs = pairs[:min(20, len(pairs))]

    pair_results = []
    for a, b, raw in pairs:
        row_a = games.iloc[a]
        row_b = games.iloc[b]
        pair_results.append({
            "game_a": a,
            "game_b": b,
            "date_a": row_a["date_str"],
            "day_a": row_a["day_of_week"][:3],
            "opponent_a": row_a["opponent_abbr"],
            "date_b": row_b["date_str"],
            "day_b": row_b["day_of_week"][:3],
            "opponent_b": row_b["opponent_abbr"],
            "pref_combined": int(row_a["preference"]) + int(row_b["preference"]),
            "profit_combined": float(row_a["resale_profit"]) + float(row_b["resale_profit"]),
            "combined_value": float(raw),
        })

    pair_results.sort(key=lambda x: -x["combined_value"])
    return pair_results[:top_k]


def estimate_survival(
    games: pd.DataFrame,
    feature_matrix: np.ndarray,
    snake_order: list[tuple[int, int]],
    current_pick_idx: int,
    my_slot: int,
    opponent_models: dict[int, OpponentModel],
    taken_games: set[int],
    game_ids: list[int],
    n_sims: int = 300,
) -> dict[int, float]:
    """Estimate survival probability for each game to the next turn."""
    n_games = len(games)
    resale_norm = games["resale_profit_norm"].values.astype(np.float64)
    jewish_mask = games["is_jewish_holiday"].values

    available = np.ones((n_sims, n_games), dtype=bool)
    for gid in taken_games:
        available[:, gid] = False

    sampled_sigmas = {}
    for slot, model in opponent_models.items():
        s = np.clip(np.random.normal(model.sigma, model.sigma_std, size=n_sims), 0.0, 1.0)
        sampled_sigmas[slot] = s.astype(np.float64)

    opp_v_att = {}
    opp_dmu = {}
    for slot, model in opponent_models.items():
        opp_v_att[slot] = model.compute_v_att(feature_matrix).astype(np.float64)
        dmu = np.ones(n_games, dtype=np.float64)
        for gid in range(n_games):
            cats = get_game_categories(games, gid)
            dmu[gid] = model.get_dmu_decay(cats)
        opp_dmu[slot] = dmu

    for pick_slot, round_num in snake_order[current_pick_idx + 1:]:
        if pick_slot == my_slot:
            break
        if pick_slot not in opponent_models:
            continue

        model = opponent_models[pick_slot]
        v_att = opp_v_att[pick_slot]
        dmu = opp_dmu[pick_slot]
        sigma = sampled_sigmas.get(pick_slot, np.full(n_sims, 0.5))

        u = compute_opponent_utilities(
            feature_matrix, resale_norm, v_att, sigma, dmu,
            jewish_mask, model.is_observant,
        )
        T = temperature_for_round(round_num) * model.temperature_mult
        picks = softmax_sample_vectorized(u, available, T)
        idx = np.arange(n_sims)
        available[idx, picks] = False

    survival = {}
    for gid in game_ids:
        survival[gid] = float(available[:, gid].mean())
    return survival


def generate_reason(games, gid, survival, my_picks, snake_order,
                    current_pick_idx, my_slot, opponent_models) -> str:
    """Generate a one-line human-readable reason for the recommendation."""
    row = games.iloc[gid]

    opponents_before = []
    for pick_slot, rnd in snake_order[current_pick_idx + 1:]:
        if pick_slot == my_slot:
            break
        if pick_slot in opponent_models:
            opponents_before.append(opponent_models[pick_slot].persona_name)

    n_picks = len(opponents_before)
    if n_picks == 0:
        return "Next pick is yours"

    calc_count = opponents_before.count("Calculator")
    fan_count = opponents_before.count("Fan")

    if survival < 0.25:
        if calc_count >= 2:
            return f"{calc_count} Calculators before you — high risk"
        return f"Only {survival*100:.0f}% survival across {n_picks} picks"

    if survival > 0.75:
        if fan_count == n_picks:
            return "Safe — only Fans between you, they won't see it"
        return f"High survival ({survival*100:.0f}%) — safe to defer"

    if row.get("must_have", False):
        return "Must-have game — grab if at risk"

    if row["resale_profit"] > 50:
        return f"High resale (+${row['resale_profit']:.0f}), moderate risk"

    cat_counts = {}
    for pid in my_picks:
        prow = games.iloc[pid]
        dt = "weekend" if prow["is_weekend"] else "weekday"
        cat_counts[dt] = cat_counts.get(dt, 0) + 1

    day_type = "weekend" if row["is_weekend"] else "weekday"
    if cat_counts.get(day_type, 0) >= 3:
        return f"Your {cat_counts[day_type]+1}th {day_type} — diminishing value"

    return f"{n_picks} picks before you — {survival*100:.0f}% survival"
