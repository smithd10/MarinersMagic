"""Mariners Snake Draft Optimizer — Streamlit App."""

import streamlit as st
import pandas as pd
import numpy as np
import os
import json

from engine.game_data import (
    load_games, build_feature_matrix, normalize_games, OPPONENT_ABBREVS,
    get_game_categories, compute_feature_prevalence,
)
from engine.personas import OpponentModel, PERSONAS, PERSONA_NAMES
from engine.value_functions import compute_my_values_array
from engine.mc_engine import recommend_picks, recommend_pairs, estimate_survival
from engine.snake_draft import generate_snake_order, get_my_pick_indices, is_pair_pick
from engine.state_manager import DraftState

st.set_page_config(
    page_title="Draft Optimizer",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Night Cockpit CSS
# ---------------------------------------------------------------------------
NIGHT_CSS = """
<style>
:root {
    --bg: #0D1117; --surface: #161B22; --surface-hover: #21262D;
    --border: #30363D; --accent: #58A6FF; --green: #3FB950;
    --amber: #D29922; --red: #F85149; --text1: #E6EDF3;
    --text2: #8B949E; --text3: #484F58;
}
#MainMenu, footer, .stDeployButton, header {visibility: hidden; display: none;}
.block-container {padding-top: 0.5rem !important; padding-bottom: 6rem !important;}
section[data-testid="stSidebar"] {display:none;}

/* Game row buttons */
div[data-testid="stHorizontalBlock"] button[kind="secondary"],
.game-btn > button {
    min-height: 52px !important; width: 100% !important;
    text-align: left !important; font-size: 15px !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    border-radius: 0 !important; border: none !important;
    border-bottom: 1px solid var(--border) !important;
    padding: 8px 16px !important; white-space: nowrap !important;
}
div[data-testid="stHorizontalBlock"] button[kind="secondary"]:active {
    background-color: var(--surface-hover) !important;
}
/* Recommendation cards */
div[data-testid="stContainer"] {
    border-left: 3px solid var(--border) !important;
    background: var(--surface) !important;
    border-radius: 4px !important; margin-bottom: 8px !important;
    padding: 12px !important;
}
/* Tabs bottom bar on mobile */
@media (max-width: 768px) {
    .stTabs [data-baseweb="tab-list"] {
        position: fixed; bottom: 0; left: 0; right: 0;
        z-index: 999; background: var(--bg);
        border-top: 1px solid var(--border);
        padding-bottom: env(safe-area-inset-bottom, 34px);
        justify-content: center;
    }
    .stTabs [data-baseweb="tab-list"] button {
        min-height: 48px !important; font-size: 14px !important;
        flex: 1 !important;
    }
    .block-container { padding-bottom: 100px !important; }
}
/* Status banner */
.status-bar {
    background: var(--surface); padding: 8px 16px; border-radius: 4px;
    font-size: 13px; color: var(--text2); margin-bottom: 4px;
    display: flex; justify-content: space-between; align-items: center;
}
.alert-banner {
    background: var(--amber); color: #000; padding: 8px 16px;
    border-radius: 4px; font-weight: 600; text-align: center;
    margin-bottom: 8px; animation: pulse 1s ease-in-out infinite alternate;
}
.alert-now {
    background: var(--green); color: #000;
}
@keyframes pulse { from {opacity:0.8;} to {opacity:1;} }
/* Urgency dots */
.dot-red { color: var(--red); } .dot-amber { color: var(--amber); } .dot-green { color: var(--green); }
.mono { font-family: "SF Mono", "Cascadia Mono", "Consolas", monospace; font-size: 14px; }
/* Taken game row */
.taken-row { opacity: 0.4; }
/* Filter chips */
.chip {
    display: inline-block; padding: 6px 14px; margin: 2px 4px;
    border-radius: 16px; font-size: 13px; cursor: pointer;
    border: 1px solid var(--border); color: var(--text2);
}
.chip-active { background: var(--accent); color: #fff; border-color: var(--accent); }
/* Persona grid */
.persona-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 6px; }
.persona-btn {
    text-align: center; padding: 10px 4px; border-radius: 8px;
    border: 1px solid var(--border); background: var(--surface);
    font-size: 12px; cursor: pointer; min-height: 44px;
}
/* Rec rank */
.rank { font-size: 28px; font-weight: 700; color: var(--accent); }
</style>
"""
st.markdown(NIGHT_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "mariners_2026_pricing_analysis.csv")
FALLBACK_PATH = os.path.join(os.path.dirname(__file__), "mariners_home_games_2026.csv")


def init_session():
    if "state" not in st.session_state:
        st.session_state.state = DraftState()
    if "games" not in st.session_state:
        path = DATA_PATH if os.path.exists(DATA_PATH) else FALLBACK_PATH
        st.session_state.games = load_games(path)
    if "snake_order" not in st.session_state:
        st.session_state.snake_order = generate_snake_order(9, len(st.session_state.games))
    if "recs_cache" not in st.session_state:
        st.session_state.recs_cache = None
    if "filter_mode" not in st.session_state:
        st.session_state.filter_mode = "All"


init_session()
state: DraftState = st.session_state.state
games: pd.DataFrame = st.session_state.games
snake_order = st.session_state.snake_order

FAMILY_COLORS = [
    "#F97583", "#B392F0", "#79C0FF", "#56D364",
    "#D2A8FF", "#FFA657", "#FF7B72", "#7EE787",
]


def get_family_color(slot: int) -> str:
    if slot == state.my_slot:
        return "#58A6FF"
    idx = slot - 1
    if slot > state.my_slot:
        idx -= 1
    return FAMILY_COLORS[idx % len(FAMILY_COLORS)]


def apply_overrides():
    """Apply preference/availability overrides to games dataframe."""
    for gid, pref in state.preference_overrides.items():
        if 0 <= gid < len(games):
            games.at[gid, "preference"] = pref
    for gid, avail in state.availability_overrides.items():
        if 0 <= gid < len(games):
            games.at[gid, "available"] = avail
    for gid in state.must_have_ids:
        if 0 <= gid < len(games):
            games.at[gid, "must_have"] = True
    normalize_games(games)
    st.session_state.games = games


apply_overrides()


def invalidate_recs():
    st.session_state.recs_cache = None


# ---------------------------------------------------------------------------
# Mode router
# ---------------------------------------------------------------------------
if state.mode == "live":
    render_mode = "live"
else:
    render_mode = "predraft"


# ============================= PRE-DRAFT MODE =============================
def render_predraft():
    st.markdown("## Pre-Draft Setup")

    data_tab, strategy_tab, scenario_tab = st.tabs(["Data", "Strategy", "Scenarios"])

    with data_tab:
        st.markdown("### Game Schedule")
        uploaded = st.file_uploader("Upload game CSV (optional — default data loaded)", type=["csv"])
        if uploaded:
            st.session_state.games = load_games(uploaded)
            games = st.session_state.games
            st.session_state.snake_order = generate_snake_order(9, len(games))
            st.success(f"Loaded {len(games)} games")

        display_df = games[["date_str", "day_of_week", "opponent_abbr", "time",
                            "preference", "resale_value", "face_value", "resale_profit",
                            "pricing_tier", "available", "jewish_holiday"]].copy()
        display_df.columns = ["Date", "Day", "Opp", "Time", "Pref", "Resale$",
                              "Face$", "Profit$", "Tier", "Avail", "Holiday"]
        st.dataframe(display_df, use_container_width=True, height=400)

        st.markdown("### Override Preferences")
        st.caption("Select a game to override its preference score (1-10) or availability.")
        col1, col2, col3 = st.columns(3)
        with col1:
            game_options = [f"{games.iloc[i]['date_str']} {games.iloc[i]['opponent_abbr']}"
                            for i in range(len(games))]
            selected = st.selectbox("Game", game_options, key="override_game")
            sel_idx = game_options.index(selected) if selected else 0
        with col2:
            new_pref = st.number_input("Preference (1-10)", 1, 10,
                                       int(games.iloc[sel_idx]["preference"]),
                                       key="override_pref")
        with col3:
            new_avail = st.checkbox("Available", games.iloc[sel_idx]["available"],
                                    key="override_avail")

        if st.button("Apply Override", key="apply_override"):
            state.preference_overrides[sel_idx] = new_pref
            state.availability_overrides[sel_idx] = new_avail
            apply_overrides()
            invalidate_recs()
            st.success(f"Updated {games.iloc[sel_idx]['date_str']} {games.iloc[sel_idx]['opponent_abbr']}")

    with strategy_tab:
        st.markdown("### Your Strategy")
        col1, col2 = st.columns(2)
        with col1:
            state.w_att = st.slider("Attendance weight", 0.0, 1.0, state.w_att, 0.05,
                                    key="w_att_slider")
            state.w_profit = round(1.0 - state.w_att, 2)
            st.caption(f"Profit weight: {state.w_profit:.2f}")
            state.hassle_discount_on = st.toggle("Hassle discount (penalize sell-only games)",
                                                  state.hassle_discount_on, key="hassle_toggle")
        with col2:
            st.markdown("#### Must-Have Games")
            must_options = [f"{games.iloc[i]['date_str']} {games.iloc[i]['opponent_abbr']}"
                            for i in range(len(games))]
            must_selected = st.multiselect("Pin must-have games (max 3)", must_options,
                                           default=[must_options[i] for i in state.must_have_ids
                                                    if i < len(must_options)],
                                           max_selections=3, key="must_have_select")
            state.must_have_ids = [must_options.index(s) for s in must_selected]
            apply_overrides()

        st.markdown("### Opponent Personas")
        state.init_opponents()
        cols = st.columns(4)
        for i, (slot, model) in enumerate(sorted(state.opponent_models.items())):
            with cols[i % 4]:
                new_persona = st.selectbox(
                    model.name, PERSONA_NAMES,
                    index=PERSONA_NAMES.index(model.persona_name),
                    key=f"persona_{slot}",
                )
                if new_persona != model.persona_name:
                    model.set_persona(new_persona)
                    invalidate_recs()
                is_obs = st.checkbox("Observant", model.is_observant, key=f"obs_{slot}")
                model.is_observant = is_obs

    with scenario_tab:
        st.markdown("### Scenario Planning")
        sim_slot = st.selectbox("Simulate draft slot", list(range(1, 10)), key="sim_slot")
        if st.button("Run Scenario Simulation", key="run_scenario"):
            state.my_slot = sim_slot
            state.init_opponents()
            snake = generate_snake_order(9, len(games))
            feature_matrix = build_feature_matrix(games)

            with st.spinner("Running Monte Carlo simulation..."):
                recs = recommend_picks(
                    games, snake, 0, sim_slot, [], state.opponent_models,
                    set(), state.w_att, state.w_profit, state.hassle_discount,
                    n_sims=300, top_k=10,
                )

            st.markdown(f"#### Top picks for Slot {sim_slot}")
            for i, r in enumerate(recs):
                surv_class = "dot-red" if r["survival_pct"] < 30 else (
                    "dot-amber" if r["survival_pct"] < 65 else "dot-green")
                st.markdown(f"""
                **#{i+1}** {r['day']} {r['date']} {r['opponent']} {r['promo_icon']}
                — Pref {r['preference']} · +${r['resale_profit']:.0f} ·
                <span class="{surv_class}">{r['survival_pct']:.0f}%</span> survival
                — *{r['reason']}*
                """, unsafe_allow_html=True)

    st.divider()
    if st.button("Switch to Live Draft Mode", type="primary", key="go_live"):
        state.mode = "live"
        state.init_opponents()
        invalidate_recs()
        st.rerun()


# ============================= LIVE DRAFT MODE =============================
def render_live():
    current_picker = state.get_current_picker(snake_order)
    current_round = state.get_current_round(snake_order)
    picks_to_me = state.picks_until_my_turn(snake_order)
    is_my_turn = current_picker == state.my_slot
    total_picks = len(snake_order)
    is_pair = is_my_turn and is_pair_pick(snake_order, state.current_pick_idx, state.my_slot)

    # Status bar
    picker_name = f"Family {current_picker}" if current_picker != state.my_slot else "YOU"
    st.markdown(f"""<div class="status-bar">
        <span>R{current_round} · Pick {state.current_pick_idx + 1}/{total_picks} · {picker_name}</span>
        <span>{len(state.my_picks)} picks</span>
    </div>""", unsafe_allow_html=True)

    # Alert banner
    if is_my_turn:
        st.markdown('<div class="alert-banner alert-now">YOUR PICK</div>',
                    unsafe_allow_html=True)
    elif picks_to_me <= 3 and picks_to_me > 0:
        st.markdown(f'<div class="alert-banner">YOUR TURN IN {picks_to_me}</div>',
                    unsafe_allow_html=True)

    board_tab, recs_tab, me_tab = st.tabs(["BOARD", "RECS", "ME"])

    # ---- BOARD TAB ----
    with board_tab:
        render_board(current_picker, is_my_turn)

    # ---- RECS TAB ----
    with recs_tab:
        render_recs(is_my_turn, is_pair)

    # ---- ME TAB ----
    with me_tab:
        render_me()


def render_board(current_picker, is_my_turn):
    filter_mode = st.session_state.filter_mode
    col_filters = st.columns(4)
    filter_labels = ["All", "Avail", "Wknd", "Premium"]
    for i, label in enumerate(filter_labels):
        with col_filters[i]:
            if st.button(label, key=f"filter_{label}",
                         type="primary" if filter_mode == label else "secondary",
                         use_container_width=True):
                st.session_state.filter_mode = label
                st.rerun()

    taken_set = state.get_all_taken()
    feature_matrix = build_feature_matrix(games)

    for gid in range(len(games)):
        row = games.iloc[gid]
        is_taken = gid in taken_set
        is_mine = gid in state.my_picks
        is_available_game = not is_taken

        # Filtering
        if filter_mode == "Avail" and is_taken:
            continue
        if filter_mode == "Wknd" and not row["is_weekend"]:
            continue
        if filter_mode == "Premium" and row["opponent_tier"] != "premium":
            continue

        date_str = row["date"].strftime("%b %-d") if hasattr(row["date"], "strftime") else str(row["date_str"])[5:]
        day_str = row["day_of_week"][:3]
        opp_str = row["opponent_abbr"]
        promo = row.get("promo_icon", "")
        profit_str = f"+${row['resale_profit']:.0f}" if row["resale_profit"] >= 0 else f"-${abs(row['resale_profit']):.0f}"

        if is_mine:
            dot = "🔵"
            label = f"{dot} {date_str} {day_str}  {opp_str} {promo}  {profit_str}"
            st.markdown(f"<div style='padding:8px 16px;min-height:52px;display:flex;align-items:center;"
                        f"border-bottom:1px solid #30363D;color:#58A6FF;font-size:15px;'>"
                        f"{label}</div>", unsafe_allow_html=True)
        elif is_taken:
            taker = state.taken_games[gid]
            color = get_family_color(taker)
            label = f"⬤ {date_str} {day_str}  {opp_str} {promo}  F{taker}"
            st.markdown(f"<div class='taken-row' style='padding:8px 16px;min-height:52px;"
                        f"display:flex;align-items:center;border-bottom:1px solid #30363D;"
                        f"font-size:15px;color:{color};'>{label}</div>",
                        unsafe_allow_html=True)
        else:
            label = f"○ {date_str} {day_str}  {opp_str} {promo}  {profit_str}"
            if st.button(label, key=f"pick_{gid}", use_container_width=True):
                handle_pick(gid, current_picker, is_my_turn, feature_matrix)
                st.rerun()


def handle_pick(gid, current_picker, is_my_turn, feature_matrix):
    game_features = feature_matrix[gid]
    available_mask = np.array([
        i not in state.get_all_taken() for i in range(len(games))
    ])
    prevalence = compute_feature_prevalence(feature_matrix, available_mask)
    cats = get_game_categories(games, gid)

    if is_my_turn:
        state.record_my_pick(gid)
    else:
        state.record_opponent_pick(gid, current_picker, game_features, prevalence, cats)

    invalidate_recs()


def render_recs(is_my_turn, is_pair):
    if is_my_turn:
        st.markdown(f"**YOUR PICK** · Round {state.get_current_round(snake_order)}")
    else:
        picks_to = state.picks_until_my_turn(snake_order)
        st.markdown(f"Your next pick in **{picks_to}**")

    if is_pair:
        st.markdown("**PAIR PICK — choose two**")

    recs = st.session_state.recs_cache
    if recs is None:
        with st.spinner("Computing recommendations..."):
            taken = state.get_all_taken()
            if is_pair:
                recs = recommend_pairs(
                    games, snake_order, state.current_pick_idx, state.my_slot,
                    state.my_picks, state.opponent_models, taken,
                    state.w_att, state.w_profit, state.hassle_discount,
                    n_sims=200, top_k=5,
                )
            else:
                recs = recommend_picks(
                    games, snake_order, state.current_pick_idx, state.my_slot,
                    state.my_picks, state.opponent_models, taken,
                    state.w_att, state.w_profit, state.hassle_discount,
                    n_sims=300, top_k=8,
                )
            st.session_state.recs_cache = recs

    if not recs:
        st.info("No recommendations available.")
        return

    if is_pair and recs and "game_a" in recs[0]:
        render_pair_recs(recs)
    else:
        render_single_recs(recs)


def render_single_recs(recs):
    for i, r in enumerate(recs[:3]):
        surv = r["survival_pct"]
        if surv < 30:
            border_color = "#F85149"
            dot_html = '<span class="dot-red">⬤</span>'
        elif surv < 65:
            border_color = "#D29922"
            dot_html = '<span class="dot-amber">⬤</span>'
        else:
            border_color = "#3FB950"
            dot_html = '<span class="dot-green">⬤</span>'

        with st.container(border=True):
            st.markdown(f"""
            <div style="border-left:3px solid {border_color};padding-left:12px;">
            <span class="rank">#{i+1}</span>
            <strong>{r['day']} {r['date']} {r['opponent']} {r['promo_icon']}</strong><br>
            <span class="mono">Pref {r['preference']}  ·  +${r['resale_profit']:.0f}  ·  E {r['evona']:.1f}</span><br>
            {dot_html} <span class="mono">{surv:.0f}% survival</span><br>
            <em>{r['reason']}</em>
            </div>
            """, unsafe_allow_html=True)

    if len(recs) > 3:
        with st.expander("Show picks #4-8"):
            for i, r in enumerate(recs[3:], start=4):
                st.markdown(f"**#{i}** {r['day']} {r['date']} {r['opponent']} — "
                            f"Pref {r['preference']} · +${r['resale_profit']:.0f} · "
                            f"{r['survival_pct']:.0f}% — *{r['reason']}*")


def render_pair_recs(recs):
    for i, r in enumerate(recs[:3]):
        with st.container(border=True):
            st.markdown(f"""
            <span class="rank">PAIR #{i+1}</span><br>
            <strong>{r['day_a']} {r['date_a']} {r['opponent_a']}</strong><br>
            + <strong>{r['day_b']} {r['date_b']} {r['opponent_b']}</strong><br>
            <span class="mono">Combined: Pref {r['pref_combined']}  ·  +${r['profit_combined']:.0f}</span>
            """, unsafe_allow_html=True)


def render_me():
    st.markdown(f"### My Picks ({len(state.my_picks)} of ~{len(snake_order) // 9})")

    if state.my_picks:
        cats = {"weekend": 0, "weekday": 0, "premium": 0, "mid": 0, "low": 0}
        for gid in state.my_picks:
            row = games.iloc[gid]
            if row["is_weekend"]:
                cats["weekend"] += 1
            else:
                cats["weekday"] += 1
            cats[row["opponent_tier"]] += 1
        st.markdown(f"{cats['weekend']} wknd · {cats['weekday']} wkday · "
                    f"{cats['premium']} premium · {cats['mid']} mid · {cats['low']} low")

        for gid in state.my_picks:
            row = games.iloc[gid]
            date_str = row["date"].strftime("%b %-d") if hasattr(row["date"], "strftime") else row["date_str"]
            st.markdown(f"🔵 {date_str} {row['day_of_week'][:3]} {row['opponent_abbr']}  "
                        f"Pref {row['preference']}")
    else:
        st.caption("No picks yet.")

    st.markdown("---")
    st.markdown("### Opponents")
    state.init_opponents()
    cols = st.columns(4)
    for i, (slot, model) in enumerate(sorted(state.opponent_models.items())):
        with cols[i % 4]:
            current_idx = PERSONA_NAMES.index(model.persona_name)
            color = get_family_color(slot)
            new_p = st.selectbox(
                f"F{slot}", PERSONA_NAMES, index=current_idx,
                key=f"live_persona_{slot}",
            )
            if new_p != model.persona_name:
                model.set_persona(new_p)
                invalidate_recs()
                st.rerun()
            st.caption(f"{len(model.picks)} picks · {PERSONAS[model.persona_name]['icon']}")

    st.markdown("---")

    # Restore code
    code = state.to_restore_code()
    st.text_input("Restore code", code, key="restore_display", disabled=True,
                  help="Copy this code to restore your draft state if the app crashes")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("Download State", state.to_json(),
                           "draft_state.json", "application/json", key="dl_state")
    with col2:
        uploaded_state = st.file_uploader("Load State", type=["json"], key="upload_state")
        if uploaded_state:
            try:
                loaded = DraftState.from_json(uploaded_state.read().decode())
                st.session_state.state = loaded
                invalidate_recs()
                st.success("State restored!")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to load state: {e}")

    st.markdown("---")
    paste_code = st.text_input("Paste restore code", key="paste_restore",
                               placeholder="Paste code here to restore...")
    if paste_code:
        try:
            restored = DraftState.from_restore_code(paste_code)
            st.session_state.state = restored
            invalidate_recs()
            st.success("Restored from code!")
            st.rerun()
        except Exception as e:
            st.error(f"Invalid restore code: {e}")

    st.markdown("---")
    if st.button("Undo Last Action", key="undo_btn", use_container_width=True):
        if state.undo():
            invalidate_recs()
            st.toast("Action undone")
            st.rerun()
        else:
            st.warning("Nothing to undo")

    st.markdown("---")
    if st.button("Back to Pre-Draft Mode", key="back_predraft"):
        state.mode = "predraft"
        st.rerun()


# ============================= DRAFT SLOT SETUP =============================
def render_slot_setup():
    st.markdown("## Draft Night")
    st.markdown("Enter your draft slot to begin.")

    slot = st.selectbox("Your draft slot", list(range(1, 10)), key="slot_select")
    state.my_slot = slot

    names = []
    st.markdown("#### Family Names (optional)")
    cols = st.columns(3)
    for i in range(9):
        with cols[i % 3]:
            name = st.text_input(f"Slot {i+1}", f"Family {i+1}", key=f"fname_{i}")
            names.append(name)

    if st.button("Start Draft", type="primary", key="start_draft"):
        state.my_slot = slot
        state.init_opponents(names)
        state.mode = "live"
        st.session_state.snake_order = generate_snake_order(9, len(games))
        invalidate_recs()
        st.rerun()


# ============================= MAIN ROUTER =============================
if render_mode == "predraft":
    render_predraft()
elif render_mode == "live":
    if not state.opponent_models:
        render_slot_setup()
    else:
        render_live()
