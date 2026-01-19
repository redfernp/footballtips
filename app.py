import math
from datetime import datetime
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st


# ----------------------------
# MUST BE FIRST Streamlit call
# ----------------------------
st.set_page_config(page_title="Correct Score Predictor (Entertainment Mode)", layout="wide")

st.title("Correct Score Predictor (Entertainment Mode)")
st.write("More lively correct scores using xG + goals/points + full odds logic (for entertainment, still grounded).")

# ✅ Put uploader right here so it ALWAYS renders
uploaded = st.file_uploader("Upload FootyStats CSV", type=["csv"])

if uploaded is None:
    st.info("Upload your CSV to generate predictions.")
    st.stop()


# ----------------------------
# Utilities
# ----------------------------
def safe_float(x) -> Optional[float]:
    try:
        if pd.isna(x):
            return None
        v = float(x)
        if not np.isfinite(v):
            return None
        return v
    except Exception:
        return None


def parse_date_time(date_val, time_val) -> Tuple[str, str]:
    date_out = ""
    if isinstance(date_val, (datetime, pd.Timestamp)):
        date_out = date_val.strftime("%b %d %Y")
    else:
        s = str(date_val).strip()
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%b %d %Y", "%d %b %Y"):
            try:
                dt = datetime.strptime(s, fmt)
                date_out = dt.strftime("%b %d %Y")
                break
            except Exception:
                pass
        if not date_out:
            date_out = s

    time_out = ""
    if isinstance(time_val, (datetime, pd.Timestamp)):
        time_out = time_val.strftime("%-I:%M %p") if hasattr(time_val, "strftime") else str(time_val)
    else:
        s = str(time_val).strip()
        for fmt in ("%H:%M", "%H:%M:%S", "%I:%M %p", "%I:%M%p"):
            try:
                t = datetime.strptime(s, fmt)
                time_out = t.strftime("%-I:%M %p")
                break
            except Exception:
                pass
        if not time_out:
            time_out = s

    return date_out, time_out


def find_column(df: pd.DataFrame, candidates) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def parse_footystats_date_gmt(value) -> Tuple[str, str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "", ""

    s = str(value).strip()
    s = s.replace(" - ", " ").replace("-", " ")
    s = s.replace("am", " AM").replace("pm", " PM")
    s = s.replace("AM", " AM").replace("PM", " PM")
    s = " ".join(s.split())

    dt = pd.to_datetime(s, errors="coerce")
    if pd.isna(dt):
        return str(value).strip(), ""

    date_out = dt.strftime("%b %d %Y")
    time_out = dt.strftime("%-I:%M %p")
    return date_out, time_out


def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def odds_to_prob(odds: Optional[float]) -> Optional[float]:
    if odds is None or odds <= 1e-9:
        return None
    return 1.0 / odds


def overround_normalize(implied_probs: Dict[str, float]) -> Dict[str, float]:
    s = sum(implied_probs.values())
    if s <= 0:
        return implied_probs
    return {k: v / s for k, v in implied_probs.items()}


def total_goals_prob_over_25(lam_total: float) -> float:
    p0 = poisson_pmf(0, lam_total)
    p1 = poisson_pmf(1, lam_total)
    p2 = poisson_pmf(2, lam_total)
    return max(0.0, min(1.0, 1.0 - (p0 + p1 + p2)))


def solve_lam_total_from_over25(p_over25: float) -> float:
    p_over25 = max(0.001, min(0.999, p_over25))
    lo, hi = 0.05, 7.0
    for _ in range(60):
        mid = (lo + hi) / 2
        pmid = total_goals_prob_over_25(mid)
        if pmid < p_over25:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def build_score_matrix(lam_h: float, lam_a: float, max_goals: int) -> np.ndarray:
    p = np.zeros((max_goals + 1, max_goals + 1), dtype=float)
    for i in range(max_goals + 1):
        pi = poisson_pmf(i, lam_h)
        for j in range(max_goals + 1):
            p[i, j] = pi * poisson_pmf(j, lam_a)
    s = p.sum()
    if s > 0:
        p /= s
    return p


def pick_score_entertaining(p: np.ndarray, adventurousness: float, temperature: float, seed: int) -> Tuple[int, int]:
    rng = np.random.default_rng(seed)

    T = max(0.6, float(temperature))
    alpha = 1.0 / T
    w = np.power(np.clip(p, 1e-12, 1.0), alpha)

    k = 0.35 * float(adventurousness)
    if k > 0:
        max_i, max_j = w.shape[0] - 1, w.shape[1] - 1
        for i in range(max_i + 1):
            for j in range(max_j + 1):
                w[i, j] *= math.exp(k * (i + j))

    w_sum = w.sum()
    if w_sum <= 0:
        idx = np.unravel_index(np.argmax(p), p.shape)
        return int(idx[0]), int(idx[1])

    w = w / w_sum
    flat = w.ravel()
    choice = rng.choice(len(flat), p=flat)
    i, j = np.unravel_index(choice, w.shape)
    return int(i), int(j)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def compute_lambdas_from_features(
    r: pd.Series,
    col_xg_h: str,
    col_xg_a: str,
    col_h_gf: Optional[str],
    col_a_ga: Optional[str],
    col_a_gf: Optional[str],
    col_h_ga: Optional[str],
    col_h_pts: Optional[str],
    col_a_pts: Optional[str],
    over25_odds: Optional[float],
    under25_odds: Optional[float],
    home_odds: Optional[float],
    draw_odds: Optional[float],
    away_odds: Optional[float],
    btts_yes_odds: Optional[float],
    btts_no_odds: Optional[float],
    w_xg: float,
    w_goals: float,
    points_tilt: float,
    totals_weight: float,
    tilt_weight: float,
    btts_weight: float,
) -> Tuple[float, float]:

    xg_h = safe_float(r.get(col_xg_h)) or 0.0
    xg_a = safe_float(r.get(col_xg_a)) or 0.0
    lam_h = max(0.05, xg_h)
    lam_a = max(0.05, xg_a)

    parts_h = []
    if col_h_gf:
        v = safe_float(r.get(col_h_gf))
        if v is not None:
            parts_h.append(v)
    if col_a_ga:
        v = safe_float(r.get(col_a_ga))
        if v is not None:
            parts_h.append(v)

    parts_a = []
    if col_a_gf:
        v = safe_float(r.get(col_a_gf))
        if v is not None:
            parts_a.append(v)
    if col_h_ga:
        v = safe_float(r.get(col_h_ga))
        if v is not None:
            parts_a.append(v)

    if parts_h:
        lam_h = (w_xg * lam_h) + (w_goals * (sum(parts_h) / len(parts_h)))
    if parts_a:
        lam_a = (w_xg * lam_a) + (w_goals * (sum(parts_a) / len(parts_a)))

    if col_h_pts and col_a_pts:
        hp = safe_float(r.get(col_h_pts))
        ap = safe_float(r.get(col_a_pts))
        if hp is not None and ap is not None:
            diff = (hp - ap) / 30.0
            diff = clamp(diff, -1.0, 1.0) * points_tilt
            lam_h *= (1.0 + 0.12 * diff)
            lam_a *= (1.0 - 0.12 * diff)
            lam_h = max(0.02, lam_h)
            lam_a = max(0.02, lam_a)

    p_over = odds_to_prob(over25_odds)
    p_under = odds_to_prob(under25_odds)
    if p_over is not None and p_under is not None:
        probs = overround_normalize({"over": p_over, "under": p_under})
        p_over_norm = probs["over"]
        lam_total_x = max(0.05, lam_h + lam_a)
        lam_total_m = solve_lam_total_from_over25(p_over_norm)

        lam_total_target = (1 - totals_weight) * lam_total_x + totals_weight * lam_total_m
        scale = lam_total_target / lam_total_x
        lam_h *= scale
        lam_a *= scale

    pH = odds_to_prob(home_odds)
    pD = odds_to_prob(draw_odds)
    pA = odds_to_prob(away_odds)
    if pH is not None and pD is not None and pA is not None:
        probs = overround_normalize({"H": pH, "D": pD, "A": pA})
        pHn, pAn = probs["H"], probs["A"]
        raw = (pHn - pAn)
        delta = clamp(raw * 0.35, -0.18, 0.18) * tilt_weight

        lam_total = max(0.05, lam_h + lam_a)
        lam_h = max(0.02, lam_h * (1 + delta))
        lam_a = max(0.02, lam_a * (1 - delta))
        new_total = lam_h + lam_a
        if new_total > 0:
            lam_h *= lam_total / new_total
            lam_a *= lam_total / new_total

    by = odds_to_prob(btts_yes_odds)
    bn = odds_to_prob(btts_no_odds)
    if by is not None and bn is not None:
        probs = overround_normalize({"yes": by, "no": bn})
        p_yes = probs["yes"]
        mult = 1.0 + (clamp(p_yes, 0.35, 0.70) - 0.50) * 0.55
        mult = (1 - btts_weight) * 1.0 + btts_weight * mult
        lam_h *= mult
        lam_a *= mult

    lam_h = clamp(lam_h, 0.05, 4.5)
    lam_a = clamp(lam_a, 0.05, 4.5)
    return lam_h, lam_a


# ----------------------------
# Sidebar controls (after uploader)
# ----------------------------
with st.sidebar:
    st.header("Style controls")
    adventurousness = st.slider("Adventurousness (push to higher totals)", 0.0, 1.0, 0.55, 0.05)
    temperature = st.slider("Variety (temperature)", 0.7, 2.0, 1.25, 0.05)
    max_goals = st.slider("Max goals considered per team", 4, 10, 7, 1)

    st.divider()
    st.header("Feature blending")
    w_xg = st.slider("Weight on xG", 0.0, 1.0, 0.60, 0.05)
    w_goals = 1.0 - w_xg
    points_tilt = st.slider("Points tilt strength", 0.0, 1.0, 0.35, 0.05)

    st.divider()
    st.header("Market anchors")
    use_market = st.checkbox("Use market anchors", value=True)
    totals_weight = st.slider("Totals anchor (O/U 2.5) strength", 0.0, 1.0, 0.80, 0.05)
    tilt_weight = st.slider("1X2 tilt strength", 0.0, 1.0, 0.60, 0.05)
    btts_weight = st.slider("BTTS influence (if BTTS odds exist)", 0.0, 1.0, 0.65, 0.05)

# ----------------------------
# Load data (after uploader)
# ----------------------------
df = pd.read_csv(uploaded)

# Core columns
COL_COUNTRY = "Country"
COL_LEAGUE = "League"
COL_HOME = "Home Team"
COL_AWAY = "Away Team"
COL_XG_H = "Home Team Pre-Match xG"
COL_XG_A = "Away Team Pre-Match xG"

# Date/time candidates
DATE_CANDIDATES = ["Date", "Match Date", "Match_Date", "Date GMT", "Match Date GMT"]
TIME_CANDIDATES = ["Time", "Match Time", "Match_Time", "Time GMT", "Match Time GMT"]
DATETIME_CANDIDATES = ["date_GMT", "Date_GMT", "DateTime", "Datetime", "Date Time", "Match DateTime", "Match Datetime"]

col_date = find_column(df, DATE_CANDIDATES)
col_time = find_column(df, TIME_CANDIDATES)
col_dt = find_column(df, DATETIME_CANDIDATES)

# Optional odds columns
COL_O25 = "Odds_Over25"
COL_U25 = "Odds_Under25"
COL_H = "Odds_Home_Win"
COL_D = "Odds_Draw"
COL_A = "Odds_Away_Win"

# BTTS candidates
BTTS_YES_CANDS = ["Odds_BTTS_Yes", "Odds_BothTeamsToScore_Yes", "Odds_BTTSYes", "BTTS Yes Odds"]
BTTS_NO_CANDS = ["Odds_BTTS_No", "Odds_BothTeamsToScore_No", "Odds_BTTSNo", "BTTS No Odds"]
COL_BTTS_Y = find_column(df, BTTS_YES_CANDS)
COL_BTTS_N = find_column(df, BTTS_NO_CANDS)

# Goals + conceded + points candidates
HOME_GF_CANDS = ["Home Avg Goals Scored", "Home Team Avg Goals Scored", "home_goals_for_avg"]
AWAY_GA_CANDS = ["Away Avg Goals Conceded", "Away Team Avg Goals Conceded", "away_goals_against_avg"]
AWAY_GF_CANDS = ["Away Avg Goals Scored", "Away Team Avg Goals Scored", "away_goals_for_avg"]
HOME_GA_CANDS = ["Home Avg Goals Conceded", "Home Team Avg Goals Conceded", "home_goals_against_avg"]
HOME_PTS_CANDS = ["Home Points", "Home Team Points", "home_points", "Home Team Pre-Match Points"]
AWAY_PTS_CANDS = ["Away Points", "Away Team Points", "away_points", "Away Team Pre-Match Points"]

COL_H_GF = find_column(df, HOME_GF_CANDS)
COL_A_GA = find_column(df, AWAY_GA_CANDS)
COL_A_GF = find_column(df, AWAY_GF_CANDS)
COL_H_GA = find_column(df, HOME_GA_CANDS)
COL_H_PTS = find_column(df, HOME_PTS_CANDS)
COL_A_PTS = find_column(df, AWAY_PTS_CANDS)

# Validate required columns
has_date_time = (col_date is not None and col_time is not None) or (col_dt is not None)
missing_required = []
if not has_date_time:
    missing_required.append("Date/Time (need Date+Time OR date_GMT)")
for c in [COL_COUNTRY, COL_LEAGUE, COL_HOME, COL_AWAY, COL_XG_H, COL_XG_A]:
    if c not in df.columns:
        missing_required.append(c)

if missing_required:
    st.error(f"Missing required columns: {missing_required}")
    st.write("Detected columns:", list(df.columns))
    st.stop()

rows_out = []
for idx, r in df.iterrows():
    xg_h = safe_float(r.get(COL_XG_H))
    xg_a = safe_float(r.get(COL_XG_A))
    if xg_h is None or xg_a is None:
        continue

    over25_odds = safe_float(r.get(COL_O25)) if use_market else None
    under25_odds = safe_float(r.get(COL_U25)) if use_market else None
    home_odds = safe_float(r.get(COL_H)) if use_market else None
    draw_odds = safe_float(r.get(COL_D)) if use_market else None
    away_odds = safe_float(r.get(COL_A)) if use_market else None

    btts_yes_odds = safe_float(r.get(COL_BTTS_Y)) if (use_market and COL_BTTS_Y) else None
    btts_no_odds = safe_float(r.get(COL_BTTS_N)) if (use_market and COL_BTTS_N) else None

    lam_h, lam_a = compute_lambdas_from_features(
        r=r,
        col_xg_h=COL_XG_H,
        col_xg_a=COL_XG_A,
        col_h_gf=COL_H_GF,
        col_a_ga=COL_A_GA,
        col_a_gf=COL_A_GF,
        col_h_ga=COL_H_GA,
        col_h_pts=COL_H_PTS,
        col_a_pts=COL_A_PTS,
        over25_odds=over25_odds,
        under25_odds=under25_odds,
        home_odds=home_odds,
        draw_odds=draw_odds,
        away_odds=away_odds,
        btts_yes_odds=btts_yes_odds,
        btts_no_odds=btts_no_odds,
        w_xg=w_xg,
        w_goals=w_goals,
        points_tilt=points_tilt,
        totals_weight=totals_weight,
        tilt_weight=tilt_weight,
        btts_weight=btts_weight,
    )

    p = build_score_matrix(lam_h, lam_a, max_goals=max_goals)
    seed = int((idx + 1) * 10007)
    hg, ag = pick_score_entertaining(p, adventurousness, temperature, seed)

    if col_dt is not None:
        date_str, time_str = parse_footystats_date_gmt(r.get(col_dt))
    else:
        date_str, time_str = parse_date_time(r.get(col_date), r.get(col_time))

    rows_out.append({
        "Date": date_str,
        "Time": time_str,
        "Country": str(r.get(COL_COUNTRY)),
        "League": str(r.get(COL_LEAGUE)),
        "Home Team": str(r.get(COL_HOME)),
        "Home Prediction": int(hg),
        "Away Prediction": int(ag),
        "Away Team": str(r.get(COL_AWAY)),
    })

out = pd.DataFrame(rows_out)

st.subheader("Predictions")
st.dataframe(out, use_container_width=True)

st.subheader("Copy/paste into Google Sheets")
tsv = out.to_csv(sep="\t", index=False)
st.text_area("TSV (Ctrl/Cmd+A then copy)", tsv, height=220)

st.download_button(
    "Download CSV",
    data=out.to_csv(index=False).encode("utf-8"),
    file_name="correct_score_predictions.csv",
    mime="text/csv",
)
