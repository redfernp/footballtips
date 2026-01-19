import math
from datetime import datetime
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st


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
    """
    Normalize date and time to:
    Date: 'Dec 10 2025'
    Time: '7:45 PM'
    """
    # Date
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

    # Time
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
    """
    FootyStats example:
      'Jan 18 2026 - 1:00pm'
      'Jan 18 2026 - 1:00 pm'
    Output:
      ('Jan 18 2026', '1:00 PM')
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "", ""

    s = str(value).strip()

    # Normalize separators and AM/PM
    s = s.replace(" - ", " ").replace("-", " ")
    s = s.replace("am", " AM").replace("pm", " PM")
    s = s.replace("AM", " AM").replace("PM", " PM")
    s = " ".join(s.split())

    # Parse with pandas (handles a lot of variants)
    dt = pd.to_datetime(s, errors="coerce")
    if pd.isna(dt):
        # fallback: keep original in Date, blank time
        return str(value).strip(), ""

    date_out = dt.strftime("%b %d %Y")
    time_out = dt.strftime("%-I:%M %p")  # works on Streamlit Cloud (Linux)
    return date_out, time_out


def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def overround_normalize(implied_probs: Dict[str, float]) -> Dict[str, float]:
    s = sum(implied_probs.values())
    if s <= 0:
        return implied_probs
    return {k: v / s for k, v in implied_probs.items()}


def odds_to_prob(odds: Optional[float]) -> Optional[float]:
    if odds is None or odds <= 1e-9:
        return None
    return 1.0 / odds


def total_goals_prob_over_25(lam_total: float) -> float:
    # P(Total >= 3) = 1 - P(0) - P(1) - P(2), Total ~ Poisson(lam_total)
    p0 = poisson_pmf(0, lam_total)
    p1 = poisson_pmf(1, lam_total)
    p2 = poisson_pmf(2, lam_total)
    return max(0.0, min(1.0, 1.0 - (p0 + p1 + p2)))


def solve_lam_total_from_over25(p_over25: float) -> float:
    """
    Find lam such that P(Total >= 3) ~= p_over25 for Poisson total.
    Simple bisection on [0.05, 6.0].
    """
    p_over25 = max(0.001, min(0.999, p_over25))
    lo, hi = 0.05, 6.0
    for _ in range(50):
        mid = (lo + hi) / 2
        pmid = total_goals_prob_over_25(mid)
        if pmid < p_over25:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def dixon_coles_adjust(p_matrix: np.ndarray, lam_h: float, lam_a: float, rho: float) -> np.ndarray:
    """
    Apply Dixon-Coles low-score adjustment to (0,0), (1,0), (0,1), (1,1).
    rho typically small, e.g. -0.10 to +0.10.
    """
    adj = p_matrix.copy()

    def tau(x, y):
        if x == 0 and y == 0:
            return 1 - (lam_h * lam_a * rho)
        if x == 0 and y == 1:
            return 1 + (lam_h * rho)
        if x == 1 and y == 0:
            return 1 + (lam_a * rho)
        if x == 1 and y == 1:
            return 1 - rho
        return 1.0

    for x in [0, 1]:
        for y in [0, 1]:
            adj[x, y] = adj[x, y] * tau(x, y)

    s = adj.sum()
    if s > 0:
        adj /= s
    return adj


def correct_score_from_lambdas(
    lam_h: float,
    lam_a: float,
    max_goals: int = 6,
    rho: float = 0.0,
) -> Tuple[int, int]:
    """
    Build a score probability matrix and return the mode (most likely scoreline).
    """
    p = np.zeros((max_goals + 1, max_goals + 1), dtype=float)
    for i in range(max_goals + 1):
        pi = poisson_pmf(i, lam_h)
        for j in range(max_goals + 1):
            p[i, j] = pi * poisson_pmf(j, lam_a)

    if abs(rho) > 1e-9:
        p = dixon_coles_adjust(p, lam_h, lam_a, rho)

    idx = np.unravel_index(np.argmax(p), p.shape)
    return int(idx[0]), int(idx[1])


def apply_market_anchors(
    lam_h: float,
    lam_a: float,
    over25_odds: Optional[float],
    under25_odds: Optional[float],
    home_odds: Optional[float],
    draw_odds: Optional[float],
    away_odds: Optional[float],
    totals_weight: float,
    tilt_weight: float,
) -> Tuple[float, float]:
    """
    1) Total goals anchor using O/U 2.5.
    2) Home-v-away tilt using 1X2.
    totals_weight and tilt_weight in [0,1], where 0=ignore, 1=full anchor.
    """
    lam_h2, lam_a2 = lam_h, lam_a

    # --- Total goals anchor ---
    p_over = odds_to_prob(over25_odds)
    p_under = odds_to_prob(under25_odds)

    if p_over is not None and p_under is not None:
        probs = overround_normalize({"over": p_over, "under": p_under})
        p_over_norm = probs["over"]

        lam_total_xg = max(0.05, lam_h2 + lam_a2)
        lam_total_mkt = solve_lam_total_from_over25(p_over_norm)

        lam_total_target = (1 - totals_weight) * lam_total_xg + totals_weight * lam_total_mkt

        scale = lam_total_target / lam_total_xg
        lam_h2 *= scale
        lam_a2 *= scale

    # --- 1X2 tilt anchor (shifts split, preserves total) ---
    pH = odds_to_prob(home_odds)
    pD = odds_to_prob(draw_odds)
    pA = odds_to_prob(away_odds)

    if pH is not None and pD is not None and pA is not None:
        probs = overround_normalize({"H": pH, "D": pD, "A": pA})
        pHn, pAn = probs["H"], probs["A"]

        raw = (pHn - pAn)
        delta = max(-0.15, min(0.15, raw * 0.30))  # conservative
        delta = delta * tilt_weight

        lam_total = max(0.05, lam_h2 + lam_a2)

        lam_h2 = max(0.02, lam_h2 * (1 + delta))
        lam_a2 = max(0.02, lam_a2 * (1 - delta))

        new_total = lam_h2 + lam_a2
        if new_total > 0:
            lam_h2 *= lam_total / new_total
            lam_a2 *= lam_total / new_total

    return lam_h2, lam_a2


# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(page_title="Correct Score Predictor (xG + Markets)", layout="wide")
st.title("Correct Score Predictor (xG + Market Anchors)")
st.write("Upload your FootyStats CSV, generate score predictions, and copy/paste into Google Sheets.")

with st.sidebar:
    st.header("Model controls")
    use_market = st.checkbox("Use market anchors (recommended if odds present)", value=True)
    totals_weight = st.slider("Totals anchor strength (O/U 2.5)", 0.0, 1.0, 0.7, 0.05)
    tilt_weight = st.slider("1X2 tilt strength", 0.0, 1.0, 0.5, 0.05)

    st.divider()
    use_dc = st.checkbox("Apply Dixon–Coles low-score adjustment", value=True)
    rho = st.slider(
        "Dixon–Coles rho (negative increases 0-0/1-1 slightly)",
        -0.2, 0.2, -0.08, 0.01
    ) if use_dc else 0.0

    st.divider()
    max_goals = st.slider("Max goals considered per team", 4, 10, 6, 1)
    st.caption("Most leagues are fine with 6. Use 7–8 for high-scoring leagues.")

uploaded = st.file_uploader("Upload FootyStats CSV", type=["csv"])
default_path_hint = "TOOL + API - Correct Scores goals auto tips creator V2.0 - Upload Daily Correct Scores Tips Footystats File.csv"
st.caption(f"If you’re testing locally, your file name looks like: {default_path_hint}")

if not uploaded:
    st.stop()

df = pd.read_csv(uploaded)

# Expected columns in your feed
COL_COUNTRY = "Country"
COL_LEAGUE = "League"
COL_HOME = "Home Team"
COL_AWAY = "Away Team"
COL_XG_H = "Home Team Pre-Match xG"
COL_XG_A = "Away Team Pre-Match xG"

# Date/time candidates (supports separate columns OR a combined date_GMT column)
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

# Validate required columns
has_date_time = (col_date is not None and col_time is not None) or (col_dt is not None)

missing_required = []
if not has_date_time:
    missing_required.append("Date/Time (need Date+Time columns OR a date_GMT column)")

for c in [COL_COUNTRY, COL_LEAGUE, COL_HOME, COL_AWAY, COL_XG_H, COL_XG_A]:
    if c not in df.columns:
        missing_required.append(c)

if missing_required:
    st.error(f"Missing required columns: {missing_required}")
    st.write("Detected columns:", list(df.columns))
    st.stop()

rows_out = []
for _, r in df.iterrows():
    lam_h = safe_float(r.get(COL_XG_H))
    lam_a = safe_float(r.get(COL_XG_A))

    if lam_h is None or lam_a is None:
        continue

    lam_h = max(0.05, lam_h)
    lam_a = max(0.05, lam_a)

    if use_market:
        lam_h, lam_a = apply_market_anchors(
            lam_h=lam_h,
            lam_a=lam_a,
            over25_odds=safe_float(r.get(COL_O25)),
            under25_odds=safe_float(r.get(COL_U25)),
            home_odds=safe_float(r.get(COL_H)),
            draw_odds=safe_float(r.get(COL_D)),
            away_odds=safe_float(r.get(COL_A)),
            totals_weight=totals_weight,
            tilt_weight=tilt_weight,
        )

    hg, ag = correct_score_from_lambdas(
        lam_h=lam_h,
        lam_a=lam_a,
        max_goals=max_goals,
        rho=rho if use_dc else 0.0,
    )

    # Date + Time from either combined datetime or separate columns
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
