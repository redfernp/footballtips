import math
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st


# ----------------------------
# MUST BE FIRST Streamlit call
# ----------------------------
st.set_page_config(page_title="Correct Score Predictor (Entertainment Mode)", layout="wide")

st.title("Correct Score Predictor (Entertainment Mode)")
st.write("Correct score tips using simplified PPG * xG / win-odds logic (entertainment mode, simplified model).")

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


def round_half_up(x: float) -> int:
    """Normal mathematical rounding: halves round up (away from zero)."""
    if x >= 0:
        return int(math.floor(x + 0.5))
    return int(math.ceil(x - 0.5))


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


def compute_simple_correct_scores(
    r: pd.Series,
    col_home_ppg_current: str,
    col_away_ppg_current: str,
    col_home_xg: str,
    col_away_xg: str,
    col_odds_home: str,
    col_odds_away: str,
) -> Tuple[int, int]:
    """
    Home goals = round_half_up( (Home PPG Current * Home Pre-Match xG) / Odds_Home_Win )
    Away goals = round_half_up( (Away PPG Current * Away Pre-Match xG) / Odds_Away_Win )
    """
    hp = safe_float(r.get(col_home_ppg_current))
    ap = safe_float(r.get(col_away_ppg_current))
    hxg = safe_float(r.get(col_home_xg))
    axg = safe_float(r.get(col_away_xg))
    oh = safe_float(r.get(col_odds_home))
    oa = safe_float(r.get(col_odds_away))

    # If anything critical missing, fall back to 0–0 (or you can choose to skip the row)
    if hp is None or hxg is None or oh is None or oh <= 0:
        home_goals = 0
    else:
        home_goals = round_half_up((hp * hxg) / oh)

    if ap is None or axg is None or oa is None or oa <= 0:
        away_goals = 0
    else:
        away_goals = round_half_up((ap * axg) / oa)

    # Safety clamps (optional but sensible for scorelines)
    home_goals = max(0, min(10, home_goals))
    away_goals = max(0, min(10, away_goals))
    return home_goals, away_goals


# ----------------------------
# Load data
# ----------------------------
df = pd.read_csv(uploaded)

# Core columns
COL_COUNTRY = "Country"
COL_LEAGUE = "League"
COL_HOME = "Home Team"
COL_AWAY = "Away Team"

COL_HOME_PPG_CURR = "Home Team Points Per Game (Current)"
COL_AWAY_PPG_CURR = "Away Team Points Per Game (Current)"

COL_XG_H = "Home Team Pre-Match xG"
COL_XG_A = "Away Team Pre-Match xG"

COL_H = "Odds_Home_Win"
COL_A = "Odds_Away_Win"

# Date/time candidates
DATE_CANDIDATES = ["Date", "Match Date", "Match_Date", "Date GMT", "Match Date GMT"]
TIME_CANDIDATES = ["Time", "Match Time", "Match_Time", "Time GMT", "Match Time GMT"]
DATETIME_CANDIDATES = ["date_GMT", "Date_GMT", "DateTime", "Datetime", "Date Time", "Match DateTime", "Match Datetime"]

col_date = find_column(df, DATE_CANDIDATES)
col_time = find_column(df, TIME_CANDIDATES)
col_dt = find_column(df, DATETIME_CANDIDATES)

# Validate required columns
has_date_time = (col_date is not None and col_time is not None) or (col_dt is not None)

missing_required = []
if not has_date_time:
    missing_required.append("Date/Time (need Date+Time OR date_GMT)")

for c in [COL_COUNTRY, COL_LEAGUE, COL_HOME, COL_AWAY, COL_HOME_PPG_CURR, COL_AWAY_PPG_CURR, COL_XG_H, COL_XG_A, COL_H, COL_A]:
    if c not in df.columns:
        missing_required.append(c)

if missing_required:
    st.error(f"Missing required columns: {missing_required}")
    st.write("Detected columns:", list(df.columns))
    st.stop()

rows_out = []
for idx, r in df.iterrows():
    hg, ag = compute_simple_correct_scores(
        r=r,
        col_home_ppg_current=COL_HOME_PPG_CURR,
        col_away_ppg_current=COL_AWAY_PPG_CURR,
        col_home_xg=COL_XG_H,
        col_away_xg=COL_XG_A,
        col_odds_home=COL_H,
        col_odds_away=COL_A,
    )

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
