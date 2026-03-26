import math
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st


# ----------------------------
# MUST BE FIRST Streamlit call
# ----------------------------
st.set_page_config(page_title="Football Tips Generator", layout="wide")

st.title("Football Tips Generator")
st.write("Upload a FootyStats CSV to generate tips across multiple markets.")

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
    try:
        time_out = dt.strftime("%-I:%M %p")
    except ValueError:
        time_out = dt.strftime("%I:%M %p").lstrip("0")
    return date_out, time_out


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
        try:
            time_out = time_val.strftime("%-I:%M %p")
        except ValueError:
            time_out = time_val.strftime("%I:%M %p").lstrip("0")
    else:
        s = str(time_val).strip()
        for fmt in ("%H:%M", "%H:%M:%S", "%I:%M %p", "%I:%M%p"):
            try:
                t = datetime.strptime(s, fmt)
                try:
                    time_out = t.strftime("%-I:%M %p")
                except ValueError:
                    time_out = t.strftime("%I:%M %p").lstrip("0")
                break
            except Exception:
                pass
        if not time_out:
            time_out = s

    return date_out, time_out


def get_date_time(r: pd.Series, col_dt, col_date, col_time) -> Tuple[str, str]:
    if col_dt is not None:
        return parse_footystats_date_gmt(r.get(col_dt))
    return parse_date_time(r.get(col_date), r.get(col_time))


def make_tsv_block(df: pd.DataFrame, label: str):
    st.subheader(f"Copy/paste {label} into Google Sheets")
    tsv = df.to_csv(sep="\t", index=False)
    st.text_area("TSV (Ctrl/Cmd+A then copy)", tsv, height=200, key=f"tsv_{label}")


def make_download_button(df: pd.DataFrame, filename: str, label: str):
    st.download_button(
        f"Download {label} CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        key=f"dl_{label}",
    )


# ----------------------------
# Load & common column detection
# ----------------------------
df = pd.read_csv(uploaded)

# Drop rows where date_unix is missing (blank/error rows from spreadsheet)
if "date_unix" in df.columns:
    df = df[df["date_unix"].notna() & (df["date_unix"].astype(str).str.strip() != "")]
    df = df[~df["date_unix"].astype(str).str.startswith("#")]

# Filter: only include games where all key odds columns are present and > 0.1
ODDS_FILTER_COLS = ["Odds_Home_Win", "Odds_BTTS_Yes", "Odds_Over15", "Odds_Over25"]
present_odds_cols = [c for c in ODDS_FILTER_COLS if c in df.columns]
for col in present_odds_cols:
    df = df[pd.to_numeric(df[col], errors="coerce").fillna(0) > 0.1]

DATE_CANDIDATES = ["Date", "Match Date", "Match_Date", "Date GMT", "Match Date GMT"]
TIME_CANDIDATES = ["Time", "Match Time", "Match_Time", "Time GMT", "Match Time GMT"]
DATETIME_CANDIDATES = ["date_GMT", "Date_GMT", "DateTime", "Datetime", "Date Time", "Match DateTime", "Match Datetime"]

col_date = find_column(df, DATE_CANDIDATES)
col_time = find_column(df, TIME_CANDIDATES)
col_dt = find_column(df, DATETIME_CANDIDATES)

# Sort games by date/time ascending
if "date_unix" in df.columns:
    df = df.sort_values("date_unix", ascending=True).reset_index(drop=True)
elif col_dt is not None:
    df = df.sort_values(col_dt, ascending=True).reset_index(drop=True)

COL_COUNTRY = "Country"
COL_LEAGUE = "League"
COL_HOME = "Home Team"
COL_AWAY = "Away Team"

BASE_COLS = [COL_COUNTRY, COL_LEAGUE, COL_HOME, COL_AWAY]
has_date_time = (col_date is not None and col_time is not None) or (col_dt is not None)

# ----------------------------
# League Filters
# ----------------------------
TOP_LEAGUES = [
    # UEFA club competitions
    ("Europe", "UEFA Champions League"),
    ("Europe", "UEFA Europa League"),
    ("Europe", "UEFA Europa Conference League"),
    # Big 5 + top European leagues
    ("England", "Premier League"),
    ("Spain", "La Liga"),
    ("Germany", "Bundesliga"),
    ("France", "Ligue 1"),
    ("Italy", "Serie A"),
    ("Portugal", "Liga NOS"),
    ("Netherlands", "Eredivisie"),
    ("Belgium", "Pro League"),
    ("Turkey", "Süper Lig"),
    ("Scotland", "Premiership"),
    # Americas
    ("Brazil", "Serie A"),
    ("Argentina", "Primera División"),
    ("Mexico", "Liga MX"),
    ("USA", "MLS"),
    # Asia / Rest of World
    ("Russia", "Russian Premier League"),
    ("Japan", "J1 League"),
    ("South Korea", "K League 1"),
]

TOP_AND_SECOND_TIER = TOP_LEAGUES + [
    # --- 2nd tiers of big leagues ---
    ("England", "Championship"),
    ("Germany", "2. Bundesliga"),
    ("Spain", "Segunda División"),
    ("France", "Ligue 2"),
    ("Italy", "Serie B"),
    ("Portugal", "LigaPro"),
    ("Netherlands", "Eerste Divisie"),
    ("Belgium", "First Division B"),
    ("Turkey", "1. Lig"),
    ("Scotland", "Championship"),
    ("Brazil", "Serie B"),
    ("Argentina", "Prim B Nacional"),
    ("Mexico", "Ascenso MX"),
    ("Russia", "FNL"),
    ("Japan", "J2 League"),
    ("South Korea", "K League 2"),
    # --- Top flights of second-tier European countries ---
    ("Austria", "Bundesliga"),
    ("Switzerland", "Super League"),
    ("Denmark", "Superliga"),
    ("Norway", "Eliteserien"),
    ("Sweden", "Allsvenskan"),
    ("Finland", "Veikkausliiga"),
    ("Republic of Ireland", "Premier Division"),
    ("Greece", "Super League"),
    ("Croatia", "Prva HNL"),
    ("Czech Republic", "First League"),
    ("Poland", "Ekstraklasa"),
    ("Romania", "Liga I"),
    ("Serbia", "SuperLiga"),
    ("Hungary", "NB I"),
    ("Slovakia", "Super Liga"),
    ("Bulgaria", "First League"),
    ("Israel", "Israeli Premier League"),
    ("Cyprus", "First Division"),
    ("Slovenia", "PrvaLiga"),
    ("Kazakhstan", "Kazakhstan Premier League"),
    ("Azerbaijan", "Premyer Liqası"),
    # --- Top flights of notable non-European countries ---
    ("Saudi Arabia", "Professional League"),
    ("Qatar", "Stars League"),
    ("UAE", "Arabian Gulf League"),
    ("China", "Chinese Super League"),
    ("Australia", "A-League"),
    ("Thailand", "Thai League T1"),
    ("Indonesia", "Liga 1"),
    ("South Africa", "Premier Soccer League"),
    ("Egypt", "Egyptian Premier League"),
    ("Chile", "Primera División"),
    ("Colombia", "Categoria Primera A"),
    ("Peru", "Primera División"),
    ("Ecuador", "Primera Categoría Serie A"),
    ("Paraguay", "Division Profesional"),
    # --- International competitions ---
    ("International", "International Friendlies"),
    ("International", "WC Qualification Europe"),
    ("International", "WC Qualification Africa"),
    ("International", "WC Qualification Asia"),
    ("International", "WC Qualification CONCACAF"),
    ("International", "Africa Cup of Nations"),
    ("International", "CONCACAF Champions League"),
    ("International", "FIFA World Cup U20"),
    ("Asia", "AFC Champions League"),
    ("Asia", "AFC Cup"),
    ("Africa", "CAF Confederations Cup"),
    ("South America", "Copa Libertadores"),
    ("South America", "Copa Sudamericana"),
    ("Europe", "UEFA U21 Championship Qualification"),
]

BIG_BETTING_COMPETITIONS = [
    # England - top 4 leagues + cups
    ("England", "Premier League"),
    ("England", "Championship"),
    ("England", "EFL League One"),
    ("England", "EFL League Two"),
    ("England", "FA Cup"),
    ("England", "League Cup"),
    ("England", "EFL Trophy"),
    # Scotland - top 2 leagues + cups
    ("Scotland", "Premiership"),
    ("Scotland", "Championship"),
    ("Scotland", "Scottish League Cup"),
    ("Scotland", "Challenge Cup"),
    # Spain - top league + cup
    ("Spain", "La Liga"),
    ("Spain", "Copa del Rey"),
    # Italy - top league + cup
    ("Italy", "Serie A"),
    ("Italy", "Coppa Italia"),
    # Germany - top league + cup
    ("Germany", "Bundesliga"),
    ("Germany", "DFB Pokal"),
    # France - top league + cup
    ("France", "Ligue 1"),
    ("France", "Coupe de France"),
    # Netherlands - top league + cup
    ("Netherlands", "Eredivisie"),
    ("Netherlands", "KNVB Cup"),
    # UEFA club competitions
    ("Europe", "UEFA Champions League"),
    ("Europe", "UEFA Europa League"),
    ("Europe", "UEFA Europa Conference League"),
    ("Europe", "UEFA U21 Championship Qualification"),
    # All international competitions
    ("International", "International Friendlies"),
    ("International", "WC Qualification Europe"),
    ("International", "WC Qualification Africa"),
    ("International", "WC Qualification Asia"),
    ("International", "WC Qualification CONCACAF"),
    ("International", "Africa Cup of Nations"),
    ("International", "CONCACAF Champions League"),
    ("International", "FIFA World Cup U20"),
    ("Asia", "AFC Champions League"),
    ("Asia", "AFC Cup"),
    ("Africa", "CAF Confederations Cup"),
    ("South America", "Copa Libertadores"),
    ("South America", "Copa Sudamericana"),
]

league_filter = st.sidebar.radio(
    "League filter",
    ["All Leagues", "Top Leagues only", "Top + 2nd Tier Leagues + Internationals", "Big Betting Competitions"],
    index=1,
)

if league_filter == "Top Leagues only":
    mask = pd.Series(False, index=df.index)
    for country, league in TOP_LEAGUES:
        mask |= (df[COL_COUNTRY] == country) & (df[COL_LEAGUE] == league)
    df = df[mask].reset_index(drop=True)
    if df.empty:
        st.warning("No matches found for top leagues in this CSV.")
        st.stop()
elif league_filter == "Top + 2nd Tier Leagues + Internationals":
    mask = pd.Series(False, index=df.index)
    for country, league in TOP_AND_SECOND_TIER:
        mask |= (df[COL_COUNTRY] == country) & (df[COL_LEAGUE] == league)
    df = df[mask].reset_index(drop=True)
    if df.empty:
        st.warning("No matches found for selected leagues in this CSV.")
        st.stop()
elif league_filter == "Big Betting Competitions":
    mask = pd.Series(False, index=df.index)
    for country, league in BIG_BETTING_COMPETITIONS:
        mask |= (df[COL_COUNTRY] == country) & (df[COL_LEAGUE] == league)
    df = df[mask].reset_index(drop=True)
    if df.empty:
        st.warning("No matches found for Big Betting Competitions in this CSV.")
        st.stop()

missing_base = []
if not has_date_time:
    missing_base.append("Date/Time (need Date+Time OR date_GMT)")
for c in BASE_COLS:
    if c not in df.columns:
        missing_base.append(c)

if missing_base:
    st.error(f"Missing base columns: {missing_base}")
    st.write("Detected columns:", list(df.columns))
    st.stop()


# ----------------------------
# Tabs
# ----------------------------
tab_correct, tab_btts, tab_over15, tab_over25, tab_1x2 = st.tabs([
    "Correct Score", "BTTS", "Over/Under 1.5", "Over/Under 2.5", "1X2"
])


# ============================================================
# TAB 1: Correct Score
# ============================================================
with tab_correct:
    st.header("Correct Score Predictions")
    st.caption("Formula: Home Goals = round((Home PPG × Home xG) / Odds_Home_Win), same for away.")

    COL_HOME_PPG_CURR = "Home Team Points Per Game (Current)"
    COL_AWAY_PPG_CURR = "Away Team Points Per Game (Current)"
    COL_XG_H = "Home Team Pre-Match xG"
    COL_XG_A = "Away Team Pre-Match xG"
    COL_OH = "Odds_Home_Win"
    COL_OA = "Odds_Away_Win"

    cs_missing = [c for c in [COL_HOME_PPG_CURR, COL_AWAY_PPG_CURR, COL_XG_H, COL_XG_A, COL_OH, COL_OA]
                  if c not in df.columns]

    if cs_missing:
        st.warning(f"Missing columns for Correct Score: {cs_missing}")
    else:
        def compute_correct_score(r):
            hp = safe_float(r.get(COL_HOME_PPG_CURR))
            ap = safe_float(r.get(COL_AWAY_PPG_CURR))
            hxg = safe_float(r.get(COL_XG_H))
            axg = safe_float(r.get(COL_XG_A))
            oh = safe_float(r.get(COL_OH))
            oa = safe_float(r.get(COL_OA))

            home_goals = round_half_up((hp * hxg) / oh) if all(v is not None for v in [hp, hxg, oh]) and oh > 0 else 0
            away_goals = round_half_up((ap * axg) / oa) if all(v is not None for v in [ap, axg, oa]) and oa > 0 else 0

            home_goals = max(0, min(10, home_goals))
            away_goals = max(0, min(10, away_goals))
            return home_goals, away_goals

        rows_cs = []
        for _, r in df.iterrows():
            hg, ag = compute_correct_score(r)
            date_str, time_str = get_date_time(r, col_dt, col_date, col_time)
            rows_cs.append({
                "Date": date_str,
                "Time": time_str,
                "Country": str(r.get(COL_COUNTRY)),
                "League": str(r.get(COL_LEAGUE)),
                "Home Team": str(r.get(COL_HOME)),
                "Home Prediction": int(hg),
                "Away Prediction": int(ag),
                "Away Team": str(r.get(COL_AWAY)),
            })

        out_cs = pd.DataFrame(rows_cs)
        st.dataframe(out_cs, use_container_width=True)
        make_tsv_block(out_cs, "Correct Score")
        make_download_button(out_cs, "correct_score_predictions.csv", "Correct Score")


# ============================================================
# TAB 2: BTTS
# ============================================================
with tab_btts:
    st.header("BTTS Tips")
    st.caption("Formula: BTTS Avg < 35 → NO | 35–64 → NO BET | ≥ 65 → YES")

    COL_BTTS_AVG = "BTTS Average"
    COL_BTTS_YES_ODDS = "Odds_BTTS_Yes"
    COL_BTTS_NO_ODDS = "Odds_BTTS_No"

    btts_missing = [c for c in [COL_BTTS_AVG] if c not in df.columns]

    if btts_missing:
        st.warning(f"Missing columns for BTTS: {btts_missing}")
    else:
        def compute_btts_tip(r):
            avg = safe_float(r.get(COL_BTTS_AVG))
            if avg is None:
                return "NO BET", None

            if avg < 35:
                tip = "NO"
                odds = safe_float(r.get(COL_BTTS_NO_ODDS))
            elif avg < 65:
                tip = "NO BET"
                odds = safe_float(r.get(COL_BTTS_YES_ODDS))
            else:
                tip = "YES"
                odds = safe_float(r.get(COL_BTTS_YES_ODDS))

            return tip, odds

        rows_btts = []
        for _, r in df.iterrows():
            tip, odds = compute_btts_tip(r)
            date_str, time_str = get_date_time(r, col_dt, col_date, col_time)
            rows_btts.append({
                "Date": date_str,
                "Time": time_str,
                "Country": str(r.get(COL_COUNTRY)),
                "League": str(r.get(COL_LEAGUE)),
                "Home Team": str(r.get(COL_HOME)),
                "Away Team": str(r.get(COL_AWAY)),
                "BTTS Tip": tip,
                "Odds": odds,
            })

        out_btts = pd.DataFrame(rows_btts)

        # Filter controls
        show_btts = st.radio(
            "Show tips",
            ["Exclude NO BET", "All", "YES only", "NO only"],
            horizontal=True,
            key="btts_filter",
        )
        filtered_btts = out_btts.copy()
        if show_btts == "YES only":
            filtered_btts = filtered_btts[filtered_btts["BTTS Tip"] == "YES"]
        elif show_btts == "NO only":
            filtered_btts = filtered_btts[filtered_btts["BTTS Tip"] == "NO"]
        elif show_btts == "Exclude NO BET":
            filtered_btts = filtered_btts[filtered_btts["BTTS Tip"] != "NO BET"]

        st.dataframe(filtered_btts, use_container_width=True)
        make_tsv_block(filtered_btts, "BTTS")
        make_download_button(filtered_btts, "btts_tips.csv", "BTTS")


# ============================================================
# TAB 3: Over/Under 1.5
# ============================================================
with tab_over15:
    st.header("Over/Under 1.5 Goals Tips")
    st.caption("Formula: Over15 Avg < 49 → UNDER | 49–69 → NO BET | ≥ 70 → OVER")

    COL_OVER15_AVG = "Over15 Average"
    COL_OVER15_ODDS = "Odds_Over15"
    COL_UNDER15_ODDS = "Odds_Under15"

    o15_missing = [c for c in [COL_OVER15_AVG] if c not in df.columns]

    if o15_missing:
        st.warning(f"Missing columns for Over/Under 1.5: {o15_missing}")
    else:
        def compute_over15_tip(r):
            avg = safe_float(r.get(COL_OVER15_AVG))
            if avg is None:
                return "NO BET", None

            if avg < 49:
                tip = "UNDER"
                odds = safe_float(r.get(COL_UNDER15_ODDS))
            elif avg < 70:
                tip = "NO BET"
                odds = safe_float(r.get(COL_OVER15_ODDS))
            else:
                tip = "OVER"
                odds = safe_float(r.get(COL_OVER15_ODDS))

            return tip, odds

        rows_o15 = []
        for _, r in df.iterrows():
            tip, odds = compute_over15_tip(r)
            date_str, time_str = get_date_time(r, col_dt, col_date, col_time)
            rows_o15.append({
                "Date": date_str,
                "Time": time_str,
                "Country": str(r.get(COL_COUNTRY)),
                "League": str(r.get(COL_LEAGUE)),
                "Home Team": str(r.get(COL_HOME)),
                "Away Team": str(r.get(COL_AWAY)),
                "1.5 Goals Tip": tip,
                "Odds": odds,
            })

        out_o15 = pd.DataFrame(rows_o15)

        show_o15 = st.radio(
            "Show tips",
            ["Exclude NO BET", "All", "OVER only", "UNDER only"],
            horizontal=True,
            key="o15_filter",
        )
        filtered_o15 = out_o15.copy()
        if show_o15 == "OVER only":
            filtered_o15 = filtered_o15[filtered_o15["1.5 Goals Tip"] == "OVER"]
        elif show_o15 == "UNDER only":
            filtered_o15 = filtered_o15[filtered_o15["1.5 Goals Tip"] == "UNDER"]
        elif show_o15 == "Exclude NO BET":
            filtered_o15 = filtered_o15[filtered_o15["1.5 Goals Tip"] != "NO BET"]

        st.dataframe(filtered_o15, use_container_width=True)
        make_tsv_block(filtered_o15, "Over15")
        make_download_button(filtered_o15, "over15_tips.csv", "Over15")


# ============================================================
# TAB 4: Over/Under 2.5
# ============================================================
with tab_over25:
    st.header("Over/Under 2.5 Goals Tips")
    st.caption("Formula: Over25 Avg < 35 → UNDER | 35–64 → NO BET | ≥ 65 → OVER")

    COL_OVER25_AVG = "Over25 Average"
    COL_OVER25_ODDS = "Odds_Over25"
    COL_UNDER25_ODDS = "Odds_Under25"

    o25_missing = [c for c in [COL_OVER25_AVG] if c not in df.columns]

    if o25_missing:
        st.warning(f"Missing columns for Over/Under 2.5: {o25_missing}")
    else:
        def compute_over25_tip(r):
            avg = safe_float(r.get(COL_OVER25_AVG))
            if avg is None:
                return "NO BET", None

            if avg < 35:
                tip = "UNDER"
                odds = safe_float(r.get(COL_UNDER25_ODDS))
            elif avg < 65:
                tip = "NO BET"
                odds = safe_float(r.get(COL_OVER25_ODDS))
            else:
                tip = "OVER"
                odds = safe_float(r.get(COL_OVER25_ODDS))

            return tip, odds

        rows_o25 = []
        for _, r in df.iterrows():
            tip, odds = compute_over25_tip(r)
            date_str, time_str = get_date_time(r, col_dt, col_date, col_time)
            rows_o25.append({
                "Date": date_str,
                "Time": time_str,
                "Country": str(r.get(COL_COUNTRY)),
                "League": str(r.get(COL_LEAGUE)),
                "Home Team": str(r.get(COL_HOME)),
                "Away Team": str(r.get(COL_AWAY)),
                "2.5 Goals Tip": tip,
                "Odds": odds,
            })

        out_o25 = pd.DataFrame(rows_o25)

        show_o25 = st.radio(
            "Show tips",
            ["Exclude NO BET", "All", "OVER only", "UNDER only"],
            horizontal=True,
            key="o25_filter",
        )
        filtered_o25 = out_o25.copy()
        if show_o25 == "OVER only":
            filtered_o25 = filtered_o25[filtered_o25["2.5 Goals Tip"] == "OVER"]
        elif show_o25 == "UNDER only":
            filtered_o25 = filtered_o25[filtered_o25["2.5 Goals Tip"] == "UNDER"]
        elif show_o25 == "Exclude NO BET":
            filtered_o25 = filtered_o25[filtered_o25["2.5 Goals Tip"] != "NO BET"]

        st.dataframe(filtered_o25, use_container_width=True)
        make_tsv_block(filtered_o25, "Over25")
        make_download_button(filtered_o25, "over25_tips.csv", "Over25")


# ============================================================
# TAB 5: 1X2
# ============================================================
with tab_1x2:
    st.header("1X2 Tips")
    st.caption("Formula: Home PPG > 2.4 → HOME WIN | Away PPG > 2.5 → AWAY WIN | Home PPG = Away PPG → DRAW")

    COL_HOME_PPG = "Home Team Points Per Game (Pre-Match)"
    COL_AWAY_PPG = "Away Team Points Per Game (Pre-Match)"

    x12_missing = [c for c in [COL_HOME_PPG, COL_AWAY_PPG] if c not in df.columns]

    if x12_missing:
        st.warning(f"Missing columns for 1X2: {x12_missing}")
    else:
        def compute_1x2_tip(r):
            home_ppg = safe_float(r.get(COL_HOME_PPG))
            away_ppg = safe_float(r.get(COL_AWAY_PPG))

            if home_ppg is None or away_ppg is None:
                return "NO TIP", None

            if home_ppg > 2.4:
                odds = safe_float(r.get("Odds_Home_Win"))
                return ("NO TIP", None) if odds is None or odds >= 4.0 else ("HOME WIN", odds)
            elif away_ppg > 2.5:
                odds = safe_float(r.get("Odds_Away_Win"))
                return ("NO TIP", None) if odds is None or odds >= 4.0 else ("AWAY WIN", odds)
            elif home_ppg == away_ppg:
                odds = safe_float(r.get("Odds_Draw"))
                return ("NO TIP", None) if odds is None or odds >= 4.0 else ("DRAW", odds)
            else:
                return "NO TIP", None

        rows_1x2 = []
        for _, r in df.iterrows():
            tip, odds = compute_1x2_tip(r)
            date_str, time_str = get_date_time(r, col_dt, col_date, col_time)
            rows_1x2.append({
                "Date": date_str,
                "Time": time_str,
                "Country": str(r.get(COL_COUNTRY)),
                "League": str(r.get(COL_LEAGUE)),
                "Home Team": str(r.get(COL_HOME)),
                "Away Team": str(r.get(COL_AWAY)),
                "1X2 Result Tip": tip,
                "Odds": odds,
            })

        out_1x2 = pd.DataFrame(rows_1x2)

        show_1x2 = st.radio(
            "Show tips",
            ["Exclude NO TIP", "All", "HOME WIN only", "AWAY WIN only", "DRAW only"],
            horizontal=True,
            key="x12_filter",
        )
        filtered_1x2 = out_1x2.copy()
        if show_1x2 == "HOME WIN only":
            filtered_1x2 = filtered_1x2[filtered_1x2["1X2 Result Tip"] == "HOME WIN"]
        elif show_1x2 == "AWAY WIN only":
            filtered_1x2 = filtered_1x2[filtered_1x2["1X2 Result Tip"] == "AWAY WIN"]
        elif show_1x2 == "DRAW only":
            filtered_1x2 = filtered_1x2[filtered_1x2["1X2 Result Tip"] == "DRAW"]
        elif show_1x2 == "Exclude NO TIP":
            filtered_1x2 = filtered_1x2[filtered_1x2["1X2 Result Tip"] != "NO TIP"]

        st.dataframe(filtered_1x2, use_container_width=True)
        make_tsv_block(filtered_1x2, "1X2")
        make_download_button(filtered_1x2, "1x2_tips.csv", "1X2")
