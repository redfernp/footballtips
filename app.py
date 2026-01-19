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
    Input like: 'Jan 18 2026 - 1:00pm'
    Output: ('Jan 18 2026', '1:00 PM')
    """
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


def pick_score_entertaining(
    p: np.ndarray,
    adventurousness: float,
    temperature: float,
    seed: int
) -> Tuple[int, int]:
    """
    - temperature > 1 flattens distribution -> more variety
    - adventurousness biases toward higher total goals
    """
    rng = np.random.default_rng(seed)

    # Temperature transform
    # (higher T => more flat). Using power alpha = 1/T
    T = max(0.6, float(temperature))
    alpha = 1.0 / T
    w = np.power(np.clip(p, 1e-12, 1.0), alpha)

    # Adventure bias toward higher totals
    # multiplier exp(k*(i+j))
    k = 0.35 * float(adventurousness)  # 0..0.35
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
    """
    Build lambdas using:
    - xG base
    - avg goals scored + conceded (if present)
    - points tilt (if present)
    - market total goals anchor (O/U 2.5)
    - market tilt anchor (1X2)
    - BTTS nudges away from sterile low-score shapes
    """

    # --- base from xG ---
    xg_h = safe_float(r.get(col_xg_h)) or 0.0
    xg_a = safe_float(r.get(col_xg_a)) or 0.0
    lam_h = max(0.05, xg_h)
    lam_a = max(0.05, xg_a)

    # --- add “form-like” goals info if available ---
    # Home attack + Away defense
    parts_h = []
    if col_h_gf:
        v = safe_float(r.get(col_h_gf))
        if v is not None:
            parts_h.append(v)
    if col_a_ga:
        v = safe_float(r.get(col_a_ga))
        if v is not None:
            parts_h.append(v)

    # Away attack + Home defense
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

    # --- points tilt (small, just to push one side up a bit) ---
    if col_h_pts and col_a_pts:
        hp = safe_float(r.get(col_h_pts))
        ap = safe_float(r.get(col_a_pts))
        if hp is not None and ap is not None:
            # normalize by a typical range (0..90-ish season points)
            diff = (hp - ap) / 30.0  # conservative
            diff = clamp(diff, -1.0, 1.0) * points_tilt  # points_tilt slider
            # increase home and reduce away slightly, then renormalize later
            lam_h *= (1.0 + 0.12 * diff)
            lam_a *= (1.0 - 0.12 * diff)
            lam_h = max(0.02, lam_h)
            lam_a = max(0.02, lam_a)

    # --- market total goals anchor from O/U 2.5 ---
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

    # --- 1X2 tilt anchor (who gets more of the goals) ---
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

    # --- BTTS logic nudges away from “too many 1-0/0-1/1-1” if BTTS Yes is strong ---
    # We do this by gently increasing both lambdas when market likes BTTS Yes.
    by = odds_to_prob(btts_yes_odds)
    bn = odds_to_prob(btts_no_odds)
    if by is not None and bn is not None:
        probs = overround_normalize({"yes": by, "no": bn})
        p_yes = probs["yes"]
        # map p_yes in [0.35..0.70] to a multiplier range ~[0.95..1.12]
        mult = 1.0 + (clamp(p_yes, 0.35, 0.70) - 0.50) * 0.55
        mult = (1 - btts_weight) * 1.0 + btts_weight * mult
        lam_h *= mult
        lam_a *= mult

    # Final clamps
    lam_h = clamp(lam_h, 0.05, 4.5)
    lam_a = clamp(lam_a, 0.05, 4.5)
    return lam_h, lam_a


# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(page_title="Correct Score Predictor (Entertainment Mode)", layout="wide")
st.title("Correct Score Predictor (Entertainment Mode)")
st.write("More lively correct scores using xG + goals/points + full odds logic (for entertainment, still grounded).")

with st.sidebar:
    st.header("Style controls")
    adventurousness = st.slider("Adventurousness (push to higher totals)", 0.0, 1.0, 0.55, 0.05)
