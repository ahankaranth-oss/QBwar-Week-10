#!/usr/bin/env python3
"""
TRUE PLAY-BY-PLAY QVAR 4.7 ENGINE, EXPORTER & INTERACTIVE DASHBOARD
-------------------------------------------------------------------
1. CLI mode (writes CSV + JSON files for Substack / Datawrapper / a website):
       python qvar_47_app.py

2. Web dashboard mode (Streamlit):
       pip install streamlit pandas numpy
       streamlit run qvar_47_app.py

3. Use as a library (e.g. from a FastAPI / Flask backend):
       from qvar_47_app import run_qvar_47_model, load_default_data
       results = run_qvar_47_model(load_default_data())

WHAT CHANGED vs. THE ORIGINAL SCRIPT
  * Streamlit detection fixed (uses Streamlit's own runtime check).
  * Sack / INT terms rewritten. Before: every sack and INT ADDED WAR, so taking
    more sacks / throwing more picks raised a QB's score. Now: the adjustment is
    SYMMETRIC around the league-average rate (a QB only gets credit for sacks or
    INTs ABOVE expected, and is docked for being BELOW expected), and only for
    the share of that variance attributed to non-QB factors (line, tipped balls).
    This is regression-to-the-mean logic, not a volume bonus.
  * Full-season projection now shrinks each QB's per-game rate toward the
    league mean so tiny samples (3-5 games) don't extrapolate wildly.
  * Input validation, divide-by-zero guards, JSON export, charts, and a
    Substack-ready trimmed table.

NOTE: The dataset below is hard-coded exactly as supplied and has NOT been
independently verified. Check/refresh it (e.g. from nflverse) before publishing.
"""

import sys
import json
import warnings

import numpy as np
import pandas as pd

# ==============================================================================
# 1. MODEL & 2026 SALARY CAP CONSTANTS
# ==============================================================================
DEFAULT_CONFIG = {
    "epa_per_war": 32.0,          # EPA per 1.0 QVAR WAR
    "repl_epa_per_db": 0.020,     # replacement-level EPA per dropback
    "yacoe_discount": 0.75,       # share of YAC-over-expected NOT credited to QB
    "epa_per_yac_yd": 0.055,      # EPA value of one excess YAC yard
    # --- Sack / INT adjustments (symmetric around league average) ---
    "epa_per_sack_lost": 1.00,    # avg EPA cost of one sack
    "sack_line_share": 0.50,      # share of sack variance blamed on OL/scheme, not QB
    "epa_per_int_lost": 4.50,     # avg EPA cost of one interception
    "int_luck_share": 0.25,       # share of INT variance blamed on luck/tips/drops
    # --- Schedule + market ---
    "dsos_multiplier": 1.00,      # weight on leave-one-out pass-defense dSOS
    "nfl_cap_2026_m": 301.20,     # 2026 NFL salary cap ($M)
    "repl_qb_cost_m": 4.00,       # replacement-level QB cost ($M) = 0.0 WAR
    "dollars_per_war_m": 13.65,   # market $M per 1.0 WAR
    # --- Projection ---
    "shrinkage_games": 6.0,       # prior weight (in games) pulling rate to mean
}

# ==============================================================================
# 2. DATASET (THROUGH WEEK 10 MNF) -- UNVERIFIED, AS SUPPLIED
# ==============================================================================
COLUMNS = [
    "QB", "Team", "Contract", "G", "G_Rem", "DB", "Tot_EPA", "Cmp", "Att",
    "PYds", "CAY", "YAC", "ADoT", "TD", "INT", "Sacks", "CPOE",
    "dSOS_WAR", "Tip_WP_WAR", "Cap_2026_M", "AAV_M",
]

WEEK_10_2026_MASTER_DATA = [
    ("Joe Burrow", "CIN", "Vet Ext ($55M)", 9, 8, 368, 108.8, 254, 340, 2795, 1664, 1143, 7.58, 27, 6, 17, +9.6, +0.067, +0.085, 48.00, 55.00),
    ("Brock Purdy", "SF", "Vet Ext ($53M)", 9, 8, 297, 107.1, 185, 264, 2369, 1270, 1099, 8.20, 20, 5, 12, +5.1, +0.040, +0.030, 38.50, 53.00),
    ("Dak Prescott", "DAL", "Vet Ext ($60M)", 10, 7, 425, 73.8, 248, 387, 2738, 1735, 1026, 8.33, 21, 6, 23, +0.5, -0.030, +0.045, 56.00, 60.00),
    ("Trevor Lawrence", "JAX", "Vet Ext ($55M)", 9, 8, 284, 85.9, 172, 262, 2284, 1301, 1001, 8.65, 21, 5, 12, +3.1, +0.055, +0.045, 42.00, 55.00),
    ("Drake Maye", "NE", "Rookie (Yr 3)", 10, 7, 325, 51.4, 186, 285, 2330, 1472, 858, 9.55, 18, 11, 27, +6.6, -0.056, +0.160, 10.00, 9.16),
    ("Jared Goff", "DET", "Vet Ext ($53M)", 9, 8, 331, 69.8, 206, 305, 2437, 1316, 1130, 7.61, 18, 4, 20, +3.0, +0.033, -0.015, 46.00, 53.00),
    ("Josh Allen", "BUF", "Vet Ext ($55M)", 9, 8, 302, 76.9, 167, 257, 2223, 1103, 1120, 9.24, 13, 7, 16, +3.5, -0.022, +0.040, 47.50, 55.00),
    ("Jordan Love", "GB", "Vet Ext ($55M)", 10, 7, 396, 55.3, 232, 368, 2838, 1688, 1152, 9.64, 20, 7, 22, +2.3, +0.058, +0.065, 44.50, 55.00),
    ("C.J. Stroud", "HOU", "Rookie (Yr 4)", 9, 8, 347, 54.6, 202, 323, 2462, 1526, 949, 9.25, 19, 3, 17, +1.4, +0.117, +0.015, 11.50, 9.07),
    ("Caleb Williams", "CHI", "Rookie (Yr 3)", 7, 8, 227, 43.9, 156, 227, 1546, 875, 671, 7.82, 8, 5, 13, +6.4, +0.020, +0.090, 10.80, 9.87),
    ("Kirk Cousins", "MIN/LV", "Vet Bridge", 10, 7, 369, 37.9, 223, 346, 2547, 1434, 1113, 8.01, 22, 9, 17, +1.3, +0.062, -0.030, 27.50, 30.00),
    ("Patrick Mahomes", "KC", "Vet Restruct", 9, 8, 328, 59.9, 199, 298, 2385, 836, 1552, 6.08, 19, 4, 12, +2.4, +0.105, +0.040, 52.50, 63.10),
    ("Matthew Stafford", "LAR", "Vet Ext ($55M)", 10, 7, 390, 33.6, 248, 372, 2622, 1325, 1297, 7.83, 19, 9, 14, -0.2, +0.073, +0.030, 49.50, 55.00),
    ("Lamar Jackson", "BAL", "Vet Ext ($52M)", 5, 6, 112, 41.1, 68, 97, 967, 540, 427, 10.39, 6, 1, 7, +7.4, +0.040, +0.010, 51.20, 52.00),
    ("Geno Smith", "NYJ", "Vet Bridge", 10, 7, 333, 25.1, 196, 290, 2226, 1196, 1030, 7.70, 12, 3, 25, +4.1, +0.087, +0.010, 26.50, 25.00),
    ("Jayden Daniels", "WAS", "Rookie (Yr 3)", 8, 8, 258, 21.5, 163, 252, 1499, 828, 671, 8.45, 10, 2, 17, +4.8, +0.050, +0.035, 10.30, 9.44),
    ("Bryce Young", "CAR", "Rookie (Yr 4)", 9, 8, 376, 18.9, 203, 338, 2397, 1329, 1069, 8.27, 17, 6, 28, +0.3, +0.075, +0.065, 12.10, 9.49),
    ("Tyler Shough", "NO", "Rookie (Yr 2)", 9, 8, 385, 1.6, 229, 343, 2172, 1244, 929, 7.50, 15, 7, 25, +1.1, +0.055, +0.050, 2.50, 2.30),
    ("Drew Lock", "SEA", "Vet Backup", 4, 4, 94, 16.7, 59, 82, 720, 423, 304, 9.10, 5, 0, 5, +5.5, +0.015, +0.035, 4.50, 4.50),
    ("Deshaun Watson", "CLE", "Fully Gtd*", 9, 7, 354, 1.0, 191, 292, 1909, 924, 985, 6.60, 8, 5, 35, +0.2, +0.088, -0.045, 44.00, 46.00),
    ("Cam Ward", "TEN", "Rookie (Yr 2)", 9, 8, 338, -3.7, 203, 314, 2021, 990, 1031, 6.80, 13, 7, 27, -0.5, +0.095, -0.010, 9.80, 9.80),
    ("Bo Nix", "DEN", "Rookie (Yr 3)", 9, 8, 310, 6.4, 174, 287, 1673, 683, 990, 5.60, 11, 7, 15, -1.1, -0.050, -0.010, 5.10, 4.65),
    ("J.J. McCarthy", "NYG", "Rookie (Yr 3)", 3, 7, 89, 9.7, 51, 80, 608, 297, 311, 7.60, 3, 2, 7, +1.5, +0.008, +0.010, 6.20, 5.46),
    ("Sam Darnold", "SEA", "Vet Bridge", 8, 3, 237, -6.9, 128, 212, 1356, 730, 626, 6.60, 9, 7, 18, -2.5, +0.040, -0.010, 18.50, 18.50),
    ("Michael Penix Jr.", "ATL", "Rookie (Yr 3)", 8, 7, 266, -8.7, 147, 242, 1730, 814, 916, 7.02, 10, 7, 12, -4.1, +0.038, -0.055, 8.90, 7.22),
    ("Jacoby Brissett", "BAL/ARI", "Vet Bridge", 9, 2, 385, -18.8, 226, 354, 2059, 1142, 1160, 7.90, 10, 8, 21, -1.2, +0.165, +0.040, 12.50, 12.50),
    ("Aaron Rodgers", "PIT", "Vet 1-Yr Deal", 9, 8, 345, -34.5, 190, 317, 1827, 796, 1038, 7.20, 12, 7, 23, -2.1, +0.022, -0.085, 22.50, 22.50),
    ("Justin Herbert", "LAC", "Vet Ext ($52.5M)", 9, 8, 314, -70.1, 158, 261, 1546, 932, 614, 7.37, 8, 9, 31, -4.4, +0.088, +0.130, 46.50, 52.50),
    ("Baker Mayfield", "TB", "Vet Ext ($55M)", 9, 8, 170, -44.0, 86, 142, 922, 380, 542, 7.50, 3, 4, 18, -3.5, +0.040, -0.010, 36.80, 55.00),
    ("Daniel Jones", "IND", "Vet Ext ($44M)", 10, 7, 362, -75.4, 193, 324, 1830, 889, 941, 7.27, 10, 11, 22, -3.9, +0.015, -0.025, 24.50, 44.00),
]


def load_default_data() -> pd.DataFrame:
    """Returns the built-in Week 10 dataset as a DataFrame."""
    return pd.DataFrame(WEEK_10_2026_MASTER_DATA, columns=COLUMNS)


def validate_data(df: pd.DataFrame) -> None:
    """Raises a clear error if columns are missing or key fields are unusable."""
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    for c in ("G", "DB", "Cmp", "Att"):
        if (df[c] <= 0).any():
            bad = df.loc[df[c] <= 0, "QB"].tolist()
            raise ValueError(f"Column '{c}' must be > 0 (check: {bad})")
    if (df["Cmp"] > df["Att"]).any():
        bad = df.loc[df["Cmp"] > df["Att"], "QB"].tolist()
        raise ValueError(f"Completions exceed attempts for: {bad}")


# ==============================================================================
# 3. CORE QVAR 4.7 ENGINE
# ==============================================================================
def run_qvar_47_model(df: pd.DataFrame, cfg: dict = None) -> pd.DataFrame:
    """
    Computes True Play-by-Play QVAR 4.7 and 2026 contract surplus value.

    True_QVAR_47 = Base_PBP_WAR + xYAC_WAR_Adj + Sack_INT_Adj + Tip_WP_WAR + dSOS_Adj
    """
    cfg = {**DEFAULT_CONFIG, **(cfg or {})}
    validate_data(df)

    out = df.copy()
    epa_per_war = cfg["epa_per_war"]

    # --- Descriptive splits ---------------------------------------------------
    out["EPA_DB"] = (out["Tot_EPA"] / out["DB"]).round(3)
    out["CAY_Cmp"] = (out["CAY"] / out["Cmp"]).round(2)
    out["YAC_Cmp"] = (out["YAC"] / out["Cmp"]).round(2)
    out["YAC_Pct"] = (out["YAC"] / (out["CAY"] + out["YAC"]).replace(0, np.nan) * 100).round(1)

    # --- Pillar 1: Base WAR over replacement ---------------------------------
    out["Base_PBP_WAR"] = ((out["Tot_EPA"] - cfg["repl_epa_per_db"] * out["DB"]) / epa_per_war).round(3)

    # --- Pillar 2: YAC-over-expected discount --------------------------------
    # Shorter throws naturally generate more YAC, so expectation falls as ADoT rises.
    exp_yac_per_cmp = 5.20 - 0.12 * (out["ADoT"] - 7.80)
    excess_yac_yds = out["YAC"] - out["Cmp"] * exp_yac_per_cmp
    out["xYAC_WAR_Adj"] = (
        -cfg["yacoe_discount"] * excess_yac_yds * cfg["epa_per_yac_yd"] / epa_per_war
    ).round(3)

    # --- Pillar 3: Sack & INT variance adjustment (symmetric, mean-centered) --
    # Expected rates are calibrated to the pooled sample so the adjustment nets ~0
    # across the league. QBs with MORE sacks/INTs than expected get partial credit
    # (some of it was line/luck); QBs with FEWER get docked by the same logic.
    lg_sack_rate = out["Sacks"].sum() / out["DB"].sum()
    lg_int_rate = out["INT"].sum() / out["Att"].sum()
    exp_sacks = out["DB"] * lg_sack_rate
    exp_ints = out["Att"] * lg_int_rate

    sack_adj = (out["Sacks"] - exp_sacks) * cfg["epa_per_sack_lost"] * cfg["sack_line_share"]
    int_adj = (out["INT"] - exp_ints) * cfg["epa_per_int_lost"] * cfg["int_luck_share"]
    out["Sack_INT_Adj"] = ((sack_adj + int_adj) / epa_per_war).round(3)

    # Tip / win-probability (garbage-time) adjustment is supplied directly in WAR.
    out["Tip_WP_WAR"] = out["Tip_WP_WAR"].round(3)

    # --- Pillar 4: Leave-one-out pass-defense strength of schedule -----------
    out["dSOS_Adj"] = (out["dSOS_WAR"] * cfg["dsos_multiplier"]).round(3)

    # --- Total ---------------------------------------------------------------
    out["True_QVAR_47"] = (
        out["Base_PBP_WAR"] + out["xYAC_WAR_Adj"] + out["Sack_INT_Adj"]
        + out["Tip_WP_WAR"] + out["dSOS_Adj"]
    ).round(3)
    out["WAR_47_PG"] = (out["True_QVAR_47"] / out["G"]).round(3)

    # --- Projection with shrinkage toward league-mean per-game rate -----------
    k = cfg["shrinkage_games"]
    mean_rate = out["WAR_47_PG"].mean()
    shrunk_rate = (out["G"] * out["WAR_47_PG"] + k * mean_rate) / (out["G"] + k)
    out["Proj_Rate_PG"] = shrunk_rate.round(3)
    out["Proj_Full_WAR"] = (out["True_QVAR_47"] + out["G_Rem"] * shrunk_rate).round(2)

    # --- Contract valuation ---------------------------------------------------
    out["Fair_Mkt_Val_M"] = (cfg["repl_qb_cost_m"] + out["Proj_Full_WAR"] * cfg["dollars_per_war_m"]).round(2)
    out["Fair_Mkt_Val_M"] = out["Fair_Mkt_Val_M"].clip(lower=0)   # a QB can't be worth negative $
    out["Surplus_vs_Cap_M"] = (out["Fair_Mkt_Val_M"] - out["Cap_2026_M"]).round(2)
    out["Surplus_vs_AAV_M"] = (out["Fair_Mkt_Val_M"] - out["AAV_M"]).round(2)
    out["Cap_Pct"] = (out["Cap_2026_M"] / cfg["nfl_cap_2026_m"] * 100).round(2)

    out = out.sort_values("True_QVAR_47", ascending=False).reset_index(drop=True)
    out.index = out.index + 1
    out.index.name = "Rank"
    return out


# ==============================================================================
# 4. EXPORT HELPERS (CSV / JSON / SUBSTACK TABLE)
# ==============================================================================
SUBSTACK_COLS = ["QB", "Team", "G", "EPA_DB", "CPOE", "YAC_Pct", "True_QVAR_47",
                 "Proj_Full_WAR", "Cap_2026_M", "Fair_Mkt_Val_M", "Surplus_vs_Cap_M"]


def export_all(results: pd.DataFrame, prefix: str = "qvar_47") -> list:
    """Writes every file you'd need for Substack, Datawrapper, or a website."""
    files = []
    results.to_csv(f"{prefix}_full_leaderboard.csv", index=True); files.append(f"{prefix}_full_leaderboard.csv")
    results.sort_values("Surplus_vs_Cap_M", ascending=False).to_csv(f"{prefix}_cap_surplus.csv", index=False)
    files.append(f"{prefix}_cap_surplus.csv")
    results[SUBSTACK_COLS].to_csv(f"{prefix}_substack_table.csv", index=True); files.append(f"{prefix}_substack_table.csv")
    # JSON for a website front-end (D3, Chart.js, React, etc.)
    with open(f"{prefix}_data.json", "w") as f:
        json.dump(json.loads(results.reset_index().to_json(orient="records")), f, indent=2)
    files.append(f"{prefix}_data.json")
    return files


# ==============================================================================
# 5. STREAMLIT DASHBOARD
# ==============================================================================
def _show_df(st, data, **kw):
    """Works on both old and new Streamlit (use_container_width vs width='stretch')."""
    try:
        st.dataframe(data, width="stretch", **kw)
    except TypeError:
        st.dataframe(data, use_container_width=True, **kw)


def run_streamlit_dashboard():
    import streamlit as st

    st.set_page_config(page_title="QVAR 4.7 NFL QB Valuation Engine", layout="wide")
    st.title("🏈 True Play-by-Play QVAR 4.7 & Cap Surplus Engine (2026)")
    st.caption("Play-by-play EPA, Air-vs-YAC discounting, symmetric sack/INT variance adjustment, "
               "win-probability adjustment, and leave-one-out pass-defense strength of schedule.")

    st.sidebar.header("⚙️ Model Weights")
    yac_disc = st.sidebar.slider("YAC-over-expected discount", 0.0, 1.0, DEFAULT_CONFIG["yacoe_discount"], 0.05)
    sack_share = st.sidebar.slider("Sack share blamed on line (not QB)", 0.0, 1.0, DEFAULT_CONFIG["sack_line_share"], 0.05)
    int_share = st.sidebar.slider("INT share blamed on luck", 0.0, 1.0, DEFAULT_CONFIG["int_luck_share"], 0.05)
    dsos_mult = st.sidebar.slider("Pass-defense dSOS weight", 0.0, 2.0, DEFAULT_CONFIG["dsos_multiplier"], 0.1)
    shrink = st.sidebar.slider("Projection shrinkage (games)", 0.0, 16.0, DEFAULT_CONFIG["shrinkage_games"], 1.0)
    war_price = st.sidebar.number_input("Market price per 1.0 WAR ($M)", 8.0, 20.0, DEFAULT_CONFIG["dollars_per_war_m"], 0.25)

    cfg = {
        "yacoe_discount": yac_disc, "sack_line_share": sack_share, "int_luck_share": int_share,
        "dsos_multiplier": dsos_mult, "shrinkage_games": shrink, "dollars_per_war_m": war_price,
    }

    uploaded = st.sidebar.file_uploader("Optional: upload your own data (CSV, same columns)", type="csv")
    df_raw = pd.read_csv(uploaded) if uploaded is not None else load_default_data()

    try:
        results = run_qvar_47_model(df_raw, cfg)
    except ValueError as e:
        st.error(f"Data problem: {e}")
        st.stop()

    tab1, tab2, tab3 = st.tabs(["📊 Leaderboard", "💰 Contract Surplus", "📈 Charts"])

    with tab1:
        cols = ["QB", "Team", "G", "DB", "EPA_DB", "CPOE", "CAY_Cmp", "YAC_Cmp", "YAC_Pct",
                "Base_PBP_WAR", "xYAC_WAR_Adj", "Sack_INT_Adj", "Tip_WP_WAR", "dSOS_Adj",
                "True_QVAR_47", "WAR_47_PG"]
        _show_df(st, results[cols], height=650)

    with tab2:
        cap_cols = ["QB", "Team", "Contract", "True_QVAR_47", "Proj_Full_WAR",
                    "Fair_Mkt_Val_M", "Cap_2026_M", "Surplus_vs_Cap_M"]
        a, b = st.columns(2)
        with a:
            st.subheader("🟢 Top 12 Surplus Contracts")
            _show_df(st, results.sort_values("Surplus_vs_Cap_M", ascending=False)[cap_cols].head(12))
        with b:
            st.subheader("🔴 Bottom 12 Contracts")
            _show_df(st, results.sort_values("Surplus_vs_Cap_M")[cap_cols].head(12))

    with tab3:
        st.subheader("True QVAR 4.7 by Quarterback")
        st.bar_chart(results.set_index("QB")["True_QVAR_47"])
        st.subheader("Fair Market Value vs. 2026 Cap Hit ($M)")
        st.scatter_chart(results, x="Cap_2026_M", y="Fair_Mkt_Val_M")
        st.subheader("Contract Surplus ($M)")
        st.bar_chart(results.sort_values("Surplus_vs_Cap_M").set_index("QB")["Surplus_vs_Cap_M"])

    st.download_button("⬇️ Download full results (CSV)", results.to_csv().encode(), "qvar_47_results.csv", "text/csv")


# ==============================================================================
# 6. ENTRYPOINT
# ==============================================================================
def _running_in_streamlit() -> bool:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from streamlit.runtime.scriptrunner import get_script_run_ctx
            return get_script_run_ctx() is not None
    except Exception:
        return False


if __name__ == "__main__":
    if _running_in_streamlit():
        run_streamlit_dashboard()
    else:
        board = run_qvar_47_model(load_default_data())
        written = export_all(board)
        print("Exported:", ", ".join(written), "\n")
        pd.set_option("display.width", 220)
        print(board[["QB", "Team", "G", "EPA_DB", "YAC_Pct", "Sack_INT_Adj",
                     "True_QVAR_47", "Proj_Full_WAR", "Fair_Mkt_Val_M", "Surplus_vs_Cap_M"]].to_string())
