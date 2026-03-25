#!/usr/bin/env python
# -*- coding: utf-8 -*-
import glob
import argparse, os, json
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import pandas as pd
import optuna
from datetime import timedelta

# ---------------------------
# GLOBAL CONFIGURATION
# ---------------------------
DEFAULT_TZ = "America/New_York"
LONDON_START, LONDON_END = "03:00", "07:00"
NY_START, NY_END = "07:00", "12:00"
RISK_PER_TRADE = 100.0
FLASH_TRADE_PENALTY = 1000

# --- Walk-Forward Optimization (WFO) Parameters ---
# WFO is critical to avoid optimization look-ahead bias (overfitting).
WFO_TRAINING_MONTHS = 1  # Data used for Optuna optimization
WFO_TEST_MONTHS = 1      # Unseen data used to test the best parameters

# --- Realistic Stop Loss Buffer ---
# This is crucial to prevent impossibly tight stops and huge RR ratios.
# Set this to your typical spread + small buffer (e.g., 1 pip or 10 points for XAUUSD)
SL_BUFFER_POINTS = 0.00010 

# ---------------------------
# DATA STRUCTURES
# ---------------------------
@dataclass
class Trade:
    day: pd.Timestamp
    side: str
    entry_time: pd.Timestamp
    entry_price: float
    sl: float
    tp: float
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl_dollars: float = 0.0
    rr: float = 0.0
    sweep_level: float = 0.0
    sweep_time: Optional[pd.Timestamp] = None
    bos_time: Optional[pd.Timestamp] = None
    risk_in_price: float = 0.0
    four_h_trend: str = "NA"
    one_h_trend: str = "NA"
    sl_anchor_price: float = 0.0
    bos_pivot_level: float = 0.0
    sl_buffer: float = SL_BUFFER_POINTS
    wfo_step: int = 0


# ---------------------------
# UTILITY FUNCTIONS
# ---------------------------

def parse_args():
    p = argparse.ArgumentParser("ICT-style London Sweep & BOS backtest (1m)")
    p.add_argument("--datadir", required=True, help="Path to the directory containing monthly CSV data files.") # <-- CHANGE
    p.add_argument("--tz", default=DEFAULT_TZ, help="Timezone of the timestamps.")
    p.add_argument("--outdir", default="./out", help="Where to write results")
    p.add_argument("--london", default=f"{LONDON_START}-{LONDON_END}", help="London window HH:MM-HH:MM")
    p.add_argument("--ny", default=f"{NY_START}-{NY_END}", help="NY window HH:MM-HH:MM")
    p.add_argument("--risk", type=float, default=RISK_PER_TRADE)
    return p.parse_args()


def parse_window(s: str) -> Tuple[pd.Timedelta, pd.Timedelta]:
    def fix_time(t: str):
        parts = t.split(":")
        if len(parts) == 1: t = f"{parts[0]}:00:00"
        elif len(parts) == 2: t = f"{parts[0]}:{parts[1]}:00"
        return pd.to_timedelta(t)
    a, b = s.split("-")
    return fix_time(a), fix_time(b)

import glob # <-- ADD THIS IMPORT AT THE TOP
# ... (other imports)
from datetime import timedelta

# ---------------------------
# UTILITY FUNCTIONS (REPLACE THE ENTIRE FUNCTION)
# ---------------------------
# ... (parse_args, parse_window functions)

def load_mt_csv(dir_path: str, tz: str) -> pd.DataFrame:
    """
    Loads and combines multiple CSV files from a directory, sorts them by name, 
    and performs the standard data cleaning/indexing.
    """
    all_files = sorted(glob.glob(os.path.join(dir_path, "*.csv")))
    
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in the directory: {dir_path}")

    dfs = []
    
    # Standard column names expected from your current function:
    col_names=["Date","Time","Open","High","Low","Close","Volume"]
    
    for path in all_files:
        print(f"   -> Reading {os.path.basename(path)}")
        df = pd.read_csv(path, header=None, names=col_names)
        df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format="%Y.%m.%d %H:%M")
        dfs.append(df)

    # Concatenate all DataFrames
    df_combined = pd.concat(dfs, ignore_index=True)
    
    # Set index and localize timezone on the combined data
    df_combined.set_index('DateTime', inplace=True)
    # Ensure correct time order and drop duplicates (which can happen at month boundaries)
    df_combined.sort_index(inplace=True) 
    df_combined.drop_duplicates(inplace=True) 
    
    # Apply timezone localization
    # Use 'NaT' handling as per your original script
    df_combined.index = df_combined.index.tz_localize(tz, nonexistent="shift_forward", ambiguous="NaT")
    
    print(f"Successfully combined {len(all_files)} files into a single DataFrame.")
    
    return df_combined[["Open","High","Low","Close"]].astype(float).sort_index()


# --- FIX: Pivot Calculation (Lagging Window) ---
def pivot_highs_lows(df: pd.DataFrame, k: int) -> Tuple[pd.Series, pd.Series]:
    """Identifies confirmed pivot highs/lows using a lagging window (no look-ahead)."""
    if k % 2 == 0: raise ValueError("pivot window must be odd")
    window_size = 2 * k + 1
    
    # Calculate max/min in the full confirmation window
    roll_max = df["High"].rolling(window=window_size, min_periods=window_size).max()
    roll_min = df["Low"].rolling(window=window_size, min_periods=window_size).min()
    
    # We shift the OHLC data forward by k to align with the *confirmed* time index.
    is_ph = (df["High"].shift(-k) == roll_max.shift(-k)).fillna(False)
    is_pl = (df["Low"].shift(-k) == roll_min.shift(-k)).fillna(False)
    
    return is_ph, is_pl

def session_slice(day_df: pd.DataFrame, start: pd.Timedelta, end: pd.Timedelta) -> pd.DataFrame:
    d0 = day_df.index[0].normalize()
    s, e = d0 + start, d0 + end
    return day_df.loc[(day_df.index>=s) & (day_df.index<=e)]

def calculate_trends(df_1m: pd.DataFrame, pivot_k_1h=7, pivot_k_4h=7):
    # This remains the same, but relies on the fixed pivot_highs_lows.
    df_1h = df_1m.resample("1h").agg({'Open':'first','High':'max','Low':'min','Close':'last'}).dropna()
    df_4h = df_1m.resample("4h").agg({'Open':'first','High':'max','Low':'min','Close':'last'}).dropna()

    def detect_trend(df, pivot_k):
        is_ph, is_pl = pivot_highs_lows(df, pivot_k)
        trend_dict = {}
        last_trend = None
        for t in df.index:
            ph_t_series = is_ph.loc[:t][is_ph.loc[:t]].index
            pl_t_series = is_pl.loc[:t][is_pl.loc[:t]].index

            if len(ph_t_series) and len(pl_t_series):
                last_ph_t = ph_t_series[-1]
                last_pl_t = pl_t_series[-1]
                if last_ph_t > last_pl_t: last_trend = "Downtrend"
                elif last_pl_t > last_ph_t: last_trend = "Uptrend"
            trend_dict[t] = last_trend
        return trend_dict

    trend_1h_full = pd.Series(detect_trend(df_1h, pivot_k_1h)).reindex(df_1m.index, method='ffill')
    trend_4h_full = pd.Series(detect_trend(df_4h, pivot_k_4h)).reindex(df_1m.index, method='ffill')

    return pd.DataFrame({'1h_trend': trend_1h_full, '4h_trend': trend_4h_full}).ffill()

def is_valid_trend(row, tf_filter):
    trend_4h = row.get("4h_trend")
    trend_1h = row.get("1h_trend")
    
    if tf_filter == "none": return True, True
    elif tf_filter == "1h": return trend_1h == "Uptrend", trend_1h == "Downtrend"
    elif tf_filter == "4h": return trend_4h == "Uptrend", trend_4h == "Downtrend"
    elif tf_filter == "1h+4h": return (trend_1h == "Uptrend" and trend_4h == "Uptrend"), (trend_1h == "Downtrend" and trend_4h == "Downtrend")
    return False, False


# ---------------------------
# CORE BACKTEST LOGIC
# ---------------------------

def backtest_london_sweep_bos(df, london_window, ny_window, pivot, sl_mult, tp_mult, tf_filter, risk_per_trade, wfo_step=0):
    
    trades: List[Trade] = []
    flash_trades = 0
    total_days_triggered = 0
    
    # Calculate trends and pivots once for the entire window
    trend_df = calculate_trends(df)
    df = df.join(trend_df)
    is_ph_full, is_pl_full = pivot_highs_lows(df, pivot) 

    for day, day_df in df.groupby(df.index.date):
        day_df = df.loc[str(day)]
        if day_df.empty: continue

        lon = session_slice(day_df, *london_window)
        if lon.empty: continue
        london_high, london_low = float(lon["High"].max()), float(lon["Low"].min())

        ny = session_slice(day_df, *ny_window)
        if ny.empty: continue
        
        # Sweep Detection (Find the FIRST breach)
        bullish_sweep_mask = ny["Low"] < london_low
        bullish_sweep_time = bullish_sweep_mask[bullish_sweep_mask].index[0] if bullish_sweep_mask.any() else None
        bearish_sweep_mask = ny["High"] > london_high
        bearish_sweep_time = bearish_sweep_mask[bearish_sweep_mask].index[0] if bearish_sweep_mask.any() else None
        
        sweep_taken = False
        
        for t, row in ny.iterrows():
            if sweep_taken: break
            is_bullish, is_bearish = is_valid_trend(row, tf_filter)
            
            # --- Bullish Entry Check (Long) ---
            if bullish_sweep_time is not None and t >= bullish_sweep_time and is_bullish:
                sweep_range = ny.loc[bullish_sweep_time:t]
                sl_anchor_price = float(sweep_range["Low"].min())
                
                ph_indices = is_ph_full.loc[bullish_sweep_time:t]
                if len(ph_indices[ph_indices]) > 0:
                    last_ph_t = ph_indices[ph_indices].index[-1]
                    bos_pivot_level = float(df.loc[last_ph_t, "High"])
                    
                    if row["Close"] > bos_pivot_level:
                        entry_time, entry_price = t, float(row["Close"])
                        
                        # FIX: Apply SL Buffer and calculate final SL
                        sl_raw_anchor = sl_anchor_price - SL_BUFFER_POINTS
                        risk_in_price_anchor = abs(entry_price - sl_raw_anchor)
                        risk_in_price = risk_in_price_anchor * sl_mult
                        sl = entry_price - risk_in_price
                        
                        risk_price_diff = abs(entry_price - sl) # This is the final 1R risk in price

                        # --- CORRECTED TP LINE ---
                        tp = entry_price + (risk_price_diff * tp_mult)
                        # -------------------------

                        trade_size = risk_per_trade / risk_price_diff if risk_price_diff > 0 else 0
                        
                        trade = Trade(
                            day=pd.Timestamp(day), side="long", entry_time=entry_time, entry_price=entry_price,
                            sl=sl, tp=tp, sweep_level=london_low, sweep_time=bullish_sweep_time, bos_time=t,
                            risk_in_price=risk_in_price, four_h_trend=row.get("4h_trend", "NA"), 
                            one_h_trend=row.get("1h_trend", "NA"), sl_anchor_price=sl_anchor_price, 
                            bos_pivot_level=bos_pivot_level, wfo_step=wfo_step
                        )
                        sweep_taken = True
                        break

            # --- Bearish Entry Check (Short) ---
            if bearish_sweep_time is not None and t >= bearish_sweep_time and is_bearish:
                sweep_range = ny.loc[bearish_sweep_time:t]
                sl_anchor_price = float(sweep_range["High"].max())

                pl_indices = is_pl_full.loc[bearish_sweep_time:t]
                if len(pl_indices[pl_indices]) > 0:
                    last_pl_t = pl_indices[pl_indices].index[-1]
                    bos_pivot_level = float(df.loc[last_pl_t, "Low"])
                    
                    if row["Close"] < bos_pivot_level:
                        entry_time, entry_price = t, float(row["Close"])
                        
                        # FIX: Apply SL Buffer and calculate final SL
                        sl_raw_anchor = sl_anchor_price + SL_BUFFER_POINTS
                        risk_in_price_anchor = abs(sl_raw_anchor - entry_price)
                        risk_in_price = risk_in_price_anchor * sl_mult
                        sl = entry_price + risk_in_price

                        risk_price_diff = abs(sl - entry_price) # This is the final 1R risk in price

                        # --- CORRECTED TP LINE ---
                        tp = entry_price - (risk_price_diff * tp_mult) 
                        # -------------------------

                        trade_size = risk_per_trade / risk_price_diff if risk_price_diff > 0 else 0
                            
                        trade = Trade(
                            day=pd.Timestamp(day), side="short", entry_time=entry_time, entry_price=entry_price,
                            sl=sl, tp=tp, sweep_level=london_high, sweep_time=bearish_sweep_time, bos_time=t,
                            risk_in_price=risk_in_price, four_h_trend=row.get("4h_trend", "NA"), 
                            one_h_trend=row.get("1h_trend", "NA"), sl_anchor_price=sl_anchor_price, 
                            bos_pivot_level=bos_pivot_level, wfo_step=wfo_step
                        )
                        sweep_taken = True
                        break

        # 3. Process Trade Exit (Robust High/Low Check)
        if sweep_taken:
            total_days_triggered += 1
            exit_time, exit_price, exit_reason = None, None, None
            
            risk_price_diff = abs(trade.entry_price - trade.sl)
            trade_size = risk_per_trade / risk_price_diff if risk_price_diff > 0 else 0
            
            post = ny.loc[ny.index >= trade.entry_time] 

            for tt, rr in post.iterrows():
                
                # Check for SL (Long: Low <= SL, Short: High >= SL)
                if trade.side == "long" and rr["Low"] <= trade.sl:
                    exit_time, exit_price, exit_reason = tt, trade.sl, "sl"
                    pnl_dollars = (trade.sl - trade.entry_price) * trade_size
                    rr_val = pnl_dollars / risk_per_trade
                    break
                elif trade.side == "short" and rr["High"] >= trade.sl:
                    exit_time, exit_price, exit_reason = tt, trade.sl, "sl"
                    pnl_dollars = (trade.entry_price - trade.sl) * trade_size
                    rr_val = pnl_dollars / risk_per_trade
                    break

                # Check for TP (Long: High >= TP, Short: Low <= TP)
                if trade.side == "long" and rr["High"] >= trade.tp:
                    exit_time, exit_price, exit_reason = tt, trade.tp, "tp"
                    pnl_dollars = (trade.tp - trade.entry_price) * trade_size
                    rr_val = pnl_dollars / risk_per_trade
                    break
                elif trade.side == "short" and rr["Low"] <= trade.tp:
                    exit_time, exit_price, exit_reason = tt, trade.tp, "tp"
                    pnl_dollars = (trade.entry_price - trade.tp) * trade_size
                    rr_val = pnl_dollars / risk_per_trade
                    break
            
            # Close at end of NY session if no exit
            if exit_time is None:
                exit_time = ny.index[-1]
                exit_price = float(ny.iloc[-1]["Close"])
                exit_reason = "close"
                pnl_dollars = (exit_price - trade.entry_price) * trade_size if trade.side == "long" else (trade.entry_price - exit_price) * trade_size
                rr_val = pnl_dollars / risk_per_trade 
            
            if exit_time == trade.entry_time: flash_trades += 1
            
            trade.exit_time, trade.exit_price, trade.exit_reason = exit_time, exit_price, exit_reason
            trade.pnl_dollars, trade.rr = pnl_dollars, rr_val
            trades.append(trade)

    if trades:
        tdf = pd.DataFrame([asdict(t) for t in trades])
        wins = int((tdf["exit_reason"] == "tp").sum())
        losses = int((tdf["exit_reason"] == "sl").sum())
        closes = int((tdf["exit_reason"] == "close").sum())
        wr = 100.0 * wins / max(1, wins + losses)
        summary = {
            "trades": len(tdf), "wins": wins, "losses": losses, "closed_no_hit": closes,
            "win_rate_pct": wr, "net_pnl_dollars": float(tdf["pnl_dollars"].sum()), 
            "days_triggered": total_days_triggered, "flash_trades": flash_trades
        }
    else:
        tdf = pd.DataFrame()
        summary = { "trades": 0, "wins": 0, "losses": 0, "closed_no_hit": 0, "win_rate_pct": 0.0, 
                    "net_pnl_dollars": 0.0, "days_triggered": 0, "flash_trades": flash_trades }
    
    return tdf, summary

# ---------------------------
# OPTUNA & WFO FUNCTIONS
# ---------------------------

def objective(trial, df_train, london_window, ny_window, risk_per_trade, wfo_step):
    params = {
        "pivot": trial.suggest_int("pivot", 3, 19, step=2),
        "sl_mult": trial.suggest_float("sl_mult", 0.5, 3.0),
        "tp_mult": trial.suggest_float("tp_mult", 0.5, 5.0), 
        "tf_filter": trial.suggest_categorical("tf_filter", ["none", "1h", "4h", "1h+4h"]),
    }
    
    trades_df, summary = backtest_london_sweep_bos(
        df_train, london_window=london_window, ny_window=ny_window, **params,
        risk_per_trade=risk_per_trade, wfo_step=wfo_step
    )
    
    penalty = summary["flash_trades"] * FLASH_TRADE_PENALTY
    return summary["net_pnl_dollars"] - penalty

# ---------------------------
# OPTUNA & WFO FUNCTIONS (REPLACE THE ENTIRE FUNCTION)
# ---------------------------

def walk_forward_optimization(df, london_window, ny_window, risk_per_trade, outdir):
    all_test_trades = []
    wfo_summaries: List[Dict[str, Any]] = [] # New list for monthly setup summaries
    wfo_step = 1
    start_date = df.index.min().normalize()
    current_train_end = start_date + pd.DateOffset(months=WFO_TRAINING_MONTHS)
    
    print("\nStarting Walk-Forward Optimization...")
    print(f"Training Window: {WFO_TRAINING_MONTHS} Months | Testing Window: {WFO_TEST_MONTHS} Months")
    print("-" * 50)
    
    while current_train_end < df.index.max():
        train_start = current_train_end - pd.DateOffset(months=WFO_TRAINING_MONTHS)
        test_end = current_train_end + pd.DateOffset(months=WFO_TEST_MONTHS)
        
        # Check to ensure the test period does not go beyond the end of the data
        if test_end > df.index.max(): 
            test_end = df.index.max()
        
        df_train_wfo = df.loc[train_start:current_train_end - timedelta(minutes=1)]
        df_test_wfo = df.loc[current_train_end:test_end]
        
        # Break if training or testing data is too small or missing
        if df_train_wfo.empty or df_test_wfo.empty or len(df_test_wfo) < 10: break

        print(f"🔄 WFO Step {wfo_step}: Optimizing from {train_start.date()} to {current_train_end.date()}")
        
        study = optuna.create_study(
            direction="maximize",
            storage=f"sqlite:///{os.path.join(outdir, f'study_{wfo_step}.db')}",
            study_name=f"wfo_step_{wfo_step}", load_if_exists=False
        )
        study.optimize(
            lambda trial: objective(trial, df_train_wfo, london_window, ny_window, risk_per_trade, wfo_step), 
            n_trials=50
        )
        
        best_params = study.best_params
        best_objective_value = study.best_value
        
        print(f"   ✅ Best parameters found: {best_params}")
        
        print(f"   🧪 Testing on UNSEEN data from {current_train_end.date()} to {test_end.date()}")
        
        trades_test_df, summary_test = backtest_london_sweep_bos(
            df_test_wfo, london_window=london_window, ny_window=ny_window, 
            **best_params, risk_per_trade=risk_per_trade, wfo_step=wfo_step
        )
        
        if not trades_test_df.empty: 
            # 1. Save the specific test month's trade log
            month_filename = f"WFO_Step_{wfo_step}_Trades_Test_{current_train_end.strftime('%Y-%m')}.csv"
            trades_test_df.to_csv(os.path.join(outdir, month_filename), index=False)
            all_test_trades.append(trades_test_df)

            # 2. Collect the setup/summary for the master file
            wfo_summaries.append({
                "wfo_step": wfo_step,
                "train_end_date": current_train_end.date().isoformat(),
                "test_start_date": current_train_end.date().isoformat(),
                "test_end_date": test_end.date().isoformat(),
                "best_pnl_in_sample": best_objective_value,
                "net_pnl_out_of_sample": summary_test['net_pnl_dollars'],
                "trades_out_of_sample": summary_test['trades'],
                "best_params": json.dumps(best_params), # Use json.dumps to save dictionary as string
            })
            
        print(f"   💰 Test Profit: ${summary_test['net_pnl_dollars']:.2f}")
        
        # Roll the window forward by the length of the test period (1 month)
        current_train_end = current_train_end + pd.DateOffset(months=WFO_TEST_MONTHS) 
        wfo_step += 1
        
    # --- Final saving changes at the end of the function ---
    final_trades_df = pd.concat(all_test_trades, ignore_index=True) if all_test_trades else pd.DataFrame()
    
    # Save the master setup log
    if wfo_summaries:
        wfo_summary_df = pd.DataFrame(wfo_summaries)
        wfo_summary_df.to_csv(os.path.join(outdir, "WFO_Best_Setups_Summary.csv"), index=False)
        print(f"\nConfiguration summary saved to: {os.path.join(outdir, 'WFO_Best_Setups_Summary.csv')}")

    return final_trades_df

#2.3 advance version

def parameter_sweep(df, london_window, ny_window, risk_per_trade):
    results = []

    pivots = list(range(3, 17, 2))
    sl_mults = np.linspace(0.5, 2.0, 4)
    tp_mults = np.linspace(1.0, 4.0, 4)
    tf_filters = ["none", "1h", "4h"]

    total_runs = len(pivots)*len(sl_mults)*len(tp_mults)*len(tf_filters)
    run = 1

    for pivot in pivots:
        for sl in sl_mults:
            for tp in tp_mults:
                for tf in tf_filters:

                    print(f"Run {run}/{total_runs} → pivot={pivot}, sl={sl}, tp={tp}, tf={tf}")
                    run += 1

                    trades, summary = backtest_london_sweep_bos(
                        df,
                        london_window=london_window,
                        ny_window=ny_window,
                        pivot=pivot,
                        sl_mult=sl,
                        tp_mult=tp,
                        tf_filter=tf,
                        risk_per_trade=risk_per_trade
                    )

                    results.append({
                        "pivot": pivot,
                        "sl_mult": sl,
                        "tp_mult": tp,
                        "tf_filter": tf,
                        "trades": summary["trades"],
                        "winrate": summary["win_rate_pct"],
                        "pnl": summary["net_pnl_dollars"]
                    })

    return pd.DataFrame(results)


# def monte_carlo_simulation(trades_df, n_sim=500):
    equity_results = []

    rr = trades_df["rr"].values

    for i in range(n_sim):
        shuffled = np.random.permutation(rr)
        equity = np.cumsum(shuffled)
        equity_results.append(equity[-1])  # final result only

    equity_results = np.array(equity_results)

    print("\nMonte Carlo Results:")
    print(f"Mean final R: {equity_results.mean():.2f}")
    print(f"Min final R: {equity_results.min():.2f}")
    print(f"Max final R: {equity_results.max():.2f}")
    print(f"Probability of profit: {(equity_results > 0).mean()*100:.2f}%")

    return equity_results

# ---------------------------
# MAIN EXECUTION
# ---------------------------

# def main():
    print("Starting...")
    args = parse_args()
    london_window = parse_window(args.london)
    ny_window = parse_window(args.ny)

    os.makedirs(args.outdir, exist_ok=True)
    
    print("Loading data...")
    # --- CHANGE THIS LINE ---
    # Pass the data directory path to the loading function
    df = load_mt_csv(args.datadir, args.tz) 
    # ------------------------
    
    final_test_trades_df = walk_forward_optimization(
        df, london_window, ny_window, args.risk, args.outdir
    )

    print("\n" + "=" * 50)
    print("🚀 Walk-Forward Optimization Complete.")
    
    if not final_test_trades_df.empty:
        total_pnl = final_test_trades_df["pnl_dollars"].sum()
        
        final_test_trades_df.to_csv(os.path.join(args.outdir, "WFO_Trades_Test_Combined.csv"))
        
        print(f"📈 Total Out-of-Sample Net PnL (Test Data): **${total_pnl:.2f}**")
        print(f"Total Trades: {len(final_test_trades_df)}")
        print(f"Combined Test Trade Log saved to: {os.path.join(args.outdir, 'WFO_Trades_Test_Combined.csv')}")
    else:
        print("❌ No trades were executed in the test windows.")
    
    print("=" * 50)
    print("\nBacktest complete.")
    # df_results = parameter_sweep(df, london_window, ny_window, args.risk)
    # df_results.to_csv(os.path.join(args.outdir, "parameter_sweep.csv"), index=False)
def monte_carlo_simulation(trades_df, n_sim=500):
    rr = trades_df["rr"].values

    final_results = []
    max_drawdowns = []

    for _ in range(n_sim):
        shuffled = np.random.permutation(rr)
        equity = np.cumsum(shuffled)

        final_results.append(equity[-1])

        # ✅ Calculate drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = equity - peak
        max_dd = drawdown.min()

        max_drawdowns.append(max_dd)

    final_results = np.array(final_results)
    max_drawdowns = np.array(max_drawdowns)

    print("\nMonte Carlo Results:")
    print(f"Mean final R: {final_results.mean():.2f}")
    print(f"Worst final R: {final_results.min():.2f}")
    print(f"Best final R: {final_results.max():.2f}")
    print(f"Probability of profit: {(final_results > 0).mean()*100:.2f}%")
    print(f"Worst drawdown: {max_drawdowns.min():.2f}")
    print(f"Average drawdown: {max_drawdowns.mean():.2f}")

    return final_results, max_drawdowns
# def main():
#     print("Starting PARAMETER SWEEP...")
#     args = parse_args()
#     london_window = parse_window(args.london)
#     ny_window = parse_window(args.ny)

#     os.makedirs(args.outdir, exist_ok=True)

#     print("Loading data...")
#     df = load_mt_csv(args.datadir, args.tz)

#     print("Running parameter sweep...")
#     # df_results = parameter_sweep(df, london_window, ny_window, args.risk)

#     # df_results.to_csv(os.path.join(args.outdir, "parameter_sweep.csv"), index=False)

#     print("✅ Sweep complete.")

def main():
    print("Starting MONTE CARLO TEST...")
    args = parse_args()
    london_window = parse_window(args.london)
    ny_window = parse_window(args.ny)

    os.makedirs(args.outdir, exist_ok=True)

    print("Loading data...")
    df = load_mt_csv(args.datadir, args.tz)

    # ✅ FIXED STRATEGY (from your good zone)
    trades, summary = backtest_london_sweep_bos(
        df,
        london_window=london_window,
        ny_window=ny_window,
        pivot=11,
        sl_mult=1.0,
        tp_mult=3.0,
        tf_filter="none",
        risk_per_trade=args.risk
    )

    print("Backtest summary:", summary)

    trades.to_csv(os.path.join(args.outdir, "selected_trades.csv"), index=False)

    # ✅ RUN MONTE CARLO
    equity_curves = monte_carlo_simulation(trades)

    print("Monte Carlo completed. Check results manually.")
if __name__=="__main__":
    main()
    