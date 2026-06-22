"""Multi-sleeve portfolio of the locked Slopes strategy (a la TwoSleeves).

Three independent sleeves, each running the locked config
(up=20, down=80, vol-window=10, vol-exit=80, cooldown=20, park-stop=18%):

  Sleeve 1: TECL -> park XLK
  Sleeve 2: SPXL -> park SPY
  Sleeve 3: TQQQ -> park QQQ

Capital is split equally across sleeves and rebalanced back to equal weight
on the first trading bar of each calendar year. Each sleeve independently
rotates vehicle <-> park <-> cash via its own signal.

Benchmarks:
  - Equal-weight buy & hold of the three 3x vehicles (annually rebalanced)
  - The single TECL/XLK locked strategy (our flagship single sleeve)
"""

import argparse

import numpy as np
import pandas as pd

from slope_backtest import load_prices, backtest, stats

SLEEVES = [("TECL", "XLK"), ("SPXL", "SPY"), ("TQQQ", "QQQ")]
START = "2010-06-01"

# Locked config applied to every sleeve.
CFG = dict(up_window=20, down_window=80, vol_window=10, vol_exit_thresh=80.0,
           cooldown=20, park_stop=0.18)


def sleeve_returns(veh, prk):
    """Daily strategy return + vehicle (B&H) return for one sleeve, by date."""
    df = load_prices(f"{veh}.json")
    park = load_prices(f"{prk}.json").rename(columns={"price": "park_price"})
    df = df.merge(park, on="date", how="inner").reset_index(drop=True)
    bt = backtest(df, CFG["up_window"], CFG["down_window"], 100_000.0,
                  vol_window=CFG["vol_window"], vol_exit_thresh=CFG["vol_exit_thresh"],
                  cooldown=CFG["cooldown"], park_stop=CFG["park_stop"])
    bt = bt.dropna(subset=["slope"])
    return bt[["date", "strat_ret", "asset_ret", "in_market"]].rename(
        columns={"strat_ret": f"s_{veh}", "asset_ret": f"v_{veh}",
                 "in_market": f"in_{veh}"})


def equal_weight_curve(ret_matrix, dates, start=100_000.0):
    """Equal-weight N-sleeve equity, rebalanced to equal weight each new year."""
    n, k = ret_matrix.shape
    eq = np.full(k, start / k)
    total = np.empty(n)
    prev_year = dates.iloc[0].year
    for i in range(n):
        eq = eq * (1 + ret_matrix[i])
        t = eq.sum()
        if dates.iloc[i].year > prev_year:
            eq = np.full(k, t / k)          # annual rebalance to equal weight
            prev_year = dates.iloc[i].year
        total[i] = t
    return pd.Series(total, index=dates.index)


def curve_stats(curve, dates):
    ret = curve.pct_change().fillna(0.0)
    s = stats(curve, dates, ret)
    s["calmar"] = s["cagr_pct"] / abs(s["max_dd_pct"]) if s["max_dd_pct"] else float("nan")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-date", default=START)
    args = ap.parse_args()

    # Collect each sleeve's return series, align on common dates.
    merged = None
    for veh, prk in SLEEVES:
        sr = sleeve_returns(veh, prk)
        merged = sr if merged is None else merged.merge(sr, on="date", how="inner")
    merged = merged[merged["date"] >= pd.Timestamp(args.start_date)].reset_index(drop=True)
    dates = merged["date"]

    vehs = [v for v, _ in SLEEVES]
    strat_R = merged[[f"s_{v}" for v in vehs]].to_numpy()
    bh_R = merged[[f"v_{v}" for v in vehs]].to_numpy()

    port = equal_weight_curve(strat_R, dates)
    bh_port = equal_weight_curve(bh_R, dates)
    tecl_only = 100_000.0 * (1 + merged["s_TECL"]).cumprod()

    ps = curve_stats(port, dates)
    bs = curve_stats(bh_port, dates)
    ts = curve_stats(tecl_only, dates)

    # Per-sleeve standalone (over the common window) for context.
    print("=" * 86)
    print("  MULTI-SLEEVE PORTFOLIO  —  TECL/XLK + SPXL/SPY + TQQQ/QQQ  (equal-wt, annual rebal)")
    print("=" * 86)
    print(f"  Config per sleeve: up=20 d=80 volW=10 volExit=80 cooldown=20 park-stop=18%")
    print(f"  Period: {dates.iloc[0].date()} -> {dates.iloc[-1].date()} "
          f"({ps['years']:.1f} yrs)")
    print("-" * 86)
    print(f"  {'sleeve (standalone)':<24}{'CAGR%':>9}{'MaxDD%':>9}{'Sharpe':>9}{'Calmar':>9}{'inMkt%':>9}")
    for v in vehs:
        eq = 100_000.0 * (1 + merged[f"s_{v}"]).cumprod()
        ss = curve_stats(eq, dates)
        im = merged[f"in_{v}"].mean() * 100
        print(f"  {v:<24}{ss['cagr_pct']:>9.1f}{ss['max_dd_pct']:>9.1f}"
              f"{ss['sharpe']:>9.2f}{ss['calmar']:>9.2f}{im:>9.0f}")
    print("-" * 86)
    print(f"  {'PORTFOLIO (3 sleeves)':<24}{ps['cagr_pct']:>9.1f}{ps['max_dd_pct']:>9.1f}"
          f"{ps['sharpe']:>9.2f}{ps['calmar']:>9.2f}")
    print(f"  {'TECL/XLK alone (flagship)':<24}{ts['cagr_pct']:>9.1f}{ts['max_dd_pct']:>9.1f}"
          f"{ts['sharpe']:>9.2f}{ts['calmar']:>9.2f}")
    print(f"  {'EW B&H of 3x vehicles':<24}{bs['cagr_pct']:>9.1f}{bs['max_dd_pct']:>9.1f}"
          f"{bs['sharpe']:>9.2f}{bs['calmar']:>9.2f}")
    print("-" * 86)
    print(f"  Final equity:  PORTFOLIO ${ps['final_equity']:,.0f}   "
          f"TECL-alone ${ts['final_equity']:,.0f}   EW-B&H ${bs['final_equity']:,.0f}")
    print("-" * 86)
    split = pd.Timestamp("2018-01-01")
    for lbl, mask in [("Half A (2010-2017)", dates < split),
                      ("Half B (2018-2026)", dates >= split)]:
        sub = port[mask.to_numpy()]
        d = dates[mask]
        hs = curve_stats((sub / sub.iloc[0] * 100_000.0).reset_index(drop=True),
                         d.reset_index(drop=True))
        print(f"  {lbl}: PORTFOLIO CAGR {hs['cagr_pct']:5.1f}%  DD {hs['max_dd_pct']:6.1f}%  "
              f"Sharpe {hs['sharpe']:.2f}  Calmar {hs['calmar']:.2f}")
    print("=" * 86)

    out = pd.DataFrame({"date": dates, "portfolio": port.values,
                        "ew_bh": bh_port.values, "tecl_only": tecl_only.values})
    out.to_csv("slope_portfolio_equity.csv", index=False)
    print("  Saved: slope_portfolio_equity.csv")


if __name__ == "__main__":
    main()
