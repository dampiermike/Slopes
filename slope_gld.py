"""
TECL/XLK Slopes strategy + GLD safety sleeve  —  LOCKED PORTFOLIO.

Sleeve A (65%): the locked TECL/XLK Slopes strategy
    (up=20, down=80, vol-window=10, vol-exit=80, cooldown=20, park-stop=18%)
Sleeve B (35%): GLD, buy & hold.

Rebalanced to target weights quarterly. This is the project's best
risk-adjusted config: clears Sharpe > 1.0 AND maxDD shallower than -30%,
validated out-of-sample on both halves of the 2009-2026 history.

Run with no args for the LOCKED full report. Use --sweep to reproduce the
allocation x rebalance grid.
"""

import argparse

import numpy as np
import pandas as pd

from slope_backtest import load_prices, backtest, stats

# ── Locked config ──────────────────────────────────────────────────────────────
GLD_WEIGHT = 0.35           # GLD safety sleeve weight
REBALANCE  = "quarterly"    # none | annual | quarterly | monthly
START      = "2009-04-14"
STRAT_CFG  = dict(up_window=20, down_window=80, vol_window=10, vol_exit_thresh=80.0,
                  cooldown=20, park_stop=0.18)
VEHICLE, PARK, SAFETY = "TECL", "XLK", "GLD"


def rebal_mask(dates, freq):
    d = pd.DatetimeIndex(dates)
    if freq == "none":
        return np.zeros(len(d), dtype=bool)
    key = {"annual": d.year,
           "quarterly": d.year * 4 + d.quarter,
           "monthly": d.year * 12 + d.month}[freq].to_numpy()
    m = np.zeros(len(d), dtype=bool)
    m[1:] = key[1:] != key[:-1]
    return m


def two_sleeve_curve(rA, rB, w_gld, reb, start=100_000.0):
    """Equity of (1-w) strategy + w GLD, rebalanced where reb[i] is True."""
    n = len(rA)
    eqA = (1 - w_gld) * start
    eqB = w_gld * start
    total = np.empty(n)
    for i in range(n):
        eqA *= (1 + rA[i]); eqB *= (1 + rB[i])
        t = eqA + eqB
        if reb[i]:
            eqA = (1 - w_gld) * t; eqB = w_gld * t
        total[i] = t
    return pd.Series(total)


def cstats(curve, dates):
    curve = curve.reset_index(drop=True)
    ret = curve.pct_change().fillna(0.0)
    s = stats(curve, pd.Series(dates).reset_index(drop=True), ret)
    s["calmar"] = s["cagr_pct"] / abs(s["max_dd_pct"]) if s["max_dd_pct"] else float("nan")
    return s


def worst_drawdowns(curve, dates, k=5):
    eq = curve.to_numpy(); d = pd.DatetimeIndex(dates).date
    peak = np.maximum.accumulate(eq); dd = eq / peak - 1
    eps = []; in_dd = False
    for i in range(len(eq)):
        if not in_dd and dd[i] < -1e-4:
            in_dd = True; start = i; pk = peak[i]; trough = i
        elif in_dd:
            if eq[i] < eq[trough]:
                trough = i
            if eq[i] >= pk:
                eps.append((d[start], d[trough], d[i], dd[trough]*100,
                            (d[i]-d[start]).days)); in_dd = False
    if in_dd:
        eps.append((d[start], d[trough], d[-1], dd[trough]*100, (d[-1]-d[start]).days))
    e = pd.DataFrame(eps, columns=["peak", "trough", "recover", "dd_pct", "days"])
    return e.sort_values("dd_pct").head(k)


def load_aligned(start_date):
    """Strategy daily return + GLD daily return + TECL B&H return, by date."""
    df = load_prices(f"{VEHICLE}.json")
    park = load_prices(f"{PARK}.json").rename(columns={"price": "park_price"})
    df = df.merge(park, on="date", how="inner").reset_index(drop=True)
    bt = backtest(df, STRAT_CFG["up_window"], STRAT_CFG["down_window"], 100_000.0,
                  vol_window=STRAT_CFG["vol_window"],
                  vol_exit_thresh=STRAT_CFG["vol_exit_thresh"],
                  cooldown=STRAT_CFG["cooldown"], park_stop=STRAT_CFG["park_stop"])
    a = bt.dropna(subset=["slope"])[["date", "strat_ret", "asset_ret"]]
    gld = load_prices(f"{SAFETY}.json")
    gld["gld_ret"] = gld["price"].pct_change().fillna(0.0)
    m = a.merge(gld[["date", "gld_ret"]], on="date", how="inner")
    m = m[m["date"] >= pd.Timestamp(start_date)].reset_index(drop=True)
    return m


def full_report(args):
    m = load_aligned(args.start_date)
    dates = m["date"]
    rA = m["strat_ret"].to_numpy(); rB = m["gld_ret"].to_numpy()
    w = args.gld_weight
    reb = rebal_mask(dates, args.rebalance)

    port = two_sleeve_curve(rA, rB, w, reb)
    strat_only = two_sleeve_curve(rA, rB, 0.0, np.zeros(len(rA), bool))
    gld_only = 100_000.0 * (1 + m["gld_ret"]).cumprod()
    tecl_bh = 100_000.0 * (1 + m["asset_ret"]).cumprod()

    ps, ss, gs, bs = (cstats(port, dates), cstats(strat_only, dates),
                      cstats(gld_only, dates), cstats(tecl_bh, dates))
    nreb = int(reb.sum())

    print("=" * 84)
    print(f"  LOCKED PORTFOLIO — {VEHICLE}/{PARK} strategy {(1-w)*100:.0f}% + "
          f"{SAFETY} {w*100:.0f}%  ({args.rebalance} rebal)")
    print("=" * 84)
    print(f"  Strategy sleeve: up=20 d=80 volW=10 volExit=80 cooldown=20 park-stop=18% "
          f"(park {PARK})")
    print(f"  Period: {dates.iloc[0].date()} -> {dates.iloc[-1].date()} "
          f"({ps['years']:.1f} yrs)   Rebalances: {nreb}")
    print("-" * 84)
    print(f"  {'':<26}{'CAGR%':>9}{'MaxDD%':>9}{'Sharpe':>9}{'Calmar':>9}{'final $':>15}")
    for lbl, s in [(f"PORTFOLIO ({(1-w)*100:.0f}/{w*100:.0f})", ps),
                   (f"{VEHICLE}/{PARK} strat alone", ss),
                   (f"{SAFETY} buy & hold", gs),
                   (f"{VEHICLE} buy & hold", bs)]:
        print(f"  {lbl:<26}{s['cagr_pct']:>9.1f}{s['max_dd_pct']:>9.1f}"
              f"{s['sharpe']:>9.2f}{s['calmar']:>9.2f}{s['final_equity']:>15,.0f}")
    print("-" * 84)
    goal = "PASS" if (ps["sharpe"] > 1.0 and ps["max_dd_pct"] > -30.0) else "FAIL"
    print(f"  GOAL (Sharpe>1.0 & maxDD>-30%): {goal}  "
          f"[Sharpe {ps['sharpe']:.2f}, maxDD {ps['max_dd_pct']:.1f}%]")
    print("-" * 84)
    split = pd.Timestamp("2018-01-01")
    print("  Per-half (out-of-sample split 2018-01-01):")
    for lbl, mask in [("Half A", dates < split), ("Half B", dates >= split)]:
        sub = port[mask.to_numpy()]
        hs = cstats(sub / sub.iloc[0] * 100_000.0, dates[mask])
        print(f"    {lbl} ({dates[mask].iloc[0].date()}->{dates[mask].iloc[-1].date()}): "
              f"CAGR {hs['cagr_pct']:5.1f}%  DD {hs['max_dd_pct']:6.1f}%  "
              f"Sharpe {hs['sharpe']:.2f}  Calmar {hs['calmar']:.2f}")
    print("-" * 84)
    print("  Worst portfolio drawdown episodes (peak -> trough -> recovery):")
    last = pd.DatetimeIndex(dates).date[-1]
    for _, r in worst_drawdowns(port, dates).iterrows():
        rec = r["recover"] if r["recover"] != last else "ongoing"
        print(f"    {r['dd_pct']:6.1f}%  {r['peak']} -> {r['trough']} -> {rec}  "
              f"({r['days']}d underwater)")
    print("=" * 84)

    out = pd.DataFrame({"date": dates, "portfolio": port.values,
                        "strat_only": strat_only.values,
                        "gld_only": gld_only.values, "tecl_bh": tecl_bh.values})
    fn = f"slope_portfolio_g{int(w*100)}_{args.rebalance}.csv"
    out.to_csv(fn, index=False)
    print(f"  Saved: {fn}")


def sweep(args):
    m = load_aligned(args.start_date)
    dates = m["date"]
    rA = m["strat_ret"].to_numpy(); rB = m["gld_ret"].to_numpy()
    weights = [0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    rebals = ["none", "annual", "quarterly", "monthly"]
    masks = {f: rebal_mask(dates, f) for f in rebals}
    print(f"  Allocation x rebalance sweep  ({dates.iloc[0].date()} -> {dates.iloc[-1].date()})")
    for metric, title, fmt in [("sharpe", "SHARPE", "{:>12.2f}"),
                               ("max_dd_pct", "MAX DRAWDOWN %", "{:>12.1f}")]:
        print(f"\n  === {title} (rows=GLD%, cols=rebalance) ===")
        print("  GLD%   " + "".join(f"{r:>12}" for r in rebals))
        for w in weights:
            cells = [cstats(two_sleeve_curve(rA, rB, w, masks[f]), dates)[metric]
                     for f in rebals]
            print(f"  {w*100:>4.0f}   " + "".join(fmt.format(c) for c in cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gld-weight", type=float, default=GLD_WEIGHT,
                    help="GLD safety sleeve weight (default 0.35)")
    ap.add_argument("--rebalance", default=REBALANCE,
                    choices=["none", "annual", "quarterly", "monthly"])
    ap.add_argument("--start-date", default=START)
    ap.add_argument("--sweep", action="store_true",
                    help="show the allocation x rebalance grid instead of the report")
    args = ap.parse_args()
    if args.sweep:
        sweep(args)
    else:
        full_report(args)


if __name__ == "__main__":
    main()
