"""Out-of-sample robustness check for the slope + vol-regime strategy.

Splits TECL history into two halves (default split 2018-01-01). For each
config we compute the position series over the FULL history (so signals at
the split boundary use real prior data, no warm-up gap), then score the
SAME position series separately on each half.

Reports:
  1. The current global winners scored on each half individually.
  2. A walk-forward: optimize on half A -> test on half B, and vice versa.
     If the in-sample best still beats buy & hold out-of-sample, the edge
     is more likely real than curve-fit.
"""

import argparse

import numpy as np
import pandas as pd

from slope_backtest import load_prices, rolling_slope, compute_vol, stats

UP_WINDOWS   = [20, 40, 60, 80, 100, 120, 150, 200]
DOWN_WINDOWS = [5, 10, 15, 20, 30, 40, 60, 80]
VOL_PERIODS  = [10, 15, 20, 30, 40]
VOL_ENTRIES  = [None, 90, 120]
VOL_EXITS    = [None, 60, 70, 80, 100, 120, 150]

SPLIT = "2018-01-01"


def position_series(slopes, vol, up_w, down_w, vol_entry, vol_exit):
    """Full-history boolean in_market array for one config."""
    s_up, s_down = slopes[up_w], slopes[down_w]
    buy  = (s_up.shift(1)   <= 0) & (s_up   > 0)
    sell = (s_down.shift(1) >= 0) & (s_down < 0)
    b, sl = buy.to_numpy(), sell.to_numpy()
    v = vol.to_numpy()
    n = len(b)
    in_mkt = np.zeros(n, dtype=bool)
    held = False
    for i in range(1, n):
        vp = v[i - 1]
        if not held:
            if b[i - 1] and (vol_entry is None
                             or (not np.isnan(vp) and vp <= vol_entry)):
                held = True
        else:
            do_vol = (vol_exit is not None
                      and not np.isnan(vp) and vp >= vol_exit)
            if sl[i - 1] or do_vol:
                held = False
        in_mkt[i] = held
    return in_mkt


def score(df, asset_ret, in_mkt, mask):
    """Stats for the strategy and B&H over the rows selected by `mask`."""
    sub = df[mask]
    sret = pd.Series(np.where(in_mkt[mask.to_numpy()], asset_ret[mask], 0.0))
    aret = asset_ret[mask].reset_index(drop=True)
    dates = sub["date"].reset_index(drop=True)
    seq = 100_000.0 * (1 + sret).cumprod()
    beq = 100_000.0 * (1 + aret).cumprod()
    st = stats(seq, dates, sret)
    bh = stats(beq, dates, aret)
    st["calmar"] = st["cagr_pct"] / abs(st["max_dd_pct"]) if st["max_dd_pct"] else float("nan")
    bh["calmar"] = bh["cagr_pct"] / abs(bh["max_dd_pct"]) if bh["max_dd_pct"] else float("nan")
    st["inmkt"] = in_mkt[mask.to_numpy()].mean() * 100
    return st, bh


def cfgstr(c):
    ve = "off" if c[3] is None else f"{c[3]:.0f}"
    vx = "off" if c[4] is None else f"{c[4]:.0f}"
    return f"up={c[0]:>3} down={c[1]:>3} vWin={c[2]:>2} vEntry={ve:>3} vExit={vx:>3}"


def line(tag, st):
    return (f"{tag:<10} CAGR {st['cagr_pct']:6.1f}% | DD {st['max_dd_pct']:6.1f}% | "
            f"Sharpe {st['sharpe']:.2f} | Calmar {st['calmar']:.2f} | "
            f"inMkt {st.get('inmkt', float('nan')):4.0f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="TECL.json")
    ap.add_argument("--split", default=SPLIT)
    args = ap.parse_args()

    df = load_prices(args.data)
    split_ts = pd.Timestamp(args.split)
    asset_ret = df["price"].pct_change().fillna(0.0)
    mask_a = df["date"] < split_ts       # first half
    mask_b = df["date"] >= split_ts      # second half

    a0, a1 = df[mask_a]["date"].iloc[0].date(), df[mask_a]["date"].iloc[-1].date()
    b0, b1 = df[mask_b]["date"].iloc[0].date(), df[mask_b]["date"].iloc[-1].date()
    print(f"Half A (in/out): {a0} -> {a1}  ({mask_a.sum()} bars)")
    print(f"Half B (in/out): {b0} -> {b1}  ({mask_b.sum()} bars)\n")

    all_w = sorted(set(UP_WINDOWS) | set(DOWN_WINDOWS))
    slopes = {w: rolling_slope(df["price"], w) for w in all_w}
    vols = {p: compute_vol(df["price"], p) for p in VOL_PERIODS}

    # Build every config's full-history position series + per-half scores.
    configs, recs = [], []
    for u in UP_WINDOWS:
        for d in DOWN_WINDOWS:
            for vp in VOL_PERIODS:
                for ve in VOL_ENTRIES:
                    for vx in VOL_EXITS:
                        if ve is None and vx is None and vp != VOL_PERIODS[0]:
                            continue
                        cfg = (u, d, vp, ve, vx)
                        im = position_series(slopes, vols[vp], u, d, ve, vx)
                        sa, _ = score(df, asset_ret, im, mask_a)
                        sb, _ = score(df, asset_ret, im, mask_b)
                        configs.append(cfg)
                        recs.append(dict(cfg=cfg,
                                         a_sharpe=sa["sharpe"], a_calmar=sa["calmar"],
                                         a_cagr=sa["cagr_pct"], a_dd=sa["max_dd_pct"],
                                         a_in=sa["inmkt"],
                                         b_sharpe=sb["sharpe"], b_calmar=sb["calmar"],
                                         b_cagr=sb["cagr_pct"], b_dd=sb["max_dd_pct"],
                                         b_in=sb["inmkt"]))
    res = pd.DataFrame(recs)
    print(f"Scored {len(res)} configs on both halves.\n")

    # Buy & hold per half (in_market always True):
    ones = np.ones(len(df), dtype=bool)
    _, bha = score(df, asset_ret, ones, mask_a)
    _, bhb = score(df, asset_ret, ones, mask_b)
    bha["inmkt"], bhb["inmkt"] = 100.0, 100.0

    # ── 1. Global winners scored on each half ─────────────────────────────────
    GLOBAL = [
        (20, 80, 10, None, 80),   # max-Sharpe global winner
        (20, 40, 20, 90, 60),     # max-Calmar global winner
        (20, 40, 20, None, 70),   # earlier exit-only keeper
        (20, 40, 10, None, None),  # baseline up20/d40 (vol off; vWin irrelevant)
    ]
    print("=" * 78)
    print("  1. GLOBAL WINNERS — same config scored on each half")
    print("=" * 78)
    for cfg in GLOBAL:
        r = res[res["cfg"] == cfg]
        if r.empty:
            print(f"  [missing] {cfgstr(cfg)}"); continue
        r = r.iloc[0]
        print(f"\n  {cfgstr(cfg)}")
        print("    " + line("Half A", dict(cagr_pct=r['a_cagr'], max_dd_pct=r['a_dd'],
              sharpe=r['a_sharpe'], calmar=r['a_calmar'], inmkt=r['a_in'])))
        print("    " + line("Half B", dict(cagr_pct=r['b_cagr'], max_dd_pct=r['b_dd'],
              sharpe=r['b_sharpe'], calmar=r['b_calmar'], inmkt=r['b_in'])))
    print("\n  Buy & hold reference:")
    print("    " + line("B&H A", bha))
    print("    " + line("B&H B", bhb))

    # ── 2. Walk-forward: optimize on one half, test on the other ──────────────
    def walk(opt_col, test_prefix, opt_prefix, opt_half, test_half, metric):
        top = res.sort_values(opt_col, ascending=False).head(5)
        print(f"\n  Optimize on Half {opt_half} by {metric}, test on Half {test_half}:")
        for _, r in top.iterrows():
            print(f"    {cfgstr(r['cfg'])}")
            print(f"        IS (Half {opt_half}):  Sharpe {r[opt_prefix+'sharpe']:.2f} "
                  f"Calmar {r[opt_prefix+'calmar']:.2f} CAGR {r[opt_prefix+'cagr']:5.1f}% "
                  f"DD {r[opt_prefix+'dd']:6.1f}%")
            print(f"        OOS(Half {test_half}): Sharpe {r[test_prefix+'sharpe']:.2f} "
                  f"Calmar {r[test_prefix+'calmar']:.2f} CAGR {r[test_prefix+'cagr']:5.1f}% "
                  f"DD {r[test_prefix+'dd']:6.1f}%")

    print("\n" + "=" * 78)
    print("  2. WALK-FORWARD — optimize in-sample, test out-of-sample")
    print("=" * 78)
    print(f"\n  OOS buy&hold to beat:  Half B Sharpe {bhb['sharpe']:.2f} "
          f"Calmar {bhb['calmar']:.2f} | Half A Sharpe {bha['sharpe']:.2f} "
          f"Calmar {bha['calmar']:.2f}")
    walk("a_sharpe", "b_", "a_", "A", "B", "Sharpe")
    walk("b_sharpe", "a_", "b_", "B", "A", "Sharpe")
    walk("a_calmar", "b_", "a_", "A", "B", "Calmar")
    walk("b_calmar", "a_", "b_", "B", "A", "Calmar")

    res_out = res.copy()
    res_out["cfg"] = res_out["cfg"].apply(lambda c: cfgstr(c))
    res_out.to_csv("slope_oos_results.csv", index=False)
    print("\nWrote slope_oos_results.csv")


if __name__ == "__main__":
    main()
