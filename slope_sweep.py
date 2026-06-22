"""Sweep upslope (buy) and downslope (sell) lookback windows for the
slope-crossover strategy on TECL, now with the volatility-regime filter
(entry gate + exit trigger) as additional sweep dimensions.

Reuses the engine in slope_backtest.py.
"""

import argparse

import numpy as np
import pandas as pd

from slope_backtest import load_prices, rolling_slope, compute_vol, stats, trade_log

UP_WINDOWS   = [20, 40, 60, 80, 100, 120, 150, 200]
DOWN_WINDOWS = [5, 10, 15, 20, 30, 40, 60, 80]
# Volatility lookback windows to sweep (shorter = faster/more extreme readings).
VOL_PERIODS  = [10, 15, 20, 30, 40]
# None = filter off. Entry gate mostly costs return (median TECL vol ~51%),
# so keep its grid small; the exit trigger is the value-add, so sweep it finely.
VOL_ENTRIES  = [None, 90, 120]
VOL_EXITS    = [None, 60, 70, 80, 100, 120, 150]


def run_combo(df, slopes, vol, vol_period, up_w, down_w, vol_entry, vol_exit,
              start=100_000.0):
    s_up   = slopes[up_w]
    s_down = slopes[down_w]
    buy  = (s_up.shift(1)   <= 0) & (s_up   > 0)
    sell = (s_down.shift(1) >= 0) & (s_down < 0)

    n = len(df)
    in_mkt = np.zeros(n, dtype=bool)
    held = False
    b, sl = buy.to_numpy(), sell.to_numpy()
    v = vol.to_numpy()
    for i in range(1, n):
        vp = v[i - 1]                       # regime known at the signal close
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

    asset_ret = df["price"].pct_change().fillna(0.0)
    strat_ret = pd.Series(np.where(in_mkt, asset_ret, 0.0), index=df.index)

    # Stats only over the period where all windows (incl. vol) are warmed up.
    warm = max(up_w, down_w, vol_period)
    v_df = df.iloc[warm:].copy()
    vret = strat_ret.iloc[warm:]
    eq = start * (1 + vret).cumprod()
    st = stats(eq, v_df["date"].reset_index(drop=True), vret.reset_index(drop=True))

    tdf = v_df.copy()
    tdf["in_market"] = in_mkt[warm:]
    trades = trade_log(tdf.reset_index(drop=True))
    maxdd = st["max_dd_pct"]
    calmar = st["cagr_pct"] / abs(maxdd) if maxdd else float("nan")
    return {
        "up": up_w,
        "down": down_w,
        "vwin": vol_period,
        "ventry": vol_entry,
        "vexit": vol_exit,
        "cagr": st["cagr_pct"],
        "maxdd": maxdd,
        "sharpe": st["sharpe"],
        "calmar": calmar,
        "final": st["final_equity"],
        "time_in_mkt": in_mkt[warm:].mean() * 100,
        "trades": len(trades),
    }


def heat(df, value_col, fmt):
    piv = df.pivot(index="up", columns="down", values=value_col)
    cols = piv.columns.tolist()
    header = "up\\down" + "".join(f"{c:>9}" for c in cols)
    print(header)
    for up in piv.index:
        row = "".join(fmt.format(piv.loc[up, c]) for c in cols)
        print(f"{up:>7}" + row)


def vtag(x):
    return "off" if x is None else f"{x:.0f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="TECL.json")
    args = ap.parse_args()

    df = load_prices(args.data)
    all_w = sorted(set(UP_WINDOWS) | set(DOWN_WINDOWS))
    print(f"Precomputing {len(all_w)} slope windows + "
          f"{len(VOL_PERIODS)} vol windows {VOL_PERIODS} ...")
    slopes = {w: rolling_slope(df["price"], w) for w in all_w}
    vols = {p: compute_vol(df["price"], p) for p in VOL_PERIODS}

    rows = []
    for u in UP_WINDOWS:
        for d in DOWN_WINDOWS:
            for vp in VOL_PERIODS:
                for ve in VOL_ENTRIES:
                    for vx in VOL_EXITS:
                        # Baseline (both filters off) is vol-window-independent;
                        # run it once (under the first vol window) to avoid dupes.
                        if ve is None and vx is None and vp != VOL_PERIODS[0]:
                            continue
                        rows.append(run_combo(df, slopes, vols[vp], vp,
                                              u, d, ve, vx))
    res = pd.DataFrame(rows)
    print(f"Ran {len(res)} combos "
          f"({len(UP_WINDOWS)}×{len(DOWN_WINDOWS)} windows × "
          f"{len(VOL_PERIODS)} vWin × {len(VOL_ENTRIES)} entries × "
          f"{len(VOL_EXITS)} exits, baseline deduped)\n")

    # Baseline heatmaps (vol filter fully off) for reference.
    base = res[(res["ventry"].isna()) & (res["vexit"].isna())]
    print("=== BASELINE (vol filter off) — CAGR % (rows=up/buy, cols=down/sell) ===")
    heat(base, "cagr", "{:>9.1f}")
    print("\n=== BASELINE — Max Drawdown % ===")
    heat(base, "maxdd", "{:>9.1f}")
    print("\n=== BASELINE — Sharpe ===")
    heat(base, "sharpe", "{:>9.2f}")

    def show(title, sort_col, n=15):
        print(f"\n--- Top {n} by {title} (all vol configs) ---")
        top = res.sort_values(sort_col, ascending=False).head(n)
        for _, r in top.iterrows():
            print(f"up={r['up']:>3} down={r['down']:>3} vWin={int(r['vwin']):>2} "
                  f"vEntry={vtag(r['ventry']):>3} vExit={vtag(r['vexit']):>3} | "
                  f"CAGR {r['cagr']:6.1f}% | DD {r['maxdd']:6.1f}% | "
                  f"Sharpe {r['sharpe']:.2f} | Calmar {r['calmar']:.2f} | "
                  f"inMkt {r['time_in_mkt']:4.0f}% | {int(r['trades'])} trades")

    show("Calmar (CAGR/|MaxDD|)", "calmar")
    show("Sharpe", "sharpe")
    show("CAGR", "cagr")

    # Focused ladder: best (window, vWin) combo by Calmar, varied across exits.
    best = res.sort_values("calmar", ascending=False).iloc[0]
    bu, bd, bvw = int(best["up"]), int(best["down"]), int(best["vwin"])
    base_row = res[(res["up"] == bu) & (res["down"] == bd)
                   & (res["ventry"].isna()) & (res["vexit"].isna())]
    ladder = res[(res["up"] == bu) & (res["down"] == bd) & (res["vwin"] == bvw)
                 & (res["ventry"].isna()) & (res["vexit"].notna())]
    ladder = pd.concat([base_row, ladder]).sort_values("vexit", na_position="first")
    print(f"\n--- Vol-exit ladder at the top Calmar combo "
          f"(up={bu}/down={bd}/vWin={bvw}, entry off) ---")
    for _, r in ladder.iterrows():
        print(f"  vExit={vtag(r['vexit']):>3} | CAGR {r['cagr']:6.1f}% | "
              f"DD {r['maxdd']:6.1f}% | Sharpe {r['sharpe']:.2f} | "
              f"Calmar {r['calmar']:.2f} | inMkt {r['time_in_mkt']:4.0f}%")

    # How the vol lookback window itself ranks (best Calmar achievable per vWin).
    print("\n--- Best Calmar achievable per vol lookback window ---")
    for vp in VOL_PERIODS:
        sub = res[res["vwin"] == vp]
        r = sub.sort_values("calmar", ascending=False).iloc[0]
        print(f"  vWin={vp:>2} | best Calmar {r['calmar']:.2f} "
              f"(up={int(r['up'])}/down={int(r['down'])} "
              f"vEntry={vtag(r['ventry'])}/vExit={vtag(r['vexit'])}) | "
              f"CAGR {r['cagr']:5.1f}% DD {r['maxdd']:6.1f}% Sharpe {r['sharpe']:.2f}")

    res.to_csv("slope_sweep_results.csv", index=False)
    print("\nWrote slope_sweep_results.csv")


if __name__ == "__main__":
    main()
