"""
Slope-crossover backtest.

Strategy
--------
- Each day compute the slope of a linear regression fit to the last N
  adjusted-close prices (N defaults to 100).
- When the slope crosses from negative to positive -> BUY TECL (fully invested).
- When the slope crosses from positive to negative -> SELL (leave TECL).
- A volatility-regime filter gates the position (see compute_vol):
    entry blocked when realized vol > --vol-entry-max,
    forced exit when realized vol >= --vol-exit-thresh.
- Signals are generated at the close of day t (slope/vol known at the close).
  To avoid look-ahead, trades execute at the NEXT day's close.
- When NOT in TECL, capital is parked in --park (default XLK) rather than cash.

LOCKED config (robust global winner, see slope_oos.py walk-forward):
    up=20  down=80  vol-window=10  vol-exit=80  vol-entry=off
    park=XLK  cooldown=20 (after losing legs)  park-stop=18% (XLK->cash)

Signal security: TECL (real adjusted closes from inception).
"""

import argparse
import json

import numpy as np
import pandas as pd


def load_prices(path: str) -> pd.DataFrame:
    with open(path) as f:
        rows = json.load(f)
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    # Use adjusted close for everything (handles splits/dividends).
    df["price"] = df["adjusted_close"].astype(float)
    return df[["date", "price"]]


def compute_vol(prices: pd.Series, window: int = 20) -> pd.Series:
    """Annualized realized volatility (%) over `window` bars of log returns.

    Matches the TwoSleeves convention: sample variance (÷ N-1), ×252, ×100.
    NaN until the first full window is available.
    """
    lr = np.log(prices / prices.shift(1))
    vol = lr.rolling(window).std(ddof=1) * np.sqrt(252) * 100.0
    return vol


def rolling_slope(prices: pd.Series, window: int) -> pd.Series:
    """Slope of an OLS line fit to the last `window` prices, per day.

    x is 0..window-1 so the slope is in price-units per day.
    """
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    x_dev = x - x_mean
    denom = (x_dev ** 2).sum()

    vals = prices.to_numpy(dtype=float)
    out = np.full(len(vals), np.nan)
    for i in range(window - 1, len(vals)):
        y = vals[i - window + 1 : i + 1]
        out[i] = (x_dev * (y - y.mean())).sum() / denom
    return pd.Series(out, index=prices.index)


def backtest(df: pd.DataFrame, up_window: int, down_window: int,
             start_equity: float = 100_000.0,
             vol_window: int = 20, vol_entry_max: float | None = None,
             vol_exit_thresh: float | None = None,
             cooldown: int = 0, park_stop: float | None = None):
    df = df.copy()
    # Slow slope confirms the uptrend (buy); fast slope reacts to weakness (sell).
    df["slope_up"] = rolling_slope(df["price"], up_window)
    df["slope_down"] = rolling_slope(df["price"], down_window)
    df["slope_up_prev"] = df["slope_up"].shift(1)
    df["slope_down_prev"] = df["slope_down"].shift(1)
    # Combined slope column only exists once both windows are warmed up.
    df["slope"] = df["slope_up"].where(df["slope_down"].notna())

    # Volatility regime filter (known at the close of day t, same as the slope).
    df["vol"] = compute_vol(df["price"], vol_window)

    # Signal day t (acted on at close of t+1).
    buy_signal = (df["slope_up_prev"] <= 0) & (df["slope_up"] > 0)
    sell_signal = (df["slope_down_prev"] >= 0) & (df["slope_down"] < 0)

    n = len(df)
    in_market = np.zeros(n, dtype=bool)   # position held over day i (close[i-1] -> close[i])
    held = False
    # Position for day i is decided by the signal on day i-1 (execute at close i).
    sig_buy = buy_signal.to_numpy()
    sig_sell = sell_signal.to_numpy()
    vol = df["vol"].to_numpy()
    price = df["price"].to_numpy()
    cool = 0          # bars remaining before a new TECL entry is allowed
    entry_px = 0.0    # TECL entry price of the current leg (for cooldown P&L test)
    for i in range(1, n):
        if cool > 0:
            cool -= 1
        v = vol[i - 1]   # regime known at the close of the signal day
        if not held:
            # ENTRY: slope buy crossing, calm-vol regime, and not in cooldown.
            if (sig_buy[i - 1] and cool == 0
                    and (vol_entry_max is None
                         or (not np.isnan(v) and v <= vol_entry_max))):
                held = True
                entry_px = price[i - 1]
        else:
            # EXIT: slope sell crossing OR a high-vol regime (independent of slope).
            vol_exit = (vol_exit_thresh is not None
                        and not np.isnan(v) and v >= vol_exit_thresh)
            if sig_sell[i - 1] or vol_exit:
                # Cooldown only after a LOSING leg — pause re-entry into leverage.
                if cooldown > 0 and price[i - 1] / entry_px - 1 < 0:
                    cool = cooldown
                held = False
        in_market[i] = held
    df["in_market"] = in_market

    # Daily returns of TECL; strategy earns them only when in_market.
    df["asset_ret"] = df["price"].pct_change().fillna(0.0)
    # When out of TECL, park in the alternate security (XLK) instead of cash.
    if "park_price" in df:
        df["park_ret"] = df["park_price"].pct_change().fillna(0.0)
    else:
        df["park_ret"] = 0.0

    # Third tier: while parked, if the park asset falls >= park_stop from the
    # stint's entry price, rotate to cash (0%) for the rest of that out-of-TECL
    # stint. Guards against the park asset itself crashing.
    if park_stop is not None and "park_price" in df:
        park_px = df["park_price"].to_numpy()
        park_active = np.ones(n, dtype=bool)   # True = holding XLK, False = cash
        state = "tecl"
        stint_entry = 0.0
        for i in range(n):
            if in_market[i]:
                state = "tecl"
                continue
            if state == "tecl":          # first bar of a new park stint
                state = "park"
                stint_entry = park_px[i]
            if state == "park":
                if park_px[i] <= stint_entry * (1 - park_stop):
                    state = "cash"
                    park_active[i] = False
            else:                         # state == "cash" for rest of stint
                park_active[i] = False
        df["park_active"] = park_active
        out_ret = np.where(park_active, df["park_ret"], 0.0)
    else:
        out_ret = df["park_ret"]
    df["strat_ret"] = np.where(df["in_market"], df["asset_ret"], out_ret)

    df["strat_equity"] = start_equity * (1 + df["strat_ret"]).cumprod()
    df["bh_equity"] = start_equity * (1 + df["asset_ret"]).cumprod()
    return df


def trade_log(df: pd.DataFrame) -> pd.DataFrame:
    trades = []
    pos = df["in_market"].to_numpy()
    dates = df["date"].to_numpy()
    price = df["price"].to_numpy()
    entry_i = None
    for i in range(len(df)):
        if pos[i] and entry_i is None:
            entry_i = i
        elif not pos[i] and entry_i is not None:
            trades.append((dates[entry_i], price[entry_i], dates[i], price[i]))
            entry_i = None
    if entry_i is not None:  # still open at end
        trades.append((dates[entry_i], price[entry_i], dates[-1], price[-1]))
    t = pd.DataFrame(trades, columns=["entry_date", "entry_px", "exit_date", "exit_px"])
    if not t.empty:
        t["ret_pct"] = (t["exit_px"] / t["entry_px"] - 1) * 100
        t["days"] = (t["exit_date"] - t["entry_date"]).dt.days
    return t


def stats(equity: pd.Series, dates: pd.Series, strat_ret: pd.Series) -> dict:
    total_ret = equity.iloc[-1] / equity.iloc[0] - 1
    years = (dates.iloc[-1] - dates.iloc[0]).days / 365.25
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1
    roll_max = equity.cummax()
    dd = equity / roll_max - 1
    max_dd = dd.min()
    ann_vol = strat_ret.std() * np.sqrt(252)
    sharpe = (strat_ret.mean() * 252) / ann_vol if ann_vol > 0 else float("nan")
    return {
        "final_equity": equity.iloc[-1],
        "total_return_pct": total_ret * 100,
        "cagr_pct": cagr * 100,
        "max_dd_pct": max_dd * 100,
        "sharpe": sharpe,
        "years": years,
    }


def main():
    ap = argparse.ArgumentParser()
    # Defaults = LOCKED robust config (see slope_oos.py walk-forward).
    ap.add_argument("--up-window", type=int, default=20,
                    help="slope window for the BUY trigger (negative->positive)")
    ap.add_argument("--down-window", type=int, default=80,
                    help="slope window for the SELL trigger (positive->negative)")
    ap.add_argument("--data", default="TECL.json")
    ap.add_argument("--start", type=float, default=100_000.0)
    ap.add_argument("--start-date", default=None,
                    help="restrict STATS to dates >= this (YYYY-MM-DD); "
                         "indicators/positions still warm up on full history")
    ap.add_argument("--vol-window", type=int, default=10,
                    help="lookback for annualized realized vol")
    ap.add_argument("--vol-entry-max", type=float, default=None,
                    help="block entry when vol exceeds this %% (omit to disable)")
    ap.add_argument("--vol-exit-thresh", type=float, default=80.0,
                    help="force exit when vol reaches this %% (omit to disable)")
    ap.add_argument("--park", default="XLK.json",
                    help="security held when out of TECL ('none' or '' = cash)")
    ap.add_argument("--cooldown", type=int, default=20,
                    help="bars to block TECL re-entry after a LOSING leg (0=off)")
    ap.add_argument("--park-stop", type=float, default=18.0,
                    help="rotate park->cash if park falls this %% from stint entry "
                         "(omit/0 = never)")
    args = ap.parse_args()

    df = load_prices(args.data)
    park_name = "CASH"
    if args.park and args.park.lower() not in ("none", "cash"):
        park = load_prices(args.park).rename(columns={"price": "park_price"})
        df = df.merge(park, on="date", how="inner").reset_index(drop=True)
        park_name = args.park.replace(".json", "")
    park_stop_frac = (args.park_stop / 100.0) if args.park_stop else None
    bt = backtest(df, args.up_window, args.down_window, args.start,
                  vol_window=args.vol_window,
                  vol_entry_max=args.vol_entry_max,
                  vol_exit_thresh=args.vol_exit_thresh,
                  cooldown=args.cooldown,
                  park_stop=park_stop_frac)

    # Drop the warm-up period (before slope exists) for fair stats.
    valid = bt.dropna(subset=["slope"]).reset_index(drop=True)
    if args.start_date:
        valid = valid[valid["date"] >= pd.Timestamp(args.start_date)].reset_index(drop=True)
    valid["strat_equity"] = args.start * (1 + valid["strat_ret"]).cumprod()
    valid["bh_equity"] = args.start * (1 + valid["asset_ret"]).cumprod()

    s = stats(valid["strat_equity"], valid["date"], valid["strat_ret"])
    bh = stats(valid["bh_equity"], valid["date"], valid["asset_ret"])
    trades = trade_log(valid)

    def calmar(st):
        return st["cagr_pct"] / abs(st["max_dd_pct"]) if st["max_dd_pct"] else float("nan")

    def worst_drawdowns(equity, dates, k=5):
        eq = equity.to_numpy(); d = dates.dt.date.to_numpy()
        peak = np.maximum.accumulate(eq); dd = eq / peak - 1
        eps = []; in_dd = False
        for i in range(len(eq)):
            if not in_dd and dd[i] < -1e-4:
                in_dd = True; start = i; pk = peak[i]; trough = i
            elif in_dd:
                if eq[i] < eq[trough]:
                    trough = i
                if eq[i] >= pk:
                    eps.append((d[start], d[trough], d[i], dd[trough] * 100,
                                (d[i] - d[start]).days)); in_dd = False
        if in_dd:
            eps.append((d[start], d[trough], d[-1], dd[trough] * 100,
                        (d[-1] - d[start]).days))
        e = pd.DataFrame(eps, columns=["peak", "trough", "recover", "dd_pct", "days"])
        return e.sort_values("dd_pct").head(k)

    def half_stats(mask, label):
        m = valid[mask].reset_index(drop=True)
        seq = args.start * (1 + m["strat_ret"]).cumprod()
        beq = args.start * (1 + m["asset_ret"]).cumprod()
        hs = stats(seq, m["date"], m["strat_ret"]); hb = stats(beq, m["date"], m["asset_ret"])
        print(f"  {label} ({m['date'].iloc[0].date()}->{m['date'].iloc[-1].date()}): "
              f"STRAT CAGR {hs['cagr_pct']:5.1f}% DD {hs['max_dd_pct']:6.1f}% "
              f"Sharpe {hs['sharpe']:.2f} Calmar {calmar(hs):.2f}  |  "
              f"TECL-B&H Sharpe {hb['sharpe']:.2f} Calmar {calmar(hb):.2f}")

    vmin = "off" if args.vol_entry_max is None else f"{args.vol_entry_max:.0f}%"
    vmax = "off" if args.vol_exit_thresh is None else f"{args.vol_exit_thresh:.0f}%"
    pstop = "off" if not args.park_stop else f"{args.park_stop:.0f}%"
    cd = "off" if not args.cooldown else f"{args.cooldown}d"
    print("=" * 78)
    print("  SLOPE-CROSSOVER + VOL-REGIME + XLK-PARK  —  FULL REPORT")
    print("=" * 78)
    print(f"  Signal: TECL  slope up={args.up_window}d / down={args.down_window}d")
    print(f"  Vol filter: window={args.vol_window}d  entry<={vmin}  exit>={vmax}")
    print(f"  Park out-of-TECL: {park_name}   park-stop: {pstop}->cash   "
          f"cooldown(losing legs): {cd}")
    print(f"  Period: {valid['date'].iloc[0].date()} -> {valid['date'].iloc[-1].date()} "
          f"({s['years']:.1f} yrs)   Time in TECL: {valid['in_market'].mean()*100:.1f}%")
    print("-" * 78)
    print(f"  {'metric':<18}{'STRATEGY':>16}{'TECL B&H':>16}{'edge':>12}")
    print(f"  {'final equity':<18}{s['final_equity']:>16,.0f}{bh['final_equity']:>16,.0f}"
          f"{s['final_equity']/bh['final_equity']:>11.2f}x")
    print(f"  {'CAGR %':<18}{s['cagr_pct']:>16.2f}{bh['cagr_pct']:>16.2f}"
          f"{s['cagr_pct']-bh['cagr_pct']:>+12.2f}")
    print(f"  {'max drawdown %':<18}{s['max_dd_pct']:>16.2f}{bh['max_dd_pct']:>16.2f}"
          f"{s['max_dd_pct']-bh['max_dd_pct']:>+12.2f}")
    print(f"  {'Sharpe':<18}{s['sharpe']:>16.2f}{bh['sharpe']:>16.2f}"
          f"{s['sharpe']-bh['sharpe']:>+12.2f}")
    print(f"  {'Calmar':<18}{calmar(s):>16.2f}{calmar(bh):>16.2f}"
          f"{calmar(s)-calmar(bh):>+12.2f}")
    print("-" * 78)
    print("  Per-half (out-of-sample split 2018-01-01):")
    half_stats(valid["date"] < pd.Timestamp("2018-01-01"), "Half A")
    half_stats(valid["date"] >= pd.Timestamp("2018-01-01"), "Half B")
    print("-" * 78)
    if "park_price" in valid:
        valid["park_bh_equity"] = args.start * (1 + valid["park_ret"]).cumprod()
        pk = stats(valid["park_bh_equity"], valid["date"], valid["park_ret"])
        parked = (~valid["in_market"]).sum()
        in_cash = (~valid["in_market"] & ~valid.get("park_active", True)).sum() \
            if "park_active" in valid else 0
        print(f"  Park asset ({park_name}) B&H reference: CAGR {pk['cagr_pct']:.2f}% | "
              f"maxDD {pk['max_dd_pct']:.2f}% | Sharpe {pk['sharpe']:.2f}")
        print(f"  Out-of-TECL days: {parked} ({parked/len(valid)*100:.0f}%)  "
              f"of which park-stopped to cash: {in_cash} days")
    if not trades.empty:
        wins = (trades["ret_pct"] > 0).sum()
        print(f"  TECL trades: {len(trades)} | win {wins/len(trades)*100:.0f}% | "
              f"avg {trades['ret_pct'].mean():.1f}% | avg hold {trades['days'].mean():.0f}d "
              f"| best {trades['ret_pct'].max():.0f}% | worst {trades['ret_pct'].min():.0f}%")
    print("-" * 78)
    print("  Worst strategy drawdown episodes (peak -> trough -> recovery):")
    for _, r in worst_drawdowns(valid["strat_equity"], valid["date"]).iterrows():
        rec = r["recover"] if r["recover"] != valid["date"].iloc[-1].date() else "ongoing"
        print(f"    {r['dd_pct']:6.1f}%  {r['peak']} -> {r['trough']} -> {rec}  "
              f"({r['days']}d underwater)")
    print("=" * 78)

    tag = f"u{args.up_window}_d{args.down_window}"
    if args.vol_exit_thresh is not None:
        tag += f"_vx{args.vol_exit_thresh:.0f}"
    if args.cooldown:
        tag += f"_cd{args.cooldown}"
    if args.park_stop:
        tag += f"_ps{args.park_stop:.0f}"
    valid.to_csv(f"slope_equity_{tag}.csv", index=False)
    trades.to_csv(f"slope_trades_{tag}.csv", index=False)
    print(f"  Saved: slope_equity_{tag}.csv  +  slope_trades_{tag}.csv")


if __name__ == "__main__":
    main()
