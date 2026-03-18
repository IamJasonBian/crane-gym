"""
IWM Put & Call Profitability Analysis — Cross-Asset Regime Framework.

Evaluates hedge profitability across all four BTC × IWM movement regimes:

    Base Asset (BTC/Grayscale):  UP  |  DOWN
    Pegged Asset (IWM):          UP  |  DOWN

                        IWM UP          IWM DOWN
    BTC UP      risk-on rally     divergence (BTC leads)
    BTC DOWN    divergence        risk-off selloff

For each regime, values both PUT and CALL positions on IWM to determine
which hedge instrument is optimal under each condition.

Usage:
    python put_profitability.py
    python put_profitability.py --weeks 12 --hold-days 5
    python put_profitability.py --live-chain
"""

import argparse
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


# ── Black-Scholes ────────────────────────────────────────────────────────

def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def bs_put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(K - S, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


# ── Data loading ─────────────────────────────────────────────────────────

def load_ticker_history(ticker: str, days: int = 200) -> pd.DataFrame:
    """Fetch daily OHLCV from yfinance."""
    end = datetime.now()
    start = end - timedelta(days=int(days * 1.5))
    df = yf.Ticker(ticker).history(
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
    )
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df[["Open", "High", "Low", "Close", "Volume"]].tail(days)


def load_iwm_options_chain() -> Dict:
    """Fetch live IWM options chain from yfinance."""
    ticker = yf.Ticker("IWM")
    expirations = ticker.options
    chains = {}
    for exp in expirations:
        try:
            chain = ticker.option_chain(exp)
            chains[exp] = {"calls": chain.calls, "puts": chain.puts}
        except Exception:
            continue
    return chains


def rolling_realized_vol(df: pd.DataFrame, window: int = 21,
                         trading_days: int = 252) -> pd.Series:
    """Annualized close-to-close realized vol."""
    log_ret = np.log(df["Close"] / df["Close"].shift(1))
    return log_ret.rolling(window).std() * np.sqrt(trading_days)


# ── Cross-asset regime classification ────────────────────────────────────

REGIMES = {
    ("UP", "UP"):     "RISK-ON RALLY",
    ("UP", "DOWN"):   "BTC LEADS (divergence)",
    ("DOWN", "UP"):   "IWM LEADS (divergence)",
    ("DOWN", "DOWN"): "RISK-OFF SELLOFF",
}


def classify_regimes(
    btc: pd.DataFrame,
    iwm: pd.DataFrame,
    hold_days: int = 10,
) -> pd.DataFrame:
    """
    Classify each hold_days window into one of 4 regimes based on
    BTC and IWM directional moves.

    Returns DataFrame with entry/exit dates, returns for both assets,
    and regime label.
    """
    # Align on common dates
    common = btc.index.intersection(iwm.index).sort_values()
    btc_aligned = btc.loc[common]
    iwm_aligned = iwm.loc[common]

    records = []
    warmup = 21  # skip vol warmup period
    entry_indices = list(range(warmup, len(common) - hold_days, hold_days))

    for entry_idx in entry_indices:
        exit_idx = min(entry_idx + hold_days, len(common) - 1)

        btc_entry = float(btc_aligned["Close"].iloc[entry_idx])
        btc_exit = float(btc_aligned["Close"].iloc[exit_idx])
        iwm_entry = float(iwm_aligned["Close"].iloc[entry_idx])
        iwm_exit = float(iwm_aligned["Close"].iloc[exit_idx])

        btc_ret = (btc_exit - btc_entry) / btc_entry * 100
        iwm_ret = (iwm_exit - iwm_entry) / iwm_entry * 100

        btc_dir = "UP" if btc_ret >= 0 else "DOWN"
        iwm_dir = "UP" if iwm_ret >= 0 else "DOWN"
        regime = REGIMES[(btc_dir, iwm_dir)]

        records.append({
            "entry_date": common[entry_idx].strftime("%Y-%m-%d"),
            "exit_date": common[exit_idx].strftime("%Y-%m-%d"),
            "btc_entry": round(btc_entry, 2),
            "btc_exit": round(btc_exit, 2),
            "btc_ret_pct": round(btc_ret, 2),
            "btc_dir": btc_dir,
            "iwm_entry": round(iwm_entry, 2),
            "iwm_exit": round(iwm_exit, 2),
            "iwm_ret_pct": round(iwm_ret, 2),
            "iwm_dir": iwm_dir,
            "regime": regime,
        })

    return pd.DataFrame(records)


# ── Option P&L engine (PUT + CALL) ───────────────────────────────────────

def compute_option_pnl(
    regimes_df: pd.DataFrame,
    iwm_vol: pd.Series,
    iwm_df: pd.DataFrame,
    hold_days: int = 10,
    contracts: int = 10,
    r: float = 0.045,
    iv_override: Optional[float] = None,
) -> pd.DataFrame:
    """
    For each regime window, compute P&L for buying:
      - IWM ATM PUT
      - IWM 5% OTM PUT
      - IWM ATM CALL
      - IWM 5% OTM CALL

    Returns one row per window with P&L for all four instruments.
    """
    T = hold_days / 252
    records = []

    for _, row in regimes_df.iterrows():
        S = row["iwm_entry"]
        S_exit = row["iwm_exit"]

        # Find vol at entry date
        entry_dt = pd.Timestamp(row["entry_date"])
        vol_idx = iwm_vol.index.get_indexer([entry_dt], method="nearest")[0]
        vol = iv_override or float(iwm_vol.iloc[vol_idx])
        if np.isnan(vol) or vol <= 0:
            vol = 0.22

        rec = dict(row)
        rec["iv_used"] = round(vol * 100, 1)

        # PUT: ATM (K = spot) and 5% OTM (K = spot * 0.95)
        for label, K_mult in [("atm", 1.0), ("5otm", 0.95)]:
            K = round(S * K_mult, 2)
            prem = bs_put_price(S, K, T, r, vol)
            intrinsic = max(K - S_exit, 0.0)
            cost = prem * 100 * contracts
            value = intrinsic * 100 * contracts
            pnl = value - cost

            rec[f"put_{label}_strike"] = K
            rec[f"put_{label}_prem"] = round(prem, 2)
            rec[f"put_{label}_cost"] = round(cost, 2)
            rec[f"put_{label}_value"] = round(value, 2)
            rec[f"put_{label}_pnl"] = round(pnl, 2)
            rec[f"put_{label}_itm"] = "ITM" if S_exit < K else "OTM"

        # CALL: ATM (K = spot) and 5% OTM (K = spot * 1.05)
        for label, K_mult in [("atm", 1.0), ("5otm", 1.05)]:
            K = round(S * K_mult, 2)
            prem = bs_call_price(S, K, T, r, vol)
            intrinsic = max(S_exit - K, 0.0)
            cost = prem * 100 * contracts
            value = intrinsic * 100 * contracts
            pnl = value - cost

            rec[f"call_{label}_strike"] = K
            rec[f"call_{label}_prem"] = round(prem, 2)
            rec[f"call_{label}_cost"] = round(cost, 2)
            rec[f"call_{label}_value"] = round(value, 2)
            rec[f"call_{label}_pnl"] = round(pnl, 2)
            rec[f"call_{label}_itm"] = "ITM" if S_exit > K else "OTM"

        records.append(rec)

    return pd.DataFrame(records)


# ── Live chain analysis ──────────────────────────────────────────────────

def analyze_live_options(spot: float, chains: Dict, max_dte: int = 45) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyze current IWM put AND call options.
    Returns (puts_df, calls_df).
    """
    put_rows = []
    call_rows = []
    today = datetime.now().date()

    for exp_str, chain_data in chains.items():
        exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
        dte = (exp_date - today).days
        if dte < 1 or dte > max_dte:
            continue

        for side, df_key, rows in [("put", "puts", put_rows), ("call", "calls", call_rows)]:
            opts = chain_data[df_key]
            if opts.empty:
                continue

            # Near-money filter
            if side == "put":
                opts = opts[(opts["strike"] >= spot * 0.90) & (opts["strike"] <= spot * 1.02)].copy()
            else:
                opts = opts[(opts["strike"] >= spot * 0.98) & (opts["strike"] <= spot * 1.10)].copy()

            for _, row in opts.iterrows():
                K = float(row["strike"])
                bid = float(row.get("bid", 0))
                ask = float(row.get("ask", 0))
                mid = (bid + ask) / 2 if bid > 0 and ask > 0 else float(row.get("lastPrice", 0))
                volume = int(row.get("volume", 0)) if not pd.isna(row.get("volume")) else 0
                oi = int(row.get("openInterest", 0)) if not pd.isna(row.get("openInterest")) else 0
                iv = float(row.get("impliedVolatility", 0))

                if side == "put":
                    otm_pct = (spot - K) / spot * 100
                    breakeven = K - mid
                else:
                    otm_pct = (K - spot) / spot * 100
                    breakeven = K + mid

                rows.append({
                    "expiration": exp_str,
                    "dte": dte,
                    "strike": K,
                    "otm_pct": round(otm_pct, 1),
                    "bid": round(bid, 2),
                    "ask": round(ask, 2),
                    "mid": round(mid, 2),
                    "iv": round(iv * 100, 1),
                    "volume": volume,
                    "open_interest": oi,
                    "cost_pct_spot": round(mid / spot * 100, 2),
                    "breakeven": round(breakeven, 2),
                    "cost_10_contracts": round(mid * 100 * 10, 2),
                })

    return pd.DataFrame(put_rows), pd.DataFrame(call_rows)


# ── Output: Regime matrix ────────────────────────────────────────────────

def print_regime_distribution(regimes_df: pd.DataFrame):
    """Print 2x2 regime frequency matrix."""
    print("\n" + "=" * 90)
    print("REGIME DISTRIBUTION  (Base=BTC, Pegged=IWM)")
    print("=" * 90)

    total = len(regimes_df)

    print(f"\n  {'':>20s}  {'IWM UP':>18s}  {'IWM DOWN':>18s}")
    print(f"  {'':>20s}  {'─' * 18}  {'─' * 18}")

    for btc_dir in ["UP", "DOWN"]:
        cells = []
        for iwm_dir in ["UP", "DOWN"]:
            subset = regimes_df[
                (regimes_df["btc_dir"] == btc_dir) &
                (regimes_df["iwm_dir"] == iwm_dir)
            ]
            n = len(subset)
            pct = n / total * 100 if total > 0 else 0
            avg_btc = subset["btc_ret_pct"].mean() if n > 0 else 0
            avg_iwm = subset["iwm_ret_pct"].mean() if n > 0 else 0
            cells.append(f"{n:>2d} ({pct:4.0f}%) {avg_btc:+.1f}%B")
        print(f"  {'BTC ' + btc_dir:>20s}  {cells[0]:>18s}  {cells[1]:>18s}")

    print(f"\n  Total windows: {total}")
    # Rolling correlation
    btc_rets = regimes_df["btc_ret_pct"].values
    iwm_rets = regimes_df["iwm_ret_pct"].values
    if len(btc_rets) > 2:
        corr = np.corrcoef(btc_rets, iwm_rets)[0, 1]
        print(f"  BTC-IWM return correlation: {corr:.2f}")
    print()


def print_regime_option_pnl(pnl_df: pd.DataFrame):
    """Print PUT vs CALL P&L breakdown for each of the 4 regimes."""
    print("=" * 90)
    print("PUT vs CALL PROFITABILITY BY REGIME  (10 contracts per trade)")
    print("=" * 90)

    instruments = [
        ("put_atm",  "IWM ATM PUT"),
        ("put_5otm", "IWM 5% OTM PUT"),
        ("call_atm",  "IWM ATM CALL"),
        ("call_5otm", "IWM 5% OTM CALL"),
    ]

    regime_order = [
        "RISK-ON RALLY",
        "BTC LEADS (divergence)",
        "IWM LEADS (divergence)",
        "RISK-OFF SELLOFF",
    ]

    for regime in regime_order:
        subset = pnl_df[pnl_df["regime"] == regime]
        if subset.empty:
            continue

        n = len(subset)
        avg_btc = subset["btc_ret_pct"].mean()
        avg_iwm = subset["iwm_ret_pct"].mean()

        print(f"\n  {regime}  ({n} windows, avg BTC {avg_btc:+.1f}%, avg IWM {avg_iwm:+.1f}%)")
        print(f"  {'─' * 85}")
        print(f"    {'Instrument':>20s}  {'Total Cost':>10s}  {'Total Val':>10s}  "
              f"{'Net P&L':>10s}  {'Win%':>6s}  {'Avg P&L':>10s}  {'ROI':>7s}")
        print(f"    {'─' * 20}  {'─' * 10}  {'─' * 10}  {'─' * 10}  {'─' * 6}  {'─' * 10}  {'─' * 7}")

        for key, label in instruments:
            cost_col = f"{key}_cost"
            val_col = f"{key}_value"
            pnl_col = f"{key}_pnl"

            total_cost = subset[cost_col].sum()
            total_val = subset[val_col].sum()
            total_pnl = subset[pnl_col].sum()
            wins = (subset[pnl_col] > 0).sum()
            win_pct = wins / n * 100 if n > 0 else 0
            avg_pnl = subset[pnl_col].mean()
            roi = total_pnl / total_cost * 100 if total_cost > 0 else 0

            print(
                f"    {label:>20s}  ${total_cost:>9,.0f}  ${total_val:>9,.0f}  "
                f"${total_pnl:>+9,.0f}  {win_pct:>5.0f}%  ${avg_pnl:>+9,.0f}  "
                f"{roi:>+6.1f}%"
            )

    print()


def print_regime_detail_table(pnl_df: pd.DataFrame):
    """Print per-window detail showing both assets + all 4 option P&Ls."""
    print("=" * 110)
    print("WINDOW-BY-WINDOW DETAIL")
    print("=" * 110)

    print(f"\n  {'Entry':>10s}  {'BTC%':>6s}  {'IWM%':>6s}  {'Regime':>24s}  "
          f"{'PUT ATM':>9s}  {'PUT 5OTM':>9s}  {'CALL ATM':>9s}  {'CALL 5OTM':>10s}")
    print(f"  {'─' * 10}  {'─' * 6}  {'─' * 6}  {'─' * 24}  "
          f"{'─' * 9}  {'─' * 9}  {'─' * 9}  {'─' * 10}")

    for _, r in pnl_df.iterrows():
        print(
            f"  {r['entry_date']:>10s}  {r['btc_ret_pct']:>+5.1f}%  {r['iwm_ret_pct']:>+5.1f}%  "
            f"{r['regime']:>24s}  "
            f"${r['put_atm_pnl']:>+8,.0f}  ${r['put_5otm_pnl']:>+8,.0f}  "
            f"${r['call_atm_pnl']:>+8,.0f}  ${r['call_5otm_pnl']:>+9,.0f}"
        )

    # Totals
    print(f"  {'─' * 108}")
    print(
        f"  {'TOTAL':>10s}  {'':>6s}  {'':>6s}  {'':>24s}  "
        f"${pnl_df['put_atm_pnl'].sum():>+8,.0f}  "
        f"${pnl_df['put_5otm_pnl'].sum():>+8,.0f}  "
        f"${pnl_df['call_atm_pnl'].sum():>+8,.0f}  "
        f"${pnl_df['call_5otm_pnl'].sum():>+9,.0f}"
    )
    print()


def print_hedge_verdict(pnl_df: pd.DataFrame):
    """Summarize which instrument is best for hedging BTC downside."""
    print("=" * 90)
    print("HEDGE VERDICT: WHICH INSTRUMENT PROTECTS BTC DOWNSIDE?")
    print("=" * 90)

    # Focus on BTC DOWN regimes (the scenarios we need protection for)
    btc_down = pnl_df[pnl_df["btc_dir"] == "DOWN"]
    btc_up = pnl_df[pnl_df["btc_dir"] == "UP"]

    instruments = [
        ("put_atm",  "IWM ATM PUT"),
        ("put_5otm", "IWM 5% OTM PUT"),
        ("call_atm",  "IWM ATM CALL"),
        ("call_5otm", "IWM 5% OTM CALL"),
    ]

    print(f"\n  When BTC falls ({len(btc_down)} windows):")
    print(f"  {'─' * 70}")

    best_pnl = -float("inf")
    best_label = ""
    best_cost_ratio = float("inf")
    best_cost_label = ""

    for key, label in instruments:
        pnl_col = f"{key}_pnl"
        cost_col = f"{key}_cost"

        if btc_down.empty:
            continue

        down_pnl = btc_down[pnl_col].sum()
        down_cost = btc_down[cost_col].sum()
        up_cost = btc_up[pnl_col].sum() if not btc_up.empty else 0  # bleed when BTC is up
        total_pnl = pnl_df[pnl_col].sum()
        total_cost = pnl_df[cost_col].sum()

        # "Protection ratio" = P&L in BTC-down / total premium spent
        protection_ratio = down_pnl / total_cost * 100 if total_cost > 0 else 0

        print(
            f"    {label:>20s}:  P&L when BTC down: ${down_pnl:>+9,.0f}  |  "
            f"Bleed when BTC up: ${up_cost:>+9,.0f}  |  "
            f"Net all: ${total_pnl:>+9,.0f}  |  "
            f"Protection ratio: {protection_ratio:>+.1f}%"
        )

        if down_pnl > best_pnl:
            best_pnl = down_pnl
            best_label = label

        # Best cost efficiency: highest (down_pnl - up_bleed) / total_cost
        net_hedge = down_pnl + up_cost  # up_cost is negative (bleed)
        if total_cost > 0 and net_hedge / total_cost > -best_cost_ratio:
            best_cost_ratio = -net_hedge / total_cost
            best_cost_label = label

    # BTC-IWM correlation in down periods
    if not btc_down.empty and len(btc_down) > 1:
        corr_down = np.corrcoef(
            btc_down["btc_ret_pct"].values,
            btc_down["iwm_ret_pct"].values
        )[0, 1]
    else:
        corr_down = float("nan")

    print(f"\n  Findings:")
    print(f"    Best absolute hedge:    {best_label} (${best_pnl:+,.0f} in BTC-down windows)")
    print(f"    Best cost efficiency:   {best_cost_label}")
    if not np.isnan(corr_down):
        print(f"    BTC-IWM corr in down:   {corr_down:.2f}")
        if corr_down > 0.5:
            print(f"    -> HIGH correlation in selloffs: IWM PUTs ARE effective BTC hedge")
        elif corr_down > 0.2:
            print(f"    -> MODERATE correlation: IWM puts provide partial BTC hedge")
        else:
            print(f"    -> LOW correlation in selloffs: IWM puts are NOT a reliable BTC proxy hedge")

    # Divergence analysis
    diverged = pnl_df[
        ((pnl_df["btc_dir"] == "DOWN") & (pnl_df["iwm_dir"] == "UP")) |
        ((pnl_df["btc_dir"] == "UP") & (pnl_df["iwm_dir"] == "DOWN"))
    ]
    pct_diverged = len(diverged) / len(pnl_df) * 100 if len(pnl_df) > 0 else 0
    print(f"\n    Divergence frequency:   {len(diverged)}/{len(pnl_df)} windows ({pct_diverged:.0f}%)")
    if pct_diverged > 30:
        print(f"    -> WARNING: Assets diverge often — hedge slippage risk is material")
    else:
        print(f"    -> Assets mostly move together — proxy hedge thesis holds")

    print()


def print_live_chain(puts_df: pd.DataFrame, calls_df: pd.DataFrame, spot: float):
    """Print current put AND call options available."""
    for side_label, df in [("PUT", puts_df), ("CALL", calls_df)]:
        if df.empty:
            continue
        print(f"\n  {'─' * 90}")
        print(f"  LIVE IWM {side_label} OPTIONS (spot: ${spot:.2f})")
        print(f"  {'─' * 90}")

        for exp in sorted(df["expiration"].unique()):
            exp_opts = df[df["expiration"] == exp].sort_values("strike", ascending=(side_label == "CALL"))
            dte = exp_opts.iloc[0]["dte"]
            print(f"\n    Exp: {exp} (DTE {dte})")
            print(f"    {'Strike':>7s}  {'OTM%':>5s}  {'Bid':>6s}  {'Ask':>6s}  {'Mid':>6s}  "
                  f"{'IV%':>5s}  {'Vol':>6s}  {'OI':>7s}  {'Cost%':>6s}  {'10ct $':>9s}")

            for _, r in exp_opts.iterrows():
                print(
                    f"    ${r['strike']:>5.0f}  {r['otm_pct']:>4.1f}%  "
                    f"${r['bid']:>5.2f}  ${r['ask']:>5.2f}  ${r['mid']:>5.2f}  "
                    f"{r['iv']:>4.1f}%  {r['volume']:>5d}  {r['open_interest']:>6d}  "
                    f"{r['cost_pct_spot']:>5.2f}%  ${r['cost_10_contracts']:>8,.0f}"
                )
    print()


# ── Main ─────────────────────────────────────────────────────────────────

def run(
    weeks: int = 12,
    hold_days: int = 10,
    contracts: int = 10,
    iv_override: Optional[float] = None,
    live_chain: bool = False,
    max_dte: int = 45,
    btc_ticker: str = "BTC-USD",
):
    print("=" * 90)
    print("BTC x IWM CROSS-ASSET HEDGE PROFITABILITY")
    print("  PUT vs CALL  |  All 4 Regime Quadrants")
    print("=" * 90)
    print(f"  Lookback:   {weeks} weeks  |  Hold period: {hold_days} trading days")
    print(f"  Contracts:  {contracts}  |  IV: {'market RV' if not iv_override else f'{iv_override:.0%}'}")
    print(f"  Base asset: {btc_ticker}  |  Pegged asset: IWM")
    print()

    # ── Load both assets ─────────────────────────────────────────────
    days = weeks * 7 + 60
    print("Loading historical data...")
    btc = load_ticker_history(btc_ticker, days=days)
    iwm = load_ticker_history("IWM", days=days)

    btc_spot = float(btc["Close"].iloc[-1])
    iwm_spot = float(iwm["Close"].iloc[-1])
    print(f"  {btc_ticker}: {len(btc)} bars, {btc.index[0].date()} to {btc.index[-1].date()}, spot ${btc_spot:,.2f}")
    print(f"  IWM:     {len(iwm)} bars, {iwm.index[0].date()} to {iwm.index[-1].date()}, spot ${iwm_spot:.2f}")

    # Realized vols
    btc_rv = rolling_realized_vol(btc, trading_days=365)  # crypto = 365
    iwm_rv = rolling_realized_vol(iwm, trading_days=252)
    btc_rv_now = float(btc_rv.dropna().iloc[-1]) if len(btc_rv.dropna()) > 0 else 0.80
    iwm_rv_now = float(iwm_rv.dropna().iloc[-1]) if len(iwm_rv.dropna()) > 0 else 0.20
    print(f"  {btc_ticker} 21d RV: {btc_rv_now:.1%}  |  IWM 21d RV: {iwm_rv_now:.1%}")
    print(f"  Vol ratio: {btc_rv_now / iwm_rv_now:.1f}x  (BTC is {btc_rv_now / iwm_rv_now:.1f}x more volatile)")
    print()

    # ── Classify regimes ─────────────────────────────────────────────
    print("Classifying BTC x IWM regimes...")
    regimes = classify_regimes(btc, iwm, hold_days=hold_days)
    if regimes.empty:
        print("  Insufficient overlapping data for regime classification.\n")
        return

    print_regime_distribution(regimes)

    # ── Compute option P&L per window ────────────────────────────────
    print("Computing PUT and CALL P&L for each regime window...")
    pnl_df = compute_option_pnl(
        regimes,
        iwm_vol=iwm_rv,
        iwm_df=iwm,
        hold_days=hold_days,
        contracts=contracts,
        iv_override=iv_override,
    )

    print_regime_option_pnl(pnl_df)
    print_regime_detail_table(pnl_df)
    print_hedge_verdict(pnl_df)

    # ── Live chain ───────────────────────────────────────────────────
    if live_chain:
        print("Fetching live IWM options chain (puts + calls)...")
        chains = load_iwm_options_chain()
        if chains:
            puts_df, calls_df = analyze_live_options(iwm_spot, chains, max_dte=max_dte)
            print_live_chain(puts_df, calls_df, iwm_spot)

    # ── Aggregate summary ────────────────────────────────────────────
    print("=" * 90)
    print("AGGREGATE SUMMARY")
    print("=" * 90)

    for key, label in [("put_atm", "ATM PUT"), ("put_5otm", "5%OTM PUT"),
                       ("call_atm", "ATM CALL"), ("call_5otm", "5%OTM CALL")]:
        total_cost = pnl_df[f"{key}_cost"].sum()
        total_pnl = pnl_df[f"{key}_pnl"].sum()
        roi = total_pnl / total_cost * 100 if total_cost > 0 else 0
        print(f"  {label:>12s}:  cost ${total_cost:>9,.0f}  |  net P&L ${total_pnl:>+9,.0f}  |  ROI {roi:>+6.1f}%")

    print()


def parse_args():
    p = argparse.ArgumentParser(description="BTC x IWM Cross-Asset Hedge Profitability")
    p.add_argument("--weeks", type=int, default=12, help="Lookback weeks (default: 12)")
    p.add_argument("--hold-days", type=int, default=10, help="Hold period in trading days (default: 10)")
    p.add_argument("--contracts", type=int, default=10, help="Contracts per trade (default: 10)")
    p.add_argument("--iv", type=float, default=None, help="Override IWM IV (e.g. 0.25)")
    p.add_argument("--live-chain", action="store_true", help="Fetch live options chain")
    p.add_argument("--max-dte", type=int, default=45, help="Max DTE for live chain (default: 45)")
    p.add_argument("--btc-ticker", type=str, default="BTC-USD",
                   help="Base asset ticker (default: BTC-USD, can use GBTC)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        weeks=args.weeks,
        hold_days=args.hold_days,
        contracts=args.contracts,
        iv_override=args.iv,
        live_chain=args.live_chain,
        max_dte=args.max_dte,
        btc_ticker=args.btc_ticker,
    )
