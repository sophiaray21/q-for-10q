import argparse
import importlib
import os
import time
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from sentiment import sentiment_from_file

# the module name starts with a digit so a normal import statement is a syntax error
tenQ = importlib.import_module("10QForm")

'''
State vector builder (ticket #4). Wrangles financials, market data and MD&A sentiment
into one row per stock-quarter and writes the csv that train.py / backtest.py already
expect: columns ticker, quarter_end, close, f0..f10.

The 11 features (this is also the ticket #5 state space definition):
    f0  revenue growth, year over year          \
    f1  net income margin                        |
    f2  operating cash flow / revenue            | 10-Q financials (SEC XBRL)
    f3  liabilities / assets                     |
    f4  diluted EPS change, year over year      /
    f5  trailing 1-quarter log return           \
    f6  daily volatility over the quarter        | market data (yfinance)
    f7  volume change vs prior quarter           |
    f8  trailing 12-month log return            /
    f9  MD&A sentiment scalar                   \  sentence-transformers (ticket #3)
    f10 change in sentiment vs prior quarter    /

Two design decisions worth knowing about:

1. Financials come from the SEC companyfacts XBRL API, not from parsing the filing
   HTML tables. The API returns properly tagged values (Revenues, NetIncomeLoss, ...)
   for any company, where table positions differ per company and per quarter.
   Wrinkle: cash flow items are reported year-to-date, so quarterly OCF has to be
   recovered by differencing consecutive YTD values within a fiscal year - see
   _duration_series().

2. Everything is aligned to the FILING date, not the quarter end. A 10-Q becomes
   public ~5 weeks after the quarter closes; pairing its features with quarter-end
   prices would hand the agent information the market didn't have yet (look-ahead
   bias). So market features use only data up to the filing date, and the `close`
   column - which reward.py turns into returns - is the first close on/after filing.

Normalization: features are z-scored with stats fit on the EARLIEST train_frac of
rows (chronologically, pooled across tickers), matching backtest.py's walk-forward
split. Fitting on the full history would leak future means/stds into the past.

Usage:
    python3 build_states.py --tickers AAPL MSFT JPM --name "Your Name" \
        --email you@example.com --limit 12 --out states.csv
'''

SEC_REQUEST_GAP_S = 0.15      # SEC asks for <=10 requests/sec; stay well under

# XBRL tag fallbacks, in priority order: companies switch tags across the years
# (e.g. Revenues -> RevenueFromContractWithCustomerExcludingAssessedTax after 2018)
# and banks report total net revenue under RevenuesNetOfInterestExpense
REVENUE_TAGS = ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues",
                "SalesRevenueNet", "RevenuesNetOfInterestExpense"]
NET_INCOME_TAGS = ["NetIncomeLoss"]
OCF_TAGS = ["NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"]
ASSETS_TAGS = ["Assets"]
LIABILITIES_TAGS = ["Liabilities"]
EQUITY_TAGS = ["StockholdersEquity",
               "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"]
EPS_TAGS = ["EarningsPerShareDiluted", "EarningsPerShareBasic"]


def _sec_headers(name, email):
    return {"User-Agent": f"{name} {email}", "Accept-Encoding": "gzip, deflate"}


def fetch_companyfacts(name, email, cik):
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    r = requests.get(url, headers=_sec_headers(name, email), timeout=60)
    r.raise_for_status()
    time.sleep(SEC_REQUEST_GAP_S)
    return r.json()


def _parse_date(s):
    return datetime.strptime(s, "%Y-%m-%d").date()


def _entries_for_tags(facts, tags, unit_pref=("USD", "USD/shares")):
    '''All fact entries for the first tags that exist, keeping tag priority order.'''
    gaap = facts.get("facts", {}).get("us-gaap", {})
    out = []
    for tag in tags:
        if tag not in gaap:
            continue
        units = gaap[tag]["units"]
        unit = next((u for u in unit_pref if u in units), next(iter(units)))
        out.append(units[unit])
    return out


def _duration_series(facts, tags):
    '''
    {period_end: quarterly value} for a duration concept (revenue, net income, ...).

    XBRL entries come in two flavors: clean ~90-day quarters, and year-to-date
    windows (always for cash flow, sometimes for the rest). YTD pairs that share a
    fiscal-year start and end one quarter apart are differenced to recover the
    missing quarter.
    '''
    quarterly = {}
    for entries in _entries_for_tags(facts, tags):
        windows = []
        for e in entries:
            if e.get("form") not in ("10-Q", "10-K") or "start" not in e:
                continue
            windows.append((_parse_date(e["start"]), _parse_date(e["end"]), e["val"]))
        for start, end, val in windows:
            if 60 <= (end - start).days <= 100:
                quarterly.setdefault(end, val)
        # recover quarters hidden inside YTD windows: two windows sharing a fiscal-year
        # start whose ends are one quarter apart difference to a clean quarter. The
        # shorter window can itself be a plain Q1 (its "YTD" IS the quarter), which is
        # why this pairs across ALL windows, not just the >100-day ones.
        for s1, e1, v1 in windows:
            for s2, e2, v2 in windows:
                if s1 == s2 and 60 <= (e2 - e1).days <= 100:
                    quarterly.setdefault(e2, v2 - v1)
    return quarterly


def _instant_series(facts, tags):
    '''{period_end: value} for a point-in-time concept (assets, liabilities, ...).'''
    series = {}
    for entries in _entries_for_tags(facts, tags):
        for e in entries:
            if e.get("form") in ("10-Q", "10-K"):
                series.setdefault(_parse_date(e["end"]), e["val"])
    return series


def _lookup(series, target, tol_days=10):
    '''Value whose period end is within tol_days of target, else None. Filings
    occasionally report period ends a few days off the calendar quarter.'''
    if target in series:
        return series[target]
    best = None
    for d, v in series.items():
        gap = abs((d - target).days)
        if gap <= tol_days and (best is None or gap < best[0]):
            best = (gap, v)
    return best[1] if best else None


def financial_features(facts, quarter_end):
    '''f0..f4 for one quarter, np.nan where the data just isn't there.'''
    rev = _duration_series(facts, REVENUE_TAGS)
    ni = _duration_series(facts, NET_INCOME_TAGS)
    ocf = _duration_series(facts, OCF_TAGS)
    eps = _duration_series(facts, EPS_TAGS)
    assets = _instant_series(facts, ASSETS_TAGS)
    liab = _instant_series(facts, LIABILITIES_TAGS)
    equity = _instant_series(facts, EQUITY_TAGS)

    year_ago = quarter_end - timedelta(days=365)
    rev_q = _lookup(rev, quarter_end)
    rev_yr = _lookup(rev, year_ago)
    ni_q = _lookup(ni, quarter_end)
    ocf_q = _lookup(ocf, quarter_end)
    eps_q = _lookup(eps, quarter_end)
    eps_yr = _lookup(eps, year_ago)
    a_q = _lookup(assets, quarter_end)
    l_q = _lookup(liab, quarter_end)
    if l_q is None:  # some filers skip the Liabilities total; derive it
        e_q = _lookup(equity, quarter_end)
        if a_q is not None and e_q is not None:
            l_q = a_q - e_q

    f0 = rev_q / rev_yr - 1 if rev_q and rev_yr else np.nan
    f1 = ni_q / rev_q if ni_q is not None and rev_q else np.nan
    f2 = ocf_q / rev_q if ocf_q is not None and rev_q else np.nan
    f3 = l_q / a_q if l_q is not None and a_q else np.nan
    # absolute EPS change, not percent: EPS near zero makes percent changes explode
    f4 = eps_q - eps_yr if eps_q is not None and eps_yr is not None else np.nan
    return [f0, f1, f2, f3, f4]


def market_features(hist, filing_date):
    '''f5..f8 using ONLY closes up to the filing date, plus the first close on/after
    it (the price the agent could actually trade at). Returns (features, close).'''
    past = hist[hist.index.date <= filing_date]
    future = hist[hist.index.date >= filing_date]
    if len(past) < 260 or len(future) == 0:
        return [np.nan] * 4, np.nan
    px = past["Close"].to_numpy()
    vol = past["Volume"].to_numpy()
    logret = np.diff(np.log(px[-64:]))
    f5 = float(np.log(px[-1] / px[-64]))
    f6 = float(np.std(logret))
    f7 = float(vol[-63:].mean() / vol[-126:-63].mean() - 1) if vol[-126:-63].mean() > 0 else np.nan
    f8 = float(np.log(px[-1] / px[-253]))
    return [f5, f6, f7, f8], float(future["Close"].iloc[0])


def filing_sentiment(name, email, ticker, cik, filing, cache, filings_dir="filings"):
    '''f9 for one filing, downloading the document if needed. cache maps
    accessionNumber -> score so reruns skip the embedding work.'''
    accn = filing["accessionNumber"]
    if accn in cache:
        return cache[accn]
    dest_dir = os.path.join(filings_dir, ticker)
    path = os.path.join(dest_dir, filing["primaryDocument"])
    if not os.path.exists(path):
        url = tenQ.get_10Q_from_filings(cik, accn, filing["primaryDocument"])
        path = tenQ.download_from_url(name, email, url, dest_dir=dest_dir)
        time.sleep(SEC_REQUEST_GAP_S)
    score, n_chars = sentiment_from_file(path)
    if n_chars == 0:
        print(f"    warning: no MD&A section found in {filing['primaryDocument']}")
        score = np.nan
    cache[accn] = score
    return score


def build_for_ticker(name, email, ticker, limit, cache):
    print(f"{ticker}: fetching filings list...")
    filings, cik = tenQ.get_10Q_filings_from_ticker(name, email, ticker)
    filings = sorted(filings[:limit], key=lambda f: f["filingDate"])  # oldest first
    facts = fetch_companyfacts(name, email, cik)
    hist = yf.Ticker(ticker).history(period="max")
    hist.index = hist.index.tz_localize(None)

    rows = []
    prev_sent = np.nan
    for filing in filings:
        quarter_end = _parse_date(filing["reportDate"])
        filing_date = _parse_date(filing["filingDate"])
        fin = financial_features(facts, quarter_end)
        mkt, close = market_features(hist, filing_date)
        f9 = filing_sentiment(name, email, ticker, cik, filing, cache)
        f10 = f9 - prev_sent if not (np.isnan(f9) or np.isnan(prev_sent)) else np.nan
        prev_sent = f9
        rows.append({"ticker": ticker, "quarter_end": quarter_end.isoformat(),
                     "close": close,
                     **{f"f{i}": v for i, v in enumerate(fin + mkt + [f9, f10])}})
        print(f"  {quarter_end} (filed {filing_date}): sentiment {f9:+.3f}" if not np.isnan(f9)
              else f"  {quarter_end} (filed {filing_date}): sentiment n/a")
    return rows


def normalize(df, train_frac=0.7):
    '''Z-score f0..f9 using stats from the earliest train_frac of rows (by date,
    pooled across tickers) so the held-out period never influences the scaling.
    f10 is a difference of the already-normalized-scale f9, left as is.'''
    feature_cols = [c for c in df.columns if c.startswith("f")]
    df = df.sort_values(["quarter_end", "ticker"]).reset_index(drop=True)
    cut = max(int(len(df) * train_frac), 2)
    fit = df.iloc[:cut]
    for c in feature_cols:
        mean, std = fit[c].mean(), fit[c].std()
        if not np.isfinite(std) or std == 0:
            std = 1.0
        df[c] = (df[c] - mean) / std
    return df


def main():
    p = argparse.ArgumentParser(description="Build the state-vector csv (ticket #4)")
    p.add_argument("--tickers", nargs="+", required=True)
    p.add_argument("--name", default="Sophia Ray")
    p.add_argument("--email", default="sophia@example.com")
    p.add_argument("--limit", type=int, default=12, help="10-Qs per ticker (newest N)")
    p.add_argument("--out", default="states.csv")
    p.add_argument("--train-frac", type=float, default=0.7,
                   help="fraction of history used to fit normalization stats - keep "
                        "in sync with backtest.py's split_stocks default")
    p.add_argument("--cache", default="sentiment_cache.csv")
    args = p.parse_args()

    cache = {}
    if os.path.exists(args.cache):
        cached = pd.read_csv(args.cache)
        cache = dict(zip(cached["accession"], cached["sentiment"]))
        print(f"loaded {len(cache)} cached sentiment scores from {args.cache}")

    rows = []
    for ticker in args.tickers:
        try:
            rows.extend(build_for_ticker(args.name, args.email, ticker, args.limit, cache))
        except Exception as e:
            print(f"{ticker}: FAILED ({e}) - skipping")

    pd.DataFrame({"accession": list(cache), "sentiment": list(cache.values())}
                 ).to_csv(args.cache, index=False)

    df = pd.DataFrame(rows)
    feature_cols = [c for c in df.columns if c.startswith("f")]
    before = len(df)
    df["f10"] = df["f10"].fillna(0.0)  # first quarter per ticker has no prior sentiment
    df = df.dropna(subset=feature_cols + ["close"])
    if before - len(df):
        print(f"dropped {before - len(df)}/{before} rows with missing features")

    df = normalize(df, args.train_frac)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df)} stock-quarters x {len(feature_cols)} features to {args.out}")
    print("next: python3 train.py", args.out, " and  python3 backtest.py", args.out)


if __name__ == "__main__":
    main()
