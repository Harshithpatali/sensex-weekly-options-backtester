"""
BSE Equity Derivatives (FO) historical bhavcopy downloader and normaliser.

Official endpoints (as of research 2026):
  UDiFF (>= 2024-07-08):
    https://www.bseindia.com/download/BhavCopy/Derivative/BhavCopy_BSE_FO_0_0_0_{YYYYMMDD}_F_0000.CSV
  Legacy (< 2024-07-08):
    https://www.bseindia.com/download/BhavCopy/Derivative/bhavcopy{DD-MM-YY}.zip

SENSEX options coverage begins roughly May 2023 (BSE equity derivatives relaunch).
Never fabricate premiums. Missing files are recorded in the collection manifest.
"""

from __future__ import annotations

import io
import logging
import time
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from sensex_options.config import DEFAULT_PATHS, SENSEX_OPTIONS_LAUNCH

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.bseindia.com/",
}

UDIFF_START = datetime(2024, 7, 8)


def _udiff_url(d: datetime) -> str:
    return (
        f"https://www.bseindia.com/download/BhavCopy/Derivative/"
        f"BhavCopy_BSE_FO_0_0_0_{d.strftime('%Y%m%d')}_F_0000.CSV"
    )


def _legacy_url(d: datetime) -> str:
    return (
        f"https://www.bseindia.com/download/BhavCopy/Derivative/"
        f"bhavcopy{d.strftime('%d-%m-%y')}.zip"
    )


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


def download_one_day(
    trade_date: datetime,
    raw_dir: Path,
    session: Optional[requests.Session] = None,
    timeout: int = 30,
    retries: int = 3,
) -> Tuple[Optional[Path], str]:
    """
    Download one day of BSE FO bhavcopy.

    Returns
    -------
    (path_or_None, status_message)
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    sess = session or _session()
    date_str = trade_date.strftime("%Y-%m-%d")

    if trade_date >= UDIFF_START:
        url = _udiff_url(trade_date)
        target = raw_dir / f"BSE_FO_{trade_date.strftime('%Y%m%d')}.csv"
        is_zip = False
    else:
        url = _legacy_url(trade_date)
        target = raw_dir / f"BSE_FO_{trade_date.strftime('%Y%m%d')}.zip"
        is_zip = True

    if target.exists() and target.stat().st_size > 100:
        return target, "already_exists"

    last_err = ""
    for attempt in range(1, retries + 1):
        try:
            resp = sess.get(url, timeout=timeout)
            if resp.status_code == 404:
                return None, "not_found_404"
            if resp.status_code != 200:
                last_err = f"HTTP {resp.status_code}"
                time.sleep(1.5 * attempt)
                continue
            content = resp.content
            if len(content) < 50:
                return None, "empty_or_tiny"
            target.write_bytes(content)
            return target, "downloaded"
        except requests.RequestException as e:
            last_err = str(e)
            time.sleep(1.5 * attempt)

    return None, f"failed:{last_err}"


def parse_bse_fo_file(path: Path, trade_date: Optional[datetime] = None) -> pd.DataFrame:
    """
    Parse a raw BSE FO file (CSV or ZIP) into a normalised DataFrame.

    Normalised columns:
      trade_date, expiry_date, instrument_type, symbol, strike,
      option_type, open, high, low, close, volume, open_interest, lot_size
    """
    if not path.exists():
        return pd.DataFrame()

    try:
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path, "r") as zf:
                names = zf.namelist()
                if not names:
                    return pd.DataFrame()
                with zf.open(names[0]) as f:
                    raw = pd.read_csv(f)
        else:
            raw = pd.read_csv(path)
    except Exception as e:
        logger.warning("Failed to parse %s: %s", path, e)
        return pd.DataFrame()

    if raw.empty:
        return pd.DataFrame()

    # Column name normalisation (UDiFF vs legacy)
    colmap = {c.lower().strip(): c for c in raw.columns}
    lower = {c.lower().strip(): c for c in raw.columns}

    def pick(*candidates: str) -> Optional[str]:
        for cand in candidates:
            if cand.lower() in lower:
                return lower[cand.lower()]
        return None

    # UDiFF-style columns
    tckr = pick("TckrSymb", "ticker", "symbol", "scrip_name", "contract")
    strike_c = pick("StrkPric", "strike", "strike_price", "strikeprice")
    opt_c = pick("OptnTp", "option_type", "optiontype", "opt_type", "ce_pe")
    exp_c = pick("XpryDt", "expiry", "expiry_date", "exp_date", "expirydate")
    open_c = pick("OpnPric", "open", "open_price")
    high_c = pick("HghPric", "high", "high_price")
    low_c = pick("LwPric", "low", "low_price")
    close_c = pick("ClsPric", "close", "close_price")
    settle_c = pick("SttlmPric", "settle_price", "settlement", "settle")
    last_c = pick("LastPric", "last", "last_price", "ltp")
    und_c = pick("UndrlygPric", "underlying", "underlying_price", "spot")
    vol_c = pick("TtlTradgVol", "volume", "contracts", "traded_contracts", "no_of_contracts")
    oi_c = pick("OpnIntrst", "open_interest", "oi", "openinterest")
    lot_c = pick(
        "NewBrdLotQty", "LotSize", "lot_size", "board_lot", "market_lot", "BrdLotQty"
    )
    inst_c = pick("FinInstrmTp", "instrument", "instrument_type", "inst_type")
    trad_c = pick("TradDt", "trade_date", "timestamp", "date", "BizDt")

    out = pd.DataFrame()

    # Prefer column from file; fall back to explicit trade_date argument
    if trad_c:
        out["trade_date"] = pd.to_datetime(raw[trad_c], errors="coerce").dt.normalize()
    elif trade_date is not None:
        out["trade_date"] = pd.Timestamp(trade_date).normalize()
    else:
        # try from filename
        stem = path.stem
        try:
            ds = stem.split("_")[-1]
            out["trade_date"] = pd.Timestamp(datetime.strptime(ds[:8], "%Y%m%d")).normalize()
        except Exception:
            out["trade_date"] = pd.NaT

    if exp_c:
        out["expiry_date"] = pd.to_datetime(raw[exp_c], errors="coerce").dt.normalize()
    else:
        out["expiry_date"] = pd.NaT

    if tckr:
        out["symbol"] = raw[tckr].astype(str).str.strip().str.upper()
    else:
        out["symbol"] = ""

    if strike_c:
        out["strike"] = pd.to_numeric(raw[strike_c], errors="coerce")
    else:
        out["strike"] = float("nan")

    if opt_c:
        ot = raw[opt_c].astype(str).str.strip().str.upper()
        # Normalise CE/PE / Call/Put
        ot = ot.replace({"CALL": "CE", "PUT": "PE", "C": "CE", "P": "PE"})
        out["option_type"] = ot
    else:
        out["option_type"] = ""

    if inst_c:
        out["instrument_type"] = raw[inst_c].astype(str).str.strip().str.upper()
    else:
        # Infer from option_type presence
        out["instrument_type"] = out["option_type"].apply(
            lambda x: "OPTIDX" if x in ("CE", "PE") else "UNKNOWN"
        )

    for src, dst in [
        (open_c, "open"),
        (high_c, "high"),
        (low_c, "low"),
        (close_c, "close"),
        (settle_c, "settle"),
        (last_c, "last"),
        (und_c, "underlying"),
        (vol_c, "volume"),
        (oi_c, "open_interest"),
        (lot_c, "lot_size"),
    ]:
        if src:
            out[dst] = pd.to_numeric(raw[src], errors="coerce")
        else:
            out[dst] = float("nan") if dst != "lot_size" else pd.NA

    # Sanitise premium: on expiry BSE sometimes sets ClsPric ≈ underlying index.
    # Prefer last / settle / intrinsic when close looks like the index level.
    out = _sanitise_premiums(out)

    # Filter to SENSEX related contracts when possible
    sensex_mask = out["symbol"].str.contains("SENSEX", case=False, na=False)
    if sensex_mask.any():
        out = out[sensex_mask].copy()

    out = out.dropna(subset=["strike"], how="all")
    return out.reset_index(drop=True)


def _sanitise_premiums(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix pathological closes where ClsPric equals (or nearly) the underlying index.
    Uses last price, settlement, or CE intrinsic max(0, underlying - strike).
    Does not invent prices when no usable field exists.
    """
    if df.empty or "close" not in df.columns:
        return df
    out = df.copy()
    und = out["underlying"] if "underlying" in out.columns else pd.Series(np.nan, index=out.index)
    last = out["last"] if "last" in out.columns else pd.Series(np.nan, index=out.index)
    settle = out["settle"] if "settle" in out.columns else pd.Series(np.nan, index=out.index)
    strike = out["strike"]
    opt = out.get("option_type", pd.Series("", index=out.index))

    # close looks like underlying (within 0.5%) and is huge vs typical weekly premium
    close_is_index = (
        und.notna()
        & out["close"].notna()
        & (und > 1000)
        & ((out["close"] - und).abs() / und < 0.005)
    )
    # or close > 50% of strike (implausible for a weekly premium quote)
    close_absurd = out["close"].notna() & strike.notna() & (out["close"] > strike * 0.5)

    bad = close_is_index | close_absurd
    if not bad.any():
        return out

    # Prefer last if sensible
    use_last = bad & last.notna() & (last < strike * 0.5)
    out.loc[use_last, "close"] = last[use_last]

    still_bad = bad & ~use_last
    use_settle = still_bad & settle.notna() & (settle < strike * 0.5)
    out.loc[use_settle, "close"] = settle[use_settle]

    still_bad = still_bad & ~use_settle
    # Intrinsic for CE at expiry-like rows
    is_ce = opt.astype(str).str.upper().isin(["CE", "CALL", "C"])
    intrinsic = (und - strike).clip(lower=0)
    use_intr = still_bad & is_ce & und.notna() & strike.notna()
    out.loc[use_intr, "close"] = intrinsic[use_intr]

    return out


def download_bse_options_range(
    start: str,
    end: str,
    raw_dir: Optional[Path] = None,
    sleep: float = 0.8,
) -> pd.DataFrame:
    """
    Download BSE FO bhavcopies for a date range and build a collection manifest.

    Does not invent data. Holidays / missing files are logged.
    """
    raw_dir = raw_dir or DEFAULT_PATHS.raw_options
    raw_dir.mkdir(parents=True, exist_ok=True)
    reports = DEFAULT_PATHS.reports
    reports.mkdir(parents=True, exist_ok=True)

    start_d = datetime.strptime(start, "%Y-%m-%d")
    end_d = datetime.strptime(end, "%Y-%m-%d")
    if start_d < datetime.strptime(SENSEX_OPTIONS_LAUNCH, "%Y-%m-%d"):
        logger.info(
            "Requested start %s is before approximate SENSEX options launch %s; "
            "files before launch will mostly be missing.",
            start,
            SENSEX_OPTIONS_LAUNCH,
        )

    dates = []
    cur = start_d
    while cur <= end_d:
        if cur.weekday() < 5:  # Mon-Fri only (still may be holidays)
            dates.append(cur)
        cur += timedelta(days=1)

    sess = _session()
    manifest_rows = []
    frames: List[pd.DataFrame] = []

    for d in tqdm(dates, desc="BSE FO download"):
        path, status = download_one_day(d, raw_dir, session=sess)
        manifest_rows.append({
            "trade_date": d.strftime("%Y-%m-%d"),
            "status": status,
            "file": str(path) if path else "",
        })
        if path is not None:
            parsed = parse_bse_fo_file(path, trade_date=d)
            if not parsed.empty:
                frames.append(parsed)
        time.sleep(sleep)

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = reports / "options_collection_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    logger.info("Manifest written to %s (%d days)", manifest_path, len(manifest))

    if not frames:
        logger.warning("No option rows parsed. Check network / BSE availability / date range.")
        empty = pd.DataFrame(columns=[
            "trade_date", "expiry_date", "instrument_type", "symbol", "strike",
            "option_type", "open", "high", "low", "close", "volume",
            "open_interest", "lot_size",
        ])
        empty.to_csv(DEFAULT_PATHS.processed / "sensex_options.csv", index=False)
        return empty

    combined = pd.concat(frames, ignore_index=True)
    # Keep both Calls and Puts: the multi-strategy research suite requires both sides.
    options = combined[
        (combined["option_type"].isin(["CE", "PE"]))
        & (combined["strike"].notna())
        & (combined["strike"] > 0)
    ].copy()
    sensex = options[options["symbol"].str.contains("SENSEX", case=False, na=False)]
    if len(sensex) > 0:
        options = sensex
    out_path = DEFAULT_PATHS.processed / "sensex_options.csv"
    options.to_csv(out_path, index=False)
    logger.info("Wrote %d SENSEX Call/Put option rows to %s", len(options), out_path)
    return options


def load_options(path: Optional[Path] = None) -> pd.DataFrame:
    """Load processed SENSEX options data."""
    path = path or DEFAULT_PATHS.processed / "sensex_options.csv"
    if not path.exists():
        logger.warning("Options file not found: %s", path)
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["trade_date", "expiry_date"])
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.normalize()
    df["expiry_date"] = pd.to_datetime(df["expiry_date"]).dt.normalize()
    return df
