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

def _parse_bse_date(values: pd.Series) -> pd.Series:
    """Parse ISO, compact YYYYMMDD, and day-first BSE dates without ambiguity."""
    text = values.astype("string").str.strip()
    compact = text.str.fullmatch(r"\d{8}", na=False)
    iso = text.str.match(r"^\d{4}-\d{2}-\d{2}(?:[ T].*)?$", na=False)

    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    if iso.any():
        parsed.loc[iso] = pd.to_datetime(text.loc[iso], errors="coerce", format="mixed")
    if compact.any():
        parsed.loc[compact] = pd.to_datetime(
            text.loc[compact], format="%Y%m%d", errors="coerce"
        )
    other = ~(iso | compact)
    if other.any():
        parsed.loc[other] = pd.to_datetime(
            text.loc[other], errors="coerce", dayfirst=True, format="mixed"
        )
    return parsed.dt.normalize()


def _remove_reversed_expiry_rows(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    """Exclude impossible option observations whose contract expired before trading."""
    if frame.empty or not {"trade_date", "expiry_date"}.issubset(frame.columns):
        return frame
    trade = pd.to_datetime(frame["trade_date"], errors="coerce")
    expiry = pd.to_datetime(frame["expiry_date"], errors="coerce")
    invalid = trade.notna() & expiry.notna() & (expiry < trade)
    count = int(invalid.sum())
    if count:
        sample = frame.loc[invalid, ["trade_date", "expiry_date"]].head(3).to_dict("records")
        logger.warning(
            "Excluded %d impossible option rows from %s (expiry before trade date); samples=%s",
            count, source, sample,
        )
        frame = frame.loc[~invalid].copy()
    return frame



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
            if is_zip and not zipfile.is_zipfile(io.BytesIO(content)):
                last_err = "invalid_zip_payload"
                time.sleep(1.5 * attempt)
                continue
            if not is_zip and content.lstrip()[:20].lower().startswith((b"<html", b"<!doctype")):
                last_err = "html_instead_of_csv"
                time.sleep(1.5 * attempt)
                continue
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
                names = [name for name in zf.namelist() if name.lower().endswith(".csv")]
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

    # UDiFF and legacy derivatives-bhavcopy columns (legacy names are uppercase).
    tckr = pick("TckrSymb", "ticker", "symbol", "scrip_name", "contract")
    strike_c = pick("StrkPric", "STRIKE_PR", "strike", "strike_price", "strikeprice")
    opt_c = pick("OptnTp", "OPTION_TYP", "Option Type", "option_type", "optiontype", "opt_type", "ce_pe")
    exp_c = pick("XpryDt", "EXPIRY_DT", "expiry", "expiry_date", "exp_date", "expirydate")
    open_c = pick("OpnPric", "open", "open_price")
    high_c = pick("HghPric", "high", "high_price")
    low_c = pick("LwPric", "low", "low_price")
    close_c = pick("ClsPric", "close", "close_price")
    settle_c = pick("SttlmPric", "SETTLE_PR", "settle_price", "settlement", "settle")
    last_c = pick("LastPric", "last", "last_price", "ltp")
    und_c = pick("UndrlygPric", "underlying", "underlying_price", "spot")
    vol_c = pick("TtlTradgVol", "CONTRACTS", "Contracts Traded", "volume", "traded_contracts", "no_of_contracts")
    oi_c = pick("OpnIntrst", "OPEN_INT", "Open Interest", "open_interest", "oi", "openinterest")
    lot_c = pick(
        "NewBrdLotQty", "LotSize", "lot_size", "board_lot", "market_lot", "BrdLotQty"
    )
    inst_c = pick("FinInstrmTp", "Contract Type", "instrument", "instrument_type", "inst_type")
    trad_c = pick("TradDt", "TIMESTAMP", "L T Date", "trade_date", "timestamp", "date", "BizDt")

    out = pd.DataFrame()

    # Prefer column from file; fall back to explicit trade_date argument
    if trad_c:
        out["trade_date"] = _parse_bse_date(raw[trad_c])
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
        out["expiry_date"] = _parse_bse_date(raw[exp_c])
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
        instrument_values = raw[inst_c].astype(str).str.strip().str.upper()
        # Legacy BSE FO archive codes: IO = index option, IF = index future.
        out["instrument_type"] = instrument_values.replace({"IO": "OPTIDX", "IF": "FUTIDX"})
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

    # Legacy BSE archives use spaced display headers (Expiry, Strike,
    # Option Type, Contract Type) and do not carry a lot-size column.
    # SENSEX contracts in this pre-UDiFF archive window used a 10-unit lot.
    legacy_schema = (
        {"expiry_dt", "strike_pr"}.issubset(lower)
        or {"expiry", "strike", "option type", "contract type"}.issubset(lower)
    )
    if legacy_schema and lot_c is None:
        sensex_rows = out["symbol"].astype(str).str.startswith("SENSEX", na=False)
        out.loc[sensex_rows, "lot_size"] = 10

    # Sanitise premium: on expiry BSE sometimes sets ClsPric ≈ underlying index.
    # Prefer last / settle / intrinsic when close looks like the index level.
    out = _sanitise_premiums(out)

    # This project is specifically a SENSEX options backtester. Never silently
    # return every index/stock derivative when a source schema changes.
    sensex_mask = out["symbol"].str.fullmatch(r"SENSEX", case=False, na=False) | out["symbol"].str.startswith("SENSEX", na=False)
    out = out[sensex_mask].copy()
    out = out.dropna(subset=["trade_date", "expiry_date", "strike"], how="any")
    out = out[out["option_type"].isin(["CE", "PE"])]
    out = out[(out["strike"] > 0) & (out["close"].isna() | (out["close"] >= 0))]
    out = _remove_reversed_expiry_rows(out, str(path))
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
    # Never replace a missing/invalid market close with theoretical intrinsic value.
    # Intrinsic payoff is not an observed executable premium; mark it missing instead.
    out.loc[still_bad, "close"] = np.nan
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
    DEFAULT_PATHS.processed.mkdir(parents=True, exist_ok=True)

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
        logger.warning("No option rows parsed. Existing processed history is preserved.")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    # Keep both Calls and Puts, and reject every non-SENSEX instrument.
    options = combined[
        combined["symbol"].astype(str).str.startswith("SENSEX", na=False)
        & combined["option_type"].isin(["CE", "PE"])
        & combined["strike"].notna()
        & (combined["strike"] > 0)
    ].copy()

    out_path = DEFAULT_PATHS.processed / "sensex_options.csv"
    compressed_path = Path(str(out_path) + ".gz")
    existing = pd.DataFrame()
    if out_path.exists():
        existing = pd.read_csv(out_path)
    elif compressed_path.exists():
        existing = pd.read_csv(compressed_path, compression="gzip")
    if not existing.empty:
        options = pd.concat([existing, options], ignore_index=True)
    if not options.empty:
        keys = ["trade_date", "expiry_date", "symbol", "strike", "option_type"]
        options = options.drop_duplicates(subset=keys, keep="last").sort_values(keys).reset_index(drop=True)
        options.to_csv(out_path, index=False)
    logger.info("Merged %d SENSEX Call/Put option rows into %s", len(options), out_path)
    return options


def load_options(path: Optional[Path] = None) -> pd.DataFrame:
    """Load processed SENSEX options data."""
    path = path or DEFAULT_PATHS.processed / "sensex_options.csv"
    compressed = Path(str(path) + ".gz")
    if not path.exists() and compressed.exists():
        path = compressed
    if not path.exists():
        logger.warning("Options file not found: %s", path)
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["trade_date", "expiry_date"], compression="gzip" if str(path).endswith(".gz") else "infer")
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.normalize()
    df["expiry_date"] = pd.to_datetime(df["expiry_date"], errors="coerce").dt.normalize()
    df = _remove_reversed_expiry_rows(df, str(path))
    return df
