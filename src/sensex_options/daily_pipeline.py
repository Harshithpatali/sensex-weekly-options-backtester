from __future__ import annotations
"""Incremental daily collection + strategy replay. Run in GitHub Actions at 06:30 Asia/Kolkata."""
import json, logging
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from sensex_options.config import DEFAULT_PATHS, SENSEX_OPTIONS_LAUNCH
from sensex_options.sources.bse import download_one_day, parse_bse_fo_file
from sensex_options.sources.index import download_sensex_index, load_index
from sensex_options.sources.bse import load_options
from sensex_options.validation import validate_all, save_validation_report
from sensex_options.strategies import run_strategy_suite, period_pnl

logging.basicConfig(level=logging.INFO)
log=logging.getLogger("daily_pipeline")

def merge_csv(new: pd.DataFrame, path: Path, keys: list[str]):
    path.parent.mkdir(parents=True,exist_ok=True)
    old=pd.read_csv(path) if path.exists() else pd.DataFrame()
    combined=pd.concat([old,new],ignore_index=True) if not old.empty else new.copy()
    if not combined.empty:
        combined=combined.drop_duplicates(subset=keys,keep="last").sort_values(keys)
        combined.to_csv(path,index=False)
    return combined

def run():
    DEFAULT_PATHS.ensure()
    # At 06:30 IST collect the most recent completed calendar day. A BSE 404 is normal on holidays.
    target=(datetime.utcnow().date()-timedelta(days=1))
    day=datetime.combine(target,datetime.min.time())
    raw_path,status=download_one_day(day,DEFAULT_PATHS.raw_options)
    log.info("BSE download %s: %s",target,status)
    if raw_path:
        daily=parse_bse_fo_file(raw_path,day)
        daily=daily[daily.symbol.str.contains("SENSEX",case=False,na=False)]
        if not daily.empty:
            merge_csv(daily,DEFAULT_PATHS.processed/"sensex_options.csv",["trade_date","expiry_date","symbol","strike","option_type"])
    # Refresh index history; merge so intermittent vendor gaps don't erase existing observations.
    end=target.isoformat()
    fresh=download_sensex_index(SENSEX_OPTIONS_LAUNCH,end,processed_path=DEFAULT_PATHS.processed/"index_refresh.csv")
    if not fresh.empty:
        merge_csv(fresh,DEFAULT_PATHS.processed/"sensex_index.csv",["date"])
    index=load_index()
    options=load_options()
    quality=validate_all(index,options)
    save_validation_report(quality)
    if quality.get("errors"):
        raise RuntimeError("Daily data validation failed: "+json.dumps(quality["errors"]))
    trades,ranking,meta=run_strategy_suite(index,options)
    if trades.empty:
        raise RuntimeError(meta.get("error","No strategy trades generated"))
    out=DEFAULT_PATHS.outputs;out.mkdir(parents=True,exist_ok=True)
    trades.to_csv(out/"strategy_trade_ledger.csv",index=False)
    ranking.to_csv(out/"strategy_leaderboard.csv",index=False)
    periods=period_pnl(trades)
    for period,frame in periods.items(): frame.to_csv(out/f"{period}_strategy_pnl.csv",index=False)
    with open(out/"daily_run.json","w") as f:
        json.dump({"run_date":datetime.utcnow().isoformat()+"Z","data_quality":quality,"strategy_run":meta,"strategy_count":len(ranking)},f,indent=2,default=str)
    log.info("Daily strategy leaderboard:\n%s",ranking.to_string(index=False))

if __name__=="__main__": run()
