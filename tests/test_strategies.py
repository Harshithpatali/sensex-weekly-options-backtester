import pandas as pd
from sensex_options.strategies import run_strategy_suite, period_pnl

def fixture():
    index=pd.DataFrame({"date":["2024-01-04","2024-01-05"],"close":[100.0,100.0]})
    rows=[]
    for typ in ("CE","PE"):
        for strike in (99.0,100.0,101.0):
            rows += [
                {"trade_date":"2024-01-04","expiry_date":"2024-01-05","symbol":"SENSEX","strike":strike,"option_type":typ,"close":5.0,"lot_size":10},
                {"trade_date":"2024-01-05","expiry_date":"2024-01-05","symbol":"SENSEX","strike":strike,"option_type":typ,"close":6.0,"lot_size":10},
            ]
    return index,pd.DataFrame(rows)

def test_strategy_suite_generates_five_baselines():
    index,options=fixture()
    trades,ranking,meta=run_strategy_suite(index,options,capital=10000,brokerage_per_order=0,slippage_points=0)
    assert set(ranking.strategy)=={"ATM_CALL","ATM_PUT","OTM_CALL_1","OTM_PUT_1","ATM_STRADDLE"}
    assert len(trades)==5
    assert ranking.net_pnl.notna().all()

def test_period_pnl_has_daily_weekly_monthly():
    index,options=fixture()
    trades,_,_=run_strategy_suite(index,options,capital=10000,brokerage_per_order=0,slippage_points=0)
    reports=period_pnl(trades)
    assert set(reports)=={"daily","weekly","monthly"}
    assert not reports["daily"].empty

def test_missing_lot_size_does_not_crash():
    index,options=fixture()
    options["lot_size"]=float("nan")
    trades,ranking,meta=run_strategy_suite(index,options)
    assert trades.empty
