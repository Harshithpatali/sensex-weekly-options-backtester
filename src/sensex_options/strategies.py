from __future__ import annotations
"""Comparable expiry-to-expiry research strategies. EOD marks are not executable fills."""
import pandas as pd

STRATEGIES = {
    "ATM_CALL": "Buy nearest-ATM call",
    "ATM_PUT": "Buy nearest-ATM put",
    "OTM_CALL_1": "Buy one strike-step OTM call",
    "OTM_PUT_1": "Buy one strike-step OTM put",
    "ATM_STRADDLE": "Buy ATM call + ATM put",
}

def _last_contract_row(frame, expiry, strike, opt_type, symbol):
    x=frame[(frame.expiry_date==expiry)&(frame.option_type==opt_type)&(frame.strike==strike)]
    if symbol is not None and "symbol" in x: x=x[x.symbol==symbol]
    return x.sort_values("trade_date").tail(1)

def run_strategy_suite(index: pd.DataFrame, options: pd.DataFrame, capital: float=10000.0, brokerage_per_order: float=20.0, slippage_points: float=0.5):
    if index.empty or options.empty:
        return pd.DataFrame(), pd.DataFrame(), {"error":"Index or options dataset is empty"}
    idx=index.copy(); opt=options.copy()
    idx["date"]=pd.to_datetime(idx.date).dt.normalize()
    opt["trade_date"]=pd.to_datetime(opt.trade_date).dt.normalize()
    opt["expiry_date"]=pd.to_datetime(opt.expiry_date).dt.normalize()
    for c in ("strike","close","lot_size"): opt[c]=pd.to_numeric(opt[c],errors="coerce")
    opt=opt.dropna(subset=["trade_date","expiry_date","strike","close"])
    opt=opt[(opt.close>0)&(opt.strike>0)&(opt.option_type.isin(["CE","PE"]))]
    trades=[]
    for expiry in sorted(opt.expiry_date.unique()):
        expiry=pd.Timestamp(expiry)
        sessions=idx.loc[idx.date<expiry,"date"]
        if sessions.empty: continue
        entry_date=sessions.max()
        spot_rows=idx[idx.date==entry_date]
        if spot_rows.empty: continue
        spot=float(spot_rows.iloc[-1].close)
        entry=opt[(opt.trade_date==entry_date)&(opt.expiry_date==expiry)]
        if entry.empty: continue
        strikes=sorted(entry.strike.dropna().unique())
        if not strikes: continue
        atm=min(strikes,key=lambda s:abs(float(s)-spot))
        step_candidates=[s for s in strikes if s>atm]
        step_down=[s for s in strikes if s<atm]
        otm_call=min(step_candidates) if step_candidates else None
        otm_put=max(step_down) if step_down else None
        exit_day=opt.loc[opt.expiry_date.eq(expiry),"trade_date"].max()
        if pd.isna(exit_day): continue
        for key,legs in {
            "ATM_CALL":[("CE",atm)],"ATM_PUT":[("PE",atm)],
            "OTM_CALL_1":[("CE",otm_call)],"OTM_PUT_1":[("PE",otm_put)],
            "ATM_STRADDLE":[("CE",atm),("PE",atm)]
        }.items():
            if any(strike is None for _,strike in legs): continue
            entry_rows=[]; exit_rows=[]
            for typ,strike in legs:
                er=entry[(entry.option_type==typ)&(entry.strike==strike)].sort_values("symbol").head(1)
                if er.empty: break
                symbol=er.iloc[0].get("symbol")
                xr=_last_contract_row(opt[opt.trade_date<=exit_day],expiry,strike,typ,symbol)
                if xr.empty: break
                entry_rows.append(er.iloc[0]); exit_rows.append(xr.iloc[0])
            if len(entry_rows)!=len(legs): continue
            lot_values=[pd.to_numeric(r.get("lot_size"),errors="coerce") for r in entry_rows]
            if any(pd.isna(v) or float(v)<=0 for v in lot_values): continue
            lots=int(lot_values[0])
            if any(int(v)!=lots for v in lot_values): continue
            entry_debit=sum(float(r.close) for r in entry_rows)+len(legs)*slippage_points
            exit_credit=sum(max(0.0,float(r.close)-slippage_points) for r in exit_rows)
            if entry_debit<=0: continue
            n_lots=int(capital//(entry_debit*lots))
            if n_lots<1: continue
            qty=n_lots*lots
            gross=(exit_credit-entry_debit)*qty
            costs=brokerage_per_order*2*len(legs)
            net=gross-costs
            trades.append({"strategy":key,"strategy_description":STRATEGIES[key],"expiry_date":expiry,"entry_date":entry_date,"exit_date":exit_day,"spot":spot,"strikes":"+".join(str(s) for _,s in legs),"legs":len(legs),"entry_debit":entry_debit,"exit_credit":exit_credit,"lot_size":lots,"lots":n_lots,"quantity":qty,"gross_pnl":round(gross,2),"estimated_costs":round(costs,2),"net_pnl":round(net,2),"return_pct":round(net/(entry_debit*qty)*100,3),"status":"completed"})
    trades_df=pd.DataFrame(trades)
    if trades_df.empty: return trades_df,pd.DataFrame(),{"error":"No complete trades; verify both CE/PE data and historical lot sizes."}
    trades_df["exit_date"]=pd.to_datetime(trades_df.exit_date)
    summary=[]
    for key,grp in trades_df.groupby("strategy"):
        pnl=grp.net_pnl
        gains=float(pnl[pnl>0].sum()); losses=float(-pnl[pnl<0].sum())
        eq=capital+pnl.cumsum()
        peak=pd.Series([capital]+eq.tolist()).cummax().iloc[1:].to_numpy()
        dd=eq.to_numpy()-peak
        dd_pct=(dd/peak*100) if len(peak) else []
        summary.append({"strategy":key,"description":STRATEGIES[key],"trades":len(grp),"net_pnl":round(float(pnl.sum()),2),"win_rate_pct":round(float((pnl>0).mean()*100),2),"profit_factor":round(gains/losses,3) if losses else None,"max_drawdown":round(float(dd.min()),2) if len(dd) else 0.0,"max_drawdown_pct":round(float(min(dd_pct)),2) if len(dd_pct) else 0.0,"avg_trade_pnl":round(float(pnl.mean()),2),"max_loss":round(float(pnl.min()),2)})
    rank=pd.DataFrame(summary).sort_values(["net_pnl","profit_factor"],ascending=[False,False],na_position="last")
    return trades_df,rank,{"capital_per_strategy":capital,"cost_assumptions":{"brokerage_per_order":brokerage_per_order,"slippage_points_per_leg_side":slippage_points},"note":"EOD close-to-close simulations; no intraday fills, bid/ask spread, market impact, margin funding or taxes beyond stated estimated brokerage/slippage."}

def period_pnl(trades: pd.DataFrame):
    if trades.empty: return {"daily":pd.DataFrame(),"weekly":pd.DataFrame(),"monthly":pd.DataFrame()}
    x=trades.copy(); x["exit_date"]=pd.to_datetime(x.exit_date)
    x["day"]=x.exit_date.dt.strftime("%Y-%m-%d")
    x["week"]=x.exit_date.dt.to_period("W-SUN").astype(str)
    x["month"]=x.exit_date.dt.to_period("M").astype(str)
    return {
      "daily":x.groupby(["day","strategy"],as_index=False).net_pnl.sum(),
      "weekly":x.groupby(["week","strategy"],as_index=False).net_pnl.sum(),
      "monthly":x.groupby(["month","strategy"],as_index=False).net_pnl.sum()
    }
