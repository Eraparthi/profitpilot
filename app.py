import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests, xml.etree.ElementTree as ET
from datetime import datetime
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed

st.set_page_config(page_title="ProfitPilot V2", page_icon="📈", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
.block-container{padding-top:1rem;padding-bottom:4rem;max-width:1100px}
div[data-testid="stMetric"]{background:var(--secondary-background);padding:12px;border-radius:12px}
div.stButton>button,div.stDownloadButton>button{width:100%;border-radius:10px;min-height:42px}
[data-testid="stTabs"] button{font-size:14px}
.small-note{font-size:12px;color:gray}
</style>
""",unsafe_allow_html=True)

st.title("📈 ProfitPilot")
st.caption("V2 · Indian stock scanner · Mobile-first · Research only")
st.info("Free-data limitation: public Yahoo Finance feeds do not reliably provide NSE pre-open auction prices or order-book imbalance. ProfitPilot V2 uses the latest available daily data plus news catalysts to create a **pre-market watchlist**, not a verified live pre-open signal. Confirm prices and announcements with NSE/BSE before acting.")

# Mobile-friendly settings
with st.expander("⚙️ Scanner settings", expanded=True):
    a,b=st.columns(2)
    with a:
        strategy=st.selectbox("Strategy",["Both","Intraday watchlist","Swing (2–10 days)"])
        risk=st.selectbox("Risk profile",["Balanced","Conservative","Aggressive"])
    with b:
        capital=st.number_input("Capital (₹, optional)",min_value=0,value=0,step=10000)
        lookback=st.selectbox("Historical data",["2y","3y","5y"],index=2)
    c,d=st.columns(2)
    with c: min_price=st.number_input("Minimum share price ₹",min_value=1.0,value=20.0)
    with d: min_value=st.number_input("Minimum avg traded value ₹ Cr",min_value=0.0,value=2.0,step=1.0)
    max_stocks=st.slider("Maximum watchlist stocks",5,30,15)

DEFAULT = [
"RELIANCE.NS","HDFCBANK.NS","ICICIBANK.NS","SBIN.NS","BHARTIARTL.NS","TCS.NS","INFY.NS","LT.NS","ITC.NS","AXISBANK.NS","KOTAKBANK.NS","BAJFINANCE.NS","MARUTI.NS","SUNPHARMA.NS","TATAMOTORS.NS",
"DIXON.NS","PERSISTENT.NS","TRENT.NS","BEL.NS","HAL.NS","BSE.NS","CDSL.NS","KAYNES.NS","KPITTECH.NS","COFORGE.NS","POLYCAB.NS","CAMS.NS","MAZDOCK.NS","RVNL.NS","IREDA.NS","MOSCHIP.NS","GANDHAR.NS","PINEALABS.NS","BIRLASOFT.NS","LAURUSLABS.NS"
]
up=st.file_uploader("Optional: upload ticker CSV (column named ticker)",type=["csv"])
if up:
    try: tickers=pd.read_csv(up)["ticker"].dropna().astype(str).str.strip().tolist()
    except Exception: tickers=DEFAULT; st.error("CSV must contain a ticker column.")
else: tickers=DEFAULT
st.caption(f"Universe: {len(tickers)} symbols · NSE symbols use .NS; BSE symbols use .BO")

def download(t):
    try:
        q=yf.download(t,period=lookback,interval="1d",auto_adjust=True,progress=False,threads=False)
        if q is None or q.empty: return t,None
        if isinstance(q.columns,pd.MultiIndex): q.columns=q.columns.get_level_values(0)
        q.columns=[str(x).title() for x in q.columns]
        q=q[[x for x in ["Open","High","Low","Close","Volume"] if x in q.columns]].dropna()
        if len(q)<65 or not set(["Open","High","Low","Close","Volume"]).issubset(q.columns): return t,None
        return t,q
    except: return t,None

@st.cache_data(ttl=900,show_spinner=False)
def fetch_all(symbols,period):
    out={}
    with ThreadPoolExecutor(max_workers=8) as pool:
        fs=[pool.submit(lambda sym: (sym,yf.download(sym,period=period,interval="1d",auto_adjust=True,progress=False,threads=False)),s) for s in symbols]
        for f in as_completed(fs):
            try:
                sym,q=f.result()
                if q is None or q.empty: continue
                if isinstance(q.columns,pd.MultiIndex): q.columns=q.columns.get_level_values(0)
                q.columns=[str(x).title() for x in q.columns]
                need=["Open","High","Low","Close","Volume"]
                if set(need).issubset(q.columns) and len(q)>65: out[sym]=q[need].dropna()
            except: pass
    return out

with st.spinner("Refreshing historical market data…"):
    hist=fetch_all(tuple(tickers),lookback)
st.caption(f"Last refresh: {datetime.now(ZoneInfo('Asia/Kolkata')).strftime('%d %b %Y, %I:%M %p IST')} · {len(hist)} symbols with usable data")
if not hist:
    st.error("No price data returned. Check internet access or try again later.")
    st.stop()

def indicators(df):
    x=df.copy()
    x["r1"]=x.Close.pct_change()
    x["r3"]=x.Close.pct_change(3)
    x["r5"]=x.Close.pct_change(5)
    x["r20"]=x.Close.pct_change(20)
    x["v20"]=x.Volume.rolling(20).mean()
    x["vr"]=x.Volume/x.v20
    x["sma20"]=x.Close.rolling(20).mean()
    x["sma50"]=x.Close.rolling(50).mean()
    x["prior_high20"]=x.High.shift(1).rolling(20).max()
    x["avgvalcr"]=(x.Close*x.Volume).rolling(20).mean()/1e7
    x["atr_pct"]=(x.High-x.Low).rolling(14).mean()/x.Close*100
    return x
ind={t:indicators(q) for t,q in hist.items()}
mkt=pd.concat([x.r1.rename(t) for t,x in ind.items()],axis=1).mean(axis=1)
mkt20=(1+mkt.tail(20).fillna(0)).prod()-1

def make_row(t,x):
    r=x.iloc[-1]
    if pd.isna(r.sma50) or pd.isna(r.vr) or pd.isna(r.avgvalcr): return None
    trend=100 if r.Close>r.sma20>r.sma50 else 65 if r.Close>r.sma20 else 15
    momentum=float(np.clip((r.r5*100+2)/8*100,0,100))
    volume=float(np.clip((r.vr-.7)/3.3*100,0,100))
    breakout=100 if r.Close>r.prior_high20 else 35 if r.Close>r.prior_high20*.97 else 0
    rs=float(np.clip(50+(r.r20-mkt20)*500,0,100))
    score=int(round(.25*trend+.20*momentum+.20*volume+.20*breakout+.15*rs))
    if risk=="Conservative" and r.avgvalcr<10: score=max(0,score-20)
    if risk=="Aggressive" and r.vr>2: score=min(100,score+5)
    return {"Ticker":t,"Close ₹":round(r.Close,2),"1D %":round(r.r1*100,2),"5D %":round(r.r5*100,2),"20D %":round(r.r20*100,2),"Vol ×":round(r.vr,2),"Avg value ₹Cr":round(r.avgvalcr,1),"ATR %":round(r.atr_pct,2),"Score":score,"Setup":"20D breakout" if r.Close>r.prior_high20 else "Near breakout" if r.Close>r.prior_high20*.97 else "Trend"}
rows=[make_row(t,x) for t,x in ind.items()]
scan=pd.DataFrame([x for x in rows if x and x["Close ₹"]>=min_price and x["Avg value ₹Cr"]>=min_value])
if not scan.empty: scan=scan.sort_values(["Score","Avg value ₹Cr"],ascending=False)
tabs=st.tabs(["🚦 Watchlist","🧪 Backtest","📰 Catalysts","📐 Position size","ℹ️ How it works"])
with tabs[0]:
    st.subheader("Potential movers")
    st.caption("Ranked watchlist based on trend, 5-day momentum, unusual volume, breakout proximity and relative strength. Score is not a probability.")
    if not scan.empty:
        top=scan.head(max_stocks)
        m1,m2,m3=st.columns(3)
        m1.metric("Stocks scanned",len(scan))
        m2.metric("Strong setups",int((scan.Score>=75).sum()))
        m3.metric("Highest score",int(scan.Score.max()))
        st.dataframe(top,use_container_width=True,hide_index=True)
        st.download_button("⬇️ Download watchlist CSV",top.to_csv(index=False).encode(),"profitpilot_v2_watchlist.csv","text/csv")
        st.markdown("**How to read the score**")
        st.write("75–100: stronger technical setup · 55–74: watch · below 55: weaker setup. Check liquidity, news, opening gap and risk before considering any trade.")
    else: st.warning("No stocks meet the current price and liquidity filters.")
    st.button("↻ Refresh market data",on_click=st.cache_data.clear)
with tabs[1]:
    st.subheader("Historical signal test")
    c1,c2,c3=st.columns(3)
    with c1: hold=st.select_slider("Holding days",[1,2,3,5,7,10],value=5)
    with c2: tgt=st.select_slider("Profit target",[2,3,5,8,10],value=3,format_func=lambda v:f"+{v}%")
    with c3: sl=st.select_slider("Stop loss",[2,3,4,5,7],value=4,format_func=lambda v:f"-{v}%")
    trades=[]
    for t,x in ind.items():
        sig=(x.Close>x.sma20)&(x.sma20>x.sma50)&(x.vr>=1.5)&(x.Close>x.prior_high20)
        if risk=="Conservative": sig &= x.avgvalcr>=10
        for i in np.flatnonzero(sig.fillna(False).values):
            entry_i=i+1
            if entry_i>=len(x) or i+hold>=len(x): continue
            entry=float(x.Open.iloc[entry_i])
            if not np.isfinite(entry) or entry<=0: continue
            exitp=float(x.Close.iloc[i+hold]); reason="Time"
            for j in range(entry_i,min(i+hold+1,len(x))):
                # Daily candles cannot reveal whether target or stop came first; conservative stop-first assumption.
                if x.Low.iloc[j]<=entry*(1-sl/100): exitp=entry*(1-sl/100);reason="Stop";break
                if x.High.iloc[j]>=entry*(1+tgt/100): exitp=entry*(1+tgt/100);reason="Target";break
            gross=(exitp/entry-1)*100
            net=gross-.25
            trades.append({"Ticker":t,"Signal date":x.index[i].strftime("%Y-%m-%d"),"Entry date":x.index[entry_i].strftime("%Y-%m-%d"),"Net %":net,"Exit":reason})
    bt=pd.DataFrame(trades)
    if not bt.empty:
        win=(bt["Net %"]>0).mean()*100
        hit=(bt["Net %"]>=tgt).mean()*100
        a,b,c,d=st.columns(4)
        a.metric("Signals",len(bt));b.metric("Win rate",f"{win:.1f}%");c.metric(f"Hit +{tgt}%",f"{hit:.1f}%");d.metric("Avg net/trade",f"{bt['Net %'].mean():+.2f}%")
        st.caption("Simplified historical simulation: signal at close, entry next session open, daily-bar stop-first rule, flat 0.25% estimated friction. This is not a verified live strategy. Current universe creates survivorship bias; no point-in-time constituents or intraday bars.")
        st.dataframe(bt.sort_values("Signal date",ascending=False).head(150),use_container_width=True,hide_index=True)
        st.download_button("⬇️ Download backtest CSV",bt.to_csv(index=False).encode(),"profitpilot_v2_backtest.csv","text/csv")
        if len(bt)<100: st.warning("Small sample. Treat the win rate as preliminary; use a much larger point-in-time sample and out-of-sample testing.")
    else: st.info("No qualifying historical signals for these settings.")
with tabs[2]:
    st.subheader("Latest company news")
    selected=st.selectbox("Company",list(hist.keys()))
    term=selected.replace(".NS","").replace(".BO","")
    if st.button("Search Google News RSS"):
        try:
            u="https://news.google.com/rss/search?q="+requests.utils.quote(term+" India stock")
            root=ET.fromstring(requests.get(u,timeout=15,headers={"User-Agent":"Mozilla/5.0"}).content)
            items=root.findall(".//item")[:10]
            for it in items:
                st.markdown(f"- [{it.findtext('title','News')}]({it.findtext('link','#')})")
                st.caption(it.findtext("pubDate",""))
            if not items: st.info("No RSS headlines returned.")
        except Exception as e: st.error(f"News feed unavailable: {e}")
    st.caption("News search is a discovery aid. Confirm material announcements on the official NSE/BSE filings pages.")
with tabs[3]:
    st.subheader("Illustrative position sizing")
    st.caption("Sizing uses a fixed risk budget and a sample 4% stop distance. Adjust the stop based on actual volatility and setup.")
    if capital>0 and not scan.empty:
        riskpct={"Conservative":.01,"Balanced":.015,"Aggressive":.02}[risk]
        q=scan.head(max_stocks).copy()
        q["Risk budget ₹"]=capital*riskpct
        q["Stop ₹"]=q["Close ₹"]*.96
        q["Shares"]=np.floor(q["Risk budget ₹"]/(q["Close ₹"]*.04)).astype(int)
        q["Position ₹"]=q["Shares"]*q["Close ₹"]
        st.metric("Per-position risk budget",f"₹{capital*riskpct:,.0f}")
        st.dataframe(q[["Ticker","Score","Close ₹","Stop ₹","Shares","Position ₹"]],use_container_width=True,hide_index=True)
    else: st.info("Set capital in Scanner settings to calculate illustrative quantities.")
with tabs[4]:
    st.subheader("Methodology and limitations")
    st.markdown("""
**Score components:** trend alignment 25%, 5-day momentum 20%, relative volume 20%, 20-day breakout 20%, relative strength vs this universe 15%.

**Intraday vs swing:** This V2 creates a pre-market *watchlist* from the latest daily bars. It does not calculate an intraday auction signal. A genuine pre-open model needs licensed indicative equilibrium price, order imbalance, pre-open volume and timestamped news.

**Historical success:** The displayed win rate is specific to the selected universe, signal definition, holding period, target and stop. It is not a probability that tomorrow's signal will work. Survivorship bias and daily-bar assumptions remain.

**Free sources:** Yahoo Finance via yfinance for historical daily OHLCV; Google News RSS for headline discovery. Availability, freshness and completeness are not guaranteed.

**Alerts:** This app must be open/refreshed to scan. For 8:45 AM IST Telegram/email alerts, it must be deployed on an always-on host with a scheduled job and credentials. No trades are ever placed.
""")
st.caption("ProfitPilot V2 · Research only · Verify live prices and exchange announcements · No guarantee of returns.")