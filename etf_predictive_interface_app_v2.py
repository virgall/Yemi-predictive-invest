"""
ETF Predictive Investment Interface v2
Run with: streamlit run etf_predictive_interface_app_v2.py

requirements.txt:
streamlit
yfinance
pandas
numpy
plotly

This app is for educational decision-support only, not financial advice.
"""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="ETF Predictive Investment Interface", page_icon="📈", layout="wide")

DEFAULT_ETFS = ["VGT", "VOO", "VFV.TO", "VTI", "QQQ", "SPY", "SCHD", "XEQT.TO", "XUU.TO", "XIC.TO", "TEC.TO"]
HORIZON_OPTIONS = {
    "1 month": 1/12,
    "3 months": 0.25,
    "6 months": 0.5,
    "1 year": 1,
    "3 years": 3,
    "5 years": 5,
    "10 years / decade": 10,
    "20 years": 20,
}
TRADING_DAYS = 252


def clean_ticker(t: str) -> str:
    return str(t).strip().upper()


@st.cache_data(ttl=60 * 30, show_spinner=False)
def load_prices(tickers: Tuple[str, ...], period: str = "10y") -> pd.DataFrame:
    tickers = tuple(clean_ticker(t) for t in tickers if clean_ticker(t))
    if not tickers:
        return pd.DataFrame()
    raw = yf.download(list(tickers), period=period, auto_adjust=True, progress=False, group_by="ticker", threads=True)
    if raw.empty:
        return pd.DataFrame()
    frames = []
    for t in tickers:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                s = raw[t]["Close"].rename(t)
            else:
                s = raw["Close"].rename(t)
            frames.append(s)
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1).dropna(how="all").ffill()


def nearest_price(prices: pd.DataFrame, ticker: str, invest_date) -> float:
    ticker = clean_ticker(ticker)
    if ticker not in prices or prices[ticker].dropna().empty:
        return np.nan
    d = pd.to_datetime(invest_date)
    s = prices[ticker].dropna()
    valid = s.loc[s.index <= d]
    if valid.empty:
        valid = s
    return float(valid.iloc[-1])


def transaction_summary(transactions: pd.DataFrame, prices: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tx = transactions.copy()
    tx["Ticker"] = tx["Ticker"].map(clean_ticker)
    tx["Investment Date"] = pd.to_datetime(tx["Investment Date"]).dt.date
    tx["Amount Invested"] = pd.to_numeric(tx["Amount Invested"], errors="coerce").fillna(0.0)
    tx["Share Price Used"] = pd.to_numeric(tx["Share Price Used"], errors="coerce")

    estimated_prices = []
    for _, r in tx.iterrows():
        entered = r["Share Price Used"]
        if pd.notna(entered) and entered > 0:
            estimated_prices.append(float(entered))
        else:
            estimated_prices.append(nearest_price(prices, r["Ticker"], r["Investment Date"]))
    tx["Price Used"] = estimated_prices
    tx["Estimated Shares Bought"] = np.where(tx["Price Used"] > 0, tx["Amount Invested"] / tx["Price Used"], 0.0)

    latest = prices.ffill().iloc[-1].to_dict() if not prices.empty else {}
    tx["Current Price"] = tx["Ticker"].map(latest)
    tx["Current Value"] = tx["Estimated Shares Bought"] * tx["Current Price"]
    tx["Gain/Loss"] = tx["Current Value"] - tx["Amount Invested"]
    tx["Gain/Loss %"] = np.where(tx["Amount Invested"] > 0, tx["Gain/Loss"] / tx["Amount Invested"], np.nan)

    holdings = tx.groupby("Ticker", as_index=False).agg(
        Shares=("Estimated Shares Bought", "sum"),
        Total_Invested=("Amount Invested", "sum"),
        Current_Value=("Current Value", "sum"),
        Gain_Loss=("Gain/Loss", "sum"),
    )
    holdings["Current Price"] = holdings["Ticker"].map(latest)
    holdings["Avg Cost"] = np.where(holdings["Shares"] > 0, holdings["Total_Invested"] / holdings["Shares"], np.nan)
    holdings["Gain/Loss %"] = np.where(holdings["Total_Invested"] > 0, holdings["Gain_Loss"] / holdings["Total_Invested"], np.nan)
    return tx, holdings


def annual_return_and_vol(series: pd.Series) -> Tuple[float, float]:
    r = series.dropna().pct_change().dropna()
    if len(r) < 60:
        return 0.07, 0.18
    # Use a blended estimate so one strange recent period does not dominate long-term forecasts.
    hist_annual = float((1 + r.mean()) ** TRADING_DAYS - 1)
    hist_vol = float(r.std() * math.sqrt(TRADING_DAYS))
    conservative_default = 0.07
    blended_return = 0.65 * hist_annual + 0.35 * conservative_default
    blended_return = float(np.clip(blended_return, -0.15, 0.18))
    hist_vol = float(np.clip(hist_vol, 0.06, 0.45))
    return blended_return, hist_vol


def technical_signal(series: pd.Series) -> Tuple[str, int, str]:
    s = series.dropna()
    if len(s) < 220:
        return "WATCH", 0, "Not enough long history for a strong trend signal."
    last = float(s.iloc[-1])
    ma50 = float(s.rolling(50).mean().iloc[-1])
    ma200 = float(s.rolling(200).mean().iloc[-1])
    mom_1m = float(s.pct_change(21).iloc[-1])
    score = 0
    reasons = []
    if last > ma50:
        score += 1; reasons.append("above 50-day trend")
    else:
        score -= 1; reasons.append("below 50-day trend")
    if ma50 > ma200:
        score += 1; reasons.append("50-day trend above 200-day trend")
    else:
        score -= 1; reasons.append("50-day trend below 200-day trend")
    if mom_1m > 0.03:
        score += 1; reasons.append("positive recent momentum")
    elif mom_1m < -0.03:
        score -= 1; reasons.append("weak recent momentum")
    if score >= 2:
        label = "ACCUMULATE"
    elif score <= -2:
        label = "WAIT / REDUCE"
    else:
        label = "HOLD / WATCH"
    return label, score, "; ".join(reasons).capitalize() + "."


def future_contribution_dates(start: date, years: float, frequency: str) -> List[pd.Timestamp]:
    if frequency == "None" or years <= 0:
        return []
    end = pd.Timestamp(start) + pd.DateOffset(days=int(years * 365.25))
    freq_map = {"Weekly": "W", "Bi-weekly": "2W", "Monthly": "MS", "Quarterly": "QS", "Yearly": "YS"}
    rng = pd.date_range(pd.Timestamp(start), end, freq=freq_map.get(frequency, "MS"))
    return list(rng)


def forecast_portfolio(
    holdings: pd.DataFrame,
    prices: pd.DataFrame,
    allocations: Dict[str, float],
    contribution_amount: float,
    contribution_frequency: str,
    years: float,
    simulations: int = 600,
) -> Dict[str, float]:
    rng = np.random.default_rng(77)
    tickers = list(holdings["Ticker"]) if not holdings.empty else list(allocations.keys())
    tickers = [t for t in tickers if t in prices.columns]
    if not tickers:
        return {"invested_future": 0, "p10": np.nan, "p50": np.nan, "p90": np.nan, "loss_probability": np.nan}

    current_values = {r["Ticker"]: float(r["Current_Value"]) for _, r in holdings.iterrows()} if not holdings.empty else {t: 0.0 for t in tickers}
    mu_vol = {t: annual_return_and_vol(prices[t]) for t in tickers}
    steps = max(1, int(years * 12))
    dt = 1 / 12
    contrib_dates = future_contribution_dates(date.today(), years, contribution_frequency)
    monthly_contrib_count = np.zeros(steps)
    if contrib_dates:
        for d in contrib_dates:
            m = min(steps - 1, max(0, int(((d.date() - date.today()).days / 365.25) * 12)))
            monthly_contrib_count[m] += 1

    totals = []
    for _ in range(simulations):
        values = current_values.copy()
        for m in range(steps):
            if monthly_contrib_count[m] > 0 and contribution_amount > 0:
                for t in tickers:
                    values[t] = values.get(t, 0.0) + contribution_amount * monthly_contrib_count[m] * allocations.get(t, 0.0)
            for t in tickers:
                mu, vol = mu_vol[t]
                shock = rng.normal((mu - 0.5 * vol * vol) * dt, vol * math.sqrt(dt))
                values[t] = max(0.0, values.get(t, 0.0) * math.exp(shock))
        totals.append(sum(values.values()))
    totals = np.array(totals)
    current_total = float(sum(current_values.values()))
    future_invested = float(len(contrib_dates) * contribution_amount)
    total_principal = float(holdings["Total_Invested"].sum() if not holdings.empty else 0) + future_invested
    return {
        "current_total": current_total,
        "invested_future": future_invested,
        "total_principal": total_principal,
        "p10": float(np.percentile(totals, 10)),
        "p50": float(np.percentile(totals, 50)),
        "p90": float(np.percentile(totals, 90)),
        "expected_gain_p50": float(np.percentile(totals, 50) - total_principal),
        "loss_probability": float(np.mean(totals < total_principal)),
    }


st.title("📈 ETF Predictive Investment Interface")
st.caption("User-driven ETF tracker: investment dates, amounts invested, planned contributions, live pricing, forecast scenarios, and buy/hold/watch signals. Educational only — not financial advice.")

with st.sidebar:
    st.header("1) Choose ETFs")
    selected_defaults = ["VGT", "VOO", "VFV.TO"]
    chosen = st.multiselect("ETFs invested in or watching", DEFAULT_ETFS, default=selected_defaults)
    custom = st.text_input("Add custom ETF tickers", placeholder="Example: ZSP.TO, XSP.TO, IVV")
    custom_list = [clean_ticker(x) for x in custom.replace(";", ",").split(",") if clean_ticker(x)]
    tickers = tuple(sorted(set([clean_ticker(x) for x in chosen] + custom_list)))
    period = st.selectbox("Historical data window", ["1y", "2y", "5y", "10y", "max"], index=3)
    st.divider()
    st.header("2) Future investing plan")
    contribution_amount = st.number_input("Amount to keep investing each time", min_value=0.0, value=500.0, step=50.0)
    contribution_frequency = st.selectbox("Contribution frequency", ["None", "Weekly", "Bi-weekly", "Monthly", "Quarterly", "Yearly"], index=3)
    horizon_label = st.selectbox("Forecast from today", list(HORIZON_OPTIONS.keys()), index=5)
    simulations = st.slider("Forecast simulation depth", 200, 2000, 600, step=100)

prices = load_prices(tickers, period)
if not tickers:
    st.info("Choose at least one ETF to begin.")
    st.stop()
if prices.empty:
    st.error("No live market data loaded. Check ticker symbols. For Canadian ETFs, use Yahoo format such as VFV.TO or XEQT.TO.")
    st.stop()

st.subheader("3) Enter investment history")
st.write("Add each investment you made. If you do not know the share price, leave it blank and the app estimates it from historical ETF prices around that date.")

default_tx = pd.DataFrame({
    "Ticker": [tickers[0] if len(tickers) > 0 else "VOO", tickers[1] if len(tickers) > 1 else "VGT"],
    "Investment Date": [date(2025, 1, 15), date(2025, 8, 1)],
    "Amount Invested": [1000.0, 1500.0],
    "Share Price Used": [np.nan, np.nan],
})

transactions = st.data_editor(
    default_tx,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "Ticker": st.column_config.SelectboxColumn("Ticker", options=list(tickers), required=True),
        "Investment Date": st.column_config.DateColumn("Investment Date", required=True),
        "Amount Invested": st.column_config.NumberColumn("Amount Invested", min_value=0.0, step=50.0, format="$%.2f"),
        "Share Price Used": st.column_config.NumberColumn("Share Price Used", min_value=0.0, step=1.0, format="$%.2f", help="Optional. Leave blank to estimate using historical prices."),
    },
    hide_index=True,
)

clean_tx = transactions.dropna(subset=["Ticker", "Investment Date"]).copy()
clean_tx = clean_tx[clean_tx["Amount Invested"].fillna(0) > 0]
calculated_tx, holdings = transaction_summary(clean_tx, prices) if not clean_tx.empty else (pd.DataFrame(), pd.DataFrame())

total_invested = float(holdings["Total_Invested"].sum()) if not holdings.empty else 0.0
total_value = float(holdings["Current_Value"].sum()) if not holdings.empty else 0.0
total_gain = total_value - total_invested

m1, m2, m3, m4 = st.columns(4)
m1.metric("ETFs loaded", len(prices.columns))
m2.metric("Total invested so far", f"${total_invested:,.2f}")
m3.metric("Current portfolio value", f"${total_value:,.2f}")
m4.metric("Current gain/loss", f"${total_gain:,.2f}", f"{(total_gain/total_invested if total_invested else 0):.2%}")

left, right = st.columns([1.2, 1])
with left:
    st.subheader("Calculated holdings")
    if holdings.empty:
        st.info("Enter at least one investment row above.")
    else:
        view_holdings = holdings.rename(columns={"Total_Invested": "Total Invested", "Current_Value": "Current Value", "Gain_Loss": "Gain/Loss"})
        st.dataframe(view_holdings.style.format({
            "Shares": "{:,.4f}", "Total Invested": "${:,.2f}", "Avg Cost": "${:,.2f}",
            "Current Price": "${:,.2f}", "Current Value": "${:,.2f}", "Gain/Loss": "${:,.2f}", "Gain/Loss %": "{:.2%}",
        }), use_container_width=True, hide_index=True)
with right:
    st.subheader("ETF signal engine")
    rows = []
    for t in prices.columns:
        label, score, reason = technical_signal(prices[t])
        ann, vol = annual_return_and_vol(prices[t])
        rows.append({"ETF": t, "Signal": label, "Score": score, "Blended Annual Return Estimate": ann, "Volatility": vol, "Reason": reason})
    st.dataframe(pd.DataFrame(rows).style.format({"Blended Annual Return Estimate": "{:.2%}", "Volatility": "{:.2%}"}), use_container_width=True, hide_index=True)

st.subheader("Price performance chart")
chart_tickers = st.multiselect("Chart ETFs", list(prices.columns), default=list(prices.columns[:min(4, len(prices.columns))]))
normalize = st.toggle("Normalize chart to compare performance", value=True)
if chart_tickers:
    fig = go.Figure()
    for t in chart_tickers:
        s = prices[t].dropna()
        y = s / s.iloc[0] * 100 if normalize and len(s) else s
        fig.add_trace(go.Scatter(x=s.index, y=y, mode="lines", name=t))
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=25, b=10), yaxis_title="Indexed to 100" if normalize else "Price")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Future contribution allocation")
st.write("Choose how future investments should be split. Set 100% total across the ETFs you want to keep buying.")
alloc_cols = st.columns(min(4, max(1, len(tickers))))
raw_alloc = {}
for i, t in enumerate(tickers):
    with alloc_cols[i % len(alloc_cols)]:
        default_pct = round(100 / len(tickers), 2) if tickers else 0
        raw_alloc[t] = st.number_input(f"{t} allocation %", min_value=0.0, max_value=100.0, value=float(default_pct), step=5.0, key=f"alloc_{t}")
alloc_sum = sum(raw_alloc.values())
allocations = {t: (v / alloc_sum if alloc_sum > 0 else 0.0) for t, v in raw_alloc.items()}
if abs(alloc_sum - 100) > 0.01:
    st.warning(f"Allocation total is {alloc_sum:.1f}%. The app will normalize it to 100% for the forecast.")

years = HORIZON_OPTIONS[horizon_label]
forecast = forecast_portfolio(holdings, prices, allocations, contribution_amount, contribution_frequency, years, simulations)

st.subheader(f"Forecast from today: {horizon_label}")
f1, f2, f3, f4 = st.columns(4)
f1.metric("Future contributions", f"${forecast['invested_future']:,.2f}")
f2.metric("Total principal by horizon", f"${forecast['total_principal']:,.2f}")
f3.metric("Median projected value", f"${forecast['p50']:,.2f}")
f4.metric("Median projected gain/loss", f"${forecast['expected_gain_p50']:,.2f}")

forecast_table = pd.DataFrame([
    {"Scenario": "Bear case / 10th percentile", "Projected Value": forecast["p10"], "Gain/Loss vs Principal": forecast["p10"] - forecast["total_principal"]},
    {"Scenario": "Median case / 50th percentile", "Projected Value": forecast["p50"], "Gain/Loss vs Principal": forecast["p50"] - forecast["total_principal"]},
    {"Scenario": "Bull case / 90th percentile", "Projected Value": forecast["p90"], "Gain/Loss vs Principal": forecast["p90"] - forecast["total_principal"]},
])
st.dataframe(forecast_table.style.format({"Projected Value": "${:,.2f}", "Gain/Loss vs Principal": "${:,.2f}"}), use_container_width=True, hide_index=True)
st.caption(f"Estimated probability of ending below total principal: {forecast['loss_probability']:.2%}")

fig2 = go.Figure()
fig2.add_trace(go.Bar(x=forecast_table["Scenario"], y=forecast_table["Projected Value"], name="Projected value"))
fig2.add_hline(y=forecast["total_principal"], line_dash="dash", annotation_text="Total principal", annotation_position="top left")
fig2.update_layout(height=360, margin=dict(l=10, r=10, t=25, b=10), yaxis_title="Portfolio value")
st.plotly_chart(fig2, use_container_width=True)

st.subheader("Transaction-level calculation")
if calculated_tx.empty:
    st.info("No transactions calculated yet.")
else:
    st.dataframe(calculated_tx.style.format({
        "Amount Invested": "${:,.2f}", "Share Price Used": "${:,.2f}", "Price Used": "${:,.2f}",
        "Estimated Shares Bought": "{:,.4f}", "Current Price": "${:,.2f}", "Current Value": "${:,.2f}",
        "Gain/Loss": "${:,.2f}", "Gain/Loss %": "{:.2%}",
    }), use_container_width=True, hide_index=True)

st.warning("Important: This app uses historical ETF price behaviour and simulation. It does not know the future. Treat signals as decision-support, not guaranteed financial advice or a command to buy/sell.")
