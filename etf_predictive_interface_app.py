"""
ETF Predictive Investment Interface
Run with: streamlit run etf_predictive_interface_app.py

Install first:
pip install streamlit yfinance pandas numpy plotly scikit-learn

This dashboard is for educational decision-support only, not financial advice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


st.set_page_config(
    page_title="ETF Predictive Investment Interface",
    page_icon="📈",
    layout="wide",
)

DEFAULT_ETFS = ["VGT", "VOO", "VFV.TO", "QQQ", "SPY", "SCHD", "VTI", "XUU.TO", "XEQT.TO"]
HORIZONS = {
    "1 week": 5,
    "1 month": 21,
    "1 year": 252,
    "5 years": 252 * 5,
    "10 years / decade": 252 * 10,
}


@dataclass
class SignalResult:
    signal: str
    score: int
    reason: str


@st.cache_data(ttl=60 * 30, show_spinner=False)
def load_prices(tickers: Tuple[str, ...], period: str = "5y") -> pd.DataFrame:
    raw = yf.download(list(tickers), period=period, auto_adjust=True, progress=False, group_by="ticker")
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
            continue
    return pd.concat(frames, axis=1).dropna(how="all") if frames else pd.DataFrame()


def moving_average_signal(series: pd.Series) -> SignalResult:
    s = series.dropna()
    if len(s) < 220:
        return SignalResult("HOLD", 0, "Not enough historical data for a strong technical signal.")
    last = float(s.iloc[-1])
    ma50 = float(s.rolling(50).mean().iloc[-1])
    ma200 = float(s.rolling(200).mean().iloc[-1])
    recent_return = float(s.pct_change(21).iloc[-1])
    vol = float(s.pct_change().rolling(63).std().iloc[-1] * math.sqrt(252))

    score = 0
    reasons = []
    if last > ma50:
        score += 1; reasons.append("price is above 50-day trend")
    else:
        score -= 1; reasons.append("price is below 50-day trend")
    if ma50 > ma200:
        score += 1; reasons.append("50-day trend is above 200-day trend")
    else:
        score -= 1; reasons.append("50-day trend is below 200-day trend")
    if recent_return > 0.03:
        score += 1; reasons.append("recent 1-month momentum is strong")
    elif recent_return < -0.03:
        score -= 1; reasons.append("recent 1-month momentum is weak")
    if vol > 0.35:
        score -= 1; reasons.append("volatility is elevated")

    if score >= 2:
        signal = "BUY / ACCUMULATE"
    elif score <= -2:
        signal = "REDUCE / WAIT"
    else:
        signal = "HOLD"
    return SignalResult(signal, score, "; ".join(reasons).capitalize() + ".")


def monte_carlo_forecast(series: pd.Series, days: int, simulations: int = 1000) -> Dict[str, float]:
    s = series.dropna()
    returns = s.pct_change().dropna()
    if len(returns) < 60:
        return {"p10": np.nan, "p50": np.nan, "p90": np.nan, "mean_return": np.nan, "loss_prob": np.nan}
    mu = returns.mean()
    sigma = returns.std()
    last = float(s.iloc[-1])
    shocks = np.random.default_rng(42).normal(mu, sigma, size=(days, simulations))
    paths = last * np.cumprod(1 + shocks, axis=0)
    final = paths[-1]
    return {
        "p10": float(np.percentile(final, 10)),
        "p50": float(np.percentile(final, 50)),
        "p90": float(np.percentile(final, 90)),
        "mean_return": float((np.mean(final) / last) - 1),
        "loss_prob": float(np.mean(final < last)),
    }


def portfolio_value(holdings: pd.DataFrame, prices: pd.DataFrame) -> Tuple[pd.DataFrame, float, float]:
    if holdings.empty or prices.empty:
        return holdings, 0.0, 0.0
    latest = prices.ffill().iloc[-1].to_dict()
    h = holdings.copy()
    h["current_price"] = h["ticker"].map(latest)
    h["market_value"] = h["shares"] * h["current_price"]
    h["cost_basis"] = h["shares"] * h["avg_cost"]
    h["gain_loss"] = h["market_value"] - h["cost_basis"]
    h["gain_loss_%"] = np.where(h["cost_basis"] > 0, h["gain_loss"] / h["cost_basis"], np.nan)
    return h, float(h["market_value"].sum()), float(h["gain_loss"].sum())


def parse_holdings(text: str) -> pd.DataFrame:
    rows = []
    for line in text.splitlines():
        if not line.strip() or line.lower().startswith("ticker"):
            continue
        parts = [p.strip() for p in line.replace(";", ",").split(",")]
        if len(parts) >= 3:
            try:
                rows.append({"ticker": parts[0].upper(), "shares": float(parts[1]), "avg_cost": float(parts[2])})
            except ValueError:
                pass
    return pd.DataFrame(rows)


st.title("📈 ETF Predictive Investment Interface")
st.caption("Live ETF dashboard with portfolio tracking, gain/loss analysis, trend signals, and scenario forecasting. Educational only — not financial advice.")

with st.sidebar:
    st.header("Tracked ETFs")
    tickers_text = st.text_area("ETF tickers", ", ".join(DEFAULT_ETFS), height=95)
    tickers = tuple(sorted({t.strip().upper() for t in tickers_text.replace("\n", ",").split(",") if t.strip()}))
    period = st.selectbox("Historical data window", ["1y", "2y", "5y", "10y", "max"], index=2)
    st.divider()
    st.header("Your holdings")
    holdings_text = st.text_area(
        "Enter: ticker, shares, average cost",
        "VGT, 2, 520\nVOO, 3, 480\nVFV.TO, 10, 135",
        height=130,
    )
    st.caption("Example: VOO, 3, 480")

prices = load_prices(tickers, period)
if prices.empty:
    st.error("No price data loaded. Check ticker symbols and internet access.")
    st.stop()

holdings = parse_holdings(holdings_text)
portfolio, total_value, total_gain = portfolio_value(holdings, prices)

m1, m2, m3, m4 = st.columns(4)
latest_prices = prices.ffill().iloc[-1]
m1.metric("Tracked ETFs", len(prices.columns))
m2.metric("Portfolio value", f"${total_value:,.2f}")
m3.metric("Unrealized gain/loss", f"${total_gain:,.2f}")
m4.metric("Last update", str(date.today()))

st.subheader("Live ETF price chart")
selected = st.multiselect("Choose ETFs to chart", list(prices.columns), default=list(prices.columns[: min(4, len(prices.columns))]))
if selected:
    fig = go.Figure()
    normalized = st.toggle("Normalize to compare performance", value=True)
    for t in selected:
        s = prices[t].dropna()
        y = s / s.iloc[0] * 100 if normalized and len(s) else s
        fig.add_trace(go.Scatter(x=s.index, y=y, mode="lines", name=t))
    fig.update_layout(height=450, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="Indexed price" if normalized else "Price")
    st.plotly_chart(fig, use_container_width=True)

left, right = st.columns([1.2, 1])
with left:
    st.subheader("Your portfolio")
    if portfolio.empty:
        st.info("Enter holdings in the sidebar to see portfolio gain/loss and forecasts.")
    else:
        st.dataframe(
            portfolio.style.format({
                "shares": "{:,.4f}", "avg_cost": "${:,.2f}", "current_price": "${:,.2f}",
                "market_value": "${:,.2f}", "cost_basis": "${:,.2f}",
                "gain_loss": "${:,.2f}", "gain_loss_%": "{:.2%}",
            }),
            use_container_width=True,
        )

with right:
    st.subheader("Signal engine")
    signal_rows = []
    for t in prices.columns:
        sig = moving_average_signal(prices[t])
        signal_rows.append({"ETF": t, "Signal": sig.signal, "Score": sig.score, "Reason": sig.reason})
    st.dataframe(pd.DataFrame(signal_rows), use_container_width=True, hide_index=True)

st.subheader("Forecasts")
forecast_ticker = st.selectbox("ETF to forecast", list(prices.columns))
s = prices[forecast_ticker].dropna()
current_price = float(s.iloc[-1])
forecast_rows = []
for label, days in HORIZONS.items():
    fc = monte_carlo_forecast(s, days)
    forecast_rows.append({
        "Horizon": label,
        "Current": current_price,
        "Bear case p10": fc["p10"],
        "Median p50": fc["p50"],
        "Bull case p90": fc["p90"],
        "Expected return": fc["mean_return"],
        "Probability of loss": fc["loss_prob"],
    })
forecast_df = pd.DataFrame(forecast_rows)
st.dataframe(
    forecast_df.style.format({
        "Current": "${:,.2f}", "Bear case p10": "${:,.2f}", "Median p50": "${:,.2f}",
        "Bull case p90": "${:,.2f}", "Expected return": "{:.2%}", "Probability of loss": "{:.2%}",
    }),
    use_container_width=True,
    hide_index=True,
)

fig2 = go.Figure()
fig2.add_trace(go.Bar(x=forecast_df["Horizon"], y=forecast_df["Expected return"], name="Expected return"))
fig2.update_layout(height=330, margin=dict(l=10, r=10, t=25, b=10), yaxis_tickformat=".0%")
st.plotly_chart(fig2, use_container_width=True)

st.subheader("Portfolio long-term scenario")
if portfolio.empty:
    st.info("Enter holdings to calculate portfolio-level scenarios.")
else:
    rows = []
    for _, row in portfolio.dropna(subset=["current_price"]).iterrows():
        ticker = row["ticker"]
        if ticker in prices:
            for label, days in HORIZONS.items():
                fc = monte_carlo_forecast(prices[ticker], days)
                rows.append({
                    "ticker": ticker,
                    "horizon": label,
                    "median_value": row["shares"] * fc["p50"],
                    "bear_value": row["shares"] * fc["p10"],
                    "bull_value": row["shares"] * fc["p90"],
                })
    port_fc = pd.DataFrame(rows)
    if not port_fc.empty:
        summary = port_fc.groupby("horizon", sort=False)[["bear_value", "median_value", "bull_value"]].sum().reset_index()
        st.dataframe(summary.style.format({"bear_value": "${:,.2f}", "median_value": "${:,.2f}", "bull_value": "${:,.2f}"}), use_container_width=True, hide_index=True)

st.warning("Important: Predictions are probabilistic estimates from historical ETF returns. They can be wrong, especially during recessions, rate shocks, crashes, or major policy changes. Do not treat the signal as guaranteed financial advice.")
