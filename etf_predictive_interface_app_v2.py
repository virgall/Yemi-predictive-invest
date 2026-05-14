import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import date, timedelta

st.set_page_config(page_title="Yemi Predictive Investment Interface", layout="wide")

st.title("📈 Predictive Investment Interface")
st.caption("Enter any stock/ETF ticker, add your investment history, and forecast possible future portfolio value.")

st.warning(
    "This app is for educational forecasting only. It is not financial advice. "
    "Forecasts are estimates based on historical market data and assumptions."
)

# -----------------------------
# Helper functions
# -----------------------------
@st.cache_data(ttl=3600)
def get_price_history(ticker: str, start: date, end: date) -> pd.DataFrame:
    data = yf.download(ticker, start=start, end=end + timedelta(days=1), progress=False, auto_adjust=True)
    if data.empty:
        return pd.DataFrame()
    data = data.reset_index()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [c[0] if c[0] else c[1] for c in data.columns]
    return data

@st.cache_data(ttl=900)
def get_current_price(ticker: str):
    try:
        info = yf.Ticker(ticker).fast_info
        price = info.get("last_price") or info.get("regular_market_price")
        return float(price) if price else None
    except Exception:
        return None

@st.cache_data(ttl=3600)
def get_name(ticker: str):
    try:
        info = yf.Ticker(ticker).get_info()
        return info.get("longName") or info.get("shortName") or ticker
    except Exception:
        return ticker

def closest_price_on_or_after(hist: pd.DataFrame, invest_date: pd.Timestamp):
    if hist.empty:
        return np.nan
    df = hist.copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    invest_date = pd.to_datetime(invest_date).tz_localize(None)
    valid = df[df["Date"] >= invest_date]
    if valid.empty:
        valid = df[df["Date"] <= invest_date]
    if valid.empty:
        return np.nan
    return float(valid.iloc[0]["Close"])

def annualized_return(hist: pd.DataFrame):
    if hist.empty or len(hist) < 30:
        return 0.07
    df = hist.dropna(subset=["Close"]).copy()
    if len(df) < 30:
        return 0.07
    first = float(df.iloc[0]["Close"])
    last = float(df.iloc[-1]["Close"])
    days = max((pd.to_datetime(df.iloc[-1]["Date"]) - pd.to_datetime(df.iloc[0]["Date"])).days, 1)
    if first <= 0 or last <= 0:
        return 0.07
    return (last / first) ** (365.25 / days) - 1

def annual_volatility(hist: pd.DataFrame):
    if hist.empty or len(hist) < 30:
        return 0.18
    returns = hist["Close"].pct_change().dropna()
    if returns.empty:
        return 0.18
    return float(returns.std() * np.sqrt(252))

def monthly_rate_from_annual(r):
    return (1 + r) ** (1/12) - 1

def months_between(start_date: date, end_date: date):
    return max((end_date.year - start_date.year) * 12 + end_date.month - start_date.month, 0)

def project_value(current_value, monthly_contribution, annual_return, years):
    months = int(round(years * 12))
    monthly_rate = monthly_rate_from_annual(annual_return)
    value = current_value
    path = []
    for m in range(1, months + 1):
        value = value * (1 + monthly_rate) + monthly_contribution
        path.append(value)
    return value, path

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("1) Choose investments")
st.sidebar.write("Use Yahoo Finance ticker format. Examples: VOO, VGT, VFV.TO, XEQT.TO, AAPL, MSFT, TSLA, NVDA.")

watchlist_text = st.sidebar.text_area(
    "Stocks/ETFs to analyze",
    value="VOO, VGT, VFV.TO",
    help="Enter any stock or ETF tickers separated by commas. Canadian tickers often need .TO, e.g., VFV.TO."
)

tickers = [t.strip().upper() for t in watchlist_text.replace("\n", ",").split(",") if t.strip()]

st.sidebar.header("2) Future investing plan")
future_frequency = st.sidebar.selectbox("Contribution frequency", ["Monthly", "Biweekly", "Weekly", "None"])
future_amount = st.sidebar.number_input("Contribution amount per period", min_value=0.0, value=500.0, step=50.0)
forecast_years = st.sidebar.slider("Forecast horizon", min_value=1, max_value=30, value=10)

if future_frequency == "Monthly":
    monthly_contribution = future_amount
elif future_frequency == "Biweekly":
    monthly_contribution = future_amount * 26 / 12
elif future_frequency == "Weekly":
    monthly_contribution = future_amount * 52 / 12
else:
    monthly_contribution = 0

st.sidebar.header("3) Forecast style")
forecast_style = st.sidebar.selectbox(
    "Return assumption",
    ["Use historical annualized return", "Conservative", "Balanced", "Aggressive", "Custom"]
)
custom_return = st.sidebar.number_input("Custom annual return %", value=7.0, step=0.5) / 100

if forecast_style == "Conservative":
    override_return = 0.04
elif forecast_style == "Balanced":
    override_return = 0.07
elif forecast_style == "Aggressive":
    override_return = 0.10
elif forecast_style == "Custom":
    override_return = custom_return
else:
    override_return = None

# -----------------------------
# Investment history input
# -----------------------------
st.subheader("Investment history")
st.write("Add what you invested and when. The app automatically pulls the historical share price for that date and calculates the estimated shares bought.")

if "transactions" not in st.session_state:
    st.session_state.transactions = pd.DataFrame([
        {"Ticker": tickers[0] if tickers else "VOO", "Investment Date": date.today() - timedelta(days=365), "Amount Invested": 1000.0}
    ])

edited = st.data_editor(
    st.session_state.transactions,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "Ticker": st.column_config.TextColumn("Ticker", help="Any Yahoo Finance ticker, e.g., VOO, VGT, VFV.TO, AAPL, NVDA"),
        "Investment Date": st.column_config.DateColumn("Investment Date"),
        "Amount Invested": st.column_config.NumberColumn("Amount Invested", min_value=0.0, step=50.0, format="$%.2f"),
    },
    hide_index=True,
)
st.session_state.transactions = edited

run = st.button("Run prediction", type="primary")

if run:
    tx = edited.copy()
    tx = tx.dropna(subset=["Ticker", "Investment Date", "Amount Invested"])
    tx["Ticker"] = tx["Ticker"].astype(str).str.strip().str.upper()
    tx = tx[tx["Ticker"] != ""]
    tx = tx[tx["Amount Invested"] > 0]

    if tx.empty:
        st.error("Please add at least one investment row.")
        st.stop()

    all_tickers = sorted(set(tickers + tx["Ticker"].tolist()))
    today = date.today()
    min_date = pd.to_datetime(tx["Investment Date"]).min().date()
    start_date = min(min_date - timedelta(days=7), today - timedelta(days=365 * 10))

    histories = {}
    price_rows = []
    portfolio_rows = []

    for ticker in all_tickers:
        hist = get_price_history(ticker, start_date, today)
        histories[ticker] = hist

    for _, row in tx.iterrows():
        ticker = row["Ticker"]
        invest_date = pd.to_datetime(row["Investment Date"])
        amount = float(row["Amount Invested"])
        hist = histories.get(ticker, pd.DataFrame())
        buy_price = closest_price_on_or_after(hist, invest_date)
        current_price = get_current_price(ticker)
        if current_price is None and not hist.empty:
            current_price = float(hist.dropna(subset=["Close"]).iloc[-1]["Close"])
        shares = amount / buy_price if buy_price and not np.isnan(buy_price) and buy_price > 0 else np.nan
        current_value = shares * current_price if current_price and not np.isnan(shares) else np.nan
        gain_loss = current_value - amount if not np.isnan(current_value) else np.nan
        gain_loss_pct = gain_loss / amount if amount > 0 and not np.isnan(gain_loss) else np.nan
        portfolio_rows.append({
            "Ticker": ticker,
            "Investment Date": invest_date.date(),
            "Amount Invested": amount,
            "Historical Price Used": buy_price,
            "Estimated Shares Bought": shares,
            "Current Price": current_price,
            "Current Value": current_value,
            "Gain/Loss": gain_loss,
            "Gain/Loss %": gain_loss_pct,
        })

    result = pd.DataFrame(portfolio_rows)

    st.subheader("Current portfolio estimate")
    total_invested = result["Amount Invested"].sum()
    total_value = result["Current Value"].sum()
    total_gain = total_value - total_invested
    total_gain_pct = total_gain / total_invested if total_invested else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total invested", f"${total_invested:,.2f}")
    c2.metric("Estimated current value", f"${total_value:,.2f}")
    c3.metric("Gain/Loss", f"${total_gain:,.2f}", f"{total_gain_pct:.2%}")
    c4.metric("Monthly future investing", f"${monthly_contribution:,.2f}")

    st.dataframe(
        result.style.format({
            "Amount Invested": "${:,.2f}",
            "Historical Price Used": "${:,.2f}",
            "Estimated Shares Bought": "{:,.4f}",
            "Current Price": "${:,.2f}",
            "Current Value": "${:,.2f}",
            "Gain/Loss": "${:,.2f}",
            "Gain/Loss %": "{:.2%}",
        }),
        use_container_width=True,
    )

    st.subheader("Forecast")
    grouped = result.groupby("Ticker", as_index=False).agg({"Current Value": "sum", "Amount Invested": "sum"})
    forecast_rows = []
    chart = go.Figure()

    for _, row in grouped.iterrows():
        ticker = row["Ticker"]
        current_value = float(row["Current Value"])
        hist = histories.get(ticker, pd.DataFrame())
        hist_return = annualized_return(hist)
        vol = annual_volatility(hist)
        base_return = override_return if override_return is not None else hist_return
        conservative = max(base_return - vol * 0.35, -0.30)
        optimistic = base_return + vol * 0.35

        weight = current_value / total_value if total_value > 0 else 1 / max(len(grouped), 1)
        ticker_monthly = monthly_contribution * weight

        horizons = {
            "1 month": 1/12,
            "6 months": 0.5,
            "1 year": 1,
            "5 years": 5,
            "10 years": 10,
            f"{forecast_years} years": forecast_years,
        }
        for label, yrs in horizons.items():
            expected, _ = project_value(current_value, ticker_monthly, base_return, yrs)
            low, _ = project_value(current_value, ticker_monthly, conservative, yrs)
            high, _ = project_value(current_value, ticker_monthly, optimistic, yrs)
            forecast_rows.append({
                "Ticker": ticker,
                "Horizon": label,
                "Expected Value": expected,
                "Conservative Case": low,
                "Optimistic Case": high,
                "Assumed Annual Return": base_return,
                "Historical Volatility": vol,
            })

        _, path = project_value(current_value, ticker_monthly, base_return, forecast_years)
        x = pd.date_range(today, periods=len(path), freq="ME")
        chart.add_trace(go.Scatter(x=x, y=path, mode="lines", name=ticker))

    forecast_df = pd.DataFrame(forecast_rows)
    st.dataframe(
        forecast_df.style.format({
            "Expected Value": "${:,.2f}",
            "Conservative Case": "${:,.2f}",
            "Optimistic Case": "${:,.2f}",
            "Assumed Annual Return": "{:.2%}",
            "Historical Volatility": "{:.2%}",
        }),
        use_container_width=True,
    )

    chart.update_layout(
        title=f"Projected value over {forecast_years} years",
        xaxis_title="Date",
        yaxis_title="Projected value",
        height=520,
    )
    st.plotly_chart(chart, use_container_width=True)

    st.subheader("Ticker charts")
    for ticker in all_tickers:
        hist = histories.get(ticker, pd.DataFrame())
        if hist.empty:
            st.error(f"No price data found for {ticker}. Check the ticker symbol. Canadian tickers usually need .TO, e.g., VFV.TO.")
            continue
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist["Date"], y=hist["Close"], mode="lines", name=ticker))
        fig.update_layout(title=f"{ticker} price history", xaxis_title="Date", yaxis_title="Adjusted close price", height=380)
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Enter your tickers and investment history, then click **Run prediction**.")
    st.markdown(
        """
        **Ticker examples**
        - US ETFs: `VOO`, `VGT`, `QQQ`, `SPY`
        - Canadian ETFs: `VFV.TO`, `XEQT.TO`, `XIU.TO`, `VUN.TO`
        - Stocks: `AAPL`, `MSFT`, `NVDA`, `TSLA`, `SHOP.TO`
        """
    )
