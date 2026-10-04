"""
Forex Market Predictor — Educational / Paper-Trading App
---------------------------------------------------------
Single-file Streamlit application.

IMPORTANT:
- This app is for educational analysis and paper trading only.
- It does NOT place real-money trades or connect to a broker for execution.
- ML predictions are probabilistic estimates, not guarantees.

Install:
    pip install streamlit yfinance pandas numpy scikit-learn

Run:
    streamlit run forex_predictor.py

Notes:
- Yahoo Finance symbols are used for historical data. Examples:
  EURUSD=X, GBPUSD=X, USDJPY=X, AUDUSD=X, XAUUSD=X
- For best results, use a sufficiently long historical period.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Forex Market Predictor",
    page_icon="📊",
    layout="wide",
)


# -----------------------------
# Technical indicators
# -----------------------------
def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df, period=14):
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()

    true_range = pd.concat(
        [high_low, high_close, low_close], axis=1
    ).max(axis=1)

    return true_range.rolling(period).mean()


def add_features(df):
    data = df.copy()

    data["SMA20"] = data["Close"].rolling(20).mean()
    data["SMA50"] = data["Close"].rolling(50).mean()
    data["EMA12"] = data["Close"].ewm(span=12, adjust=False).mean()
    data["EMA26"] = data["Close"].ewm(span=26, adjust=False).mean()

    data["MACD"] = data["EMA12"] - data["EMA26"]
    data["MACD_SIGNAL"] = data["MACD"].ewm(span=9, adjust=False).mean()
    data["MACD_HIST"] = data["MACD"] - data["MACD_SIGNAL"]

    data["RSI"] = rsi(data["Close"], 14)
    data["ATR"] = atr(data, 14)

    data["RETURN_1"] = data["Close"].pct_change(1)
    data["RETURN_5"] = data["Close"].pct_change(5)
    data["VOLATILITY_10"] = data["RETURN_1"].rolling(10).std()

    data["SMA_DISTANCE"] = (
        data["Close"] / data["SMA20"] - 1
    )

    data["EMA_DISTANCE"] = (
        data["Close"] / data["EMA26"] - 1
    )

    data["VOLUME_CHANGE"] = data["Volume"].pct_change()

    # Target: next candle closes higher = 1, otherwise 0.
    data["TARGET"] = (
        data["Close"].shift(-1) > data["Close"]
    ).astype(int)

    return data.replace([np.inf, -np.inf], np.nan)


FEATURES = [
    "SMA20",
    "SMA50",
    "EMA12",
    "EMA26",
    "MACD",
    "MACD_SIGNAL",
    "MACD_HIST",
    "RSI",
    "ATR",
    "RETURN_1",
    "RETURN_5",
    "VOLATILITY_10",
    "SMA_DISTANCE",
    "EMA_DISTANCE",
    "VOLUME_CHANGE",
]


# -----------------------------
# Data
# -----------------------------
@st.cache_data(ttl=300)
def download_data(symbol, period, interval):
    df = yf.download(
        symbol,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )

    if df is None or df.empty:
        return pd.DataFrame()

    # yfinance can return MultiIndex columns.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            col[0] if isinstance(col, tuple) else col
            for col in df.columns
        ]

    required = ["Open", "High", "Low", "Close", "Volume"]

    missing = [c for c in required if c not in df.columns]
    if missing:
        return pd.DataFrame()

    return df[required].dropna()


# -----------------------------
# Model
# -----------------------------
def prepare_model_data(df):
    data = add_features(df)

    # Remove final row because its future target is unknown.
    data = data.iloc[:-1].copy()
    data = data.dropna(subset=FEATURES + ["TARGET"])

    return data


def train_model(data, train_fraction=0.80):
    split = int(len(data) * train_fraction)

    if split < 100 or len(data) - split < 20:
        raise ValueError(
            "Not enough data for a reliable train/test split."
        )

    train = data.iloc[:split]
    test = data.iloc[split:]

    X_train = train[FEATURES]
    y_train = train["TARGET"]

    X_test = test[FEATURES]
    y_test = test["TARGET"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)
    probabilities = model.predict_proba(X_test_scaled)[:, 1]

    accuracy = accuracy_score(y_test, predictions)

    test_results = test.copy()
    test_results["Prediction"] = predictions
    test_results["Up_Probability"] = probabilities
    test_results["Down_Probability"] = 1 - probabilities

    return (
        model,
        scaler,
        accuracy,
        test_results,
    )


# -----------------------------
# Paper backtest
# -----------------------------
def run_backtest(
    results,
    initial_balance=100000,
    risk_percent=0.5,
    stop_atr=1.5,
    reward_risk=2.0,
    confidence_threshold=0.55,
):
    balance = float(initial_balance)
    equity = []
    trades = []

    for timestamp, row in results.iterrows():
        close = float(row["Close"])
        atr_value = float(row["ATR"])
        up_prob = float(row["Up_Probability"])

        if not np.isfinite(atr_value) or atr_value <= 0:
            equity.append(balance)
            continue

        if up_prob >= confidence_threshold:
            direction = "UP"
            confidence = up_prob
        elif up_prob <= 1 - confidence_threshold:
            direction = "DOWN"
            confidence = 1 - up_prob
        else:
            equity.append(balance)
            continue

        # Approximate next-candle outcome using the target.
        actual_up = int(row["TARGET"]) == 1
        correct = (
            direction == "UP" and actual_up
        ) or (
            direction == "DOWN" and not actual_up
        )

        risk_amount = balance * (risk_percent / 100)
        stop_distance = atr_value * stop_atr

        if correct:
            profit = risk_amount * reward_risk
            outcome = "WIN"
        else:
            profit = -risk_amount
            outcome = "LOSS"

        balance += profit

        trades.append({
            "Time": timestamp,
            "Direction": direction,
            "Confidence": confidence,
            "Entry": close,
            "ATR": atr_value,
            "Risk Amount": risk_amount,
            "Outcome": outcome,
            "PnL": profit,
            "Balance": balance,
            "Correct": correct,
        })

        equity.append(balance)

    trades_df = pd.DataFrame(trades)

    if not trades_df.empty:
        wins = int((trades_df["Outcome"] == "WIN").sum())
        losses = int((trades_df["Outcome"] == "LOSS").sum())
        total = len(trades_df)
        win_rate = wins / total * 100

        max_balance = trades_df["Balance"].cummax()
        drawdown = (
            trades_df["Balance"] / max_balance - 1
        ) * 100
        max_drawdown = float(drawdown.min())
    else:
        wins = losses = total = 0
        win_rate = 0
        max_drawdown = 0

    return {
        "balance": balance,
        "trades": trades_df,
        "wins": wins,
        "losses": losses,
        "total": total,
        "win_rate": win_rate,
        "max_drawdown": max_drawdown,
        "initial_balance": initial_balance,
    }


# -----------------------------
# Current prediction
# -----------------------------
def predict_current(model, scaler, df):
    featured = add_features(df)
    latest = featured.dropna(subset=FEATURES).iloc[-1:]

    X = scaler.transform(latest[FEATURES])
    probability_up = float(model.predict_proba(X)[0, 1])

    if probability_up >= 0.55:
        signal = "UP"
    elif probability_up <= 0.45:
        signal = "DOWN"
    else:
        signal = "NEUTRAL"

    confidence = (
        probability_up
        if signal == "UP"
        else 1 - probability_up
        if signal == "DOWN"
        else 0.50
    )

    return latest.iloc[0], signal, confidence, probability_up


# -----------------------------
# App UI
# -----------------------------
st.title("📊 Forex Market Predictor")
st.caption(
    "Machine-learning market analysis and paper backtesting — "
    "not financial advice and not a real trading system."
)

with st.sidebar:
    st.header("Settings")

    symbol = st.selectbox(
        "Market",
        [
            "EURUSD=X",
            "GBPUSD=X",
            "USDJPY=X",
            "AUDUSD=X",
            "USDCAD=X",
            "USDCHF=X",
            "NZDUSD=X",
            "XAUUSD=X",
        ],
        index=0,
    )

    period = st.selectbox(
        "Historical period",
        ["1y", "2y", "5y", "10y"],
        index=1,
    )

    interval = st.selectbox(
        "Candle interval",
        ["1d", "1h"],
        index=0,
    )

    initial_balance = st.number_input(
        "Paper balance",
        min_value=1000.0,
        value=100000.0,
        step=1000.0,
    )

    risk_percent = st.number_input(
        "Simulated risk per trade (%)",
        min_value=0.1,
        max_value=5.0,
        value=0.5,
        step=0.1,
    )

    confidence_threshold = st.slider(
        "Prediction threshold",
        min_value=0.50,
        max_value=0.80,
        value=0.55,
        step=0.01,
    )

    stop_atr = st.slider(
        "Stop distance (ATR multiplier)",
        0.5,
        4.0,
        1.5,
        0.1,
    )

    reward_risk = st.slider(
        "Reward / Risk",
        1.0,
        5.0,
        2.0,
        0.25,
    )

    run_button = st.button(
        "🚀 Analyze Market",
        use_container_width=True,
    )


if run_button or "analysis_done" in st.session_state:
    if run_button:
        st.session_state["analysis_done"] = True

    with st.spinner("Downloading data and training the model..."):
        df = download_data(symbol, period, interval)

    if df.empty:
        st.error(
            "No market data was returned. Try another symbol, "
            "period, or interval."
        )
        st.stop()

    data = prepare_model_data(df)

    try:
        (
            model,
            scaler,
            accuracy,
            test_results,
        ) = train_model(data)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    latest, signal, confidence, probability_up = predict_current(
        model,
        scaler,
        df,
    )

    # -------------------------
    # Prediction cards
    # -------------------------
    st.subheader(f"Current analysis: {symbol}")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Signal",
            signal,
        )

    with col2:
        st.metric(
            "UP probability",
            f"{probability_up * 100:.1f}%",
        )

    with col3:
        st.metric(
            "DOWN probability",
            f"{(1 - probability_up) * 100:.1f}%",
        )

    with col4:
        st.metric(
            "Test accuracy",
            f"{accuracy * 100:.1f}%",
        )

    st.info(
        "The signal is a statistical estimate from historical patterns. "
        "It cannot reliably predict the future price."
    )

    # -------------------------
    # Technical indicators
    # -------------------------
    st.subheader("Technical indicators")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Last price",
            f"{float(latest['Close']):.5f}",
        )

    with c2:
        st.metric(
            "RSI",
            f"{float(latest['RSI']):.2f}",
        )

    with c3:
        st.metric(
            "MACD",
            f"{float(latest['MACD']):.5f}",
        )

    with c4:
        st.metric(
            "ATR",
            f"{float(latest['ATR']):.5f}",
        )

    # -------------------------
    # Chart
    # -------------------------
    st.subheader("Price")

    chart_data = df[["Close"]].copy()
    chart_data["SMA20"] = df["Close"].rolling(20).mean()
    chart_data["SMA50"] = df["Close"].rolling(50).mean()

    st.line_chart(chart_data)

    # -------------------------
    # Paper backtest
    # -------------------------
    st.subheader("Paper backtest")

    backtest = run_backtest(
        test_results,
        initial_balance=initial_balance,
        risk_percent=risk_percent,
        stop_atr=stop_atr,
        reward_risk=reward_risk,
        confidence_threshold=confidence_threshold,
    )

    b1, b2, b3, b4 = st.columns(4)

    with b1:
        st.metric(
            "Final paper balance",
            f"{backtest['balance']:,.2f}",
        )

    with b2:
        st.metric(
            "Trades",
            backtest["total"],
        )

    with b3:
        st.metric(
            "Win rate",
            f"{backtest['win_rate']:.1f}%",
        )

    with b4:
        st.metric(
            "Max drawdown",
            f"{backtest['max_drawdown']:.2f}%",
        )

    if not backtest["trades"].empty:
        equity = backtest["trades"].set_index("Time")[
            ["Balance"]
        ]
        st.line_chart(equity)

        st.subheader("Paper trade log")
        st.dataframe(
            backtest["trades"].tail(50),
            use_container_width=True,
        )

    # -------------------------
    # Feature importance
    # -------------------------
    st.subheader("Model feature importance")

    importance = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": model.feature_importances_,
    }).sort_values(
        "Importance",
        ascending=False,
    )

    st.bar_chart(
        importance.set_index("Feature")
    )

    # -------------------------
    # Classification report
    # -------------------------
    st.subheader("Model test report")

    report = classification_report(
        test_results["TARGET"],
        test_results["Prediction"],
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).transpose()
    st.dataframe(
        report_df.round(3),
        use_container_width=True,
    )

else:
    st.markdown(
        """
        ### How to use

        1. Choose a forex market.
        2. Choose the historical period and candle interval.
        3. Click **Analyze Market**.
        4. Review the model's UP/DOWN probability.
        5. Review the historical paper backtest.
        6. Do not treat the prediction as a guaranteed trading signal.

        **Suggested starting point:** use `EURUSD=X` with daily candles and
        at least `2y` of historical data.
        """
    )

st.divider()

st.caption(
    "Educational software only. Historical backtests can be misleading "
    "and do not guarantee future performance. No live order execution is "
    "included."
)
