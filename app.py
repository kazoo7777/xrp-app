"""
XRP テクニカル分析・AI予測 Streamlit アプリ
=============================================
- ロウソク足チャート + ボリンジャーバンド + EMA
- RSI (14期間)
- MACD (12, 26, 9)
- Prophet による将来価格予測
- Gemini AI による市場分析
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prophet import Prophet
from datetime import datetime, timedelta
import os
import warnings

from dotenv import load_dotenv
from google import genai

warnings.filterwarnings("ignore")

# .env または Streamlit Secrets から API キーを読み込み
load_dotenv(override=True)
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except (KeyError, FileNotFoundError):
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# ──────────────────────────────────────────────
# ページ設定
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="XRP テクニカル分析 & AI予測",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# カスタムCSS
# ──────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 1rem;
}

h1 {
    background: linear-gradient(135deg, #00d4ff 0%, #7b2ff7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700 !important;
    letter-spacing: -0.5px;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    padding: 10px 24px;
    border-radius: 8px 8px 0 0;
    font-weight: 600;
}

div[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(0,212,255,0.08), rgba(123,47,247,0.08));
    border: 1px solid rgba(123,47,247,0.15);
    border-radius: 12px;
    padding: 16px 20px;
}

div[data-testid="stMetric"] label {
    font-size: 0.85rem;
    font-weight: 500;
    opacity: 0.8;
}

div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-size: 1.8rem;
    font-weight: 700;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0e1117 0%, #1a1f2e 100%);
}

section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label {
    font-weight: 600;
    color: #e0e0e0;
}
</style>
""",
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────
# テクニカル指標計算
# ──────────────────────────────────────────────
def calc_bollinger_bands(df: pd.DataFrame, length: int = 20, std: float = 2.0):
    """ボリンジャーバンド"""
    tp = df["Close"]
    sma = tp.rolling(window=length).mean()
    rolling_std = tp.rolling(window=length).std()
    df["BB_Upper"] = sma + std * rolling_std
    df["BB_Middle"] = sma
    df["BB_Lower"] = sma - std * rolling_std
    return df


def calc_rsi(df: pd.DataFrame, length: int = 14):
    """RSI (Relative Strength Index)"""
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / length, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df


def calc_macd(
    df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9
):
    """MACD (Moving Average Convergence Divergence)"""
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    df["MACD"] = ema_fast - ema_slow
    df["MACD_Signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    return df


def calc_ema(df: pd.DataFrame, periods: tuple) -> pd.DataFrame:
    """EMA（指数移動平均線）"""
    for p in periods:
        df[f"EMA_{p}"] = df["Close"].ewm(span=p, adjust=False).mean()
    return df


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """OHLCV データを指定ルールでリサンプリング"""
    resampled = df.resample(rule).agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }).dropna()
    return resampled


# ──────────────────────────────────────────────
# 足種（タイムフレーム）・期間 設定
# ──────────────────────────────────────────────
INTERVAL_OPTIONS = {
    "1分足": {"interval": "1m", "max_days": 7, "resample": None},
    "15分足": {"interval": "15m", "max_days": 60, "resample": None},
    "30分足": {"interval": "30m", "max_days": 60, "resample": None},
    "1時間足": {"interval": "1h", "max_days": 730, "resample": None},
    "4時間足": {"interval": "1h", "max_days": 730, "resample": "4h"},
    "8時間足": {"interval": "1h", "max_days": 730, "resample": "8h"},
    "日足": {"interval": "1d", "max_days": None, "resample": None},
    "週足": {"interval": "1wk", "max_days": None, "resample": None},
    "月足": {"interval": "1mo", "max_days": None, "resample": None},
}

PERIOD_BY_MAX_DAYS = {
    7: {"1日": "1d", "5日": "5d"},
    60: {"5日": "5d", "1ヶ月": "1mo"},
    730: {"1ヶ月": "1mo", "3ヶ月": "3mo", "6ヶ月": "6mo", "1年": "1y"},
    None: {"1ヶ月": "1mo", "3ヶ月": "3mo", "6ヶ月": "6mo", "1年": "1y", "3年": "3y", "全期間": "max"},
}

# EMA カラー設定
EMA_COLORS = {
    9: "#FF6B6B",
    21: "#FFD93D",
    50: "#6BCB77",
    100: "#4D96FF",
    200: "#9B59B6",
}

# ──────────────────────────────────────────────
# データ取得
# ──────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def fetch_xrp_data(
    period: str,
    interval: str = "1d",
    resample_rule: str = None,
    ema_periods: tuple = (),
) -> pd.DataFrame:
    """yfinance で XRP-USD データを取得し、テクニカル指標を付与する"""
    import time

    # リトライ付きでデータ取得
    for attempt in range(3):
        try:
            df = yf.download(
                "XRP-USD",
                period=period,
                interval=interval,
                auto_adjust=True,
                timeout=30,
            )
            if not df.empty:
                break
        except Exception:
            pass
        if attempt < 2:
            time.sleep(2)
    else:
        df = pd.DataFrame()

    if df.empty:
        return df

    # MultiIndex の場合にフラット化
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # リサンプリング（4h / 8h）
    if resample_rule:
        df = resample_ohlcv(df, resample_rule)

    # テクニカル指標計算
    df = calc_bollinger_bands(df)
    df = calc_rsi(df)
    df = calc_macd(df)

    # EMA 計算
    if ema_periods:
        df = calc_ema(df, ema_periods)

    df.dropna(inplace=True)
    return df


# ──────────────────────────────────────────────
# Plotly チャート作成
# ──────────────────────────────────────────────
CHART_TEMPLATE = "plotly_dark"
BG_COLOR = "rgba(14,17,23,0)"
GRID_COLOR = "rgba(255,255,255,0.06)"
UP_COLOR = "#00e676"
DOWN_COLOR = "#ff1744"
BB_FILL = "rgba(0,212,255,0.08)"
BB_LINE = "rgba(0,212,255,0.5)"
RSI_COLOR = "#FFC107"
MACD_LINE = "#00d4ff"
SIGNAL_LINE = "#ff9800"


def _add_candlestick(fig, df, row=1):
    """ロウソク足を追加"""
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            increasing_line_color=UP_COLOR,
            decreasing_line_color=DOWN_COLOR,
            increasing_fillcolor=UP_COLOR,
            decreasing_fillcolor=DOWN_COLOR,
            name="XRP-USD",
        ),
        row=row, col=1,
    )


def _add_bollinger(fig, df, row=1):
    """ボリンジャーバンドを追加"""
    fig.add_trace(
        go.Scatter(x=df.index, y=df["BB_Upper"], line=dict(color=BB_LINE, width=1),
                   name="BB Upper", showlegend=False),
        row=row, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df["BB_Lower"], line=dict(color=BB_LINE, width=1),
                   fill="tonexty", fillcolor=BB_FILL, name="BB Lower", showlegend=False),
        row=row, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df["BB_Middle"],
                   line=dict(color="rgba(0,212,255,0.35)", width=1, dash="dot"),
                   name="BB Middle", showlegend=False),
        row=row, col=1,
    )


def _add_ema(fig, df, ema_periods, row=1):
    """EMA ラインを追加"""
    for p in ema_periods:
        col_name = f"EMA_{p}"
        if col_name in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[col_name],
                    line=dict(color=EMA_COLORS.get(p, "#FFFFFF"), width=1.5),
                    name=f"EMA {p}",
                ),
                row=row, col=1,
            )


def _add_rsi(fig, df, row):
    """RSI を追加"""
    fig.add_trace(
        go.Scatter(x=df.index, y=df["RSI"], line=dict(color=RSI_COLOR, width=1.5), name="RSI"),
        row=row, col=1,
    )
    fig.add_hline(y=70, line_dash="dash", line_color="rgba(255,23,68,0.5)", line_width=1, row=row, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="rgba(0,230,118,0.5)", line_width=1, row=row, col=1)
    fig.add_hrect(y0=30, y1=70, fillcolor="rgba(255,255,255,0.03)", line_width=0, row=row, col=1)


def _add_macd(fig, df, row):
    """MACD を追加"""
    hist_colors = [UP_COLOR if v >= 0 else DOWN_COLOR for v in df["MACD_Hist"]]
    fig.add_trace(
        go.Bar(x=df.index, y=df["MACD_Hist"], marker_color=hist_colors, name="Histogram", opacity=0.6),
        row=row, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df["MACD"], line=dict(color=MACD_LINE, width=1.5), name="MACD"),
        row=row, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df["MACD_Signal"], line=dict(color=SIGNAL_LINE, width=1.5), name="Signal"),
        row=row, col=1,
    )


def _apply_layout(fig, n_rows):
    """共通レイアウトを適用"""
    fig.update_layout(
        template=CHART_TEMPLATE,
        height=550 if n_rows == 2 else 820,
        margin=dict(l=12, r=12, t=40, b=20),
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
            font=dict(size=11),
        ),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        font=dict(family="Inter, sans-serif"),
    )
    for i in range(1, n_rows + 1):
        xkey = f"xaxis{'' if i == 1 else i}"
        ykey = f"yaxis{'' if i == 1 else i}"
        fig.update_layout(**{xkey: dict(gridcolor=GRID_COLOR, showgrid=True)})
        fig.update_layout(**{ykey: dict(gridcolor=GRID_COLOR, showgrid=True)})


def build_analysis_chart(
    df: pd.DataFrame,
    display_mode: str = "全表示",
    ema_periods: tuple = (),
    interval_label: str = "日足",
) -> go.Figure:
    """表示モードに応じたチャートを生成"""

    if display_mode == "チャート + RSI":
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
            row_heights=[0.7, 0.3],
            subplot_titles=(f"XRP-USD  ローソク足 ({interval_label})", "RSI (14)"),
        )
        _add_candlestick(fig, df, row=1)
        _add_bollinger(fig, df, row=1)
        _add_ema(fig, df, ema_periods, row=1)
        _add_rsi(fig, df, row=2)
        _apply_layout(fig, 2)
        fig.update_yaxes(title_text="価格 (USD)", row=1, col=1)
        fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])

    elif display_mode == "チャート + MACD":
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
            row_heights=[0.7, 0.3],
            subplot_titles=(f"XRP-USD  ローソク足 ({interval_label})", "MACD (12, 26, 9)"),
        )
        _add_candlestick(fig, df, row=1)
        _add_bollinger(fig, df, row=1)
        _add_ema(fig, df, ema_periods, row=1)
        _add_macd(fig, df, row=2)
        _apply_layout(fig, 2)
        fig.update_yaxes(title_text="価格 (USD)", row=1, col=1)
        fig.update_yaxes(title_text="MACD", row=2, col=1)

    else:  # 全表示
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=(f"XRP-USD  ローソク足 ({interval_label})", "RSI (14)", "MACD (12, 26, 9)"),
        )
        _add_candlestick(fig, df, row=1)
        _add_bollinger(fig, df, row=1)
        _add_ema(fig, df, ema_periods, row=1)
        _add_rsi(fig, df, row=2)
        _add_macd(fig, df, row=3)
        _apply_layout(fig, 3)
        fig.update_yaxes(title_text="価格 (USD)", row=1, col=1)
        fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
        fig.update_yaxes(title_text="MACD", row=3, col=1)

    return fig


# ──────────────────────────────────────────────
# Prophet 予測
# ──────────────────────────────────────────────
@st.cache_data(ttl=600, show_spinner=False)
def run_prophet(df: pd.DataFrame, forecast_days: int):
    """Prophet で将来価格を予測"""
    prophet_df = df[["Close"]].reset_index()
    prophet_df.columns = ["ds", "y"]
    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"]).dt.tz_localize(None)

    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.15,
    )
    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)
    return forecast, model


def build_forecast_chart(df: pd.DataFrame, forecast: pd.DataFrame) -> go.Figure:
    """予測結果チャートを生成"""
    fig = go.Figure()

    # 実績
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["Close"],
            line=dict(color="#00d4ff", width=1.5), name="実績価格",
        )
    )

    # 予測
    forecast_only = forecast[forecast["ds"] > df.index[-1]]
    fig.add_trace(
        go.Scatter(
            x=forecast_only["ds"], y=forecast_only["yhat"],
            line=dict(color="#7b2ff7", width=2.5), name="予測価格",
        )
    )

    # 信頼区間
    fig.add_trace(
        go.Scatter(
            x=forecast_only["ds"], y=forecast_only["yhat_upper"],
            line=dict(width=0), showlegend=False, name="上限",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_only["ds"], y=forecast_only["yhat_lower"],
            line=dict(width=0), fill="tonexty",
            fillcolor="rgba(123,47,247,0.15)", name="信頼区間",
        )
    )

    fig.update_layout(
        template=CHART_TEMPLATE,
        height=520,
        margin=dict(l=12, r=12, t=40, b=20),
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        title=dict(text="Prophet による XRP-USD 価格予測", font=dict(size=16)),
        xaxis=dict(gridcolor=GRID_COLOR, title="日付"),
        yaxis=dict(gridcolor=GRID_COLOR, title="価格 (USD)"),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
            font=dict(size=11),
        ),
        hovermode="x unified",
        font=dict(family="Inter, sans-serif"),
    )

    return fig


# ──────────────────────────────────────────────
# Gemini AI 分析
# ──────────────────────────────────────────────
GEMINI_PROMPT_TEMPLATE = """
あなたは暗号資産（仮想通貨）の専門アナリストです。
以下の XRP-USD のテクニカルデータを分析し、今後 {forecast_days} 日間の値動きを予測してください。

## 現在の市場データ
- 現在価格: ${current_price:.4f}
- 24時間変動: {price_change_pct:+.2f}%
- 24時間出来高: {volume:,.0f}

## テクニカル指標（最新値）
- RSI (14): {rsi:.1f}
- MACD: {macd:.6f}
- MACD シグナル: {macd_signal:.6f}
- MACD ヒストグラム: {macd_hist:.6f}
- ボリンジャーバンド 上限: ${bb_upper:.4f}
- ボリンジャーバンド 中央: ${bb_middle:.4f}
- ボリンジャーバンド 下限: ${bb_lower:.4f}

## 直近 30 日間の価格推移
{price_history}

## 依頼内容
1. 上記のテクニカル指標を総合的に分析し、現在の市場状況を評価してください。
2. 今後 {forecast_days} 日間の予測される値動き（上昇・下落・横ばい）とその理由を解説してください。
3. 投資家が注意すべきリスクや重要なサポート・レジスタンスラインがあれば指摘してください。
4. 全て日本語で回答してください。

※ これは参考情報であり、投資助言ではありません。
"""


def build_gemini_prompt(df: pd.DataFrame, forecast_days: int) -> str:
    """テクニカルデータからプロンプトを生成"""
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    price_change_pct = ((latest["Close"] - prev["Close"]) / prev["Close"]) * 100

    # 直近 30 日の価格推移テーブル
    recent = df.tail(30)[["Close", "RSI", "MACD"]].copy()
    price_lines = []
    for date, row in recent.iterrows():
        d = date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date)
        price_lines.append(f"{d}: 終値=${row['Close']:.4f}  RSI={row['RSI']:.1f}  MACD={row['MACD']:.6f}")
    price_history = "\n".join(price_lines)

    return GEMINI_PROMPT_TEMPLATE.format(
        forecast_days=forecast_days,
        current_price=latest["Close"],
        price_change_pct=price_change_pct,
        volume=latest["Volume"],
        rsi=latest["RSI"],
        macd=latest["MACD"],
        macd_signal=latest["MACD_Signal"],
        macd_hist=latest["MACD_Hist"],
        bb_upper=latest["BB_Upper"],
        bb_middle=latest["BB_Middle"],
        bb_lower=latest["BB_Lower"],
        price_history=price_history,
    )


def run_gemini_analysis(prompt: str, model_name: str = "gemini-2.5-pro") -> str:
    """Gemini API にプロンプトを送信して分析結果を取得"""
    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )
    return response.text


# ──────────────────────────────────────────────
# サイドバー
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ 設定")
    st.markdown("---")

    # ── 足種選択 ─────────────────────────────
    st.markdown("##### 🕐 足種（タイムフレーム）")
    selected_interval_label = st.selectbox(
        "足種",
        options=list(INTERVAL_OPTIONS.keys()),
        index=6,  # デフォルト: 日足
        label_visibility="collapsed",
    )
    interval_config = INTERVAL_OPTIONS[selected_interval_label]
    max_days = interval_config["max_days"]

    # ── 期間選択（足種に応じた制限） ────────
    st.markdown("")
    st.markdown("##### 📅 データ取得期間")
    period_options = PERIOD_BY_MAX_DAYS[max_days]
    period_keys = list(period_options.keys())
    selected_period_label = st.selectbox(
        "期間",
        options=period_keys,
        index=len(period_keys) - 1,  # デフォルト: 最長期間
        label_visibility="collapsed",
    )
    selected_period = period_options[selected_period_label]

    # ── 表示モード ──────────────────────────
    st.markdown("")
    st.markdown("##### 📊 チャート表示")
    display_mode = st.radio(
        "表示モード",
        ["全表示", "チャート + RSI", "チャート + MACD"],
        index=0,
        label_visibility="collapsed",
    )

    # ── EMA 設定 ────────────────────────────
    st.markdown("")
    st.markdown("##### 📈 EMA（移動平均線）")
    ema_periods = st.multiselect(
        "EMA 期間",
        options=[9, 21, 50, 100, 200],
        default=[],
        format_func=lambda x: f"EMA {x}",
        label_visibility="collapsed",
    )
    if ema_periods:
        legend_html = " ".join(
            [f'<span style="color:{EMA_COLORS[p]}; font-weight:bold;">●EMA{p}</span>' for p in sorted(ema_periods)]
        )
        st.markdown(f'<div style="font-size:0.8rem;">{legend_html}</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── 予測日数 ────────────────────────────
    forecast_days = st.select_slider(
        "🔮 予測日数",
        options=[7, 14, 30, 60, 90],
        value=30,
    )

    st.markdown("---")
    st.markdown(
        """
        <div style="text-align:center; opacity:0.5; font-size:0.75rem; margin-top:2rem;">
        Powered by yfinance · Prophet · Gemini · Plotly<br>
        Built with Streamlit
        </div>
        """,
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────
# メインコンテンツ
# ──────────────────────────────────────────────
st.markdown("# 📈 XRP テクニカル分析 & AI予測")

# データ読み込み
with st.spinner(f"XRP-USD データを取得中（{selected_interval_label}）..."):
    df = fetch_xrp_data(
        period=selected_period,
        interval=interval_config["interval"],
        resample_rule=interval_config["resample"],
        ema_periods=tuple(sorted(ema_periods)),
    )

if df.empty:
    st.error("データの取得に失敗しました。インターネット接続を確認するか、別の足種・期間を選択してください。")
    st.stop()

# メトリクス表示
latest = df.iloc[-1]
prev = df.iloc[-2]
price_change = latest["Close"] - prev["Close"]
price_change_pct = (price_change / prev["Close"]) * 100

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("💰 現在価格", f"${latest['Close']:.4f}", f"{price_change_pct:+.2f}%")
with col2:
    st.metric("📊 RSI (14)", f"{latest['RSI']:.1f}")
with col3:
    st.metric("📉 MACD", f"{latest['MACD']:.6f}")
with col4:
    st.metric("📈 出来高", f"{latest['Volume']:,.0f}")

st.markdown("")

# ── タブ切り替え ─────────────────────────────
tab_chart, tab_forecast, tab_gemini = st.tabs(["📊 チャート分析", "🔮 Prophet予測", "🤖 Gemini AI分析"])

with tab_chart:
    analysis_fig = build_analysis_chart(
        df,
        display_mode=display_mode,
        ema_periods=tuple(sorted(ema_periods)),
        interval_label=selected_interval_label,
    )
    st.plotly_chart(analysis_fig, use_container_width=True, config={"scrollZoom": True})

    with st.expander("📋 直近データ (20件)"):
        display_cols = ["Open", "High", "Low", "Close", "Volume", "RSI", "MACD"]
        # EMA カラムも追加
        for p in sorted(ema_periods):
            col_name = f"EMA_{p}"
            if col_name in df.columns:
                display_cols.append(col_name)

        recent = df.tail(20)[display_cols].copy()
        col_names = ["始値", "高値", "安値", "終値", "出来高", "RSI", "MACD"]
        col_names += [f"EMA{p}" for p in sorted(ema_periods) if f"EMA_{p}" in df.columns]
        recent.columns = col_names

        fmt = {
            "始値": "${:.4f}",
            "高値": "${:.4f}",
            "安値": "${:.4f}",
            "終値": "${:.4f}",
            "出来高": "{:,.0f}",
            "RSI": "{:.1f}",
            "MACD": "{:.6f}",
        }
        for p in sorted(ema_periods):
            if f"EMA{p}" in recent.columns:
                fmt[f"EMA{p}"] = "${:.4f}"

        st.dataframe(recent.style.format(fmt), use_container_width=True, hide_index=True)

with tab_forecast:
    # Prophet は日足データが必要
    st.markdown("### 🔮 Prophet による価格予測")
    st.caption("※ Prophet 予測は日足データを使用します")

    with st.spinner("Prophet モデルを構築中..."):
        # 日足データを取得（予測用）
        df_daily = fetch_xrp_data(
            period="1y",
            interval="1d",
            resample_rule=None,
            ema_periods=(),
        )

    if not df_daily.empty:
        forecast, model = run_prophet(df_daily, forecast_days)
        forecast_fig = build_forecast_chart(df_daily, forecast)
        st.plotly_chart(forecast_fig, use_container_width=True, config={"scrollZoom": True})

        forecast_only = forecast[forecast["ds"] > df_daily.index[-1]]
        with st.expander("📋 予測データ詳細"):
            display_forecast = forecast_only[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
            display_forecast.columns = ["日付", "予測価格", "下限", "上限"]
            st.dataframe(
                display_forecast.style.format({
                    "予測価格": "${:.4f}",
                    "下限": "${:.4f}",
                    "上限": "${:.4f}",
                }),
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.warning("日足データの取得に失敗しました。")

with tab_gemini:
    if not GEMINI_API_KEY or GEMINI_API_KEY == "あなたのAPIキーをここに貼り付けてください":
        st.warning(
            "⚠️ Gemini API キーが設定されていません。\n\n"
            "`.env` ファイルに `GEMINI_API_KEY=your_key_here` を設定してください。\n\n"
            "APIキーは [Google AI Studio](https://aistudio.google.com/app/apikey) から取得できます。"
        )
    else:
        MODEL_OPTIONS = {
            "gemini-flash-latest (推奨・高速)": "gemini-flash-latest",
            "gemini-pro-latest (高性能)": "gemini-pro-latest",
        }
        selected_model_label = st.radio(
            "使用モデル",
            list(MODEL_OPTIONS.keys()),
            index=0,
            horizontal=True,
        )
        gemini_model = MODEL_OPTIONS[selected_model_label]

        prompt = build_gemini_prompt(df, forecast_days)

        with st.expander("📝 送信プロンプトを確認", expanded=False):
            st.code(prompt, language="markdown")

        if st.button("🚀 Gemini に分析を依頼する", type="primary", use_container_width=True):
            with st.spinner(f"{gemini_model} が分析中..."):
                try:
                    result = run_gemini_analysis(prompt, model_name=gemini_model)
                    st.markdown("---")
                    st.markdown("### 🤖 Gemini AI 分析レポート")
                    st.markdown(result)
                    st.markdown("---")
                    st.caption("⚠️ この分析は AI による参考情報であり、投資助言ではありません。投資判断はご自身の責任で行ってください。")
                except Exception as e:
                    error_msg = str(e)
                    if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                        st.error(
                            f"⚠️ **{gemini_model}** のレート制限に達しました。\n\n"
                            "**対処法:**\n"
                            "- 別のモデルに切り替えてください\n"
                            "- しばらく時間を置いてから再試行してください\n"
                            "- [Google AI Studio](https://ai.google.dev/gemini-api/docs/rate-limits) でクォータを確認できます"
                        )
                    else:
                        st.error(f"Gemini API エラー: {e}")
