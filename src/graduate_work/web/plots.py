"""Plotly-фигуры для веб-интерфейса трейдера."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def price_with_uncertainty(
    prices: pd.DataFrame,
    predictions: pd.DataFrame | None,
    ticker: str,
) -> str:
    """Возвращает JSON Plotly-фигуры: цена + последний прогноз с дов. интервалом."""
    sub = prices[prices["ticker"] == ticker].sort_index()
    fig = go.Figure()
    if not sub.empty:
        fig.add_trace(
            go.Scatter(
                x=sub.index,
                y=sub["close"],
                mode="lines",
                name=f"{ticker} close",
                line={"color": "#1f77b4"},
            ),
        )

    if predictions is not None and not predictions.empty:
        pred_t = predictions[predictions["ticker"] == ticker].copy()
        pred_t = pred_t.sort_values("timestamp")
        if not pred_t.empty:
            best = pred_t.loc[pred_t.groupby("timestamp")["mean"].idxmax()].copy()
            best["mean_pct"] = best["mean"] * best["horizon"]
            best["upper"] = (best["mean"] + best["std"]) * best["horizon"]
            best["lower"] = (best["mean"] - best["std"]) * best["horizon"]
            join = best.merge(
                sub[["close"]], left_on="timestamp", right_index=True, how="inner",
            )
            join["pred_close"] = join["close"] * np.exp(join["mean_pct"])
            join["upper_close"] = join["close"] * np.exp(join["upper"])
            join["lower_close"] = join["close"] * np.exp(join["lower"])
            fig.add_trace(
                go.Scatter(
                    x=join["timestamp"], y=join["upper_close"],
                    mode="lines", name="upper CI",
                    line={"color": "rgba(255,127,14,0.4)", "dash": "dot"},
                ),
            )
            fig.add_trace(
                go.Scatter(
                    x=join["timestamp"], y=join["lower_close"],
                    mode="lines", name="lower CI", fill="tonexty",
                    fillcolor="rgba(255,127,14,0.15)",
                    line={"color": "rgba(255,127,14,0.4)", "dash": "dot"},
                ),
            )
            fig.add_trace(
                go.Scatter(
                    x=join["timestamp"], y=join["pred_close"],
                    mode="lines", name="MC mean",
                    line={"color": "#ff7f0e"},
                ),
            )

    fig.update_layout(
        template="plotly_white",
        title=f"{ticker}: цена и прогноз с MC Dropout доверительным интервалом",
        xaxis_title="дата",
        yaxis_title="цена, RUB",
        legend={"orientation": "h"},
        margin={"l": 40, "r": 20, "t": 60, "b": 40},
    )
    return fig.to_json()


def equity_curve(equity: pd.Series) -> str:
    fig = go.Figure()
    if not equity.empty:
        fig.add_trace(
            go.Scatter(
                x=equity.index, y=equity.values,
                mode="lines", name="equity",
                line={"color": "#2ca02c"},
            ),
        )
    fig.update_layout(
        template="plotly_white",
        title="Кривая капитала стратегии",
        xaxis_title="дата",
        yaxis_title="капитал, RUB",
        margin={"l": 40, "r": 20, "t": 60, "b": 40},
    )
    return fig.to_json()


def random_portfolios_distribution(
    final_returns: np.ndarray,
    strategy_return: float,
    threshold: float,
) -> str:
    fig = go.Figure()
    if final_returns.size > 0:
        fig.add_trace(
            go.Histogram(
                x=final_returns * 100,
                nbinsx=40,
                name="random monkeys",
                marker={"color": "#9467bd"},
                opacity=0.7,
            ),
        )
    fig.add_vline(
        x=strategy_return * 100,
        line_color="#d62728",
        line_width=3,
        annotation_text="стратегия",
    )
    fig.add_vline(
        x=threshold * 100,
        line_color="#2ca02c",
        line_dash="dash",
        annotation_text="порог 3σ",
    )
    fig.update_layout(
        template="plotly_white",
        title="Распределение случайных портфелей и позиция стратегии",
        xaxis_title="итоговая доходность, %",
        yaxis_title="частота",
        margin={"l": 40, "r": 20, "t": 60, "b": 40},
    )
    return fig.to_json()


def per_ticker_returns(per_ticker: pd.DataFrame) -> str:
    """Bar-chart per-ticker итоговой доходности (для страницы /report)."""
    fig = go.Figure()
    if not per_ticker.empty:
        df = per_ticker.sort_values("total_return")
        colors = ["#d62728" if r < 0 else "#2ca02c" for r in df["total_return"]]
        fig.add_trace(
            go.Bar(
                x=df["ticker"],
                y=df["total_return"] * 100,
                marker_color=colors,
                name="итоговая доходность",
            ),
        )
    fig.update_layout(
        template="plotly_white",
        title="Per-ticker: итоговая доходность стратегии",
        xaxis_title="тикер",
        yaxis_title="итоговая доходность, %",
        margin={"l": 40, "r": 20, "t": 60, "b": 40},
    )
    return fig.to_json()


def signals_table(signals: pd.DataFrame) -> list[dict]:
    if signals.empty:
        return []
    cols = ["timestamp", "ticker", "horizon", "mean", "std", "action"]
    df = signals[cols].copy()
    df["timestamp"] = df["timestamp"].astype(str)
    df["mean"] = df["mean"].round(5)
    df["std"] = df["std"].round(5)
    return df.to_dict(orient="records")


