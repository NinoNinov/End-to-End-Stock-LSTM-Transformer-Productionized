from dataclasses import dataclass, field
from typing import Literal


@dataclass
class TickerConfig:
    ticker: str                           # yfinance symbol, e.g. "MSFT" or "^GSPC"
    display_name: str                     # for UI, e.g. "S&P 500"
    history_years: int = 35
    seq_len: int = 60
    feature_columns: list[str] = field(default_factory=lambda: [
        "Close", "Open", "High", "Low", "Volume_M",
        "SMA_20", "EMA_20", "RSI",
        "lag_close_1", "lag_close_2", "lag_return_1", "lag_return_2",
        "Spread_OC", "Spread_HL", "day_of_week",
    ])
    holdout_days: int = 252               # ~1 trading year held out for honest eval
    apply_stl_outlier_removal: bool = True  # Phase 4 A/B winner (lower return_mae on both tickers)
    transformer_hparams: dict = field(default_factory=lambda: dict(
        num_blocks=3, num_heads=4, key_dim=32, ffn_units=64,
        dense_units=32, dropout=0.1, lr=1e-3,
        epochs=30, batch_size=32,
    ))
    xgb_hparams: dict = field(default_factory=lambda: dict(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
    ))


TICKERS: dict[str, TickerConfig] = {
    "MSFT": TickerConfig(ticker="MSFT",  display_name="Microsoft"),
    "SPX":  TickerConfig(ticker="^GSPC", display_name="S&P 500"),
}

RANDOM_STATE = 42
