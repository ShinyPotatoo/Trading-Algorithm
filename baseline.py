"""
Strategy V1: Baseline Logistic Regression
==========================================
NVDA daily trading strategy using technical features and a logistic regression
classifier. Binary long-or-cash positions based on a probability threshold.

Install dependencies:
    pip install yfinance scikit-learn pandas numpy matplotlib
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ── Config ─────────────────────────────────────────────────────────────────────
TICKER         = 'NVDA'
START          = '2015-01-01'
END            = '2024-12-31'
RETURN_TARGET  = 0.002   # 0.2% threshold for positive label
PROB_THRESHOLD = 0.55    # min predicted probability to go long
TRAIN_RATIO    = 0.70    # fraction of data used for training

# ── Data ───────────────────────────────────────────────────────────────────────
def fetch_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
    return df

# ── Features ───────────────────────────────────────────────────────────────────
def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(window).mean()
    loss  = (-delta.clip(upper=0)).rolling(window).mean()
    rs    = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)

    # Price momentum (all shifted by 1 to avoid lookahead)
    f['log_ret']    = np.log(df['close'] / df['close'].shift(1))
    f['ret_lag1']   = f['log_ret'].shift(1)
    f['ret_5d']     = f['log_ret'].rolling(5).mean().shift(1)
    f['ret_10d']    = f['log_ret'].rolling(10).mean().shift(1)

    # Volume
    f['vol_chg']    = np.log(df['volume'] / df['volume'].shift(1)).shift(1)

    # RSI (shifted so we only use info available at open)
    f['rsi']        = compute_rsi(df['close']).shift(1)

    # EMA ratio (price relative to trend)
    ema20           = df['close'].ewm(span=20, adjust=False).mean()
    f['ema_ratio']  = (df['close'] / ema20).shift(1)

    # Rolling volatility
    f['volatility'] = f['log_ret'].rolling(20).std().shift(1)

    # Target and forward return (never used as feature)
    f['next_ret']   = f['log_ret'].shift(-1)
    f['target']     = (f['next_ret'] > RETURN_TARGET).astype(int)

    return f.dropna()

# ── Metrics ────────────────────────────────────────────────────────────────────
def print_metrics(returns: pd.Series, label: str) -> dict:
    ann_ret = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    sharpe  = ann_ret / (ann_vol + 1e-9)
    cum     = (1 + returns).cumprod()
    max_dd  = (cum / cum.cummax() - 1).min()

    print(f"\n  [{label}]")
    print(f"    Annual Return   : {ann_ret:+.2%}")
    print(f"    Annual Volatility: {ann_vol:.2%}")
    print(f"    Sharpe Ratio    : {sharpe:.2f}")
    print(f"    Max Drawdown    : {max_dd:.2%}")
    return dict(ann_ret=ann_ret, ann_vol=ann_vol, sharpe=sharpe, max_dd=max_dd)

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  Strategy V1 — Baseline Logistic Regression")
    print("=" * 55)

    # 1. Data & features
    raw  = fetch_data(TICKER, START, END)
    feat = engineer_features(raw)

    FEATURE_COLS = ['ret_lag1', 'ret_5d', 'ret_10d', 'vol_chg',
                    'rsi', 'ema_ratio', 'volatility']

    X = feat[FEATURE_COLS].values
    y = feat['target'].values

    # 2. Train / test split (temporal — no shuffling)
    split     = int(len(X) * TRAIN_RATIO)
    X_tr, X_te = X[:split], X[split:]
    y_tr      = y[:split]

    print(f"\n  Train: {feat.index[0].date()} → {feat.index[split-1].date()} ({split} days)")
    print(f"  Test : {feat.index[split].date()} → {feat.index[-1].date()} ({len(X_te)} days)")

    # 3. Scale & fit
    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)

    model   = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
    model.fit(X_tr_sc, y_tr)

    probs   = model.predict_proba(X_te_sc)[:, 1]

    # 4. Backtest
    test = feat.iloc[split:].copy()
    test['prob']     = probs
    test['position'] = (probs >= PROB_THRESHOLD).astype(float)

    # Position applies on the *next* bar (shift by 1 to avoid lookahead)
    test['strat_ret']   = test['position'].shift(1) * test['next_ret']
    test['strat_cum']   = (1 + test['strat_ret'].fillna(0)).cumprod()
    test['bh_cum']      = (1 + test['next_ret'].fillna(0)).cumprod()

    print("\nPerformance (out-of-sample):")
    print_metrics(test['strat_ret'].dropna(), 'Strategy V1')
    print_metrics(test['next_ret'].dropna(),  'Buy & Hold')

    # 5. Plot
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=False)
    fig.suptitle('Strategy V1 — Baseline Logistic Regression', fontsize=14, fontweight='bold')

    # Cumulative returns
    ax = axes[0]
    test[['strat_cum', 'bh_cum']].plot(ax=ax, color=['steelblue', 'gray'])
    ax.set_title('Cumulative Returns (Out-of-Sample)')
    ax.set_ylabel('Portfolio Value ($1 start)')
    ax.legend(['Strategy V1', 'Buy & Hold'])
    ax.grid(alpha=0.3)

    # Drawdown
    ax = axes[1]
    dd = (test['strat_cum'] / test['strat_cum'].cummax() - 1)
    dd.plot(ax=ax, color='crimson')
    ax.fill_between(test.index, dd, 0, alpha=0.3, color='crimson')
    ax.set_title('Strategy Drawdown')
    ax.set_ylabel('Drawdown')
    ax.grid(alpha=0.3)

    # Predicted probability
    ax = axes[2]
    ax.plot(test.index, probs, color='mediumorchid', linewidth=0.8, label='P(positive)')
    ax.axhline(PROB_THRESHOLD, color='orange', linestyle='--',
               label=f'Threshold ({PROB_THRESHOLD})', linewidth=1.5)
    ax.fill_between(test.index, PROB_THRESHOLD, probs,
                    where=probs >= PROB_THRESHOLD, alpha=0.25, color='green', label='Long')
    ax.set_title('Predicted Long Probability')
    ax.set_ylabel('Probability')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = 'strategy_v1_results.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\n  Chart saved → {out}")
    plt.show()


if __name__ == '__main__':
    main()
