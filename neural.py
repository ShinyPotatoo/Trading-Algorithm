"""
Strategy V3: Neural Network with Walk-Forward Validation
=========================================================
Next-generation NVDA trading strategy. Key improvements over V1/V2:

  1. PyTorch MLP — non-linear feature interactions, dropout regularisation
  2. Walk-forward (expanding window) via TimeSeriesSplit — no future leakage
  3. Broad market features — SPY, QQQ, SOXX (semis ETF), VIX
  4. Relative-strength & cross-asset momentum features
  5. Transaction-cost-aware backtest (realistic round-trip slippage)
  6. Fractional Kelly position sizing with volatility scaling
  7. Proper out-of-sample holdout (last 20% of data, never touched in training)

Install dependencies:
    pip install yfinance scikit-learn pandas numpy matplotlib torch
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

# ── Config ─────────────────────────────────────────────────────────────────────
NVDA_TICKER     = 'NVDA'
MARKET_TICKERS  = ['SPY', 'QQQ', 'SOXX', '^VIX']
START           = '2015-01-01'
END             = '2024-12-31'

RETURN_TARGET   = 0.002   # 0.2% threshold for positive label
TRAIN_RATIO     = 0.80    # final holdout = last 20%
N_SPLITS        = 5       # walk-forward folds (TimeSeriesSplit)

# MLP architecture
HIDDEN_DIMS     = [128, 64, 32]
DROPOUT         = 0.30
LR              = 3e-4
WEIGHT_DECAY    = 1e-4
EPOCHS          = 80
BATCH_SIZE      = 64
PATIENCE        = 12      # early stopping

# Position sizing
PROB_THRESHOLD  = 0.52
KELLY_FRACTION  = 0.25    # fractional Kelly (conservative)
MAX_POSITION    = 1.0     # cap exposure at 100%
TRANSACTION_COST= 0.001   # 10 bps round-trip (realistic for liquid stock)

# ── Data ───────────────────────────────────────────────────────────────────────
def fetch_all(start: str, end: str) -> pd.DataFrame:
    """Download NVDA + market ETFs/VIX and align on common trading dates."""
    tickers = [NVDA_TICKER] + MARKET_TICKERS
    raw     = yf.download(tickers, start=start, end=end,
                          auto_adjust=True, progress=False)

    # Keep only adjusted close (and volume for NVDA)
    close = raw['Close'].copy()
    close.columns = [c.replace('^', '').lower() for c in close.columns]

    vol   = raw['Volume'][[NVDA_TICKER]].copy()
    vol.columns = ['volume']

    df = close.join(vol).dropna()
    return df

# ── RSI helper ─────────────────────────────────────────────────────────────────
def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(window).mean()
    loss  = (-delta.clip(upper=0)).rolling(window).mean()
    rs    = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

# ── Feature Engineering ────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)

    nvda = df['nvda']

    # ── NVDA price features (all shifted 1 to prevent lookahead) ──────────────
    log_ret              = np.log(nvda / nvda.shift(1))
    f['nvda_ret_1']      = log_ret.shift(1)
    f['nvda_ret_5']      = log_ret.rolling(5).mean().shift(1)
    f['nvda_ret_10']     = log_ret.rolling(10).mean().shift(1)
    f['nvda_ret_20']     = log_ret.rolling(20).mean().shift(1)
    f['nvda_vol_20']     = log_ret.rolling(20).std().shift(1)   # realised vol
    f['nvda_vol_5']      = log_ret.rolling(5).std().shift(1)
    f['nvda_rsi']        = rsi(nvda).shift(1)
    f['nvda_rsi_5']      = rsi(nvda, window=5).shift(1)

    ema20                = nvda.ewm(span=20, adjust=False).mean()
    ema50                = nvda.ewm(span=50, adjust=False).mean()
    ema200               = nvda.ewm(span=200, adjust=False).mean()
    f['nvda_ema20_ratio']= (nvda / ema20).shift(1)
    f['nvda_ema50_ratio']= (nvda / ema50).shift(1)
    f['nvda_ema_cross']  = (ema20 / ema50).shift(1)
    f['nvda_trend']      = (nvda / ema200).shift(1)  # long-term trend

    # Volume features
    f['nvda_vol_chg']    = np.log(df['volume'] / df['volume'].shift(1)).shift(1)
    f['nvda_vol_ratio']  = (df['volume'] / df['volume'].rolling(20).mean()).shift(1)

    # ── Market features ───────────────────────────────────────────────────────
    for ticker in ['spy', 'qqq', 'soxx']:
        ret              = np.log(df[ticker] / df[ticker].shift(1))
        f[f'{ticker}_ret1']  = ret.shift(1)
        f[f'{ticker}_ret5']  = ret.rolling(5).mean().shift(1)
        f[f'{ticker}_vol20'] = ret.rolling(20).std().shift(1)

        ema_t            = df[ticker].ewm(span=20, adjust=False).mean()
        f[f'{ticker}_ema']   = (df[ticker] / ema_t).shift(1)

    # VIX features
    f['vix_level']       = df['vix'].shift(1)
    f['vix_chg']         = (df['vix'] / df['vix'].shift(1) - 1).shift(1)
    f['vix_ma_ratio']    = (df['vix'] / df['vix'].rolling(20).mean()).shift(1)

    # ── Relative-strength features (NVDA vs market) ──────────────────────────
    f['nvda_vs_spy']     = f['nvda_ret_5'] - f['spy_ret5']
    f['nvda_vs_qqq']     = f['nvda_ret_5'] - f['qqq_ret5']
    f['nvda_vs_soxx']    = f['nvda_ret_5'] - f['soxx_ret5']

    # ── Target & forward return ───────────────────────────────────────────────
    f['next_ret']        = log_ret.shift(-1)
    f['target']          = (f['next_ret'] > RETURN_TARGET).astype(int)

    return f.dropna()

# ── MLP Model ─────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    """Feedforward neural network with batch normalisation and dropout."""

    def __init__(self, input_dim: int, hidden_dims: list, dropout: float = 0.3):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)   # logits

# ── Training helpers ───────────────────────────────────────────────────────────
def train_epoch(model: nn.Module, loader: DataLoader,
                optimiser: optim.Optimizer, criterion: nn.Module) -> float:
    model.train()
    total_loss = 0.0
    for X_b, y_b in loader:
        optimiser.zero_grad()
        loss = criterion(model(X_b), y_b)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        total_loss += loss.item() * len(X_b)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def eval_model(model: nn.Module, X: torch.Tensor) -> np.ndarray:
    model.eval()
    logits = model(X)
    return torch.sigmoid(logits).cpu().numpy()

def fit_mlp(X_tr: np.ndarray, y_tr: np.ndarray,
            X_va: np.ndarray, y_va: np.ndarray,
            input_dim: int) -> nn.Module:
    """Train MLP with early stopping on validation loss."""
    device = torch.device('cpu')

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    X_va_t = torch.tensor(X_va, dtype=torch.float32)
    y_va_t = torch.tensor(y_va, dtype=torch.float32)

    # Class balance weights
    pos_weight = torch.tensor([(y_tr == 0).sum() / (y_tr == 1).sum() + 1e-9],
                               dtype=torch.float32)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model      = MLP(input_dim, HIDDEN_DIMS, DROPOUT).to(device)
    optimiser  = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=EPOCHS)

    dataset    = TensorDataset(X_tr_t, y_tr_t)
    loader     = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    best_val, patience_cnt, best_state = np.inf, 0, None

    for epoch in range(EPOCHS):
        train_epoch(model, loader, optimiser, criterion)
        with torch.no_grad():
            val_loss = criterion(model(X_va_t), y_va_t).item()
        scheduler.step()

        if val_loss < best_val - 1e-5:
            best_val     = val_loss
            patience_cnt = 0
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

# ── Walk-Forward Training ──────────────────────────────────────────────────────
def walk_forward_predict(X: np.ndarray, y: np.ndarray,
                          feature_cols: list) -> np.ndarray:
    """
    TimeSeriesSplit: for each fold train on all data up to that point,
    predict the test fold. Returns out-of-sample probabilities for the
    full in-sample region (train_ratio of data).
    """
    tscv   = TimeSeriesSplit(n_splits=N_SPLITS)
    probs  = np.full(len(X), np.nan)
    n_feat = len(feature_cols)

    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X)):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_te, y_te = X[te_idx], y[te_idx]

        # Minimum training size guard
        if len(X_tr) < 150 or len(np.unique(y_tr)) < 2:
            continue

        # Scale on train, apply to test
        scaler     = StandardScaler()
        X_tr_sc    = scaler.fit_transform(X_tr)
        X_te_sc    = scaler.transform(X_te)

        # Use last 15% of training fold as internal validation for early stopping
        val_split  = int(len(X_tr_sc) * 0.85)
        X_va_sc    = X_tr_sc[val_split:]
        y_va       = y_tr[val_split:]
        X_tr_sc    = X_tr_sc[:val_split]
        y_tr_inner = y_tr[:val_split]

        model = fit_mlp(X_tr_sc, y_tr_inner, X_va_sc, y_va, n_feat)

        X_te_t        = torch.tensor(X_te_sc, dtype=torch.float32)
        probs[te_idx] = eval_model(model, X_te_t)

        print(f"  Fold {fold+1}/{N_SPLITS}: trained on {len(tr_idx)} | "
              f"predicted {len(te_idx)} samples")

    return probs

# ── Kelly Position Sizing ──────────────────────────────────────────────────────
def kelly_size(probs: np.ndarray,
               volatilities: np.ndarray,
               vix: np.ndarray) -> np.ndarray:
    """
    Fractional Kelly sizing:
      kelly = f * (p - q) / (expected_win / expected_loss)
    Simplified to: fraction * (prob - 0.5) / 0.5 * vol_scale * vix_scale
    Capped at MAX_POSITION.
    """
    # Signal above neutral (0.5)
    edge        = np.clip((probs - PROB_THRESHOLD) / (1.0 - PROB_THRESHOLD), 0, 1)

    # Scale down when realised vol is elevated
    vol_norm    = volatilities / (np.nanmedian(volatilities) + 1e-9)
    vol_scale   = np.clip(1.0 / vol_norm, 0.3, 1.5)

    # Scale down when VIX is elevated (fear gauge)
    vix_norm    = vix / (np.nanmedian(vix) + 1e-9)
    vix_scale   = np.clip(1.0 / vix_norm, 0.3, 1.5)

    raw         = KELLY_FRACTION * edge * vol_scale * vix_scale
    return np.clip(raw, 0, MAX_POSITION)

# ── Backtest with Transaction Costs ───────────────────────────────────────────
def backtest(feat: pd.DataFrame, positions: np.ndarray) -> pd.DataFrame:
    res = feat.copy()
    res['position'] = positions

    # Forward shift: decision today → execution tomorrow open (approximated)
    pos_shifted      = res['position'].shift(1).fillna(0)
    turnover         = pos_shifted.diff().abs().fillna(0)
    cost             = turnover * TRANSACTION_COST

    res['strat_ret'] = pos_shifted * res['next_ret'] - cost
    res['bh_ret']    = res['next_ret']
    res['strat_cum'] = (1 + res['strat_ret'].fillna(0)).cumprod()
    res['bh_cum']    = (1 + res['bh_ret'].fillna(0)).cumprod()
    res['turnover']  = turnover
    return res

# ── Metrics ────────────────────────────────────────────────────────────────────
def print_metrics(returns: pd.Series, label: str):
    ann_ret = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    sharpe  = ann_ret / (ann_vol + 1e-9)
    cum     = (1 + returns).cumprod()
    max_dd  = (cum / cum.cummax() - 1).min()
    calmar  = ann_ret / (abs(max_dd) + 1e-9)

    print(f"\n  [{label}]")
    print(f"    Annual Return    : {ann_ret:+.2%}")
    print(f"    Annual Volatility: {ann_vol:.2%}")
    print(f"    Sharpe Ratio     : {sharpe:.2f}")
    print(f"    Calmar Ratio     : {calmar:.2f}")
    print(f"    Max Drawdown     : {max_dd:.2%}")

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  Strategy V3 — MLP Neural Network + Walk-Forward Validation")
    print("=" * 65)

    # 1. Data & features
    print("\n  Downloading market data ...")
    df   = fetch_all(START, END)
    feat = engineer_features(df)

    FEATURE_COLS = [c for c in feat.columns if c not in ('next_ret', 'target')]

    X = feat[FEATURE_COLS].values
    y = feat['target'].values

    # 2. Hard holdout — never used during walk-forward training
    split    = int(len(X) * TRAIN_RATIO)
    X_is, X_oos   = X[:split], X[split:]
    y_is, y_oos   = y[:split], y[split:]
    feat_is       = feat.iloc[:split]
    feat_oos      = feat.iloc[split:]

    print(f"\n  In-sample : {feat.index[0].date()} → {feat.index[split-1].date()} ({split} days)")
    print(f"  OOS holdout: {feat.index[split].date()} → {feat.index[-1].date()} ({len(X_oos)} days)")
    print(f"  Features  : {len(FEATURE_COLS)}")

    # 3. Walk-forward training on in-sample data
    print(f"\n  Walk-forward training ({N_SPLITS} folds) ...")
    is_probs = walk_forward_predict(X_is, y_is, FEATURE_COLS)

    # 4. Final model on all in-sample data → predict OOS
    print("\n  Training final model on full in-sample data for OOS evaluation ...")
    scaler   = StandardScaler()
    X_is_sc  = scaler.fit_transform(X_is)
    X_oos_sc = scaler.transform(X_oos)

    val_split    = int(len(X_is_sc) * 0.90)
    final_model  = fit_mlp(X_is_sc[:val_split], y_is[:val_split],
                            X_is_sc[val_split:], y_is[val_split:],
                            len(FEATURE_COLS))

    X_oos_t   = torch.tensor(X_oos_sc, dtype=torch.float32)
    oos_probs = eval_model(final_model, X_oos_t)

    # 5. Position sizing
    vix_oos   = feat_oos['vix_level'].values
    vol_oos   = feat_oos['nvda_vol_20'].values
    positions = kelly_size(oos_probs, vol_oos, vix_oos)

    # 6. Backtest (with transaction costs)
    results = backtest(feat_oos, positions)

    print("\nPerformance — OUT-OF-SAMPLE holdout (after transaction costs):")
    print_metrics(results['strat_ret'].dropna(), 'Strategy V3 (MLP Neural Net)')
    print_metrics(results['bh_ret'].dropna(),    'Buy & Hold')

    avg_daily_turnover = results['turnover'].mean()
    print(f"\n  Avg daily turnover: {avg_daily_turnover:.2%}  "
          f"(implied ~{avg_daily_turnover * 252:.0f}% annual)")

    # 7. Plot
    fig, axes = plt.subplots(5, 1, figsize=(14, 18), sharex=False)
    fig.suptitle('Strategy V3 — MLP Neural Network (OOS Holdout)',
                 fontsize=14, fontweight='bold')

    # Cumulative returns
    ax = axes[0]
    results[['strat_cum', 'bh_cum']].plot(ax=ax, color=['steelblue', 'gray'])
    ax.set_title('Cumulative Returns (OOS Holdout)')
    ax.set_ylabel('Portfolio Value')
    ax.legend(['Strategy V3 (net of costs)', 'Buy & Hold'])
    ax.grid(alpha=0.3)

    # Drawdown
    ax = axes[1]
    dd = results['strat_cum'] / results['strat_cum'].cummax() - 1
    dd.plot(ax=ax, color='crimson')
    ax.fill_between(results.index, dd, 0, alpha=0.3, color='crimson')
    ax.set_title('Strategy Drawdown')
    ax.set_ylabel('Drawdown')
    ax.grid(alpha=0.3)

    # Predicted probability
    ax = axes[2]
    ax.plot(results.index, oos_probs, color='steelblue', linewidth=0.8, label='P(long)')
    ax.axhline(PROB_THRESHOLD, color='orange', linestyle='--',
               linewidth=1.5, label=f'Threshold ({PROB_THRESHOLD})')
    ax.fill_between(results.index, PROB_THRESHOLD, oos_probs,
                    where=oos_probs >= PROB_THRESHOLD,
                    alpha=0.3, color='green', label='Long zone')
    ax.set_title('MLP Predicted Probability')
    ax.set_ylabel('Probability')
    ax.legend()
    ax.grid(alpha=0.3)

    # Position sizes
    ax = axes[3]
    pd.Series(positions, index=results.index).plot(
        ax=ax, color='darkorchid', linewidth=0.8)
    ax.fill_between(results.index, 0, positions, alpha=0.25, color='darkorchid')
    ax.set_title('Kelly Position Size')
    ax.set_ylabel('Position (fraction of capital)')
    ax.set_ylim(0, MAX_POSITION + 0.05)
    ax.grid(alpha=0.3)

    # VIX overlay
    ax = axes[4]
    results['vix_level'].plot(ax=ax, color='firebrick', linewidth=0.9, label='VIX')
    ax2 = ax.twinx()
    results['nvda_vol_20'].multiply(np.sqrt(252)).plot(
        ax=ax2, color='navy', linewidth=0.9, linestyle='--', label='NVDA Realised Vol (ann.)')
    ax.set_title('Market Stress Indicators (VIX + NVDA Volatility)')
    ax.set_ylabel('VIX', color='firebrick')
    ax2.set_ylabel('Realised Vol (ann.)', color='navy')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = 'strategy_v3_results.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\n  Chart saved → {out}")
    plt.show()


if __name__ == '__main__':
    main()
