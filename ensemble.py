"""
Strategy V2: Bootstrapped Ensemble with Uncertainty-Aware Position Sizing
=========================================================================
Extends V1 with:
  - Ensemble of bootstrapped, probability-calibrated logistic regression models
  - Prediction uncertainty from model disagreement (std of predicted probs)
  - Confidence-scaled position sizes (continuous 0–1)
  - Volatility regime filter (reduce exposure in high-vol environments)
  - TimeSeriesSplit for calibration (fixes random-CV data leakage from v1 description)

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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit

# ── Config ─────────────────────────────────────────────────────────────────────
TICKER          = 'NVDA'
START           = '2015-01-01'
END             = '2024-12-31'
RETURN_TARGET   = 0.002   # 0.2% next-day return threshold for positive label
PROB_THRESHOLD  = 0.52    # signal only when mean prob exceeds this
N_ESTIMATORS    = 60      # number of bootstrap models
UNCERTAINTY_CAP = 0.12    # std dev above which position is zeroed
VOL_WINDOW      = 20
VOL_REGIME_PCT  = 75      # percentile above which it's a "high vol" regime
VOL_REGIME_SCALE= 0.40    # scale positions down to this in high-vol
TRAIN_RATIO     = 0.70

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

    f['log_ret']        = np.log(df['close'] / df['close'].shift(1))
    f['ret_lag1']       = f['log_ret'].shift(1)
    f['ret_5d']         = f['log_ret'].rolling(5).mean().shift(1)
    f['ret_10d']        = f['log_ret'].rolling(10).mean().shift(1)
    f['ret_20d']        = f['log_ret'].rolling(20).mean().shift(1)

    f['vol_chg']        = np.log(df['volume'] / df['volume'].shift(1)).shift(1)
    f['vol_ma_ratio']   = (df['volume'] / df['volume'].rolling(20).mean()).shift(1)

    f['rsi']            = compute_rsi(df['close']).shift(1)

    ema20               = df['close'].ewm(span=20, adjust=False).mean()
    ema50               = df['close'].ewm(span=50, adjust=False).mean()
    f['ema_ratio_20']   = (df['close'] / ema20).shift(1)
    f['ema_ratio_50']   = (df['close'] / ema50).shift(1)
    f['ema_cross']      = (ema20 / ema50).shift(1)          # momentum of EMAs

    f['volatility']     = f['log_ret'].rolling(VOL_WINDOW).std().shift(1)
    f['hl_range']       = ((df['high'] - df['low']) / df['close']).shift(1)

    f['next_ret']       = f['log_ret'].shift(-1)
    f['target']         = (f['next_ret'] > RETURN_TARGET).astype(int)

    return f.dropna()

# ── Bootstrap Ensemble ─────────────────────────────────────────────────────────
class BootstrapEnsemble:
    """
    Ensemble of logistic regression models, each trained on a bootstrap resample
    and probability-calibrated with TimeSeriesSplit isotonic regression.
    """

    def __init__(self, n_estimators: int = 60, random_state: int = 42):
        self.n_estimators  = n_estimators
        self.random_state  = random_state
        self.models_: list = []
        self.scaler_       = StandardScaler()

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BootstrapEnsemble':
        X_sc = self.scaler_.fit_transform(X)
        rng  = np.random.RandomState(self.random_state)
        n    = len(X_sc)

        for i in range(self.n_estimators):
            idx      = rng.choice(n, size=n, replace=True)
            X_b, y_b = X_sc[idx], y[idx]

            # Skip degenerate bootstrap samples
            if len(np.unique(y_b)) < 2:
                continue

            base = LogisticRegression(C=0.1, max_iter=1000, random_state=i,
                                      solver='lbfgs')
            # TimeSeriesSplit for calibration avoids future-data leakage
            tscv = TimeSeriesSplit(n_splits=3)
            cal  = CalibratedClassifierCV(base, cv=tscv, method='isotonic')
            try:
                cal.fit(X_b, y_b)
                self.models_.append(cal)
            except Exception:
                continue

        print(f"  Trained {len(self.models_)} / {self.n_estimators} models.")
        return self

    def predict_with_uncertainty(self, X: np.ndarray):
        """Returns (mean_prob, std_prob) across ensemble members."""
        X_sc     = self.scaler_.transform(X)
        all_preds = np.array([m.predict_proba(X_sc)[:, 1] for m in self.models_])
        return all_preds.mean(axis=0), all_preds.std(axis=0)

# ── Position Sizing ────────────────────────────────────────────────────────────
def size_positions(probs: np.ndarray,
                   uncertainties: np.ndarray,
                   volatilities: np.ndarray) -> np.ndarray:
    """
    Continuous position size in [0, 1]:
      - Zero if mean prob < threshold
      - Scales with signal strength above threshold
      - Penalized by prediction uncertainty
      - Halved in high-volatility regimes
    """
    # 1. Signal strength: how far above threshold (normalised to [0,1])
    signal = np.clip((probs - PROB_THRESHOLD) / (1.0 - PROB_THRESHOLD), 0, 1)

    # 2. Uncertainty penalty: linear decay to 0 at UNCERTAINTY_CAP
    unc_factor = np.clip(1.0 - uncertainties / UNCERTAINTY_CAP, 0, 1)

    # 3. Volatility regime: compute rolling vol threshold on expanding history
    vol_thresh = np.percentile(volatilities, VOL_REGIME_PCT)
    vol_factor = np.where(volatilities > vol_thresh, VOL_REGIME_SCALE, 1.0)

    return signal * unc_factor * vol_factor

# ── Metrics ────────────────────────────────────────────────────────────────────
def print_metrics(returns: pd.Series, label: str):
    ann_ret = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    sharpe  = ann_ret / (ann_vol + 1e-9)
    cum     = (1 + returns).cumprod()
    max_dd  = (cum / cum.cummax() - 1).min()

    print(f"\n  [{label}]")
    print(f"    Annual Return    : {ann_ret:+.2%}")
    print(f"    Annual Volatility: {ann_vol:.2%}")
    print(f"    Sharpe Ratio     : {sharpe:.2f}")
    print(f"    Max Drawdown     : {max_dd:.2%}")

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Strategy V2 — Bootstrap Ensemble + Uncertainty Sizing")
    print("=" * 60)

    raw  = fetch_data(TICKER, START, END)
    feat = engineer_features(raw)

    FEATURE_COLS = ['ret_lag1', 'ret_5d', 'ret_10d', 'ret_20d',
                    'vol_chg', 'vol_ma_ratio', 'rsi',
                    'ema_ratio_20', 'ema_ratio_50', 'ema_cross',
                    'volatility', 'hl_range']

    X = feat[FEATURE_COLS].values
    y = feat['target'].values

    split     = int(len(X) * TRAIN_RATIO)
    X_tr, X_te = X[:split], X[split:]
    y_tr      = y[:split]

    print(f"\n  Train: {feat.index[0].date()} → {feat.index[split-1].date()} ({split} days)")
    print(f"  Test : {feat.index[split].date()} → {feat.index[-1].date()} ({len(X_te)} days)")
    print(f"\n  Fitting {N_ESTIMATORS} bootstrap models ...")

    ensemble = BootstrapEnsemble(n_estimators=N_ESTIMATORS, random_state=42)
    ensemble.fit(X_tr, y_tr)

    probs, uncertainties = ensemble.predict_with_uncertainty(X_te)

    test               = feat.iloc[split:].copy()
    test['prob']       = probs
    test['uncertainty']= uncertainties
    test['position']   = size_positions(probs, uncertainties, test['volatility'].values)

    test['strat_ret']  = test['position'].shift(1) * test['next_ret']
    test['strat_cum']  = (1 + test['strat_ret'].fillna(0)).cumprod()
    test['bh_cum']     = (1 + test['next_ret'].fillna(0)).cumprod()

    print("\nPerformance (out-of-sample):")
    print_metrics(test['strat_ret'].dropna(), 'Strategy V2')
    print_metrics(test['next_ret'].dropna(),  'Buy & Hold')

    # ── Plot ───────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(4, 1, figsize=(13, 14), sharex=False)
    fig.suptitle('Strategy V2 — Bootstrap Ensemble + Uncertainty Sizing',
                 fontsize=14, fontweight='bold')

    # 1. Cumulative returns
    ax = axes[0]
    test[['strat_cum', 'bh_cum']].plot(ax=ax, color=['steelblue', 'gray'])
    ax.set_title('Cumulative Returns (Out-of-Sample)')
    ax.set_ylabel('Portfolio Value')
    ax.legend(['Strategy V2', 'Buy & Hold'])
    ax.grid(alpha=0.3)

    # 2. Drawdown
    ax = axes[1]
    dd = test['strat_cum'] / test['strat_cum'].cummax() - 1
    dd.plot(ax=ax, color='crimson')
    ax.fill_between(test.index, dd, 0, alpha=0.3, color='crimson')
    ax.set_title('Drawdown')
    ax.set_ylabel('Drawdown')
    ax.grid(alpha=0.3)

    # 3. Probability with uncertainty band
    ax = axes[2]
    ax.plot(test.index, probs, color='steelblue', linewidth=0.9, label='Mean P(long)')
    ax.fill_between(test.index, probs - uncertainties, probs + uncertainties,
                    alpha=0.3, color='steelblue', label='±1 std (uncertainty)')
    ax.axhline(PROB_THRESHOLD, color='orange', linestyle='--',
               linewidth=1.5, label=f'Threshold ({PROB_THRESHOLD})')
    ax.set_title('Ensemble Predicted Probability + Uncertainty')
    ax.set_ylabel('Probability')
    ax.legend()
    ax.grid(alpha=0.3)

    # 4. Position sizes
    ax = axes[3]
    test['position'].plot(ax=ax, color='darkorchid', linewidth=0.8)
    ax.fill_between(test.index, 0, test['position'], alpha=0.25, color='darkorchid')
    ax.set_title('Continuous Position Size (0=cash, 1=full long)')
    ax.set_ylabel('Position')
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = 'strategy_v2_results.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\n  Chart saved → {out}")
    plt.show()


if __name__ == '__main__':
    main()
