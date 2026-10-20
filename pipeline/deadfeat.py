"""Dead-ball feature matrix (frame x features) and a tiny numpy logistic regression."""

import numpy as np

from pipeline.features import smooth_nan

NAMES = ["mps25", "mps75", "mps250", "fastfrac75", "ball_present75", "bspeed75", "bspeed250",
         "line", "outside", "slowplayers75"]


def build(f, players):
    sp = players[:, :, 2]
    mps = f["mps"]
    fast = np.nanmean(np.where(np.isnan(sp), np.nan, (sp > 2.5).astype(float)), axis=1)
    slow = np.nanmean(np.where(np.isnan(sp), np.nan, (sp < 1.0).astype(float)), axis=1)
    pres = f["present"].astype(float)
    bv = np.where(np.isnan(f["bv"]), np.nan, np.minimum(f["bv"], 40))
    cols = [
        smooth_nan(mps, 25), smooth_nan(mps, 75), smooth_nan(mps, 250),
        smooth_nan(fast, 75), smooth_nan(pres, 75),
        smooth_nan(bv, 75), smooth_nan(bv, 250),
        np.clip(f["line"], -5, 15), (f["line"] < -0.3).astype(float),
        smooth_nan(slow, 75),
    ]
    X = np.stack(cols, 1)
    return X


def prep(X, mu=None, sd=None):
    X = X.copy()
    med = np.nanmedian(X, 0) if mu is None else mu
    nanm = np.isnan(X)
    X[nanm] = np.take(med, np.where(nanm)[1])
    if mu is None:
        mu = X.mean(0); sd = X.std(0) + 1e-9
    return (X - mu) / sd, mu, sd


def fit_logreg(X, y, l2=1.0, iters=300):
    Xb = np.hstack([X, np.ones((len(X), 1))])
    w = np.zeros(Xb.shape[1])
    reg = np.eye(len(w)) * l2; reg[-1, -1] = 0
    for _ in range(iters):
        z = Xb @ w
        p = 1 / (1 + np.exp(-z))
        gr = Xb.T @ (p - y) + reg @ w
        Hs = (Xb * (p * (1 - p))[:, None]).T @ Xb + reg + 1e-6 * np.eye(len(w))
        step = np.linalg.solve(Hs, gr)
        w -= step
        if np.abs(step).max() < 1e-6:
            break
    return w


def predict_logit(X, w):
    return np.hstack([X, np.ones((len(X), 1))]) @ w
