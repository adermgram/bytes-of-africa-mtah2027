"""Per-frame features from the tactical ATD. Everything is a pure function of the tracking."""

import numpy as np

from pipeline.data import PITCH, load_atd

FPS = 25


def _interp_gaps(x, max_gap):
    """Linear interpolation across NaN gaps of at most max_gap frames. Returns (filled, was_interp)."""
    n = len(x)
    idx = np.arange(n)
    ok = ~np.isnan(x)
    out = x.copy()
    interp = np.zeros(n, bool)
    if ok.sum() < 2:
        return out, interp
    filled = np.interp(idx, idx[ok], x[ok])
    # gap length for every NaN frame = distance between neighbouring observed frames
    last = np.where(ok, idx, -1)
    last = np.maximum.accumulate(last)
    nxt = np.where(ok, idx, n + 10**9)
    nxt = np.minimum.accumulate(nxt[::-1])[::-1]
    gap = nxt - last
    use = (~ok) & (last >= 0) & (nxt < n + 10**9) & (gap <= max_gap + 1)
    out[use] = filled[use]
    interp[use] = True
    return out, interp


def smooth_nan(x, w):
    """Centred moving average that ignores NaN."""
    k = np.ones(w)
    m = ~np.isnan(x)
    s = np.convolve(np.where(m, x, 0.0), k, "same")
    c = np.convolve(m.astype(float), k, "same")
    return np.where(c > 0, s / np.maximum(c, 1), np.nan)


def compute(max_gap=50, use_broadcast=True):
    players, ball, team = load_atd()
    if use_broadcast:
        from pipeline.align import broadcast_ball
        fill = np.isnan(ball[:, 0]) & ~np.isnan(broadcast_ball()[:, 0])
        ball = ball.copy()
        ball[fill] = broadcast_ball()[fill]
    L, W = PITCH
    n = len(ball)
    bx, ix = _interp_gaps(ball[:, 0] * L, max_gap)
    by, _ = _interp_gaps(ball[:, 1] * W, max_gap)
    px = players[:, :, 0] * L
    py = players[:, :, 1] * W

    d = np.sqrt((px - bx[:, None]) ** 2 + (py - by[:, None]) ** 2)
    d = np.where(np.isnan(d), np.inf, d)
    home = team == 0
    dH = d[:, home].min(1)
    dA = d[:, ~home].min(1)
    dH[np.isnan(bx)] = np.nan
    dA[np.isnan(bx)] = np.nan
    dH[np.isinf(dH)] = np.nan
    dA[np.isinf(dA)] = np.nan

    mps = np.nanmean(players[:, :, 2], axis=1)   # mean speed of tracked players, m/s
    bv = np.full(n, np.nan)
    bv[1:] = np.sqrt(np.diff(bx) ** 2 + np.diff(by) ** 2) * FPS
    line = np.minimum.reduce([bx, L - bx, by, W - by])   # < 0 means outside the pitch
    from games import half_video_blocks, GAME
    tp = tempo(mps, half_video_blocks(GAME))
    return {"tempo": tp, "bx": bx, "by": by, "interp": ix, "dH": dH, "dA": dA, "mps": mps, "bv": bv,
            "line": line, "present": ~np.isnan(ball[:, 0])}


def mask_ball(f, hide):
    """Copy of the features with the ball removed on the frames where `hide` is True."""
    g = {k: v.copy() for k, v in f.items()}
    for k in ("bx", "by", "dH", "dA", "bv", "line"):
        g[k] = g[k].astype(float)
        g[k][hide] = np.nan
    g["interp"][hide] = False
    g["present"][hide] = False
    return g


def tempo(mps, blocks, win=75, span=6000, step=250, q=75):
    """Rolling live-play pace: the q-th percentile of smoothed mean player speed in +-span frames.

    Players tire and play slows, so a fixed 'slow means dead' threshold over-calls dead ball late in
    the match. The percentile tracks the live level (stoppages are a minority of any 8-minute
    stretch), computed per half so that halftime does not leak in.
    """
    ms = smooth_nan(mps, win)
    out = np.full(len(mps), np.nan)
    for _, a, b in blocks:
        centres = np.arange(a, b + 1, step)
        vals = []
        for c in centres:
            x = ms[max(a, c - span):min(b + 1, c + span)]
            x = x[~np.isnan(x)]
            vals.append(np.percentile(x, q) if len(x) else np.nan)
        out[a:b + 1] = np.interp(np.arange(a, b + 1), centres, vals)
    return out
