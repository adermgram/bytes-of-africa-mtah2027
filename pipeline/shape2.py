"""Extra player-only features for the team model: movement, compactness, and a ball-area proxy.

Same frame as shape.py: metres, home attacks +x (the away team is mirrored by the caller's
orientation). Everything is a function of player positions alone, so it exists on frames with no
ball.
"""

import os

import numpy as np

from pipeline.data import CACHE, PITCH
from pipeline.features import smooth_nan

FPS = 25
RADIUS = 8.0     # metres: players this close to each other are "in the same contest"


def nan_mean(a, axis):
    with np.errstate(all="ignore"):
        cnt = np.sum(~np.isnan(a), axis)
        s = np.nansum(a, axis)
        return np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)


def nan_std(a, axis):
    m = nan_mean(a, axis)
    d = a - np.expand_dims(m, axis)
    return np.sqrt(nan_mean(d * d, axis))


def hotspot(x, y, home, chunk=8000, radius=RADIUS):
    """Per frame: the position with the most players within `radius` (a proxy for where the ball is).

    Returns hx, hy (hotspot position), n_home and n_away within `radius` of it, and the distance
    between the two closest opposing players around it.
    """
    n, s = x.shape
    out = np.full((n, 5), np.nan, np.float32)
    for a in range(0, n, chunk):
        xs, ys = x[a:a + chunk].astype(np.float32), y[a:a + chunk].astype(np.float32)
        dx = xs[:, :, None] - xs[:, None, :]
        dy = ys[:, :, None] - ys[:, None, :]
        d = np.sqrt(dx * dx + dy * dy)                       # (c, S, S), NaN where a slot is missing
        near = d < radius                                    # NaN compares False
        cnt = near.sum(2)                                    # players within radius of each player
        cnt = np.where(np.isnan(xs), -1, cnt)
        k = cnt.argmax(1)
        ar = np.arange(len(k))
        ok = cnt[ar, k] > 0
        mem = near[ar, k]                                    # (c, S) players around the hotspot player
        nh = (mem & home[None]).sum(1)
        na = (mem & ~home[None]).sum(1)
        dk = np.where(np.isnan(d[ar, k]), np.inf, d[ar, k])  # distance from hotspot player to each slot
        cl_h = np.where(home[None], dk, np.inf)
        cl_a = np.where(~home[None], dk, np.inf)
        cl_h[ar, k] = np.inf
        cl_a[ar, k] = np.inf
        gap = np.minimum(cl_h.min(1), cl_a.min(1))
        res = np.stack([xs[ar, k], ys[ar, k], nh, na, gap], 1)
        res[~ok] = np.nan
        res[~np.isfinite(res)] = np.nan
        out[a:a + chunk] = res
    return out


def compute(players, team, dirH):
    """(N, F) matrix and names. `dirH` is the per-frame attack direction of the home team (+-1)."""
    L, W = PITCH
    n = len(players)
    x = players[:, :, 0] * L
    y = players[:, :, 1] * W
    spd = players[:, :, 2]
    xo = x * dirH[:, None] + (dirH[:, None] < 0) * L
    home = team == 0
    cols = {}

    for tag, m in (("h", home), ("a", ~home)):
        xs, ys, sp = xo[:, m], y[:, m], spd[:, m]
        cols[tag + "_spd"] = nan_mean(sp, 1)
        cols[tag + "_wid"] = nan_std(ys, 1)               # width of the block
        cols[tag + "_dep"] = nan_std(xs, 1)               # depth of the block
        # mean forward velocity over 1 s: attackers push forward, defenders drop
        vx = np.full_like(xs, np.nan)
        vx[FPS:] = np.clip(xs[FPS:] - xs[:-FPS], -12.0, 12.0)   # tracking jumps are not running
        # away attacks toward -x in this frame, so flip its sign: positive = toward own attacking goal
        cols[tag + "_adv"] = nan_mean(vx, 1) * (1.0 if tag == "h" else -1.0)

    hs = hotspot(xo, y, home)
    hx, hy, nh, na, gap = (hs[:, i] for i in range(5))
    cols["hs_x"], cols["hs_y"] = hx, np.abs(hy - W / 2)
    cols["hs_nh_minus_na"] = nh - na
    cols["hs_n"] = nh + na
    cols["hs_gap"] = np.minimum(gap, 20.0)

    out = {k: smooth_nan(np.asarray(v, float), 25) for k, v in cols.items()}
    names = list(out)
    X = np.stack([out[k] for k in names], 1)
    ctx = [np.stack([smooth_nan(X[:, i], w) for i in range(X.shape[1])], 1) for w in (125, 250)]
    X = np.hstack([X] + ctx)
    names = names + [k + "_5s" for k in names[:len(names)]] + [k + "_10s" for k in names[:len(names)]]
    return X, names


def features(players, team):
    """Cached extended feature matrix, aligned with shape.compute's rows."""
    from pipeline import shape
    path = os.path.join(CACHE, "shape2.npy")
    if os.path.exists(path):
        return np.load(path)
    X, _ = compute(players, team, shape.home_dir(players, team))
    os.makedirs(CACHE, exist_ok=True)
    np.save(path, X.astype(np.float32))
    return X
