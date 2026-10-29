"""Player-only features: team shape and movement, oriented so that home always attacks +x."""

import numpy as np

from pipeline.data import PITCH, load_atd
from pipeline.features import smooth_nan

FPS = 25


def attack_dir(players, team, kickoff_frame, which):
    """+1 if `which` team (0=home) attacks toward +x, judged from positions at a kickoff."""
    sl = slice(kickoff_frame, kickoff_frame + 5)
    x = np.nanmean(players[sl, :, 0], axis=0)
    own = np.nanmean(x[team == which])
    return 1.0 if own < 0.5 else -1.0


def home_dir(players, team):
    """Per-frame attack direction of the home team (+-1), from the two kickoff positions."""
    dirH = np.zeros(len(players))
    dirH[:110000] = attack_dir(players, team, 11185, 0)
    dirH[110000:] = attack_dir(players, team, 113742, 0)
    return dirH


def team_stats(x, y, mask):
    """x, y (N, S) in metres for the slots in `mask`; per-frame summary of that team."""
    xs, ys = x[:, mask], y[:, mask]
    cnt = np.sum(~np.isnan(xs), 1)
    cx = np.nanmean(xs, 1)
    cy = np.nanmean(ys, 1)
    sx = np.sort(np.where(np.isnan(xs), np.inf, xs), 1)   # ascending, NaN last
    sx = np.where(np.isinf(sx), np.nan, sx)
    k = np.maximum(cnt, 1)
    return cx, cy, sx, cnt


def compute(players=None, team=None):
    if players is None:
        players, _, team = load_atd()
    L, W = PITCH
    n = len(players)
    x = players[:, :, 0] * L
    y = players[:, :, 1] * W
    homemask = team == 0
    feats = {}
    dirH = np.zeros(n)
    # attack direction per half from kickoff positions (1st: 11185, 2nd: 113742)
    d1 = attack_dir(players, team, 11185, 0)
    d2 = attack_dir(players, team, 113742, 0)
    dirH[:110000] = d1
    dirH[110000:] = d2
    # orient: home attacks +x. For away use the mirrored frame.
    xo = x * dirH[:, None] + (dirH[:, None] < 0) * L

    hx, hy, hs, hc = team_stats(xo, y, homemask)
    ax_, ay, as_, ac = team_stats(xo, y, ~homemask)

    # advanced line = 3rd most advanced outfielder, defensive line = 3rd deepest
    # vectorised: take from the sorted rows using each row's own count
    def rank_from_top(s, c, r):
        idx = np.clip(c - 1 - r, 0, s.shape[1] - 1)
        return s[np.arange(len(s)), idx]

    cols = {
        "hx": hx, "ax": ax_, "hy": hy, "ay": ay,
        "h_front": rank_from_top(hs, hc, 2), "a_front": rank_from_top(as_, ac, 2),
        "h_back": hs[:, 2], "a_back": as_[:, 2],
        "h_len": rank_from_top(hs, hc, 0) - hs[:, 1], "a_len": rank_from_top(as_, ac, 0) - as_[:, 1],
    }
    cx_all = np.nanmean(xo, 1)
    cy_all = np.nanmean(y, 1)
    cols["cx"], cols["cy"] = cx_all, cy_all
    # for the away team, "attacking" means decreasing x in the home frame
    out = {k: smooth_nan(v, 25) for k, v in cols.items()}
    # velocities of the centroids over 2 s
    for name in ("hx", "ax"):
        s = out[name]
        v = np.full(n, np.nan)
        v[50:] = s[50:] - s[:-50]
        out["v_" + name] = v
    v = np.full(n, np.nan)
    v[50:] = out["cx"][50:] - out["cx"][:-50]
    out["v_cx"] = v
    names = list(out)
    X = np.stack([out[k] for k in names], 1)
    # longer temporal context: possession is a slow property of how the teams are set up
    ctx = [np.stack([smooth_nan(X[:, i], w) for i in range(X.shape[1])], 1) for w in (125, 250)]
    X = np.hstack([X] + ctx)
    names = names + [k + "_5s" for k in names[:len(names)]] + [k + "_10s" for k in names[:len(names)]]
    return X, names
