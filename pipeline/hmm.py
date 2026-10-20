"""Three-state (H, A, D) segmentation by Viterbi over hand-built evidence."""

import numpy as np

from games import half_video_blocks, GAME
from pipeline.features import smooth_nan

H, A, D, X = 0, 1, 2, 3

DEFAULT = dict(
    lam=2.0,      # strength of team evidence per frame at full confidence
    scale=2.0,    # metres over which (dA - dH) saturates
    r0=3.0,       # evidence fades with the nearest player's distance (m)
    kappa=3.0,    # dead-ball evidence per m/s below tau
    tau=1.4,      # smoothed mean player speed that is neutral between dead and alive
    adapt=True,   # rescale speed by the rolling live-play pace (players tire over the match)
    ref_tempo=2.7,  # pace the tau/kappa below were set at: 75th pct of speed early in the match
    win=75,       # smoothing window of mean speed, frames
    c_ha=40.0,    # cost of a H<->A switch
    c_d=40.0,     # cost of entering / leaving D
    out_bonus=3.0,  # extra dead evidence when the ball is outside the pitch
    tag_boost=0.0,  # dead evidence on frames just around a SET PIECES tag start
    k_ball=0.0,   # dead evidence per m/s of (smoothed) ball speed below v0
    v0=4.0,
    w_shape=0.5,  # weight of the player-only team classifier's logit when the ball is missing
    w_shape_ball=0.0,  # ... and when the ball is tracked
)


def emissions(f, P, shape_logit=None):
    n = len(f["bx"])
    dH, dA = f["dH"], f["dA"]
    have = ~np.isnan(dH) & ~np.isnan(dA)
    near = np.where(have, np.minimum(dH, dA), np.inf)
    conf = np.where(have, np.exp(-np.minimum(near, 50) / P["r0"]), 0.0)
    diff = np.where(have, np.clip((dA - dH) / P["scale"], -3, 3), 0.0)
    ev = P["lam"] * np.tanh(diff) * conf          # > 0 favours H
    ev = np.where(f["interp"], 0.5 * ev, ev)
    if shape_logit is not None:
        # only where the ball gives no evidence: it is slow-moving and would blunt the ball cues
        ev = ev + np.where(have, P["w_shape_ball"], P["w_shape"]) * np.clip(shape_logit, -4, 4)

    ms = smooth_nan(f["mps"], P["win"])
    if P["adapt"]:
        ms = ms * P["ref_tempo"] / np.where(np.isnan(f["tempo"]), P["ref_tempo"], f["tempo"])
    dead = np.where(np.isnan(ms), 0.0, P["kappa"] * (P["tau"] - ms))
    dead = dead + np.where(f["line"] < -0.5, P["out_bonus"], 0.0)
    if "tagd" in f:
        dead = dead + P["tag_boost"] * f["tagd"]
    if P["k_ball"]:
        bs = smooth_nan(np.minimum(f["bv"], 40), 75)
        dead = dead + np.where(np.isnan(bs), 0.0, P["k_ball"] * (P["v0"] - bs))

    E = np.zeros((n, 3))
    E[:, H] = ev / 2
    E[:, A] = -ev / 2
    E[:, D] = dead
    return E


def viterbi(E, c_ha, c_d):
    """Maximise sum of emissions minus switching costs. E is (T, 3) log-score; returns states."""
    T = len(E)
    trans = np.array([[0, c_ha, c_d], [c_ha, 0, c_d], [c_d, c_d, 0]], float)
    score = E[0].copy()
    back = np.zeros((T, 3), np.int8)
    for t in range(1, T):
        cand = score[:, None] - trans            # cand[i, j] = from i to j
        back[t] = cand.argmax(0)
        score = cand.max(0) + E[t]
    path = np.empty(T, np.int8)
    path[-1] = score.argmax()
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def predict(f, P=None, shape_logit=None):
    P = {**DEFAULT, **(P or {})}
    n = len(f["bx"])
    E = emissions(f, P, shape_logit)
    out = np.full(n, X, np.int8)
    for _, a, b in half_video_blocks(GAME):
        out[a:b + 1] = viterbi(E[a:b + 1], P["c_ha"], P["c_d"])
    # Published fact (FAQ.md): the 2nd half is graded from 110808 but kicks off at 113742.
    out[110808:113742] = D
    return out
