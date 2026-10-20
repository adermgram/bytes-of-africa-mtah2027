"""Align the broadcast ATD to the tactical timeline and use its ball to fill tactical gaps.

The two recordings are the same match on different clocks. Within each half the broadcast is a
constant integer shift of the tactical timeline (replays and cuts are absent from the ATD's ball
track over the live play), so the lag is found by masked cross-correlation of the two ball tracks:
the best lag has a mean squared error of ~0.001 (normalised pitch units) against ~0.09 for any
other lag, over ~37k frames where both feeds see the ball. Estimated at run time, not hard-coded.
"""

import numpy as np

from games import half_video_blocks, GAME
from pipeline.data import load_atd

MIN_OVERLAP = 2000


def _lag(a, b, max_lag):
    """Lag k minimising mean squared difference of a[i + k] and b[i] over frames both observe."""
    ma, mb = ~np.isnan(a), ~np.isnan(b)
    A, B = np.where(ma, a, 0.0), np.where(mb, b, 0.0)
    nf = 1 << int(np.ceil(np.log2(len(a) + len(b))))

    def cc(x, y):
        return np.fft.irfft(np.fft.rfft(x, nf) * np.conj(np.fft.rfft(y, nf)), nf)

    ks = np.arange(-max_lag, max_lag + 1)
    idx = ks % nf
    count = cc(ma.astype(float), mb.astype(float))[idx]
    sse = cc(A ** 2, mb.astype(float))[idx] + cc(ma.astype(float), B ** 2)[idx] - 2 * cc(A, B)[idx]
    mse = np.where(count > MIN_OVERLAP, sse / np.maximum(count, 1), np.inf)
    best = int(np.argmin(mse))
    return int(ks[best]), float(mse[best])


def broadcast_ball():
    """(N, 3) broadcast ball (x, y, speed) re-indexed to tactical video frames; NaN where unseen."""
    _, tball, _ = load_atd("tactical")
    _, bball, _ = load_atd("broadcast")
    out = np.full(tball.shape, np.nan, dtype=np.float32)
    for _, a, b in half_video_blocks(GAME):
        lag, mse = _lag(tball[a:b + 1, 0].astype(float), bball[:, 0].astype(float), 160000)
        if not mse < 0.01:       # no convincing alignment: leave this half untouched
            continue
        i = np.arange(a, b + 1)
        j = i - a - lag           # tactical slice index = broadcast index + lag
        ok = (j >= 0) & (j < len(bball))
        out[i[ok]] = bball[j[ok]]
    return out
