"""Ball-masking stress test: hide the ball in long stretches of the public window and score."""

import numpy as np

from pipeline.features import mask_ball
from pipeline.hmm import predict
from pipeline.score import score

A, E = 11185, 18685


def make_mask(n, seed):
    """Hide the ball in 8-40 s stretches, matching the long gaps seen across the real match."""
    r = np.random.RandomState(seed)
    hide = np.zeros(n, bool)
    i = A
    while i < E:
        i += int(r.uniform(10, 25) * 25)
        length = int(r.uniform(8, 40) * 25)
        hide[i:i + length] = True
        i += length
    hide[E:] = False
    return hide


def evaluate(f, params, z, n_masks=16):
    """Mean (spellF1, macroF1, n_spells) over the unmasked window and `n_masks` random masks."""
    rows = []
    for m in [None] + [make_mask(len(f["bx"]), s) for s in range(n_masks)]:
        ff = f if m is None else mask_ball(f, m)
        r = score(predict(ff, params, shape_logit=z))
        rows.append((r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"]))
    rows = np.array(rows)
    return rows[0], rows[1:].mean(0), rows[1:, 0].std()
