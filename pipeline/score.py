"""Score a prediction (int codes 0=H 1=A 2=D 3=X over all frames) on the public window."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

from pipeline.data import load_gt  # noqa: E402

_GT = load_gt()


def score(pred, bootstrap_n=0):
    r = pm.evaluate(_GT, pred.astype("int8"), bootstrap_n=bootstrap_n)
    return r


def brief(pred):
    r = score(pred)
    return "spellF1 %.4f  macroF1 %.4f  spells %d/%d  meanDur %.1fs/%.1fs" % (
        r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"],
        r["mean_spell_duration_s"], r["gt_mean_spell_duration_s"])
