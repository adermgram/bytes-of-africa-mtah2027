"""Full-match recovery benchmark, with paired comparisons between configurations.

The public ground truth has 17 possessions, so it cannot separate two versions of the model. This
benchmark uses the whole match instead, with no private labels:

  1. Pseudo-truth = the ball-based labels (full pipeline, ball visible) on "ball-rich" stretches,
     where the ball is tracked almost continuously. Everywhere else is `X` (not scored).
  2. Hide the ball in random 8-40 s stretches inside those ball-rich regions.
  3. Re-predict with the ball hidden and score against the pseudo-truth.

It measures how well a configuration *recovers* what the ball would have said when the ball is gone,
which is the weak spot of the model (41 % of graded frames). It does not measure truth: the
pseudo-truth is itself only ~0.5 spellF1 against the real labels, so use it to rank changes to the
ball-less machinery, not to report a score.

The player-only team model is cross-fitted: the logit scored on a block is trained on the other
blocks, so a masked stretch never sees the labels it is asked to recover.

    python -m pipeline.bench              # compare the built-in configurations
"""

import numpy as np

from games import GAME, half_video_blocks
from pipeline import shape, shape2, tags
from pipeline.data import graded_mask, load_atd
from pipeline.features import compute, mask_ball
from pipeline.hmm import predict
from pipeline.run import PARAMS
from pipeline.score import pm
from pipeline.shapemodel import fit_logit

FPS = 25
N_FOLDS = 5


def ball_rich(present, graded, win=15 * FPS, frac=0.9):
    """Graded frames whose surrounding +-15 s have the ball on at least `frac` of frames."""
    k = 2 * win + 1
    share = np.convolve(present.astype(float), np.ones(k) / k, "same")
    return graded & (share >= frac)


def folds(graded, n_folds=N_FOLDS):
    """Contiguous time blocks of the graded frames (fold id per frame, -1 where ungraded)."""
    idx = np.flatnonzero(graded)
    fid = np.full(len(graded), -1, int)
    fid[idx] = np.minimum(np.arange(len(idx)) * n_folds // len(idx), n_folds - 1)
    return fid


def crossfit_logit(Xs, labels, present, fid):
    """Shape logit where each fold is scored by a model that never saw that fold's labels."""
    z = np.zeros(len(labels))
    for k in range(fid.max() + 1):
        held = fid == k
        z_k = fit_logit(Xs, np.where(held, 3, labels), present)   # 3 = X: excluded from training
        z[held] = z_k[held]
    return z


def make_masks(rich, seed, lo=8.0, hi=40.0, gap=(10.0, 25.0)):
    """Random 8-40 s hidden stretches, clipped to the ball-rich frames."""
    r = np.random.RandomState(seed)
    hide = np.zeros(len(rich), bool)
    i = int(r.uniform(*gap) * FPS)
    while i < len(rich):
        n = int(r.uniform(lo, hi) * FPS)
        hide[i:i + n] = True
        i += n + int(r.uniform(*gap) * FPS)
    return hide & rich


class Bench:
    def __init__(self, params=None, n_masks=12):
        players, _, team = load_atd()
        self.f = compute()
        self.f["tagd"] = tags.dead_prior()
        self.Xs, _ = shape.compute(players, team)                 # features of the original model
        self.Xs_ext = np.hstack([self.Xs, shape2.features(players, team)])
        self.graded = graded_mask()
        self.params = {**PARAMS, **(params or {})}
        self.truth_codes = predict(self.f, self.params)          # ball-based, no shape model
        z_full = fit_logit(self.Xs, self.truth_codes, self.f["present"])
        self.truth_codes = predict(self.f, self.params, shape_logit=z_full)
        self.rich = ball_rich(self.f["present"], self.graded)
        self.gt = np.where(self.rich, self.truth_codes, 3).astype(np.int8)
        self.fid = folds(self.graded)
        self.zs = {"base": crossfit_logit(self.Xs, self.truth_codes, self.f["present"], self.fid),
                   "ext": crossfit_logit(self.Xs_ext, self.truth_codes, self.f["present"], self.fid)}
        self.masks = [make_masks(self.rich, s) for s in range(n_masks)]

    def run(self, params=None, z="base", model=None):
        """Per-mask (spellF1, macroF1) for one configuration. `z` picks the shape-model features;
        `model(f, hide, z)` can replace predict."""
        P = {**self.params, **(params or {})}
        zz = self.zs[z]
        rows = []
        for hide in self.masks:
            ff = mask_ball(self.f, hide)
            pred = (model or (lambda f, h, z: predict(f, P, shape_logit=z)))(ff, hide, zz)
            r = pm.evaluate(self.gt, pred.astype(np.int8), bootstrap_n=0)
            rows.append((r["spell_f1_mean"], r["macro_f1"]))
        return np.array(rows)

    def n_spells(self):
        runs = pm.valid_runs(self.gt)
        return len(pm.possession_spells(self.gt, runs))


def paired(a, b, draws=2000, seed=0):
    """Mean difference b - a over masks with a bootstrap 95 % interval and the win rate."""
    d = b - a
    r = np.random.RandomState(seed)
    boots = np.array([d[r.randint(0, len(d), len(d))].mean() for _ in range(draws)])
    return d.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5), (d > 0).mean()


def report(results, ref):
    base = results[ref]
    print("%-28s %8s %8s   %8s  %-17s %s" % ("config", "spellF1", "macroF1", "d spell", "95% CI", "wins"))
    for name, rows in results.items():
        m = rows.mean(0)
        if name == ref:
            print("%-28s %8.4f %8.4f   %8s  %-17s" % (name, m[0], m[1], "(ref)", ""))
            continue
        d, lo, hi, w = paired(base[:, 0], rows[:, 0])
        flag = "*" if lo > 0 or hi < 0 else " "
        print("%-28s %8.4f %8.4f   %+8.4f  [%+.4f,%+.4f]%s %.0f%%" % (name, m[0], m[1], d, lo, hi, flag, 100 * w))


# name -> (feature set, parameter overrides)
CONFIGS = {
    "current": ("base", {}),
    "ext, shape weight 1.0": ("ext", {"w_shape": 1.0}),
    "ext, shape weight 1.25": ("ext", {"w_shape": 1.25}),
    "ext, shape weight 1.5": ("ext", {"w_shape": 1.5}),
    "ext, shape weight 2.0": ("ext", {"w_shape": 2.0}),
}


if __name__ == "__main__":
    b = Bench()
    print("ball-rich frames %d (%.0f%% of graded), pseudo-truth possessions %d, %d masks, hidden %.0f%%\n"
          % (b.rich.sum(), 100 * b.rich.sum() / b.graded.sum(), b.n_spells(), len(b.masks),
             100 * np.mean([m.sum() / b.rich.sum() for m in b.masks])))
    results = {name: b.run(p, z) for name, (z, p) in CONFIGS.items()}
    report(results, "current")
    print("\n* = 95 %% interval excludes zero (paired over %d masks)" % len(b.masks))
