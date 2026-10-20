"""Player-only team classifier, trained on pseudo-labels from the ball-based model."""

import numpy as np

from pipeline import deadfeat as df
from pipeline.data import graded_mask


def fit_logit(Xs, labels, present, l2=100.0):
    """Logit that home has the ball, from team-shape features alone.

    Trains on frames that are well inside a ball-tracked run (>= 2 s from any gap) and that the
    ball-based model labelled H or A. No ground truth is used.
    """
    core = np.convolve((~present).astype(float), np.ones(101), "same") == 0
    train = core & graded_mask() & ((labels == 0) | (labels == 1))
    Xp, mu, sd = df.prep(Xs[train])
    w = df.fit_logreg(Xp, (labels[train] == 0).astype(float), l2=l2)
    Xall, _, _ = df.prep(Xs, mu, sd)
    return df.predict_logit(Xall, w)
