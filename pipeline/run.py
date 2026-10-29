"""Regenerate the submission from the provided data:  python -m pipeline.run [--out PATH]

Deterministic: no randomness anywhere. Needs only numpy and pandas.
"""

import argparse
import os

import numpy as np

from pipeline import shape, shape2, tags
from pipeline.data import ROOT, load_atd
from pipeline.features import compute
from pipeline.hmm import predict
from pipeline.shapemodel import fit_logit

LABELS = np.array(["H", "A", "D", "X"])
PARAMS = dict(c_ha=20.0, c_d=25.0, w_shape=1.0, tag_boost=4.0, k_ball=0.3)


def build():
    players, _, team = load_atd()
    f = compute()
    f["tagd"] = tags.dead_prior()
    Xs = np.hstack([shape.compute(players, team)[0], shape2.features(players, team)])
    first_pass = predict(f, PARAMS)                       # ball-based labels, no shape
    z = fit_logit(Xs, first_pass, f["present"])           # player-only team classifier
    return predict(f, PARAMS, shape_logit=z)


def write(codes, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as h:
        h.write("frame,label\n")
        h.writelines("%d,%s\n" % (i, l) for i, l in enumerate(LABELS[codes]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "evaluation", "results", "submission.csv"))
    args = ap.parse_args()
    codes = build()
    write(codes, args.out)
    print("wrote", args.out)
