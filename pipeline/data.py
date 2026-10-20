"""Load the tactical ATD, Smart Tagging and public ground truth as numpy arrays, cached on disk."""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAMES, GAME, half_video_blocks, require  # noqa: E402

CACHE = os.path.join(ROOT, "cache")
N = GAMES[GAME]["n_video_frames"]
PITCH = GAMES[GAME]["pitch"]
GT_PUBLIC = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")


def team_of_slot(meta_path):
    """0 = home, 1 = away for each track slot (same rule as starter/baseline.py)."""
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    return np.array([0 if p.get("teamId") == team_ids[0] else 1 for p in root.iter("Player")],
                    dtype=np.int8)


def load_atd(source="tactical"):
    """players (N, S, 3) = x, y, speed; ball (N, 3); team (S,). NaN where unobserved.

    Row i is video frame i. Pitch coordinates are normalised 0-1.
    """
    path = os.path.join(CACHE, "atd_%s.npz" % source)
    if os.path.exists(path):
        z = np.load(path)
        return z["players"], z["ball"], z["team"]

    meta, raw = require(["atd_meta", "atd_raw"], source=source)
    team = team_of_slot(meta)
    s = len(team)
    n = 0
    with open(raw) as h:
        for _ in h:
            n += 1
    players = np.full((n, s, 3), np.nan, dtype=np.float32)
    ball = np.full((n, 3), np.nan, dtype=np.float32)
    with open(raw) as h:
        for line in h:
            fid, pl, bl = line.rstrip("\n").split(":")
            i = int(fid) - 1
            players[i] = np.array(pl.replace(";", ",").split(","), dtype=np.float32).reshape(s, 3)
            ball[i] = np.array(bl.split(","), dtype=np.float32)
    os.makedirs(CACHE, exist_ok=True)
    np.savez(path, players=players, ball=ball, team=team)
    return players, ball, team


def load_tags():
    """Smart Tagging as a list of dicts with start/end video frames."""
    from starter.load_smart_tagging import load
    return load()


def load_gt():
    """Public GT as int codes over all N frames, in the scorer's code space: 0=H 1=A 2=D 3=X."""
    import pandas as pd
    lab = pd.read_csv(GT_PUBLIC)["label"].str.strip().str.upper().values
    return np.select([lab == "H", lab == "A", lab == "D"], [0, 1, 2], default=3).astype(np.int8)


def graded_mask():
    m = np.zeros(N, bool)
    for _, a, b in half_video_blocks(GAME):
        m[a:b + 1] = True
    return m
