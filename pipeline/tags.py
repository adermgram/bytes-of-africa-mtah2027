"""Smart Tagging as a dead-ball prior.

A SET PIECES tag is a 35 s interval whose start sits inside the preceding stoppage: in the public
window each tag start falls in a true dead stretch, and across all 59 restarts mean player speed
bottoms out 0-5 s after the start. So a window just around the start is very likely dead ball.
"""

import numpy as np

from pipeline.data import N, load_tags


def dead_prior(pre_s=5.0, post_s=2.0, fps=25):
    """1.0 on frames within [start - pre_s, start + post_s] of any SET PIECES tag, else 0."""
    out = np.zeros(N)
    for x in load_tags():
        if x["code"] != "SET PIECES":
            continue
        a = max(0, x["start_frame"] - int(pre_s * fps))
        out[a:x["start_frame"] + int(post_s * fps) + 1] = 1.0
    return out
