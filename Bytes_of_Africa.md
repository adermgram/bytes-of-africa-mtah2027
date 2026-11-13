# Bytes of Africa — technical document

More than a Hack 2027 · frame-level ball possession, FC Barcelona vs Athletic Club

## 1. Summary

Every graded frame is labelled `H`, `A` or `D` by one deterministic pipeline that uses only the
provided tracking data (tactical and broadcast ATD) and the Smart Tagging file. It needs numpy and
pandas, no video and no GPU. It runs in a few minutes from a cold cache (mostly the player-density
features) and in seconds once the parsed tracking and features are cached.

The core is a three-state Viterbi segmentation over hand-built evidence. Its main design decision is
driven by one fact about the data: **the ball is missing on 41 % of graded frames, and 75 % of those
frames sit in gaps longer than 15 s.** Over those stretches the ball cannot say who has possession,
so a second, player-only model carries the team decision there.

## 2. Approach

**State model.** States `H`, `A`, `D`, decoded per half with Viterbi. Each state gets a per-frame
log-score (evidence); switching costs 20 for `H↔A` and 25 for entering or leaving `D`. The costs
make possessions behave like possessions instead of per-frame flicker, which is what the ranking
metric punishes. Evidence is a sum of independent cues:

| cue | state it favours | detail |
|---|---|---|
| ball control | `H` or `A` | distance from the ball to the nearest home vs away player, `tanh((dA−dH)/2 m)`, faded by `exp(−nearest/3 m)` so passes in transit say little; halved on interpolated ball frames |
| player-only team model | `H` or `A` | see below; used **only** where the ball is absent |
| player tempo | `D` | mean tracked-player speed, 3 s smoothed, below a threshold, relative to the rolling live-play pace |
| ball outside pitch | `D` | ball more than 0.5 m beyond a line |
| stationary ball | `D` | smoothed ball speed below 4 m/s (weak) |
| set-piece tag | `D` | Smart Tagging `SET PIECES` start, from 5 s before to 2 s after |

**Player-only team model (the part that matters most).** A logistic regression on 84 team-shape
features, oriented so Barcelona always attacks toward +x (direction read from the two published
kickoff positions), each at 1 s, 5 s and 10 s of context:

* *shape* (45 features): centroid and depth of each team, position of the 3rd most advanced and 3rd
  deepest outfielder, team length, centroid velocity over 2 s;
* *movement and compactness* (`pipeline/shape2.py`, 39 features): mean speed of each team, width and
  depth of each block, mean forward velocity over 1 s (attackers push up, defenders drop), and a
  *density hotspot* — the position with the most players within 8 m, a proxy for where the ball is —
  with the home-minus-away head count around it and the gap between the closest opposing pair.

It is trained on **pseudo-labels** — the ball-based model's H/A labels on frames that are at least
2 s inside a ball-tracked run — over the whole match. No ground truth is used. Against the true
labels in the public window it reaches 0.92 team accuracy on live frames (0.90 for the 45-feature
version). It enters the HMM with weight 1.0, only on frames with no ball: it is slow-moving, and
adding it where the ball is visible measurably blunted the sharper ball cues.

**Tempo normalisation.** Mean player speed is the main dead-ball cue, but players tire: the median
falls from about 2.1 m/s early to about 1.0–1.3 m/s late, and a fixed threshold labelled up to
about 70 % of some late 5-minute blocks dead. Speed is therefore rescaled by a rolling 75th percentile
(±4 min, per half) so the threshold means the same thing at minute 5 and minute 85.

**Broadcast ball fill.** The broadcast ATD is on another clock, but within each half it is a constant
integer shift of the tactical one. The shift is found at run time by masked cross-correlation of the
two ball tracks (best-lag squared error ≈ 0.001 against ≈ 0.09 elsewhere, over ≈ 37 k frames where
both see the ball). Broadcast ball positions fill tactical gaps, raising graded-frame ball coverage
from 59.3 % to 64.5 %. The gain is small because both feeds usually lose the ball at the same moments.

## 3. Assets used, tried and dropped

| asset | use |
|---|---|
| tactical ATD | main input: players, ball, speeds |
| broadcast ATD | ball positions only, aligned to the tactical timeline, as a gap filler |
| Smart Tagging | `SET PIECES` start times only, as a dead-ball prior. Other codes are phases, not possessions, and `Team` is `N/A` on every throw-in, so they were not used |
| published facts | half boundaries, kickoff frames, and that the 2nd half is dead (`D`) until the 113,742 kickoff (FAQ) |
| homography, both videos | **not used.** The tracking-only model already runs in minutes on a laptop; a ball detector over 199 k frames is the one thing that could cover the long gaps, and is listed under future work |

Tried and rejected: (a) treating a *short* ball dropout as a stoppage — only 6 % of gaps under 6 s
overlap a restart, and two of three such gaps in the public window are live play; (b) learning the
dead-ball detector from the public window — it has 4 dead stretches, and blocked cross-validation
was worse than a hand threshold, so the detector is built from interpretable cues instead;
(c) quadratic feature interactions in the team model — no gain; (d) gradient boosting instead of
logistic regression for the team model — no better than the linear model on held-out blocks, so the
pipeline stays numpy and pandas only; (e) switching costs that depend on how long the ball has been
missing (x0.6 to x1.6 on the H<->A cost, x0.7 and x1.4 on the dead-ball cost) — best case +0.005
spellF1 on the benchmark of section 5, far inside the noise, so the costs stay constant.

## 4. Possession rules implemented

* A possession is a maximal run of one team with the ball alive; a dead stretch always ends it.
* During a pass or a loose ball the previous team keeps possession. A hand-over needs sustained
  evidence (the switching cost), which is a proxy for "controlled" possession; a deflection, a
  single touch or an aerial duel without subsequent control does not flip the label.
* Dead ball is `D`, never a team. Dead ball is inferred from collective slow-down of the players,
  the ball leaving the pitch, a stationary ball, and the restart tags.
* No abstentions: every graded frame gets `H`, `A` or `D`. Ungraded frames (before kickoff, halftime,
  after the whistle) are written `X`.

## 5. Validation — and its limits

The public window is 5 minutes with 17 possessions and a ball that is 88 % visible, against 57 %
over the match, so it cannot rank anything and was **not** used to tune the model beyond choosing
cue weights on broad plateaus. Instead we use a **ball-masking stress test**: hide the ball in random
8–40 s stretches (about 56 % of the window, harsher than the real 41 %), 16 random masks, and score
against the public truth. Reported: mean, with the standard deviation over masks.

| configuration | public spellF1 | public macroF1 | masked spellF1 (sd) | masked macroF1 |
|---|---|---|---|---|
| 1. ball-evidence HMM only | 0.471 | 0.797 | 0.376 (0.102) | 0.750 |
| 2. + tempo-normalised dead cue | 0.471 | 0.797 | 0.368 (0.096) | 0.744 |
| 3. + set-piece prior, ball-speed cue | 0.509 | 0.833 | 0.384 (0.099) | 0.773 |
| 4. + player-only team model | 0.509 | 0.833 | 0.514 (0.079) | 0.841 |
| 5. + broadcast ball fill | 0.509 | 0.833 | 0.517 (0.085) | 0.835 |
| 6. + movement/compactness/hotspot features, weight 1.0 (**submitted**) | 0.509 | 0.833 | 0.502 (0.074) | 0.847 |
| *published floor `baseline_smooth50`* | *0.326* | *0.621* | | |

Reading it honestly: the team-shape model is the big gain (+0.13 spellF1 when the ball is hidden).
Row 6 does **not** show an improvement here: 0.502 against 0.517 is well inside one standard
deviation, and macroF1 moves up by 0.012. This window cannot separate the two, which is why row 6
was chosen on the larger benchmark below instead.
The tempo normalisation does not show here because the window is early in the match; its justification
is the late-match analysis above, not this table. The broadcast fill is a marginal gain. Differences
below about 0.04 spellF1 are noise at this sample size. Switching costs sit on a plateau: any `H↔A`
cost from 14 to 28 and any dead cost from 20 to 35 scores 0.50–0.54 masked.

**Full-match recovery benchmark** (`python -m pipeline.bench`). The 5-minute window has 17
possessions, so changes to the ball-less machinery are judged on the whole match with no private
labels. The pseudo-truth is the ball-based labels on the 35 % of graded frames where the ball is
tracked on at least 90 % of the surrounding 30 s (175 possessions). The ball is hidden in random
8–40 s stretches there (about 58 % of those frames, 12 masks), the model is re-run, and the result is
scored against the pseudo-truth. The team model is cross-fitted over five time blocks, so a hidden
stretch never sees the labels it must recover. Configurations are compared on the same masks, with a
bootstrap interval on the paired difference.

| configuration | spellF1 | difference | 95 % interval |
|---|---|---|---|
| 45 features, weight 0.5 | 0.648 | reference | |
| 84 features, weight 0.5 | 0.668 | +0.020 | +0.008 to +0.031 |
| 84 features, weight 1.0 (**submitted**) | 0.680 | +0.032 | +0.021 to +0.044 |
| 84 features, weights 1.25 / 1.5 / 2.0 | 0.669 / 0.673 / 0.670 | +0.022 / +0.026 / +0.022 | all above zero |
| 84 features, weight 0.75 without the ball and 0.25 where the ball is visible | 0.556 | −0.092 | −0.115 to −0.070 |

The weight sits on a plateau (any value from 0.75 to 2.0 gives +0.02 to +0.03), and using the shape
model where the ball is visible is clearly harmful. The switching costs were checked on the same
benchmark with the earlier 45-feature model: 14/20 and 28/35 (H<->A / dead) both scored lower than the
chosen 20/25 (−0.028 and −0.045, intervals below zero), so the costs are not just a plateau but close
to the best values we found. They were not re-tuned for the 84-feature model. These scores measure how well the ball-based
labels are recovered, not agreement with the true labels, so they rank changes to the ball-less
machinery and are not a result to quote as a score.

Sanity check on the full match (no labels needed): about 333 possessions with a mean of about 13.5 s
(public truth: 14.2 s), about 105 dead stretches, Barcelona 56 % of live play. The previous model
gave about 310 possessions of 14.1 s: the larger shape weight switches team slightly more often.

## 6. Limitations and what we would fix with more time

* **Long ball-less gaps** (about a third of the match) rest on team shape alone, which is good for
  who has the ball over seconds but not for exactly when it changes hands. A ball detector on the
  video, or the homography to project it, is the natural fix.
* **Rapid hand-overs under about 4 s** (several in the public window) are missed; the switching cost
  that prevents flicker also removes them.
* **Short dead stretches** (about 5 s) are blurred by 3 s tempo smoothing, costing boundary IoU.
* The dead-ball cues were calibrated on 4 dead stretches plus the restart tags; the tempo
  normalisation reference (2.7 m/s) is an early-match value.
* Team-model training labels inherit the ball model's biases (a home-share prior of ≈ 0.6).
* The benchmark in section 5 can only judge the ball-less machinery. Anything that depends on the
  ball being visible, such as dead-ball detection or hand-over timing, is judged on the 17
  possessions of the public window, which is too few to rank two close versions.

## 7. Reproduction

```bash
pip install -r requirements-solution.txt
# put the match data bundle in data/ exactly as described in DATA.md (or set MTAH_DATA_DIR)
python -m pipeline.run --out evaluation/results/Bytes_of_Africa.csv
python evaluation/validate_submission.py evaluation/results/Bytes_of_Africa.csv
python evaluation/run_evaluation.py
```

The pipeline has no randomness: two runs from a cold cache produce byte-identical files (verified
by hash). Code is in `pipeline/` (`run.py` is the entry point; `hmm.py` the model; `shape.py`,
`shape2.py` and `shapemodel.py` the player-only team model; `bench.py` the full-match recovery
benchmark; `align.py` the broadcast alignment; `stress.py` the
masking test). Tested with Python 3.13, numpy 2.5.3, pandas 3.0.6. The match data and the parsed
cache (`cache/`) are gitignored and are not in this repository.
