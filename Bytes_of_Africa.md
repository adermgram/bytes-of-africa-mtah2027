# Bytes of Africa — technical document

More than a Hack 2027 · frame-level ball possession, FC Barcelona vs Athletic Club

## 1. Summary

Every graded frame is labelled `H`, `A` or `D` by one deterministic pipeline that uses only the
provided tracking data (tactical and broadcast ATD) and the Smart Tagging file. It needs numpy and
pandas, no video, no GPU, and runs in well under 30 seconds (about 5 s once the parsed tracking is cached).

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

**Player-only team model (the part that matters most).** A logistic regression on team-shape
features, oriented so Barcelona always attacks toward +x (direction read from the two published
kickoff positions): centroid and depth of each team, position of the 3rd most advanced and 3rd
deepest outfielder, team length, centroid velocity over 2 s, each at 1 s, 5 s and 10 s of context.
It is trained on **pseudo-labels** — the ball-based model's H/A labels on frames that are at least
2 s inside a ball-tracked run — over the whole match. No ground truth is used. Against the true
labels in the public window it reaches 0.88 team accuracy on live frames. It enters the HMM with
weight 0.5, only on frames with no ball: it is slow-moving, and adding it where the ball is visible
measurably blunted the sharper ball cues.

**Tempo normalisation.** Mean player speed is the main dead-ball cue, but players tire: the median
falls from about 2.1 m/s early to about 1.0–1.3 m/s late, and a fixed threshold labelled up to
65 % of some late 5-minute blocks dead. Speed is therefore rescaled by a rolling 75th percentile
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
| homography, both videos | **not used.** The tracking-only model already runs in seconds; a ball detector over 199 k frames is the one thing that could cover the long gaps, and is listed under future work |

Tried and rejected: (a) treating a *short* ball dropout as a stoppage — only 6 % of gaps under 6 s
overlap a restart, and two of three such gaps in the public window are live play; (b) learning the
dead-ball detector from the public window — it has 4 dead stretches, and blocked cross-validation
was worse than a hand threshold, so the detector is built from interpretable cues instead;
(c) quadratic feature interactions in the team model — no gain.

## 4. Possession rules implemented

* A possession is a maximal run of one team with the ball alive; a dead stretch always ends it.
* During a pass or a loose ball the previous team keeps possession. A hand-over needs sustained
  evidence (the switching cost), which is a proxy for "controlled" possession; a deflection or a
  single touch does not flip the label.
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
| 5. + broadcast ball fill (**submitted**) | 0.509 | 0.833 | 0.517 (0.085) | 0.835 |
| *published floor `baseline_smooth50`* | *0.326* | *0.621* | | |

Reading it honestly: the team-shape model is the big gain (+0.13 spellF1 when the ball is hidden).
The tempo normalisation does not show here because the window is early in the match; its justification
is the late-match analysis above, not this table. The broadcast fill is a marginal gain. Differences
below about 0.04 spellF1 are noise at this sample size. Switching costs sit on a plateau: any `H↔A`
cost from 14 to 28 and any dead cost from 20 to 35 scores 0.50–0.54 masked.

Sanity check on the full match (no labels needed): about 310 possessions with a mean of about 14 s
(public truth: 14.2 s), about 100 dead stretches, Barcelona 54–60 % of live play.

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

## 7. Reproduction

```bash
pip install -r requirements-solution.txt
# put the match data bundle in data/ exactly as described in DATA.md (or set MTAH_DATA_DIR)
python -m pipeline.run --out evaluation/results/Bytes_of_Africa.csv
python evaluation/validate_submission.py evaluation/results/Bytes_of_Africa.csv
python evaluation/run_evaluation.py
```

The pipeline has no randomness: two runs from a cold cache produce byte-identical files (verified
by hash). Code is in `pipeline/` (`run.py` is the entry point; `hmm.py` the model; `shape.py` and
`shapemodel.py` the player-only team model; `align.py` the broadcast alignment; `stress.py` the
masking test). Tested with Python 3.13, numpy 2.5.3, pandas 3.0.6. The match data and the parsed
cache (`cache/`) are gitignored and are not in this repository.
