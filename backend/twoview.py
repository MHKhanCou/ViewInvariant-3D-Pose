"""
Two-view demonstrator: one instant of Human3.6M seen by two synchronized
cameras, raw against canonical, with the measured cross-view distance.

This is the claim the thesis actually makes, which a single image cannot show.
The rendering and the measurement are `presentation.render.two_view` — the same
code that produces Figure 1.1 — so the numbers here and in the report come from
one implementation.
"""

import os

import numpy as np

from presentation.render import IMG, two_view

_GROUPS = None  # {(subject, action): [cameras]} — cached, the .npz load is slow

ACTION_NAMES = {
    "act_02": "Directions", "act_03": "Discussion", "act_04": "Eating",
    "act_05": "Greeting", "act_06": "Phoning", "act_07": "Posing",
    "act_08": "Purchases", "act_09": "Sitting", "act_10": "SittingDown",
    "act_11": "Smoking", "act_12": "Photo", "act_13": "Waiting",
    "act_14": "Walking", "act_15": "WalkDog", "act_16": "WalkTogether",
}


def _groups():
    global _GROUPS
    if _GROUPS is None:
        from evaluation.h36m_replication import (OUT_DIR, aggregate_by_video,
                                                 parse_video)
        meta = np.load(os.path.join(OUT_DIR, "meta.npz"), allow_pickle=True)
        pn = np.load(os.path.join(OUT_DIR, "preds.npz"))
        out = {}
        for vid, v in aggregate_by_video(meta, pn, int(pn["n_clips"])).items():
            s, a, c = parse_video(vid)
            out.setdefault((s, a), {})[c] = len(v["pred"])
        _GROUPS = out
    return _GROUPS


def sequences():
    """Labels for the sequence dropdown, e.g. 'S9 Walking'."""
    return ["%s %s" % (s, ACTION_NAMES.get(a, a)) for s, a in sorted(_groups())]


def _parse(label):
    subj, name = label.split(" ", 1)
    action = next(k for k, v in ACTION_NAMES.items() if v == name)
    return subj, action


def camera_pairs(label):
    cams = sorted(_groups()[_parse(label)])
    return ["%s vs %s" % (a, b)
            for i, a in enumerate(cams) for b in cams[i + 1:]]


def n_frames(label):
    return min(_groups()[_parse(label)].values())


def run(label, pair=None, frame=None, auto=True):
    """Render one instant. Returns (image_path, markdown_summary)."""
    subj, action = _parse(label)
    cams = None if auto or not pair else tuple(pair.split(" vs "))
    idx = None if auto else int(frame)

    _, d = two_view(subject=subj, action=action, frame=idx, cams=cams,
                    out="_app_twoview.png")

    gap = d["raw_mm"] - d["oracle_mm"]
    closed = 100.0 * (d["raw_mm"] - d["canonical_mm"]) / gap if gap > 1e-8 else 0.0
    return os.path.join(IMG, "_app_twoview.png"), (
        "### %s %s - cameras %s and %s, frame %d\n\n"
        "| | mean joint distance |\n|---|---|\n"
        "| Raw camera-frame | **%.1f mm** |\n"
        "| Canonical body-frame | **%.1f mm** |\n"
        "| Procrustes oracle (needs both views) | %.1f mm |\n\n"
        "**%.1f%% reduction**, closing %.0f%% of the gap to the oracle floor.\n\n"
        "%s"
        % (subj, ACTION_NAMES.get(action, action), d["cameras"][0],
           d["cameras"][1], d["frame"], d["raw_mm"], d["canonical_mm"],
           d["oracle_mm"], d["reduction_pct"], closed,
           "_Frame chosen as the median improvement for this sequence, not the "
           "most favourable one._" if auto else
           "_Frame chosen by hand; the reported headline uses median frames._"))
