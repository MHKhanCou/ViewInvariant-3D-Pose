"""
How much of the multi-scale result is mechanical?

An external reviewer raised a circularity concern about the per-limb frames, and
inspection of the definitions confirms it. In SEGMENTS_LONGAXIS every limb is
built from exactly the joints it is then scored on:

    left_arm   ids [14,15,16]   y=(14,15)  x=(15,16)
    right_arm  ids [11,12,13]   y=(11,12)  x=(12,13)
    left_leg   ids [1,2,3]      y=(1,2)    x=(2,3)
    right_leg  ids [4,5,6]      y=(4,5)    x=(5,6)

Root-centred, a three-joint segment has two free points and so six coordinates.
A frame built by Gram-Schmidt from those same three points removes three
rotational degrees of freedom, leaving the canonical configuration determined by
two segment lengths and the angle between them. The cross-view distance of such
a segment therefore cannot measure orientation disagreement at all: orientation
has been removed by construction, not by the method working.

The control is to compare each level against a Procrustes oracle computed on the
SAME joints. The oracle applies the optimal rotation with knowledge of both
views, so it is the floor no rotation-based method can beat. If canonical is at
the oracle floor, the frame has removed everything a rotation could remove and
the number is mechanical. If canonical sits well above the floor, there is real
orientation disagreement left for the frame to have reduced, and the result
means what it appears to mean.

The global frame is included as the comparison that matters: it is built from
joints 0, 8, 1 and 4 and scored on all seventeen, so thirteen of the seventeen
are held out and it is not circular in this way.

Run:  ./venv/Scripts/python.exe -m evaluation.multiscale_control
      ./venv/Scripts/python.exe -m evaluation.multiscale_control --preds <cache> --tag <name>
"""

import argparse
import itertools
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from canonical.multiscale import SEGMENTS
from evaluation.h36m_crossview import EVAL_STRIDE, canonicalize_stream
from evaluation.h36m_multiscale import (SEGMENT_DISPLAY, SEGMENTS_LONGAXIS,
                                        SEGMENTS_SYMMETRIC, canonicalize_with)
from evaluation.h36m_replication import OUT_DIR as PRED_DIR
from evaluation.h36m_replication import aggregate_by_video, parse_video
from evaluation.oracle import procrustes_align

OUT_DIR = os.path.join(REPO_ROOT, "thesis_artifacts", "multiscale_control")

SEGMENT_SETS = {"shipped": SEGMENTS, "symmetric": SEGMENTS_SYMMETRIC,
                "long_axis": SEGMENTS_LONGAXIS}


def constructor_joints(spec):
    """The joints a segment's frame is built from."""
    _ids, _root, (y0, y1), (x0, x1) = spec
    return {y0, y1, x0, x1}


def segment_oracle(a, b):
    """Mean per-joint distance after the optimal rigid rotation of a onto b."""
    R, _, _ = procrustes_align(a.astype(np.float32), b.astype(np.float32))
    aligned = (a - a.mean(axis=0)) @ R.T + b.mean(axis=0)
    return float(np.mean(np.linalg.norm(aligned - b, axis=1)))


def run(videos, stride=EVAL_STRIDE):
    groups = {}
    for vid, v in videos.items():
        subj, action, cam = parse_video(vid)
        groups.setdefault((subj, action), {})[cam] = v

    # accumulate sum of distances and count, per (set, segment)
    acc = {}

    def add(key, can, orc):
        s = acc.setdefault(key, [0.0, 0.0, 0])
        s[0] += can
        s[1] += orc
        s[2] += 1

    for (subj, action), cams in sorted(groups.items()):
        if len(cams) < 4:
            continue
        order = sorted(cams)
        n = min(len(cams[c]["pred"]) for c in order)
        sel = np.arange(0, n, stride)

        per_cam = {}
        for c in order:
            P = cams[c]["pred"][sel]
            P = P - P[:, 0:1, :]
            gcan, gok = canonicalize_stream(P)
            per_cam[c] = {"raw": P, "gcan": gcan, "gok": gok}
            for sname, table in SEGMENT_SETS.items():
                # canonicalize_with returns one dict per frame, keyed by level,
                # whose "joints" are already the segment's own joints in that
                # segment's frame.
                per_cam[c][sname] = [canonicalize_with(P[i], table)
                                     for i in range(len(P))]

        for a, b in itertools.combinations(order, 2):
            A, B = per_cam[a], per_cam[b]
            keep = A["gok"] & B["gok"]
            for i in np.flatnonzero(keep):
                # Global frame: built from 0, 8, 1, 4; scored on all seventeen.
                add(("global", "global"),
                    float(np.mean(np.linalg.norm(
                        A["gcan"][i] - B["gcan"][i], axis=1))),
                    segment_oracle(A["gcan"][i], B["gcan"][i]))
                for sname, table in SEGMENT_SETS.items():
                    for seg in table:
                        la, lb = A[sname][i][seg], B[sname][i][seg]
                        if not (la["valid"] and lb["valid"]):
                            continue
                        ca = np.asarray(la["joints"], dtype=np.float64)
                        cb = np.asarray(lb["joints"], dtype=np.float64)
                        add((sname, seg),
                            float(np.mean(np.linalg.norm(ca - cb, axis=1))),
                            segment_oracle(ca, cb))

    rows = []
    for (sname, seg), (scan, sorc, cnt) in sorted(acc.items()):
        if sname == "global":
            n_ids, n_ctor = 17, 4
        else:
            spec = SEGMENT_SETS[sname][seg]
            n_ids = len(spec[0])
            n_ctor = len(constructor_joints(spec) & set(spec[0]))
        rows.append({
            "set": sname,
            "segment": seg,
            "display": SEGMENT_DISPLAY.get(seg, seg),
            "n_scored": n_ids,
            "n_scored_that_build_the_frame": n_ctor,
            "fully_circular": bool(n_ctor == n_ids),
            "canonical_mm": scan / cnt,
            "oracle_mm": sorc / cnt,
            "headroom_ratio": (scan / cnt) / (sorc / cnt) if sorc > 0 else float("nan"),
            "n_frame_pairs": cnt,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    meta = np.load(os.path.join(PRED_DIR, "meta.npz"), allow_pickle=True)
    path = args.preds or os.path.join(PRED_DIR, "preds.npz")
    pn = np.load(path)
    videos = aggregate_by_video(meta, pn, int(pn["n_clips"]))

    rows = run(videos)

    print("=" * 92)
    print("HOW MUCH OF EACH LEVEL IS REMOVED BY CONSTRUCTION?")
    print("=" * 92)
    print("  headroom = canonical / per-segment Procrustes oracle.")
    print("  1.00 means the frame has removed everything a rotation could;")
    print("  the measurement then cannot show orientation disagreement.\n")
    print("  %-11s %-11s %8s %10s %10s %10s  %s"
          % ("set", "segment", "scored", "builders", "canon mm", "oracle mm",
             "headroom"))
    for r in rows:
        flag = "  <-- fully circular" if r["fully_circular"] else ""
        print("  %-11s %-11s %8d %10d %10.1f %10.1f %9.2fx%s"
              % (r["set"], r["display"], r["n_scored"],
                 r["n_scored_that_build_the_frame"], r["canonical_mm"],
                 r["oracle_mm"], r["headroom_ratio"], flag))

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "control%s.json"
                       % (("_" + args.tag) if args.tag else ""))
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "prediction_cache": os.path.basename(path)},
                  f, indent=1)
    print("\nSaved: %s" % out)


if __name__ == "__main__":
    main()
