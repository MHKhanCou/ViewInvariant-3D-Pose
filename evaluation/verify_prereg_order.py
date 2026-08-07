"""
Verify that every pre-registration was committed BEFORE its own result.

This is the single claim the thesis's credibility rests on, it is checkable by
an examiner in one command, and until now nothing checked it. For each
PREREGISTRATION.md the script finds the commit that introduced it and the
earliest commit that introduced any result file in the same directory, and
asserts the first strictly precedes the second in both commit time and history.

It also reports pre-registrations with no result and results with no
pre-registration, either of which an examiner would notice.

Run:  ./venv/Scripts/python.exe -m evaluation.verify_prereg_order
      ./venv/Scripts/python.exe -m evaluation.verify_prereg_order --json
"""

import argparse
import glob
import json
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(REPO_ROOT, "thesis_artifacts")

# Result files are anything in the directory that is not the pre-registration
# itself: the JSON artifacts and the RESULT.md record.
PREREG = "PREREGISTRATION.md"


def git(*args):
    out = subprocess.run(("git",) + args, cwd=REPO_ROOT, capture_output=True,
                         text=True)
    return out.stdout.strip()


def first_commit(relpath):
    """(sha, unix_time, subject) of the commit that introduced a path."""
    line = git("log", "--diff-filter=A", "--follow", "--format=%H|%ct|%s",
               "--", relpath)
    if not line:
        line = git("log", "--diff-filter=A", "--format=%H|%ct|%s", "--", relpath)
    if not line:
        return None
    sha, ct, subj = line.split("\n")[-1].split("|", 2)
    return sha, int(ct), subj


def is_ancestor(a, b):
    r = subprocess.run(("git", "merge-base", "--is-ancestor", a, b),
                       cwd=REPO_ROOT, capture_output=True)
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    rows, problems = [], []
    prereg_dirs = sorted(os.path.dirname(p) for p in
                         glob.glob(os.path.join(ART, "*", PREREG)))

    for d in prereg_dirs:
        name = os.path.basename(d)
        rel_pre = os.path.relpath(os.path.join(d, PREREG), REPO_ROOT).replace("\\", "/")
        pre = first_commit(rel_pre)
        if pre is None:
            problems.append("%s: pre-registration is not committed" % name)
            continue

        results = [f for f in sorted(os.listdir(d)) if f != PREREG]
        if not results:
            rows.append({"experiment": name, "prereg_sha": pre[0][:7],
                         "prereg_time": pre[1], "result_sha": None,
                         "status": "no result yet"})
            continue

        earliest = None
        for f in results:
            rel = os.path.relpath(os.path.join(d, f), REPO_ROOT).replace("\\", "/")
            c = first_commit(rel)
            if c and (earliest is None or c[1] < earliest[1]):
                earliest = c + (f,)
        if earliest is None:
            problems.append("%s: results exist on disk but none is committed" % name)
            continue

        ok_time = pre[1] < earliest[1]
        ok_hist = pre[0] == earliest[0] or is_ancestor(pre[0], earliest[0])
        same = pre[0] == earliest[0]
        if same:
            status = "SAME COMMIT - pre-registration not separable from result"
            problems.append("%s: %s" % (name, status))
        elif ok_time and ok_hist:
            status = "ok"
        else:
            status = "OUT OF ORDER"
            problems.append("%s: result %s precedes pre-registration" % (name, earliest[0][:7]))

        rows.append({"experiment": name, "prereg_sha": pre[0][:7],
                     "prereg_time": pre[1], "result_sha": earliest[0][:7],
                     "result_time": earliest[1], "first_result_file": earliest[3],
                     "gap_seconds": earliest[1] - pre[1], "status": status})

    # Artifact directories holding results but no pre-registration at all.
    for d in sorted(glob.glob(os.path.join(ART, "*"))):
        if not os.path.isdir(d) or os.path.exists(os.path.join(d, PREREG)):
            continue
        if any(f.endswith("RESULT.md") for f in os.listdir(d)):
            problems.append("%s: has a RESULT.md but no PREREGISTRATION.md"
                            % os.path.basename(d))

    if args.json:
        print(json.dumps({"rows": rows, "problems": problems}, indent=1))
        return 0 if not problems else 1

    print("=" * 78)
    print("PRE-REGISTRATION ORDERING: does each criterion precede its result?")
    print("=" * 78)
    print("  %-22s %-9s %-9s %10s  %s"
          % ("experiment", "prereg", "result", "gap", "status"))
    for r in rows:
        gap = r.get("gap_seconds")
        gs = "-" if gap is None else ("%.1f h" % (gap / 3600.0) if gap >= 3600
                                      else "%d m" % (gap // 60))
        print("  %-22s %-9s %-9s %10s  %s"
              % (r["experiment"], r["prereg_sha"], r.get("result_sha") or "-",
                 gs, r["status"]))

    n_ok = sum(1 for r in rows if r["status"] == "ok")
    print("\n  %d pre-registrations, %d verified in order" % (len(rows), n_ok))
    if problems:
        print("\n  PROBLEMS")
        for p in problems:
            print("   - %s" % p)
        return 1
    print("  No result precedes its own criterion.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
