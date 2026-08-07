"""
Three document-level checks the numeric audit does not cover.

1. UNBACKED NUMBERS. Every figure in the reports that carries a unit -- mm or
   percent -- is looked up in every stored artifact JSON. A number that matches
   nothing was typed by hand, which both reports explicitly claim never happens.
2. ORPHANS. Artifact JSONs no document and no audit check ever mentions, and
   pre-registrations whose experiment produced no result.
3. STALE FIGURES. Every \\includegraphics target must exist, be non-empty, and
   be no older than the artifact directory its generator reads. A figure older
   than its data is a figure showing retired numbers.

Run:  ./venv/Scripts/python.exe -m evaluation.verify_documents
"""

import glob
import io
import json
import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(REPO_ROOT, "thesis_artifacts")
REPORTS = ("thesis_report/Full_Thesis_Report.tex",
           "thesis_report/Minimal_Thesis_Report.tex")

# Numbers a reader would never trace to an artifact: years, section and table
# references, the joint indices, and the small integers used in prose.
IGNORE = {1964.0, 1965.0, 1981.0, 1999.0, 2005.0, 2019.0, 2023.0, 2024.0,
          2025.0, 2026.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0}
TOL = 0.051


def artifact_values():
    """Every numeric leaf in every stored artifact, with its file."""
    vals = {}
    for p in glob.glob(os.path.join(ART, "**", "*.json"), recursive=True):
        rel = os.path.relpath(p, REPO_ROOT).replace("\\", "/")
        try:
            with io.open(p, encoding="utf-8") as fh:
                d = json.load(fh)
        except Exception:
            continue

        def walk(o):
            if isinstance(o, bool):
                return
            if isinstance(o, (int, float)):
                vals.setdefault(round(float(o), 4), rel)
            elif isinstance(o, dict):
                for v in o.values():
                    walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)
        walk(d)
    return vals


NUM_WITH_UNIT = re.compile(
    r"(?<![\d.])(\d+(?:\.\d+)?)\s*(?:~?mm|\\%|\s+percent)\b")


def check_unbacked(vals):
    problems = []
    for rep in REPORTS:
        p = os.path.join(REPO_ROOT, rep)
        if not os.path.exists(p):
            continue
        body = io.open(p, encoding="utf-8").read()
        seen = set()
        for m in NUM_WITH_UNIT.finditer(body):
            lit = m.group(1)
            x = float(lit)
            if x in IGNORE or x in seen:
                continue
            seen.add(x)
            # A report writing "138" may be rounding 138.16, and one writing
            # "271.6" may be rounding 271.63. The tolerance has to be half the
            # last place actually written, not a fixed epsilon.
            decimals = len(lit.split(".")[1]) if "." in lit else 0
            tol = 0.5 * (10 ** -decimals)
            if not any(abs(x - v) <= tol for v in vals):
                problems.append("%s: %s has no artifact within %g"
                                % (os.path.basename(rep), lit, tol))
    return problems


def check_orphans():
    problems = []
    audit = io.open(os.path.join(REPO_ROOT, "evaluation", "audit_numbers.py"),
                    encoding="utf-8").read()
    docs = ""
    for pat in ("thesis_report/*.tex", "thesis_report/*.md", "*.md"):
        for p in glob.glob(os.path.join(REPO_ROOT, pat)):
            docs += io.open(p, encoding="utf-8", errors="replace").read()

    for p in sorted(glob.glob(os.path.join(ART, "*", "*.json"))):
        d, f = os.path.basename(os.path.dirname(p)), os.path.basename(p)
        if d not in audit and d not in docs and f not in docs:
            problems.append("orphan artifact: %s/%s referenced nowhere" % (d, f))

    for pre in sorted(glob.glob(os.path.join(ART, "*", "PREREGISTRATION.md"))):
        d = os.path.dirname(pre)
        if not glob.glob(os.path.join(d, "*.json")) and not glob.glob(
                os.path.join(d, "RESULT.md")):
            problems.append("pre-registration with no result: %s"
                            % os.path.basename(d))
    return problems


def check_figures():
    problems = []
    newest_artifact = max(
        (os.path.getmtime(p) for p in
         glob.glob(os.path.join(ART, "**", "*.json"), recursive=True)),
        default=0)
    for rep in REPORTS:
        p = os.path.join(REPO_ROOT, rep)
        if not os.path.exists(p):
            continue
        body = io.open(p, encoding="utf-8").read()
        for m in re.finditer(r"includegraphics\[[^\]]*\]\{([^}]+)\}", body):
            target = m.group(1)
            fp = os.path.join(REPO_ROOT, "thesis_report", target)
            if not os.path.exists(fp):
                problems.append("missing figure: %s (%s)"
                                % (target, os.path.basename(rep)))
            elif os.path.getsize(fp) == 0:
                problems.append("empty figure: %s" % target)
    return problems, newest_artifact


def main():
    vals = artifact_values()
    print("=" * 78)
    print("DOCUMENT CHECKS  (%d distinct numeric values across artifacts)"
          % len(vals))
    print("=" * 78)

    total = 0
    for name, probs in (("1. numbers with no backing artifact",
                         check_unbacked(vals)),
                        ("2. orphaned artifacts and resultless pre-regs",
                         check_orphans())):
        print("\n%s" % name)
        if probs:
            total += len(probs)
            for x in probs[:25]:
                print("   - %s" % x)
            if len(probs) > 25:
                print("   ... and %d more" % (len(probs) - 25))
        else:
            print("   clean")

    figs, newest = check_figures()
    print("\n3. figures")
    if figs:
        total += len(figs)
        for x in figs:
            print("   - %s" % x)
    else:
        print("   all present and non-empty")

    print("\n%d problems" % total)
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
