# Where this experiment's criterion lives

The sixteenth experiment (asymmetric, left-side-only distal corruption) is
pre-registered in **`../bestframe/PREREGISTRATION.md`**, committed as **7e09462**
before either it or the fifteenth was run. That one document covers both,
because they were designed and committed together.

Its result is recorded in **`../bestframe/RESULT.md`**, committed as **9744f78**.

This file is deliberately **not** named `PREREGISTRATION.md`. Naming it that
would place a pre-registration file in this directory dated after the results it
governs, which is precisely the pattern
`evaluation/verify_prereg_order.py` exists to catch. The criterion is not in
this directory; the pointer is.

Verify the ordering yourself:

```bash
git log --diff-filter=A --format='%h %ci %s' -- thesis_artifacts/bestframe/PREREGISTRATION.md
git log --diff-filter=A --format='%h %ci %s' -- thesis_artifacts/asymmetric/asymmetric.json
```
