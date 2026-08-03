import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
from evaluation.reliability import compute_reliability_score, should_abstain

# Degenerate pose: collapsed joints
degenerate = np.zeros((17, 3), dtype=np.float64)
degenerate[8] = [0, 0.001, 0]
degenerate[1] = [0.001, 0, 0]
degenerate[4] = [-0.001, 0, 0]

rel, comp, meta = compute_reliability_score(degenerate)
print("Degenerate pose reliability: %.4f" % rel)
print("Abstain: %s" % should_abstain(rel))
print("Axes valid: %s" % meta["axes_valid"])

# Good pose: standing person
good = np.array([
    [0, 0, 0], [0.05, -0.1, 0], [0.06, 0.1, 0], [0.07, 0.3, 0],
    [-0.05, -0.1, 0], [-0.06, 0.1, 0], [-0.07, 0.3, 0],
    [0, -0.15, 0], [0, -0.3, 0], [0, -0.4, 0], [0, -0.5, 0],
    [0.1, -0.28, 0], [0.2, -0.15, 0], [0.25, 0, 0],
    [-0.1, -0.28, 0], [-0.2, -0.15, 0], [-0.25, 0, 0],
], dtype=np.float64)

rel2, _, _ = compute_reliability_score(good)
print("Good pose reliability: %.4f" % rel2)
print("Abstain: %s" % should_abstain(rel2))
