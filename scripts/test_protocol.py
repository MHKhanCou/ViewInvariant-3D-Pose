import sys
sys.path.insert(0, '.')
from evaluation.protocol import build_pairing_table
table = build_pairing_table()
for t in table:
    print("%s/%s: %d pairs" % (t["subject"], t["sequence"], t["n_frames"]))
