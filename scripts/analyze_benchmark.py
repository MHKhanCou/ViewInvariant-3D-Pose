import csv
import numpy as np

csv_path = 'thesis_artifacts/example_results/benchmark_summary.csv'
with open(csv_path) as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print('Total images: %d' % len(rows))
print()

# Sort by reliability
rows.sort(key=lambda r: float(r['reliability']))

print('Sorted by reliability (lowest first):')
print('%-50s %8s %8s %10s' % ('Image', 'Reliab', 'Hard', 'Reason'))
print('-' * 80)
for r in rows:
    print('%-50s %8s %8s %10s' % (
        r['image'][:50],
        r['reliability'],
        r['hard_failure'],
        r['failure_reason']
    ))

print()
reliabilities = [float(r['reliability']) for r in rows]
print('Reliability distribution:')
buckets = {'0.5-0.6': 0, '0.6-0.7': 0, '0.7-0.8': 0, '0.8-0.9': 0, '0.9-1.0': 0}
for rel in reliabilities:
    if rel < 0.6: buckets['0.5-0.6'] += 1
    elif rel < 0.7: buckets['0.6-0.7'] += 1
    elif rel < 0.8: buckets['0.7-0.8'] += 1
    elif rel < 0.9: buckets['0.8-0.9'] += 1
    else: buckets['0.9-1.0'] += 1

for k, v in buckets.items():
    bar = '#' * v
    print('  %s: %2d %s' % (k, v, bar))

print()
print('Mean: %.4f' % np.mean(reliabilities))
print('Std:  %.4f' % np.std(reliabilities))
print('Min:  %.4f (%s)' % (np.min(reliabilities), rows[0]['image'][:40]))
print('Max:  %.4f (%s)' % (np.max(reliabilities), rows[-1]['image'][:40]))

# No hard failures - note this
n_hard = sum(1 for r in rows if r['hard_failure'] == 'True')
print('Hard failures: %d/%d' % (n_hard, len(rows)))
