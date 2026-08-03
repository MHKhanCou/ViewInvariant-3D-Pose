import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
from evaluation.reliability import compute_reliability_score, should_abstain
from backend.model_loader import get_model, get_detector
from demo_live.lifter import coco_to_h36m, N_FRAMES, normalize_screen_coordinates
from canonical.canonicalizer import Canonicalizer
import torch, cv2

model = get_model()
det = get_detector()

cap = cv2.VideoCapture('demo/video/sample_video.mp4')
cap.set(cv2.CAP_PROP_POS_FRAMES, 90)
ret, frame = cap.read()
cap.release()
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
H, W = frame.shape[:2]

kpts, scores = det.detect_with_rotation(frame_rgb)
kpts_seq = np.repeat(kpts[None], N_FRAMES, axis=0)
scores_seq = np.repeat(scores[None], N_FRAMES, axis=0)
h36m = coco_to_h36m(kpts_seq, scores_seq)

norm = normalize_screen_coordinates(h36m, w=W, h=H)
inp = torch.from_numpy(norm.astype('float32'))[None]
with torch.no_grad():
    pred = model(inp).cpu().numpy()

raw_pose = pred[0, 13]

reliability, components, metadata = compute_reliability_score(raw_pose, scores)

print("=== Reliability Score ===")
print(f"Score: {reliability:.4f}")
print(f"Abstain (< 0.5): {should_abstain(reliability)}")
print()
print("Components:")
for name, val in components.items():
    print(f"  {name:25s}: {val:.4f}")
print()
print("Metadata:")
print(f"  y_norm (torso): {metadata['y_norm']:.4f}")
print(f"  x_norm (hip):   {metadata['x_norm']:.4f}")
print(f"  z_norm (fwd):   {metadata['z_norm']:.4f}")
print(f"  axes_valid:     {metadata['axes_valid']}")
