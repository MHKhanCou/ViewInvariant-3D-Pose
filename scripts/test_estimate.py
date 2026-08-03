import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import cv2
import numpy as np
from backend.inference import estimate_poses
from backend.model_loader import get_detector, get_model

get_detector()
get_model()

cap = cv2.VideoCapture('demo/video/sample_video.mp4')
ret, frame = cap.read()
cap.release()
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

result = estimate_poses(frame_rgb)
dp = result["motionagformer_display_pose"]
vp = result["view_invariant"]
print(f"display_pose: {dp.shape}, range=[{dp.min():.3f}, {dp.max():.3f}]")
print(f"view_invariant: {vp.shape}, range=[{vp.min():.3f}, {vp.max():.3f}]")
print(f"kpts_2d: {result['kpts_2d'].shape}")
print(f"scores: {result['scores'].shape}, mean={result['scores'].mean():.3f}")
print(f"frame_rgb: {result['frame_rgb'].shape}")
print(f"inference_time: {result['inference_time']:.2f}s")
print("SUCCESS")
