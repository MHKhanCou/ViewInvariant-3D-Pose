"""Debug exactly why 90° rotation fails in detect()."""
import os, sys, numpy as np, cv2
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from demo_live.pose_detector import PoseDetector

def rotate_image(img, angle):
    if angle == 90: return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif angle == 180: return cv2.rotate(img, cv2.ROTATE_180)
    elif angle == 270: return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img

frame = cv2.imread('thesis_artifacts/rotation_test_input.jpg')
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

detector = PoseDetector(weights="yolov8n-pose.pt", device="cpu", conf=0.4)

for angle in [0, 90, 180, 270]:
    rotated = rotate_image(frame_rgb, angle)
    print(f"\n{'='*50}")
    print(f"Angle: {angle}°")

    # Raw YOLO prediction with low conf threshold
    results_raw = detector.model.predict(rotated, conf=0.01, verbose=False)
    raw_result = results_raw[0]

    if raw_result.keypoints is not None and len(raw_result.keypoints) > 0:
        print(f"  Raw: {len(raw_result.keypoints)} persons detected")
        for i in range(min(3, len(raw_result.keypoints))):
            kpts = raw_result.keypoints[i]
            xy = kpts.xy[0].cpu().numpy()
            conf = kpts.conf[0].cpu().numpy()
            valid = np.sum(conf > 0.1)
            print(f"    Person {i}: {valid}/17 keypoints valid, mean_conf={conf.mean():.3f}, "
                  f"max_conf={conf.max():.3f}")
    else:
        print(f"  Raw: No detections")

    # Standard detect() path
    kpts, scores = detector.detect(rotated)
    if kpts is not None:
        valid = np.sum(scores > 0.1)
        print(f"  detect(): {valid}/17 valid, mean={scores.mean():.3f}")
    else:
        print(f"  detect(): FAILED (returned None)")

    # Why did detect() fail? Check the filtering
    results_filtered = detector.model.predict(rotated, conf=0.4, verbose=False)
    fr = results_filtered[0]
    if fr.keypoints is not None and len(fr.keypoints) > 0:
        print(f"  With conf=0.4: {len(fr.keypoints)} persons")
        for i in range(min(3, len(fr.keypoints))):
            conf = fr.keypoints.conf[0].cpu().numpy()
            all_below = np.all(conf < 0.4)
            print(f"    Person {i}: mean_conf={conf.mean():.3f}, all_below_0.4={all_below}")
    else:
        print(f"  With conf=0.4: No detections")
