"""
Metadata capability detection for cross-view evaluation.

Inspects the processed H36M pkl to determine whether exact cross-view
pairing is possible. Returns structured reports without raising errors.

Split into:
  - prediction capability: can paired predictions be compared across views?
  - GT capability: can a metric-3D GT consistency floor be computed?
  - full evaluation: both of the above.
"""

import os
import pickle
import numpy as np
from collections import defaultdict


# ─── In-memory metadata API (testable without pkl) ───────────────────────────

def inspect_prediction_metadata(sources, cameras):
    """
    Check whether cross-view prediction pairing is possible.

    Valid pairing requires:
      - same sequence identity (ignoring ca_NN suffix)
      - same exact timestamp within that sequence
      - at least two distinct cameras

    Args:
        sources: array-like of source strings, one per frame.
        cameras: array-like of camera name strings, one per frame.

    Returns:
        dict with:
            has_exact_timestamp (bool): whether timestamps exist in source strings
            has_multi_camera_sequences (bool): whether any sequence has 2+ cameras
            can_pair (bool): prediction cross-view pairing is possible
            reasons (list[str]): why pairing is not possible (empty if can_pair)
            details (dict): diagnostic info
    """
    sources = list(sources)
    cameras = list(cameras)
    n = min(len(sources), len(cameras))

    reasons = []
    details = {}

    # Check 1: timestamps
    ts_check = _check_timestamps(sources[:n])
    details['timestamp_check'] = ts_check
    if not ts_check['found']:
        reasons.append(
            f"Source strings lack temporal timestamps. "
            f"Examples: {ts_check['samples'][:3]}"
        )

    # Check 2: multi-camera sequences
    # Group by (subject, action, subaction) as sequence key, timestamp as sync key.
    # camera_name is the view identifier (from the pkl, not from source string).
    groups = defaultdict(set)
    for i in range(n):
        parsed = _parse_source(str(sources[i]))
        seq_key = (parsed['subject'], parsed['action'], parsed['subaction'])
        ts = parsed['timestamp']
        cam = str(cameras[i])
        groups[(seq_key, ts)].add(cam)

    multi_cam_groups = {k: v for k, v in groups.items() if len(v) >= 2}
    has_multi = len(multi_cam_groups) > 0
    details['n_sequences'] = len(set(k[0] for k in groups.keys()))
    details['n_timestamps'] = len(groups)
    details['n_multi_camera_groups'] = len(multi_cam_groups)
    details['multi_camera_examples'] = list(multi_cam_groups.items())[:3]

    if not has_multi:
        reasons.append(
            f"No sequence+timestamp combination has 2+ distinct cameras. "
            f"Cannot pair frames across views."
        )

    can_pair = ts_check['found'] and has_multi

    return {
        'has_exact_timestamp': ts_check['found'],
        'has_multi_camera_sequences': has_multi,
        'can_pair': can_pair,
        'reasons': reasons,
        'details': details,
    }


def inspect_gt_metadata(gts, metric_3d_verified=False):
    """
    Check whether ground truth can provide a metric-3D canonical consistency floor.

    The z/xy range ratio is only a depth/plausibility heuristic. It cannot
    prove that arbitrary coordinates are metric 3D. The floor is only enabled
    when an explicit coordinate provenance statement (metric_3d_verified=True)
    is provided by the caller.

    Args:
        gts: numpy array of shape (N, 17, 3) — the GT joints.
        metric_3d_verified: bool, True only if the caller has independently
            verified that the GT coordinates are in metric 3D units with
            known provenance. Default False.

    Returns:
        dict with:
            depth_plausibility (bool): z/xy ratio is consistent with metric 3D
            can_compute_floor (bool): GT canonical consistency floor is meaningful
                (True ONLY if metric_3d_verified=True AND depth_plausibility=True)
            reasons (list[str]): why GT is not usable (empty if can_compute_floor)
            details (dict): diagnostic info including the heuristic ratio
    """
    gts = np.asarray(gts)
    reasons = []
    details = {}

    sample = gts[:100] if len(gts) > 100 else gts
    mean_xy = float(np.mean(np.abs(sample[:, :, :2])))
    mean_z = float(np.mean(np.abs(sample[:, :, 2:])))
    ratio = mean_z / (mean_xy + 1e-6)

    details['depth_plausibility_ratio'] = ratio
    details['mean_xy'] = mean_xy
    details['mean_z'] = mean_z
    details['metric_3d_verified'] = metric_3d_verified

    depth_plausible = ratio >= 0.1
    if not depth_plausible:
        reasons.append(
            f"Depth plausibility heuristic: z/xy ratio = {ratio:.4f} "
            f"(threshold ~0.1 for metric 3D). Coordinate ranges alone cannot "
            f"prove metric units."
        )

    if not metric_3d_verified:
        reasons.append(
            "Explicit coordinate provenance (metric_3d_verified=True) was not "
            "provided. Cannot enable GT canonical consistency floor without "
            "verified provenance."
        )

    can_floor = depth_plausible and metric_3d_verified

    return {
        'depth_plausibility': depth_plausible,
        'can_compute_floor': can_floor,
        'reasons': reasons,
        'details': details,
    }


# ─── Internal helpers ────────────────────────────────────────────────────────

def _parse_source(src):
    """
    Parse H36M source string into explicit fields.

    Handles multiple formats found in the data:
      s_09_act_02_subact_01_ca_01_01234  (test split with timestamp)
      s_09_act_02_subact_01_ca_01         (test split without timestamp)
      s_01_act_02_cam_01                  (train split)
      seq1_1000                           (synthetic/test fixtures)

    The sequence_id is (subject, action, subaction) only — it excludes
    camera_id and timestamp. This is the grouping key for cross-view pairing.

    Returns:
        dict with:
            subject:    e.g. 's_09' or None
            action:     e.g. 'act_02' or None
            subaction:  e.g. 'subact_01' or None
            camera_id:  e.g. 'ca_01' or 'cam_01' or None (view identifier)
            timestamp:  the last numeric part (>=4 digits), or None (sync key)
            sequence_id: (subject, action, subaction) joined, for grouping
    """
    parts = str(src).split('_')
    result = {
        'subject': None,
        'action': None,
        'subaction': None,
        'camera_id': None,
        'timestamp': None,
        'sequence_id': src,
    }

    # Check if last part is a numeric timestamp (>= 4 digits)
    has_timestamp = parts[-1].isdigit() and len(parts[-1]) >= 4
    if has_timestamp:
        result['timestamp'] = parts[-1]

    # Parse H36M-specific fields if format matches
    if len(parts) >= 8:
        # Format: s_09_act_02_subact_01_ca_01[_timestamp]
        result['subject'] = parts[0] + '_' + parts[1]
        result['action'] = parts[2] + '_' + parts[3]
        result['subaction'] = parts[4] + '_' + parts[5]
        result['camera_id'] = parts[6] + '_' + parts[7]
    elif len(parts) >= 6:
        # Format: s_01_act_02_cam_01
        result['subject'] = parts[0] + '_' + parts[1]
        result['action'] = parts[2] + '_' + parts[3]
        result['camera_id'] = parts[4] + '_' + parts[5]
    elif len(parts) >= 2:
        result['subject'] = parts[0]

    # sequence_id is (subject, action, subaction) — excludes camera and timestamp
    if result['subject'] and result['action'] and result['subaction']:
        result['sequence_id'] = '_'.join([result['subject'], result['action'],
                                          result['subaction']])
    elif result['subject'] and result['action']:
        result['sequence_id'] = '_'.join([result['subject'], result['action']])
    else:
        result['sequence_id'] = src

    return result


def _check_timestamps(sources, n_samples=1000):
    """Check if source strings contain numeric timestamps (last part >= 4 digits)."""
    has_ts = 0
    no_ts = 0
    samples_no_ts = []
    for src in sources[:n_samples]:
        parsed = _parse_source(str(src))
        ts = parsed['timestamp']
        if ts is not None and ts.isdigit() and len(ts) >= 4:
            has_ts += 1
        else:
            no_ts += 1
            if len(samples_no_ts) < 5:
                samples_no_ts.append(str(src))
    return {
        'found': has_ts > no_ts,
        'with_timestamp': has_ts,
        'without_timestamp': no_ts,
        'samples': samples_no_ts,
    }


# ─── Convenience wrapper ─────────────────────────────────────────────────────

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def inspect_cross_view_metadata(pkl_path=None):
    """
    Inspect H36M pkl metadata for cross-view evaluation capability.

    Returns a report combining prediction capability and GT capability.
    """
    if pkl_path is None:
        pkl_path = os.path.join(REPO_ROOT, 'data', 'motion3d',
                                'h36m_sh_conf_cam_source_final.pkl')

    if not os.path.exists(pkl_path):
        return {
            'has_exact_timestamp': False,
            'has_multi_camera_sequences': False,
            'can_compute_prediction_cross_view': False,
            'has_metric_3d_gt': False,
            'can_compute_gt_cross_view_floor': False,
            'can_compute_full_cross_view_evaluation': False,
            'reasons': [f'pkl file not found: {pkl_path}'],
            'details': {},
        }

    with open(pkl_path, 'rb') as f:
        d = pickle.load(f, encoding='latin1')

    meta = d['test']
    sources = meta['source']
    cameras = meta['camera_name']
    gts = meta['joints_2.5d_image']

    pred = inspect_prediction_metadata(sources, cameras)
    gt = inspect_gt_metadata(gts, metric_3d_verified=False)

    all_reasons = pred['reasons'] + gt['reasons']

    return {
        'has_exact_timestamp': pred['has_exact_timestamp'],
        'has_multi_camera_sequences': pred['has_multi_camera_sequences'],
        'can_compute_prediction_cross_view': pred['can_pair'],
        'depth_plausibility': gt['depth_plausibility'],
        'can_compute_gt_cross_view_floor': gt['can_compute_floor'],
        'can_compute_full_cross_view_evaluation': pred['can_pair'] and gt['can_compute_floor'],
        'reasons': all_reasons,
        'details': {'prediction': pred['details'], 'gt': gt['details']},
    }
