"""
Export predicted poses as BVH, the motion-capture format Blender, Unity and
MotionBuilder import directly.

Why this belongs in this project
--------------------------------
BVH stores a hierarchy of joint rotations relative to each joint's parent. It
has no camera in it: a BVH file describes how a body is configured, not where it
was observed from. That is precisely what canonicalization produces, so the
export is not a bolt-on convenience but the natural consumer of a body-fixed
frame. A camera-frame prediction cannot be written to BVH without first choosing
an orientation for the body, which is the problem this project solves.

What is and is not recoverable
------------------------------
Joint positions determine bone directions but not rotation about a bone's own
axis. Twist is therefore unrecoverable from a 17-joint skeleton and is set to
zero. Forearm pronation and similar motions will not appear. This is a property
of the input representation, not of the conversion, and every method that lifts
BVH from joint positions shares it.

Rotations are obtained per joint by the Kabsch algorithm over that joint's
children, which handles the hips and thorax correctly where a single-bone
alignment would be ambiguous.

Joint naming follows the Human3.6M convention used by the data (1-3 right leg,
4-6 left leg, 11-13 left arm, 14-16 right arm). The repository documents this
inconsistently in places; the geometry is unaffected by the labelling, only the
names in the output file are.

Run:  ./venv/Scripts/python.exe -m presentation.bvh_export --demo out.bvh
"""

import argparse
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

# H36M-17 kinematic tree: parent -> children.
CHILDREN = {
    0: [1, 4, 7],
    1: [2], 2: [3], 3: [],
    4: [5], 5: [6], 6: [],
    7: [8],
    8: [9, 11, 14],
    9: [10], 10: [],
    11: [12], 12: [13], 13: [],
    14: [15], 15: [16], 16: [],
}
PARENT = {c: p for p, cs in CHILDREN.items() for c in cs}

NAMES = {
    0: "Hips", 1: "RightUpLeg", 2: "RightLeg", 3: "RightFoot",
    4: "LeftUpLeg", 5: "LeftLeg", 6: "LeftFoot",
    7: "Spine", 8: "Spine1", 9: "Neck", 10: "Head",
    11: "LeftArm", 12: "LeftForeArm", 13: "LeftHand",
    14: "RightArm", 15: "RightForeArm", 16: "RightHand",
}

CHANNELS = "Zrotation Xrotation Yrotation"


def rest_offsets(poses, ref=0):
    """
    Rest-pose bone vectors: direction from a reference frame, length from the
    median over the sequence.

    Taking the median of the bone VECTORS instead would be wrong, and silently
    so. Those vectors rotate with the body, and averaging rotating vectors
    shrinks them, so the rest skeleton would come out shorter than the real one
    and no set of rotations could reproduce the input. Splitting direction from
    length avoids that: the direction comes from one real pose, and the length
    is the robust estimate over all of them.
    """
    P = np.asarray(poses, dtype=np.float64)
    off = {}
    for c, p in PARENT.items():
        v = P[ref, c] - P[ref, p]
        n = np.linalg.norm(v)
        length = float(np.median(np.linalg.norm(P[:, c] - P[:, p], axis=-1)))
        off[c] = (v / n * length) if n > 1e-12 else np.array([0.0, length, 0.0])
    return off


def _kabsch(A, B):
    """Rotation R minimising |R @ A - B|, columns are points. Proper rotation."""
    H = A @ B.T
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T


def _global_rotations(pose, off):
    """
    Per-joint global rotation aligning rest child offsets to observed ones.

    Joints with several children (hips, thorax) are solved by Kabsch over all of
    them. Joints with one child have a one-bone problem, which leaves the twist
    about that bone free; we take the minimal rotation, which sets twist to zero.
    Leaves inherit their parent's rotation, since they orient nothing.
    """
    R = {}
    for j in range(17):
        kids = CHILDREN[j]
        if not kids:
            continue
        A = np.stack([off[c] for c in kids], axis=1)
        B = np.stack([pose[c] - pose[j] for c in kids], axis=1)
        if len(kids) == 1:
            a = A[:, 0] / (np.linalg.norm(A[:, 0]) + 1e-12)
            b = B[:, 0] / (np.linalg.norm(B[:, 0]) + 1e-12)
            v = np.cross(a, b)
            c = float(np.dot(a, b))
            if np.linalg.norm(v) < 1e-10:
                R[j] = np.eye(3) if c > 0 else -np.eye(3) + 2 * np.outer(a, a)
            else:
                vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                R[j] = np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))
        else:
            R[j] = _kabsch(A, B)
    for j in range(17):
        if j not in R:
            R[j] = R.get(PARENT.get(j, 0), np.eye(3))
    return R


def _euler_zxy(M):
    """Decompose a rotation matrix into BVH ZXY Euler angles, in degrees."""
    sx = -M[1, 2]
    sx = float(np.clip(sx, -1.0, 1.0))
    x = np.arcsin(sx)
    if abs(sx) < 0.99999:
        z = np.arctan2(M[1, 0], M[1, 1])
        y = np.arctan2(M[0, 2], M[2, 2])
    else:                                   # gimbal lock: fold y into z
        z = np.arctan2(-M[0, 1], M[0, 0])
        y = 0.0
    return np.degrees([z, x, y])


def _write_hierarchy(lines, j, off, indent):
    pad = "  " * indent
    tag = "ROOT" if j == 0 else "JOINT"
    lines.append(f"{pad}{tag} {NAMES[j]}")
    lines.append(f"{pad}{{")
    o = np.zeros(3) if j == 0 else off[j]
    lines.append(f"{pad}  OFFSET {o[0]:.6f} {o[1]:.6f} {o[2]:.6f}")
    if j == 0:
        lines.append(f"{pad}  CHANNELS 6 Xposition Yposition Zposition {CHANNELS}")
    else:
        lines.append(f"{pad}  CHANNELS 3 {CHANNELS}")
    if CHILDREN[j]:
        for c in CHILDREN[j]:
            _write_hierarchy(lines, c, off, indent + 1)
    else:
        # A leaf still needs an End Site so the last bone has a length.
        p = PARENT[j]
        e = off[j] * 0.5 if np.linalg.norm(off[j]) > 1e-9 else np.array([0.0, 1.0, 0.0])
        lines.append(f"{pad}  End Site")
        lines.append(f"{pad}  {{")
        lines.append(f"{pad}    OFFSET {e[0]:.6f} {e[1]:.6f} {e[2]:.6f}")
        lines.append(f"{pad}  }}")
    lines.append(f"{pad}}}")


def poses_to_bvh(poses, fps=25.0, scale=1.0, root_motion=True):
    """
    Convert a (T, 17, 3) sequence of root-relative poses to BVH text.

    Args:
        poses: (T, 17, 3). Canonical poses give a body-fixed result, which is
            what BVH expects; camera-frame poses will import rotated.
        fps: frame rate written into the file.
        scale: multiplier applied to all lengths, for unit conversion.
        root_motion: if False the root is pinned at the origin, which is what
            you want when the poses are canonical and translation is meaningless.

    Returns:
        str, the complete BVH file.
    """
    P = np.asarray(poses, dtype=np.float64) * float(scale)
    if P.ndim == 2:
        P = P[None]
    assert P.shape[1:] == (17, 3), f"expected (T,17,3), got {P.shape}"

    off = rest_offsets(P)
    lines = ["HIERARCHY"]
    _write_hierarchy(lines, 0, off, 0)
    lines.append("MOTION")
    lines.append(f"Frames: {len(P)}")
    lines.append(f"Frame Time: {1.0 / float(fps):.6f}")

    order = [j for j in range(17)]
    for t in range(len(P)):
        R = _global_rotations(P[t], off)
        vals = []
        root_t = P[t, 0] if root_motion else np.zeros(3)
        vals += [f"{v:.6f}" for v in root_t]
        for j in order:
            local = R[j] if j == 0 else R[PARENT[j]].T @ R[j]
            vals += [f"{a:.6f}" for a in _euler_zxy(local)]
        lines.append(" ".join(vals))
    return "\n".join(lines) + "\n"


def write_bvh(path, poses, **kw):
    text = poses_to_bvh(poses, **kw)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def _demo_sequence(n=60):
    """A synthetic walking-ish sequence, so the module is runnable standalone."""
    base = np.zeros((17, 3))
    base[1] = [-0.13, -0.02, 0]; base[2] = [-0.15, -0.45, 0]; base[3] = [-0.16, -0.90, 0]
    base[4] = [0.13, -0.02, 0];  base[5] = [0.15, -0.45, 0];  base[6] = [0.16, -0.90, 0]
    base[7] = [0, 0.25, 0];      base[8] = [0, 0.50, 0]
    base[9] = [0, 0.62, 0];      base[10] = [0, 0.75, 0]
    base[11] = [0.18, 0.48, 0];  base[12] = [0.30, 0.25, 0];  base[13] = [0.35, 0.05, 0]
    base[14] = [-0.18, 0.48, 0]; base[15] = [-0.30, 0.25, 0]; base[16] = [-0.35, 0.05, 0]

    seq = np.repeat(base[None], n, axis=0)
    t = np.linspace(0, 4 * np.pi, n)
    seq[:, 2, 2] += 0.15 * np.sin(t)      # knees swing out of plane
    seq[:, 3, 2] += 0.30 * np.sin(t)
    seq[:, 5, 2] -= 0.15 * np.sin(t)
    seq[:, 6, 2] -= 0.30 * np.sin(t)
    seq[:, 12, 2] -= 0.20 * np.sin(t)     # arms counter-swing
    seq[:, 13, 2] -= 0.35 * np.sin(t)
    seq[:, 15, 2] += 0.20 * np.sin(t)
    seq[:, 16, 2] += 0.35 * np.sin(t)
    return seq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--demo", action="store_true",
                    help="write a synthetic sequence instead of reading poses")
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--scale", type=float, default=100.0,
                    help="metres to centimetres by default, which is what most "
                         "importers expect")
    args = ap.parse_args()

    poses = _demo_sequence() if args.demo else np.load(args.out.replace(".bvh", ".npy"))
    write_bvh(args.out, poses, fps=args.fps, scale=args.scale, root_motion=False)
    print("Wrote %s  (%d frames, %d joints)" % (args.out, len(poses), 17))


if __name__ == "__main__":
    main()
