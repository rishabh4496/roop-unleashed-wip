"""Synthetic face geometry for the alignment / pose tests.

Projects a known 3-D head at a chosen yaw / pitch / roll so tests can assert on
pose behaviour without needing a video, a detector, or a GPU. The 3-D points are
taken from the project's own reference face (roop.face_3d_recon._REF3D_68) so
the tests and the shipping pose code agree on what a head is shaped like.

Orthographic (weak-perspective) projection is deliberate: it is the clean case.
If a heuristic already misbehaves under orthographic projection it cannot be
rescued by perspective, and any failure is unambiguously the heuristic's.
"""

import numpy as np

# The 5 arcface keypoints, in the same order the detectors emit them:
# left eye, right eye, nose tip, left mouth corner, right mouth corner.
# Eye centres are the means of _REF3D_68[36:42] / [42:48]; nose tip is index 30;
# mouth corners are indices 48 and 54.  OpenCV convention: +x right, +y up,
# +z toward the camera.
LEFT_EYE = np.array([-0.334, 0.379, 0.170])
RIGHT_EYE = np.array([0.334, 0.379, 0.170])
NOSE_TIP = np.array([0.000, -0.098, 0.547])
MOUTH_LEFT = np.array([-0.368, -0.547, 0.299])
MOUTH_RIGHT = np.array([0.368, -0.547, 0.299])

HEAD_5PT = np.vstack([LEFT_EYE, RIGHT_EYE, NOSE_TIP, MOUTH_LEFT, MOUTH_RIGHT])


def rotation(yaw_deg, pitch_deg, roll_deg):
    y, p, r = np.radians([yaw_deg, pitch_deg, roll_deg])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    return rz @ rx @ ry


def project_kps(yaw_deg, pitch_deg=0.0, roll_deg=0.0, scale=200.0, cx=256.0, cy=256.0):
    """The 5 keypoints of a head at this pose, in image coordinates (y down)."""
    pts = HEAD_5PT @ rotation(yaw_deg, pitch_deg, roll_deg).T
    return np.column_stack([cx + scale * pts[:, 0],
                            cy - scale * pts[:, 1]]).astype(np.float32)


def decompose(matrix):
    """(uniform scale, in-plane rotation in degrees) of a 2x3 similarity."""
    a, c = matrix[0, 0], matrix[1, 0]
    return float(np.hypot(a, c)), float(np.degrees(np.arctan2(c, a)))


def fit_residual(matrix, src, dst):
    """Mean distance in pixels between the transformed src points and dst."""
    homogeneous = np.column_stack([src, np.ones(len(src))])
    return float(np.linalg.norm(homogeneous @ matrix.T - dst, axis=1).mean())
