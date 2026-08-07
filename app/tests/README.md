# Geometry tests

```
cd app
env/Scripts/python -m unittest discover -s tests -t .    # Windows
python -m unittest discover -s tests -t .                # anywhere else
```

50 tests, ~2 s. No GPU, no models, no video, no network — pure numpy/cv2 over a
synthetic head projected at a known pose (`tests/facegeom.py`), using the
project's own 3-D reference face so the tests and the shipping pose code agree
on what a head is shaped like.

Plain `unittest`, so there is nothing to install. `pytest` also discovers these
if you happen to have it.

## What is covered, and why

These target the alignment/masking geometry, which is where the recurring bugs
have been: silent, pose-dependent, and invisible on frontal footage.

| File | Guards |
|------|--------|
| `test_pose_ratios.py` | `kps_pose_ratios` is **monotonic** in yaw to 90°, roll-invariant, and returns `(None, None)` rather than NaN on degenerate keypoints |
| `test_alignment.py` | `estimate_norm` is a **bit-exact no-op** when `yaw_align` is off (1350 size × template × pose combinations); the profile path is gated off mid-angles and kills the nod-coupled rotation swing |
| `test_landmark_mask.py` | `create_landmark_mask` is **bit-identical** on upright faces and **rotationally equivariant** on tilted ones; `_mask_crop_box` never returns a degenerate rectangle |
| `test_face_overlap.py` | Two swapped faces that touch get a **boundary**, not a smear: separated faces cost nothing, the sequential composite leaves no plate showing along the join, the boundary does not move with match order, and paint order is far-to-near and stable under bbox noise |
| `test_angle_handling.py` | The three shared angle layers: the visibility polygon **never trims real face** at any pose, is off entirely near frontal, and shrinks monotonically with yaw; the off-axis fade is bounded, monotone and exactly off at 0; the trim is applied **after** the feather (before it, shrinking the matte shrank the feather 29px→18px and *sharpened* the seam); both paste call sites pass it; the default path is bit-identical |
| `test_settings_wiring.py` | A setting that reaches the UI actually reaches the render — no half-wired controls that silently do nothing |

`test_settings_wiring.py` parses `FaceSwap.jsx` and `api.py` as text rather than
importing them, which is how it can check across the Python/JavaScript boundary
with no browser and no running server. Its exception lists (`VIA_MASK_OFFSET_HELPER`,
`PREVIEW_ONLY`, `RUN_ONLY_TEMPORAL`, `RUN_ONLY_OUTPUT`) each carry the reason that
setting is legitimately absent from the other side — if you add to them, add the
reason too.

Two guarantees are load-bearing and easy to break by accident:

- **`yaw_align` off must be bit-exact.** `pose` is now the default, but `off` is
  the A/B baseline and the only way back to the pre-change output, so it has to
  stay a true no-op. The same applies to the other two angle layers: strength 0
  and the trim disabled must both be bit-identical to not having them.
- **Upright faces must be bit-identical.** Frontal is the overwhelming majority
  of frames, so any mask change there is a regression in the common case.

## Refactor safety harness

`surface_snapshot.py` is a tool, not a test. It records everything a caller
could depend on — every module-level name, every class method, every signature,
and for `api.py` the full FastAPI route table including declaration order — so a
decomposition can be *proven* to be a pure move rather than asserted to be one:

```
env/Scripts/python tests/surface_snapshot.py before.json
# ...move code...
env/Scripts/python tests/surface_snapshot.py after.json
env/Scripts/python tests/surface_snapshot.py --diff before.json after.json
```

Route order matters: FastAPI matches in declaration order, so a reordering can
silently shadow a handler without changing any path. The diff reports that
case separately.

Use it on `api.py` and `ProcessMgr.py`, which have no behavioural tests. A clean
surface diff plus a green suite is what makes "I only moved code" checkable.

## Style

Assert **properties**, not magic numbers. The bug that motivated this suite was a
heuristic that looked fine at the angles anyone tested and collapsed to exactly
zero at 90° — a threshold test would have passed. `test_monotonic_in_yaw` checks
monotonicity across the whole range instead, which is the property that was
actually violated.

Where a test encodes a known limitation rather than a requirement
(`test_pitch_ratio_degenerates_at_profile`), it says so and explains what to do
if it ever starts failing.
