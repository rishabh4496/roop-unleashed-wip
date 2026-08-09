"""Two people, two facesets, faces that touch — the repro bench.

The reported failure ("both faces flicker, go unswapped, or swap into each
other, but ONLY while the two people are close or touching") needs a clip where
two DIFFERENT selected persons interact, each bound to its own source faceset.
Neither existing video bench can produce it: angle_video and frontal_roll_video
both swap a single face.

What it does, in the order the app does it:

  1. ingest two .fsz facesets from the library exactly as source_gallery does;
  2. capture one target face per person from a frame where the two heads are
     clearly APART (the identity the whole run is judged against must not
     itself come from an ambiguous frame), plus optional extra angles;
  3. run the trimmed clip through `batch_process_with_options` in `selected`
     mode with the tracking pre-pass on — the app's own path;
  4. grade every output frame per person: was that person's face touched at
     all, and does what was pasted look like the faceset that was bound to
     them or like the OTHER one.

GRADING IS PER PERSON AND SIGNED. "It flickered" and "they swapped into each
other" produce the same per-frame difference against the plate, so a bench that
only measures "did this box change" cannot tell them apart. `own`/`other` are
cosine distances from the swapped crop's embedding to each faceset's mean, and
the column that matters is which of the two is smaller.

Usage:
    env/Scripts/python.exe tests/two_face_video.py --tag before
    env/Scripts/python.exe tests/two_face_video.py --tag after --start 200 --end 340
"""

import argparse
import csv
import os
import shutil
import sys
import zipfile

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import angle_bench as ab                     # noqa: E402
from angle_video import ensure_ffmpeg, read_frames    # noqa: E402

LIB = os.path.join(APP, "facesets")

# The grader's own blind-spot bar, deliberately NOT face_contact.CONTAM_MAX:
# a baseline run sets ROOP_EMB_CONTAM=0 to switch the fix off, and reading
# the production constant here would switch the guard off with it and report
# the baseline as having nothing to measure.
GRADE_CONTAM_MAX = 0.35


# ── material ─────────────────────────────────────────────────────────────────

def load_library_faceset(name):
    """Ingest `<name>.fsz` the way source_gallery._ingest_faceset does.

    NOT angle_bench.load_faceset: that one runs `prepare_plate` first, which is
    correct for the studio yaw strips it was written for (it cuts the
    neighbouring pose off the edge of a letterboxed plate) and destructive for
    an ordinary user faceset of loose photographs.
    """
    from roop.FaceSet import FaceSet
    from roop.face_util import extract_face_images

    path = os.path.join(LIB, name if name.endswith(".fsz") else name + ".fsz")
    if not os.path.exists(path):
        raise SystemExit(f"no such faceset: {path}")

    tmp = os.path.join(os.environ.get("TEMP", APP), "two_face_fs", name)
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)
    with zipfile.ZipFile(path) as z:
        z.extractall(tmp)

    fs = FaceSet()
    for fn in sorted(os.listdir(tmp)):
        if not fn.lower().endswith(".png"):
            continue
        p = os.path.join(tmp, fn)
        frame = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
        for fd in extract_face_images(p, (False, 0)):
            fd[0].mask_offsets = ab._mask_offsets()
            fs.faces.append(fd[0])
            fs.ref_images.append(frame)
    if not fs.faces:
        raise SystemExit(f"no faces found in {path}")
    if len(fs.faces) > 1:
        fs.AverageEmbeddings()
    return fs


def frame_at(video, idx):
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, fr = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"could not read frame {idx} of {video}")
    return fr


def separated_frame(video, stride=1, limit=400):
    """First frame holding exactly two faces with a clear gap between them.

    A capture taken while the heads already overlap can pick up the merged
    detection that spans both of them, and every later identity decision is
    then measured against a face that is half of each person.
    """
    from roop.face_util import get_all_faces
    cap = cv2.VideoCapture(video)
    i = -1
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        i += 1
        if i % stride or i > limit:
            continue
        faces = get_all_faces(fr) or []
        if len(faces) != 2:
            continue
        a, b = [f.bbox for f in faces]
        dx = max(b[0] - a[2], a[0] - b[2], 0.0)
        w = 0.5 * ((a[2] - a[0]) + (b[2] - b[0]))
        if dx > 0.25 * w:
            cap.release()
            return i, fr
    cap.release()
    raise SystemExit("no frame with two well-separated faces")


def capture_targets(frame, extra_angles=()):
    """TARGET_FACES / TARGET_FACE_GROUP for the two people, left person first.

    Left-to-right so the person index is stable across runs; the source
    facesets are given in the same order on the command line.
    """
    from roop.face_util import get_all_faces
    faces = sorted(get_all_faces(frame) or [], key=lambda f: float(f.bbox[0]))
    if len(faces) != 2:
        raise SystemExit(f"expected 2 faces in the capture frame, got {len(faces)}")
    targets, groups = list(faces), [0, 1]
    for fr in extra_angles:
        more = sorted(get_all_faces(fr) or [], key=lambda f: float(f.bbox[0]))
        if len(more) != 2:
            continue
        for person, f in enumerate(more):
            targets.append(f)
            groups.append(person)
    return targets, groups


def trim(video, start, end, out_path, fps=None):
    """Write frames [start, end) to a new clip (the bench's unit of work)."""
    cap = cv2.VideoCapture(video)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                         fps or src_fps, (w, h))
    if not vw.isOpened():
        raise SystemExit(f"could not open {out_path} for writing")
    i = -1
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        i += 1
        if i < start:
            continue
        if end and i >= end:
            break
        vw.write(fr)
    cap.release()
    vw.release()
    return out_path


def run_swap(clip_path, facesets, targets, groups, options, out_dir):
    import roop.globals as g
    from roop import ProcessMgr as _pm
    from roop import procmgr_runtime as _rt
    from roop.core import batch_process_with_options
    from roop.ProcessEntry import ProcessEntry

    _pm._SWAP_LOG = {}
    _rt.FACE_LOG = {}

    g.INPUT_FACESETS = list(facesets)
    g.TARGET_FACES = list(targets)
    g.TARGET_FACE_GROUP = list(groups)
    g.output_path = out_dir
    os.makedirs(out_dir, exist_ok=True)

    entry = ProcessEntry(clip_path, 0, 0, 30.0)
    before = set(os.listdir(out_dir))
    batch_process_with_options([entry], options, None)

    log = _pm._SWAP_LOG
    _pm._SWAP_LOG = None
    faces_log = _rt.FACE_LOG
    _rt.FACE_LOG = None
    log = (log, faces_log)

    if entry.finalname and os.path.exists(entry.finalname):
        return entry.finalname, log
    # `not f.startswith(".")`: the crash-resume path leaves `.clip__temp.segNNNN
    # .mp4` parts beside the finished file, and a part is a PREFIX of the clip —
    # picking one up produces a shorter video that grades as though the run
    # simply stopped swapping partway through.
    fresh = [f for f in os.listdir(out_dir)
             if f not in before and f.lower().endswith(".mp4")
             and not f.startswith(".")
             and os.path.join(out_dir, f) != clip_path]
    if not fresh:
        return None, log
    fresh.sort(key=lambda f: os.path.getmtime(os.path.join(out_dir, f)))
    return os.path.join(out_dir, fresh[-1]), log


# ── grading ──────────────────────────────────────────────────────────────────

def cos(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    n = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(1.0 - a.dot(b) / n) if n else float("nan")


def faceset_mean(fs):
    embs = []
    for f in fs.faces:
        e = getattr(f, "embedding", None)
        if e is not None:
            embs.append(np.asarray(e, np.float64).ravel())
    return np.mean(embs, axis=0) if embs else None


def plate_person(plate_faces, targets, groups, contam):
    """Which captured person each PLATE face is, or None when it cannot be said.

    The bench pairs faces left to right, which is right for position and wrong
    for identity: the detector regularly returns two boxes on ONE person (a
    duplicate, or one of them partial), and the second of those is then graded
    as "person 1" and reports the swap as the wrong faceset when the pipeline
    did exactly the right thing. Frames where a face cannot be attributed —
    a shared crop, or no clear nearest person — are not graded for identity at
    all rather than graded wrongly.
    """
    from roop.utilities import compute_cosine_distance as cd
    persons = {}
    for i, g in enumerate(groups):
        persons.setdefault(g, []).append(i)
    out = []
    for k, f in enumerate(plate_faces):
        emb = getattr(f, "embedding", None)
        if emb is None or contam[k] >= GRADE_CONTAM_MAX:
            out.append(None)
            continue
        ds = sorted((min(cd(targets[ti].embedding, emb) for ti in tis), g)
                    for g, tis in persons.items())
        # Nearest, and clearly nearest — 0.25 is the same margin the pipeline's
        # own track inheritance uses to call an identity decisive.
        if len(ds) > 1 and ds[1][0] - ds[0][0] < 0.25:
            out.append(None)
        else:
            out.append(ds[0][1])
    return out


def grade(plate, swapped, means, targets=None, groups=None):
    """One row per detected face: where it is, whether it moved, whose it looks
    like now. Detection is run on the PLATE so the two people are located by
    the untouched footage; a swap that fails is then still measured.

    The identity columns are BLANK on frames where the output face's own
    recognition crop is shared with its neighbour. That is not squeamishness:
    the contamination this whole investigation is about applies to the grader
    exactly as it applies to the pipeline, so a bench that reads identity there
    reports the two people as each other whatever the swap did, and would score
    a perfect fix as a failure. `contam` is carried in the CSV so the blanks can
    be counted rather than guessed at.
    """
    from roop.face_util import get_all_faces
    from roop.face_contact import crop_contamination
    rows = []
    faces = sorted(get_all_faces(plate) or [], key=lambda f: float(f.bbox[0]))
    swapped_faces = get_all_faces(swapped) or []
    contam = crop_contamination(swapped_faces)
    plate_contam = crop_contamination(faces)
    who = (plate_person(faces, targets, groups, plate_contam)
           if targets is not None else [None] * len(faces))
    for fi, f in enumerate(faces):
        x0, y0, x1, y1 = [int(round(v)) for v in f.bbox]
        x0, y0 = max(0, x0), max(0, y0)
        x1 = min(plate.shape[1], x1)
        y1 = min(plate.shape[0], y1)
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue
        d = float(np.abs(plate[y0:y1, x0:x1].astype(np.float32)
                         - swapped[y0:y1, x0:x1].astype(np.float32)).mean())
        # Nearest face detected in the OUTPUT to this plate box, for identity.
        best, bestd, bestc = None, 1e9, 0.0
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        for k, sf in enumerate(swapped_faces):
            sx = (float(sf.bbox[0]) + float(sf.bbox[2])) / 2.0
            sy = (float(sf.bbox[1]) + float(sf.bbox[3])) / 2.0
            dd = (sx - cx) ** 2 + (sy - cy) ** 2
            if dd < bestd:
                best, bestd, bestc = sf, dd, contam[k]
        ident = [float("nan")] * len(means)
        if (best is not None and bestd < ((x1 - x0) * 0.6) ** 2
                and bestc < GRADE_CONTAM_MAX):
            ident = [cos(best.embedding, m) if m is not None else float("nan")
                     for m in means]
        rows.append({"box": (x0, y0, x1, y1), "touched": d, "ident": ident,
                     "who": who[fi] if who else None,
                     "contam": bestc if best is not None else float("nan")})
    return rows


def applied_sources(entries):
    """A lookup from a plate face box to the source index pasted onto it.

    `entries` is one frame of ProcessMgr._SWAP_LOG. Matching is by centre
    distance against the swapped face's own box, tolerant to the small
    difference between the plate detection and the one the pipeline used.
    """
    def _c(b):
        return ((b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5)

    def lookup(box):
        cx, cy = _c(box)
        tol = 0.5 * (box[2] - box[0])
        best, bestd = "", tol * tol
        for bb, src in entries:
            sx, sy = _c(bb)
            d = (sx - cx) ** 2 + (sy - cy) ** 2
            if d < bestd:
                best, bestd = src, d
        return best
    return lookup


def face_reasons(entries):
    """Lookup from a plate face box to the audit buckets that face hit.

    One frame of procmgr_runtime.FACE_LOG. "It flickered" and "it was refused
    by gate X" look identical in the output, and only this says which.
    """
    def _c(b):
        return ((b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5)

    def lookup(box):
        cx, cy = _c(box)
        tol = 0.5 * (box[2] - box[0])
        best, bestd = "", tol * tol
        for bb, buckets in entries:
            sx, sy = _c(bb)
            d = (sx - cx) ** 2 + (sy - cy) ** 2
            if d < bestd:
                # `faces seen` is on every one of them; the informative bucket
                # is whatever else was hit.
                best = " | ".join(k for k in buckets if k != "faces seen")
                bestd = d
        return best
    return lookup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--video", default=r"G:/pinokio/roop-keep/sample1.mp4")
    ap.add_argument("--sources", default="harjot,ashna",
                    help="faceset names, left person first")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=0, help="0 = to the end")
    ap.add_argument("--capture", type=int, default=-1,
                    help="frame to capture the two target faces from; "
                         "-1 finds the first well-separated one")
    ap.add_argument("--capture-extra", default="",
                    help="comma-separated extra frames to add as further ANGLES "
                         "of the same two people, the way the app's 'Add angle "
                         "to Person N' button does. Use it to tell a contact "
                         "failure apart from a face simply being at a pose the "
                         "single capture does not cover.")
    ap.add_argument("--provider", default="cuda")
    ap.add_argument("--swap-model", default="inswapper")
    ap.add_argument("--enhancer", default="None")
    ap.add_argument("--mask-engine", default="None")
    ap.add_argument("--tracking", default="1")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(APP, "output", "bench_two_face"))
    args = ap.parse_args()

    ensure_ffmpeg()
    g = ab.init_pipeline(args.provider, args.swap_model, args.enhancer,
                         args.mask_engine)
    g.video_encoder = "libx264"
    g.video_quality = 12
    g.execution_threads = args.threads
    g.face_swap_mode = "selected"
    track = args.tracking != "0"
    # BOTH names. `_track_mode` (ProcessMgr) reads roop.globals.track_identities,
    # which api.py copies out of the config at run time and no bench ever did —
    # set only the CFG value and the identity-lock branch never runs, so the
    # bench silently measures the per-frame matcher instead.
    g.track_identities = track
    g.CFG.track_identities = track
    g.temporal_detection = track
    g.CFG.temporal_detection = track
    options = ab.build_options(g, args.swap_model, args.mask_engine, False)

    outdir = os.path.join(args.out, args.tag)
    work = os.path.join(outdir, "work")
    shutil.rmtree(outdir, ignore_errors=True)
    os.makedirs(work, exist_ok=True)

    names = [s.strip() for s in args.sources.split(",") if s.strip()]
    if len(names) != 2:
        raise SystemExit("--sources needs exactly two faceset names")
    facesets = [load_library_faceset(n) for n in names]
    means = [faceset_mean(fs) for fs in facesets]
    print(f"[bench] sources: {names[0]} ({len(facesets[0].faces)} faces), "
          f"{names[1]} ({len(facesets[1].faces)} faces)", flush=True)

    if args.capture >= 0:
        cap_idx, cap_frame = args.capture, frame_at(args.video, args.capture)
    else:
        cap_idx, cap_frame = separated_frame(args.video)
    print(f"[bench] target faces captured from frame {cap_idx}", flush=True)
    extra = [frame_at(args.video, int(x)) for x in args.capture_extra.split(",") if x.strip()]
    targets, groups = capture_targets(cap_frame, extra)
    if extra:
        print(f"[bench] plus {len(extra)} extra angle frame(s): {args.capture_extra}",
              flush=True)

    clip = trim(args.video, args.start, args.end,
                os.path.join(work, "clip.mp4"))
    plates = read_frames(clip)
    print(f"[bench] clip: {len(plates)} frames "
          f"[{args.start}..{args.start + len(plates)})", flush=True)

    out, (swap_log, face_log) = run_swap(clip, facesets, targets, groups, options, work)
    if not out:
        raise SystemExit("no output produced")
    swapped = read_frames(out)
    n = min(len(plates), len(swapped))
    print(f"[bench] output: {len(swapped)} frames", flush=True)

    rows = []
    for i in range(n):
        applied = applied_sources(swap_log.get(i) or [])
        reason = face_reasons(face_log.get(i) or [])
        for person, r in enumerate(grade(plates[i], swapped[i], means,
                                         targets, groups)):
            own = r["ident"][person] if person < len(r["ident"]) else float("nan")
            other = r["ident"][1 - person] if len(r["ident"]) > 1 else float("nan")
            rows.append({
                "frame": args.start + i, "person": person,
                "x0": r["box"][0], "y0": r["box"][1],
                "x1": r["box"][2], "y1": r["box"][3],
                "touched": round(r["touched"], 3),
                "contam": round(r["contam"], 3) if r["contam"] == r["contam"] else "",
                "src": applied(r["box"]),
                "who": "" if r["who"] is None else r["who"],
                "why": reason(r["box"]),
                "own": round(own, 4) if own == own else "",
                "other": round(other, 4) if other == other else "",
            })

    csv_path = os.path.join(outdir, "rows.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\n[bench] {csv_path}")
    for person in (0, 1):
        rs = [r for r in rows if r["person"] == person]
        if not rs:
            continue
        t = np.array([r["touched"] for r in rs])
        sw = t > 2.0                       # bimodal: ~1.5 untouched vs ~6.5 swapped
        graded = [r for r, s in zip(rs, sw) if s and r["own"] != "" and r["other"] != ""]
        looks_other = sum(1 for r in graded if r["other"] < r["own"])
        # The decision, not a re-measurement: which faceset the pipeline
        # actually put on this person's face. Unlike the cosine columns this
        # is exact on contact frames, which is the only place it matters.
        # Only frames where the PLATE face is confidently this person. A
        # duplicate box on the other person graded here is how a correct run
        # reports an inter-swap that never happened.
        got = [r for r in rs if r["src"] != "" and str(r["who"]) == str(person)]
        wrong_src = sum(1 for r in got if r["src"] != person)
        flips = sum(1 for a, b in zip(sw[:-1], sw[1:]) if a != b)
        print(f"  person {person} ({names[person]}): {len(rs)} frames, "
              f"swapped {int(sw.sum())} ({100.0*sw.mean():.1f}%), "
              f"on/off transitions {flips}")
        print(f"      WRONG FACESET APPLIED on {wrong_src} of {len(got)} swapped "
              f"(from the pipeline's own decision)")
        print(f"      output re-measured as the other person on {looks_other} of "
              f"{len(graded)} gradable frames")


if __name__ == "__main__":
    main()
