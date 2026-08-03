"""Pre-pass computation for ProcessMgr: identity tracks, temporal detection,
keypoint stabilisation and SAM2 mask precomputation.

Everything here runs BEFORE the swap loop and produces per-frame lookups the
loop consumes, which is why it forms a coherent slice: the swap pass then does
no detection of its own for tracked runs.

A mixin, so the method bodies move verbatim and `self` is unchanged.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty as _QueueEmpty, Full as _QueueFull
from threading import Thread
from collections import deque as _deque

import cv2
import numpy as np

from roop.procmgr_runtime import (_DEBUG_MATCH, _TRACK_EMB_MAX, _TRACK_ASSIGN_MAX,
                                  _TRACK_ASSIGN_MARGIN, _TRACK_ASSIGN_FLOOR,
                                  _TRACK_REID_MAX,
                                  _INTERP_MAX_TRAVEL, _INTERP_MAX_SCALE)

import roop.globals
from roop import session_pool
from roop.face_util import get_all_faces, analysis_pooled
from roop.utilities import compute_cosine_distance
from roop.procmgr_runtime import _prof, _gpu_guard, wait_while_paused, PROGRESS_BAR_FORMAT, _TRACK_OVERLAP_FRAC, ChunkedProgress, bar_write, publish_eta


class TrackingMixin:
    def _precompute_sam2(self, sam2_p, source_video, frame_start, frame_end, frame_count):
        """SAM2 pre-pass: dump the trimmed frames to a temp JPEG dir (0-based,
        matching the swap reader's frame_idx), detect the faces on frame 0 to seed
        the tracker, and let SAM2 propagate full-frame masks across the clip."""
        import tempfile, shutil
        from roop.face_util import get_all_faces

        tmp = tempfile.mkdtemp(prefix='sam2_')
        try:
            cap = cv2.VideoCapture(source_video)
            try:
                if frame_start and frame_start > 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
                first = None
                idx = 0
                while roop.globals.processing:
                    ret, fr = cap.read()
                    if not ret or fr is None:
                        break
                    if first is None:
                        first = fr
                    cv2.imwrite(os.path.join(tmp, f'{idx:06d}.jpg'), fr)
                    idx += 1
                    if frame_count and idx >= frame_count:
                        break
            finally:
                cap.release()

            if first is None or idx == 0:
                sam2_p.precomputed = {}
                return

            with _gpu_guard(pooled=analysis_pooled()):
                faces = get_all_faces(first) or []
            boxes = [f.bbox.astype(np.float32) for f in faces if getattr(f, 'bbox', None) is not None]
            print(f'[SAM2] seeding tracker with {len(boxes)} face(s) over {idx} frames')
            h, w = first.shape[:2]
            sam2_p.precompute(tmp, boxes, (h, w))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    @staticmethod
    def _bbox_iou(a, b):
        ax0, ay0, ax1, ay1 = a; bx0, by0, bx1, by1 = b
        ix0, iy0 = max(ax0, bx0), max(ay0, by0)
        ix1, iy1 = min(ax1, bx1), min(ay1, by1)
        iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
        return inter / ua if ua > 0 else 0.0

    def _precompute_tracks(self, source_video, frame_start, frame_end, frame_count,
                           awebp_frames=None, step=3, collect_obs=False,
                           desc='Tracking identities'):
        """Identity-lock pass 1: build tracklets (IoU + embedding association)
        across the clip, assign each tracklet to ONE source via its mean embedding,
        and store {frame_idx: [(bbox_centroid, src_index, emb_mean), ...]} for pass 2 to look
        up by nearest centroid and embedding similarity — so a person keeps the same source
        for the whole clip instead of being re-matched (and possibly flipped) every frame.

        Also serves as the shared scan for the temporal-detection pre-pass:
        step=1 detects every frame, collect_obs=True stores each track's Face
        observations ({frame_idx: Face} under track['obs']), and awebp_frames
        feeds pre-decoded animated-WebP frames instead of a VideoCapture.
        Returns the full track list (active + retired)."""
        import os
        from roop.face_util import get_all_faces, get_all_faces_in_roi

        # Skip-frames step (N=3 runs detection on 33% of frames; N=1 scans all)
        TRACK_STEP = max(1, int(step))

        # Opt-in: when exactly one track is active, detect within a padded crop
        # around its predicted bbox instead of the full frame. Same detector
        # canvas size -> same compute, but the tracked face fills much more of
        # it, improving recall on rotated/angled faces. Falls back to a
        # full-frame detect on a miss (occlusion, fast motion, re-entry), so it
        # never loses a face the old full-frame path would have found. Skipped
        # entirely with 0 or >1 active tracks to avoid extra detector calls in
        # multi-face scenes (kept identical to today's full-frame behaviour there).
        ROI_CROP = os.environ.get('ROOP_TRACK_ROI_CROP', '0') == '1'

        # active = tracks seen within STALE frames (candidates for matching);
        # retired = older tracks, kept only for the final source assignment. This
        # keeps the per-frame match loop small on long clips. emb_mean is updated
        # via EMA with outlier filtering.
        active, retired = [], []
        next_id = 0
        per_frame = {}       # frame_idx -> [(centroid(2,), track_id)]
        # Detections the appearance-only fallback was not allowed to claim.
        reid_refused = 0
        # EMB_MAX is shared with the swap-time re-association in ProcessMgr so the
        # two halves of the tracker cannot drift apart (see _TRACK_EMB_MAX).
        # REID_MAX is the tighter bar for association WITHOUT spatial evidence.
        IOU_MIN, EMB_MAX, STALE = 0.2, (_TRACK_EMB_MAX or 0.7), 15
        REID_MAX = _TRACK_REID_MAX if _TRACK_REID_MAX > 0 else EMB_MAX
        print(f'[Track] {desc}: scanning frames (step={TRACK_STEP})...')

        def _predict_bbox(t, f_idx):
            """Project a track's last bbox forward by its linear velocity to
            estimate where it should be at f_idx. Shared by _consume's
            detection-to-track association and (opt-in) ROI-crop detection."""
            predicted_bbox = t['bbox']
            dt = f_idx - t['last_seen']
            if 0 < dt <= 6 and t['vel'] is not None and np.any(t['vel']):
                proj = t['bbox'] + t['vel'] * dt
                if proj[2] > proj[0] and proj[3] > proj[1]:
                    predicted_bbox = proj
            return predicted_bbox

        # Terminal progress bar (same style as the swap phase) so the pre-pass is
        # visible in the console too, not just the web UI.
        _bar_fmt = PROGRESS_BAR_FORMAT
        pbar = ChunkedProgress(total=frame_count or 0, desc=desc, unit='frames',
                               dynamic_ncols=True, bar_format=_bar_fmt)

        # Parallelize detection across the FaceAnalysis pool (see analysis_pooled()/
        # lease_face_analyser() in face_util.py — "detection is ~43% of video time";
        # each pool worker leases its own independent instance). This pre-pass used
        # to always call get_all_faces() from this single thread, so ROOP_DETMASK_POOL
        # never sped it up even when enabled — only the swap phase benefited from it.
        # Frame decode stays sequential (one VideoCapture can't be read from multiple
        # threads), but up to pool_workers detections now run concurrently, off the
        # critical path. _consume() still runs in strict frame order (via the FIFO
        # in_flight queue below), so the tracking result is bit-identical to the
        # serial path — only the wall-clock schedule of the GPU calls changes. Falls
        # back to the exact original single-threaded call when pooling is off.
        pool_workers = session_pool.detmask_pool_size() if session_pool.detmask_pooling_enabled() else 1
        det_executor = (ThreadPoolExecutor(max_workers=pool_workers, thread_name_prefix='track_det')
                        if pool_workers > 1 else None)

        def _run_detect(fr, crop_bbox):
            if crop_bbox is not None:
                faces = get_all_faces_in_roi(fr, crop_bbox)
                if faces:
                    return faces
            return get_all_faces(fr) or []

        def _detect_one(fr, crop_bbox=None):
            # Runs inside a pool worker, one at a time per worker (ThreadPoolExecutor
            # caps concurrency at pool_workers == the analyser pool size), so this is
            # real GPU/model time, not queue-wait — lease_face_analyser() should never
            # actually block here. Tagged 'track_detect' (not 'detect') so it shows up
            # as its own STAGE TIMING line, separate from the swap phase's detect stage.
            with _prof('track_detect'), _gpu_guard(pooled=True):
                return _run_detect(fr, crop_bbox)

        def _consume(f_idx, faces):
            nonlocal active, retired, next_id, reid_refused
            # Retire tracks not seen for STALE frames so matching stays O(active).
            if active:
                fresh = []
                for t in active:
                    (fresh if t['last_seen'] >= f_idx - STALE else retired).append(t)
                active = fresh
            entries, used = [], set()
            for face in faces:
                bbox = np.asarray(face.bbox, dtype=np.float32)
                emb = np.asarray(face.embedding, dtype=np.float32)
                best, best_score = None, -1.0
                for t in active:
                    if t['id'] in used:
                        continue
                    predicted_bbox = _predict_bbox(t, f_idx)
                    iou = self._bbox_iou(bbox, predicted_bbox)
                    if iou < IOU_MIN:
                        continue

                    cos_dist = compute_cosine_distance(t['emb_mean'], emb)
                    if cos_dist > EMB_MAX:
                        continue

                    # Score: Higher IoU and lower Cosine Distance is better
                    score = iou * (1.0 - cos_dist)
                    if score > best_score:
                        best, best_score = t, score

                is_reid = False
                if best is None:
                    # Re-ID lookup: search active (not yet matched this frame) and
                    # retired tracklets for returning/moved faces. Runs once
                    # spatial continuity is lost, so it cannot use IoU — but the
                    # two things that get here are not equally well evidenced,
                    # and the bar follows the evidence:
                    #
                    #  * an ACTIVE track was seen within STALE frames. Something
                    #    is briefly wrong with the geometry — an object or another
                    #    face crossing the subject, motion blur, a fast turn, a
                    #    partial detection whose box shrank off the predicted one.
                    #    Recency stands in for the missing IoU, and the appearance
                    #    is degraded by the very event we are trying to survive,
                    #    so this keeps the primary path's EMB_MAX. Tightening it
                    #    here is what breaks occluded frames out into tracks of
                    #    their own and blinks the swap off exactly when something
                    #    passes in front of the face.
                    #
                    #  * a RETIRED track has not been seen for STALE frames. No
                    #    spatial evidence, no temporal evidence, and a face that
                    #    has been gone that long is generally re-acquired
                    #    UNoccluded, so there is no hard-frame allowance to make.
                    #    This is the claim that let a newcomer inherit a track —
                    #    see _TRACK_REID_MAX — and it takes the tighter bar.
                    best_reid, best_reid_dist = None, float('inf')
                    is_retired = False
                    near_miss = False

                    for t in active:
                        if t['id'] in used:
                            continue
                        dist = compute_cosine_distance(t['emb_mean'], emb)
                        if dist <= EMB_MAX and dist < best_reid_dist:
                            best_reid, best_reid_dist = t, dist
                            is_retired = False

                    for t in retired:
                        dist = compute_cosine_distance(t['emb_mean'], emb)
                        if dist > REID_MAX:
                            # Would have been claimed under the old shared bar.
                            near_miss = near_miss or dist <= EMB_MAX
                            continue
                        if dist < best_reid_dist:
                            best_reid, best_reid_dist = t, dist
                            is_retired = True

                    if best_reid is None and near_miss:
                        # Not lost: the face starts a track of its own below,
                        # which the source assignment then judges on its own mean.
                        reid_refused += 1

                    if best_reid is not None:
                        best = best_reid
                        if is_retired:
                            retired.remove(best)
                            active.append(best)
                        is_reid = True

                if best is None:
                    best = {
                        'id': next_id,
                        'bbox': bbox,
                        'prev_bbox': None,
                        'vel': np.zeros(4, dtype=np.float32),
                        'emb_sum': emb.astype(np.float64).copy(),
                        'emb_n': 1,
                        'emb_mean': emb.copy(),
                        'first_seen': f_idx,
                        'last_seen': f_idx
                    }
                    next_id += 1
                    active.append(best)
                else:
                    dt = f_idx - best['last_seen']
                    if dt > 0 and not is_reid:
                        best['vel'] = (bbox - best['bbox']) / dt
                        best['prev_bbox'] = best['bbox']
                    elif is_reid:
                        best['vel'] = np.zeros(4, dtype=np.float32)
                        best['prev_bbox'] = None
                    best['bbox'] = bbox
                    best['last_seen'] = f_idx

                    # Outlier filter: only update mean embedding if clean enough (distance <= 0.5)
                    dist = compute_cosine_distance(best['emb_mean'], emb)
                    if dist <= 0.5:
                        alpha = 0.25
                        best['emb_mean'] = ((1.0 - alpha) * best['emb_mean'] + alpha * emb).astype(np.float32)
                        best['emb_sum'] += emb
                        best['emb_n'] += 1

                used.add(best['id'])
                if collect_obs:
                    # Keep the full Face for this frame — the temporal
                    # pre-pass gap-fills/smooths these and the swap pass
                    # consumes them directly.
                    best.setdefault('obs', {})[f_idx] = face
                centroid = np.array([(bbox[0] + bbox[2]) * 0.5,
                                     (bbox[1] + bbox[3]) * 0.5], np.float32)
                entries.append((centroid, best['id']))
            per_frame[f_idx] = entries

        cap = None
        frame_iter = None
        if awebp_frames is not None:
            subset = awebp_frames[frame_start:frame_end] if frame_end > frame_start else awebp_frames[frame_start:]
            frame_iter = iter(subset)
        else:
            cap = cv2.VideoCapture(source_video)
            from roop.nvdec_reader import wrap_capture
            cap = wrap_capture(cap, source_video,
                               int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                               int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                               cap.get(cv2.CAP_PROP_FPS), tag='track decode')
        # (frame_idx, Future) pairs, oldest-submitted first — bounded to pool_workers
        # and always drained in this order, so consumption stays in frame order.
        in_flight = _deque()
        # ── Read-ahead decode ────────────────────────────────────────────────
        # Decoding used to run on this thread, in series with the wait for the
        # oldest detection future, so the loop cost was decode + wait per frame
        # even though the two use completely different hardware. Profiled on a
        # 35,770-frame clip: track_decode 4.93ms + track_wait 5.88ms = 10.8ms a
        # frame, against a detection pool that completed one every ~9.2ms — i.e.
        # roughly 15% of the pre-pass was the detector idling while this thread
        # pulled the next frame off the decoder. A one-thread read-ahead with a
        # small bounded queue overlaps them; frames still arrive in order
        # (single reader), so the scan is bit-identical. track_decode now times
        # the wait FOR a decoded frame, so it still reads as the ceiling if the
        # decoder is genuinely the slower half. ROOP_TRACK_READAHEAD=0 reverts.
        readahead = cap is not None and os.environ.get('ROOP_TRACK_READAHEAD', '1') != '0'
        frame_q = None
        reader = None
        reader_stop = False

        def _read_loop():
            try:
                while not reader_stop and roop.globals.processing:
                    ret_, fr_ = cap.read()
                    if not ret_ or fr_ is None:
                        break
                    while not reader_stop and roop.globals.processing:
                        try:
                            frame_q.put(fr_, timeout=0.25)
                            break
                        except _QueueFull:
                            continue        # consumer is behind (or paused) — hold
                    else:
                        return
            except Exception:
                pass
            finally:
                try:
                    frame_q.put(None, timeout=1.0)   # EOF sentinel
                except Exception:
                    pass

        try:
            if cap is not None and frame_start and frame_start > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
            if readahead:
                # Only a handful of frames of lead is needed to cover the jitter
                # between decode and detect, and these are full-resolution: a
                # deep queue would cost hundreds of MB on 4K material for no
                # extra throughput.
                frame_q = Queue(maxsize=6)
                _t = Thread(target=_read_loop, daemon=True)
                _t.start()
                # Only publish the handle once the thread is actually running:
                # join() on a thread that failed to start raises, which would
                # come out of the finally block and mask the real failure.
                reader = _t
            idx = 0
            while roop.globals.processing:
                wait_while_paused()
                if not roop.globals.processing:
                    break
                if frame_count and idx >= frame_count:
                    break
                if frame_iter is not None:
                    frame = next(frame_iter, None)
                    if frame is None:
                        break
                elif readahead:
                    with _prof('track_decode'):
                        frame = None
                        while roop.globals.processing:
                            try:
                                frame = frame_q.get(timeout=0.25)
                                break
                            except _QueueEmpty:
                                continue
                    if frame is None:
                        break
                else:
                    with _prof('track_decode'):
                        ret, frame = cap.read()
                    if not ret or frame is None:
                        break

                # Feed the UI's live view during the pre-pass too. This stage is a
                # third of a long run's wall clock and it used to leave the
                # processing box frozen on whatever the swap phase last published
                # — or empty, on the first job of a batch — which reads as a hang
                # rather than as work. These are the SOURCE frames: there is no
                # swapped output yet, and showing the footage being scanned is
                # what makes the progress legible.
                #
                # Published before the skip check so the view keeps moving at any
                # ROOP_TEMPORAL_STEP. Costs nothing per frame: live_preview
                # throttles to one publish every ROOP_LIVE_PREVIEW_MS and only
                # while the UI is actually fetching, so a skipped publish is a
                # clock read (~0.27us).
                self._publish_live(frame)

                # Skip frames to speed up detection and save memory
                if idx > 0 and idx % TRACK_STEP != 0:
                    idx += 1
                    pbar.update(1)
                    continue

                crop_bbox = _predict_bbox(active[0], idx) if ROI_CROP and len(active) == 1 else None

                if det_executor is not None:
                    in_flight.append((idx, det_executor.submit(_detect_one, frame, crop_bbox)))
                    if len(in_flight) >= pool_workers:
                        done_idx, done_fut = in_flight.popleft()
                        # If this blocks waiting on done_fut, all pool_workers are busy
                        # and the reader/consumer has caught up to the dispatch cap --
                        # i.e. detection itself (track_detect above), not this wait, is
                        # the real ceiling. Timed separately so STAGE TIMING shows which.
                        with _prof('track_wait'):
                            result = done_fut.result()
                        with _prof('track_consume'):
                            _consume(done_idx, result)
                else:
                    with _prof('track_detect'), _gpu_guard(pooled=analysis_pooled()):
                        faces = _run_detect(frame, crop_bbox)
                    with _prof('track_consume'):
                        _consume(idx, faces)

                idx += 1
                pbar.update(1)   # terminal bar
                # Drive the UI progress bar so the pre-pass isn't a silent black box.
                if self.progress_gradio is not None and (idx % 10 == 0 or idx == 1):
                    tot = frame_count or idx
                    # The pre-pass is minutes long and has its own rate, so it
                    # publishes its own ETA too — otherwise the UI would show a
                    # stale swap-stage figure through the whole of it.
                    publish_eta(pbar)
                    self.progress_gradio((idx, tot), desc=desc,
                                         total=tot, unit='frames')
                if frame_count and idx >= frame_count:
                    break
            # Drain any detections still in flight, in submission (frame) order.
            while in_flight:
                done_idx, done_fut = in_flight.popleft()
                _consume(done_idx, done_fut.result())
        finally:
            pbar.close()
            if det_executor is not None:
                det_executor.shutdown(wait=False, cancel_futures=True)
            # Stop the reader and let it leave cap.read() BEFORE releasing the
            # capture — releasing under an in-flight read is a native-level crash,
            # and the loop exits early on Stop far more often than it hits EOF.
            reader_stop = True
            if reader is not None:
                if frame_q is not None:
                    try:
                        while True:
                            frame_q.get_nowait()     # unblock a producer parked on put()
                    except _QueueEmpty:
                        pass
                reader.join(timeout=5.0)
            if cap is not None:
                cap.release()

        # How many frames the reader actually got through. The loop above exits on
        # the FIRST unreadable frame (`not ret`), which on a truncated/odd stream or
        # a decoder that behaves differently on another machine can happen long
        # before frame_count — leaving the caller with tracks that only cover a
        # prefix of the clip. The temporal pre-pass needs to know that, because
        # "no entry for this frame" is otherwise indistinguishable from "scanned it,
        # nobody was there".
        self._track_scanned = idx

        tracks = active + retired
        # Finalize each track's identity vector: replace the ONLINE EMA (alpha
        # 0.25, i.e. effectively the last ~10 observations) with the TRUE mean
        # over every accepted observation. The EMA is right for association
        # DURING the scan — a track should match what the face looks like now —
        # but as a whole-track identity it is recency-biased: a track that ends
        # on a turned head / bad pose drifts far from the captured target face
        # and fails the distance gate below, so that person is never assigned a
        # source and never gets swapped (measured on a 2-person clip: person 2's
        # track sat at 0.659 by EMA vs 0.361 by true mean, against a 0.6 gate —
        # the whole person silently stayed unswapped). emb_sum/emb_n were
        # already being accumulated for exactly this and were simply unused.
        # ROOP_TRACK_TRUEMEAN=0 restores the old recency-biased behavior.
        if os.environ.get('ROOP_TRACK_TRUEMEAN', '1') != '0':
            for t in tracks:
                n = int(t.get('emb_n') or 0)
                if n > 1 and t.get('emb_sum') is not None:
                    t['emb_mean'] = (np.asarray(t['emb_sum'], np.float64) / float(n)).astype(np.float32)

        # Assign each track to a source (person rank), once, by mean embedding.
        track_map = {t['id']: t for t in tracks}
        track_src, assign_max, refused_margin = self._assign_track_sources(tracks)

        self._track_assignments = {
            f: [(c, track_src.get(tid), track_map[tid]['emb_mean']) for (c, tid) in lst] for f, lst in per_frame.items()
        }
        persons = self._person_angle_indices()
        if _DEBUG_MATCH:
            for t in tracks:
                dd = {}
                for g, tis in persons.items():
                    embs = [getattr(self.target_face_datas[ti], 'embedding', None) for ti in tis]
                    embs = [e for e in embs if e is not None]
                    if embs and t.get('emb_mean') is not None:
                        dd[g] = round(min(compute_cosine_distance(e, t['emb_mean']) for e in embs), 3)
                bar_write(f"[TRACKASSIGN] track {t['id']}: frames={len(t.get('obs', {}))} "
                          f"obs={t.get('emb_n')} d(person)={dd} assign_max={assign_max} "
                          f"-> src={track_src.get(t['id'])}")
        matched = sum(1 for v in track_src.values() if v is not None)
        print(f'[Track] {len(tracks)} tracks over {len(per_frame)} frames, '
              f'{matched} matched to a source (gate {assign_max:.2f})'
              + (f', {refused_margin} refused as too far from their person\'s '
                 f'closest track (margin {_TRACK_ASSIGN_MARGIN})' if refused_margin else '')
              + (f'; {reid_refused} detections refused by the appearance-only '
                 f'fallback (Re-ID gate {REID_MAX})' if reid_refused else ''))
        return tracks

    def _person_angle_indices(self):
        """person group id -> the indices of its captured target angles."""
        groups = self.target_face_groups
        persons = {}
        for i, g in enumerate(groups[:len(self.target_face_datas)]):
            persons.setdefault(g, []).append(i)
        return persons

    def _assign_track_sources(self, tracks):
        """Bind each tracklet to at most one source (person rank) by mean embedding.

        Returns (track_src, assign_max, refused_by_margin).

        Three gates, in the order they run:

        1. ABSOLUTE (_TRACK_ASSIGN_MAX): the track's mean must be close enough to
           one of the person's captured angles at all. Tighter than per-frame
           matching because this decision is durable — see the constant.
        2. CONCURRENCY (_TRACK_OVERLAP_FRAC): a track that runs at the same time
           as one this person already owns is a second body, not a handoff.
        3. MARGIN (_TRACK_ASSIGN_MARGIN): a person's later tracks must sit near
           its FIRST (closest) one. Gate 2 is silent about a track that never
           coincides with the target — which is exactly a bystander's track
           fragment during a stretch where the target is off screen, the case
           where the wrong face was inheriting the swap.

        A track refused by any gate keeps its frames; they fall through to
        per-frame matching in the swap loop, so a genuine face still swaps — it
        just doesn't get identity locking.
        """
        groups = self.target_face_groups
        uniq = sorted(set(groups)) if groups else []
        rank = {g: r for r, g in enumerate(uniq)}
        single_person = len(uniq) <= 1
        threshold = self.options.face_distance_threshold

        persons = self._person_angle_indices()

        # Binding a track to a source is durable and is decided from the track's
        # MEAN embedding, so it is gated tighter than one-frame matching — the
        # loose per-frame threshold exists to carry a single bad frame, not to
        # hand a stretch of frames to a track that merely resembles the target.
        # See _TRACK_ASSIGN_MAX. A refused track still swaps via per-frame
        # matching; it just doesn't get identity locking.
        assign_max = min(threshold, _TRACK_ASSIGN_MAX) if _TRACK_ASSIGN_MAX > 0 else threshold

        candidates = []
        track_map = {t['id']: t for t in tracks}
        for t in tracks:
            t_emb = t.get('emb_mean')
            if t_emb is None:
                continue
            for g, tis in persons.items():
                embs = [getattr(self.target_face_datas[ti], 'embedding', None) for ti in tis]
                embs = [e for e in embs if e is not None]
                if not embs:
                    continue
                d = min(compute_cosine_distance(e, t_emb) for e in embs)
                if d <= assign_max:
                    candidates.append((d, g, t['id']))

        candidates.sort(key=lambda c: c[0])

        # Frames already claimed by this person. Two representations because the
        # per-frame observations only exist when the caller asked for them:
        # `collect_obs=True` is the temporal pre-pass, but the STANDALONE identity
        # -tracking pass (the `_precompute_tracks(...)` call with defaults) leaves
        # `obs` empty. Keying the guard on obs alone therefore made it a no-op in
        # exactly that mode — every track's frame set was empty, overlap was
        # always 0, and two concurrent tracks both received the same person's
        # source: the inter-swapping this guard exists to prevent. Fall back to
        # the track's [first_seen, last_seen] span there, which needs no
        # per-frame storage and answers the same question ("were these two on
        # screen at the same time?").
        person_assigned_frames = {g: set() for g in persons}
        person_assigned_spans = {g: [] for g in persons}
        # Distance of the closest track this person accepted. candidates is sorted
        # ascending, so the first acceptance is that person's best evidence.
        person_anchor = {}
        track_src = {t['id']: None for t in tracks}
        refused_margin = 0

        for d, g, tid in candidates:
            if track_src[tid] is not None:
                continue
            # Gate 3 — checked before the concurrency work because it is a
            # comparison of two floats and refuses outright. Never tighter than
            # _TRACK_ASSIGN_FLOOR: a very good anchor must not turn the margin
            # into a stricter gate than anything else applies to this person.
            if (_TRACK_ASSIGN_MARGIN > 0 and g in person_anchor
                    and d > max(person_anchor[g] + _TRACK_ASSIGN_MARGIN,
                                _TRACK_ASSIGN_FLOOR)):
                refused_margin += 1
                continue
            t = track_map[tid]
            obs = t.get('obs') or {}
            # One person can't be in two places at once, so a track that runs
            # CONCURRENTLY with one already given to this person is someone else.
            # But a handoff — the same person's track breaking and restarting over
            # an occlusion or a turn — overlaps by only a frame or two, and
            # rejecting those fragments leaves them with no source at all: every
            # one of their frames falls to per-frame matching and the person
            # flickers in and out. Require a real overlap, not an incidental one.
            if obs:
                t_frames = set(obs.keys())
                t_len = max(1, len(t_frames))
                overlap = len(person_assigned_frames[g].intersection(t_frames))
            else:
                lo = int(t.get('first_seen', t.get('last_seen', 0)))
                hi = int(t.get('last_seen', lo))
                t_len = max(1, hi - lo + 1)
                overlap = sum(max(0, min(hi, b) - max(lo, a) + 1)
                              for a, b in person_assigned_spans[g])
            if overlap and overlap > _TRACK_OVERLAP_FRAC * t_len:
                continue
            track_src[tid] = self.options.selected_index if single_person else rank[g]
            person_anchor.setdefault(g, d)
            if obs:
                person_assigned_frames[g].update(t_frames)
            else:
                person_assigned_spans[g].append((lo, hi))

        return track_src, assign_max, refused_margin

    def _precompute_temporal(self, source_video, awebp_frames, frame_start, frame_end, frame_count):
        """Temporal detection pre-pass (anti-flicker).

        Runs the tracked scan at step=1 collecting every frame's Face objects,
        then per track:
          - gap-fill: linearly interpolate bbox/kps/landmarks across detection
            misses of up to ROOP_TEMPORAL_GAP frames (default 10), so a face
            that blinks out of detection for a few frames keeps being swapped;
          - smoothing: when "Stabilize face" is on, run kps/lm106/bbox through
            the configured One Euro/EMA filter sequentially over the track
            (subsumes the kps-only 2-pass, and additionally covers the mask
            hull + mouth-restore landmarks, so mask/mouth edges stop shimmering).

        swap_faces then reads self._temporal_faces[frame_idx] instead of
        re-detecting — the swap pass stays fully multi-threaded and per-frame
        detection cost leaves the hot loop entirely. The scan also fills
        self._track_assignments, so identity locking rides along for free."""
        try:
            gap_max = int(os.environ.get('ROOP_TEMPORAL_GAP', '10') or '10')
        except ValueError:
            gap_max = 10
        # Scan stride. The pre-pass is detection-bound — profiled at ~37ms per
        # detect across a 4-instance pool, which on a long clip is the single
        # biggest block of wall-clock outside the swap itself — and stepping the
        # scan divides that cost directly. It stays at 1 by default because the
        # gap-fill that covers the skipped frames is a LINEAR interpolation:
        # fine while a head moves steadily, visibly behind on a fast turn. Raise
        # it only for footage without quick motion. Stepping past the gap limit
        # would leave the skipped frames with no faces at all, so it is capped.
        try:
            scan_step = max(1, int(os.environ.get('ROOP_TEMPORAL_STEP', '1') or '1'))
        except ValueError:
            scan_step = 1
        if scan_step > gap_max:
            print(f'[Temporal] scan step {scan_step} exceeds the gap limit {gap_max} — '
                  f'clamping to {gap_max}, or the skipped frames would not be filled.')
            scan_step = gap_max
        if scan_step > 1:
            print(f'[Temporal] scanning every {scan_step} frames; the rest are interpolated.')
        self._track_scanned = 0
        tracks = self._precompute_tracks(source_video, frame_start, frame_end, frame_count,
                                         awebp_frames=awebp_frames, step=scan_step, collect_obs=True,
                                         desc='Analyzing faces')
        self._temporal_faces = self._build_temporal_faces(tracks or [], gap_max)
        n_frames = len(self._temporal_faces)
        n_faces = sum(len(v) for v in self._temporal_faces.values())
        n_interp = sum(1 for v in self._temporal_faces.values()
                       for f in v if f.get('_interpolated'))
        n_refused = int(getattr(self, '_interp_refused', 0) or 0)
        print(f'[Temporal] {len(tracks or [])} track(s); faces on {n_frames} frames '
              f'({n_faces} total, {n_interp} gap-filled, gap limit {gap_max}'
              + (f', {n_refused} refused as unbridgeable' if n_refused else '') + ').')

        # ── Coverage guard ───────────────────────────────────────────────────
        # The swap pass trusts this cache INSTEAD of detecting, so a scan that
        # covered less than the clip would silently render every uncovered frame
        # un-swapped (empty face list -> early return), with nothing in the log.
        # The try/except around this method only catches exceptions; a short scan
        # raises nothing. So: remember how far the scan actually reached and let
        # swap_faces detect normally past that point. A deliberate Stop also ends
        # the scan early, but then the swap pass never runs those frames either —
        # so don't cry wolf about it.
        self._temporal_covered = int(getattr(self, '_track_scanned', 0) or 0)
        if frame_count and self._temporal_covered < frame_count and roop.globals.processing:
            print(f'[Temporal] WARNING: the scan reached frame {self._temporal_covered} '
                  f'of {frame_count} (the decoder stopped early). Frames past that point '
                  f'fall back to per-frame detection so they still get swapped — but they '
                  f'lose the anti-flicker smoothing.')
        if n_faces == 0:
            # Nothing to consume: a cache of "no faces anywhere" would suppress the
            # whole swap. Per-frame detection also has the ROI/rescue paths this
            # scan doesn't, so it's strictly the better fallback.
            print('[Temporal] pre-pass found no faces at all — falling back to '
                  'per-frame detection for this clip.')
            self._temporal_mode = False
            self._temporal_faces = None

    @staticmethod
    def _bridgeable(a, b, span):
        """Could a face plausibly have travelled from observation *a* to *b* over
        *span* frames? Gap-fill invents a face for every frame in between, and an
        invented face passes every identity check by construction (its embedding
        IS the track mean), so an implausible bridge paints the swap onto the
        background — see _INTERP_MAX_TRAVEL. Position/size only; identity was
        already checked when the two observations joined the track."""
        if _INTERP_MAX_TRAVEL <= 0:
            return True
        try:
            ba = np.asarray(a.bbox, np.float64)
            bb = np.asarray(b.bbox, np.float64)
        except Exception:
            return True                     # no geometry to judge on — behave as before
        wa = max(1.0, float(ba[2] - ba[0]))
        wb = max(1.0, float(bb[2] - bb[0]))
        if _INTERP_MAX_SCALE > 0 and max(wa, wb) / min(wa, wb) > _INTERP_MAX_SCALE:
            return False
        travel = float(np.hypot((ba[0] + ba[2] - bb[0] - bb[2]) * 0.5,
                                (ba[1] + ba[3] - bb[1] - bb[3]) * 0.5))
        return travel <= _INTERP_MAX_TRAVEL * float(span) * (wa + wb) * 0.5

    @staticmethod
    def _interp_face(a, b, w, emb_mean):
        """Linear blend of two Face observations at fraction w ∈ (0,1) of a→b.
        NB: copy.copy() crashes on an insightface Face (its __getattr__ returns
        None for missing dunders), so shallow-copy via the dict constructor."""
        f = type(a)(a)

        def _lerp(x, y):
            return (1.0 - w) * np.asarray(x, np.float64) + w * np.asarray(y, np.float64)

        f['bbox'] = _lerp(a.bbox, b.bbox).astype(np.float32)
        if getattr(a, 'kps', None) is not None and getattr(b, 'kps', None) is not None:
            f['kps'] = _lerp(a.kps, b.kps).astype(np.float32)
        for key in ('landmark_2d_106', 'landmark_3d_68'):
            va, vb = getattr(a, key, None), getattr(b, key, None)
            if va is not None and vb is not None and np.shape(va) == np.shape(vb):
                f[key] = _lerp(va, vb).astype(np.float32)
        # Identity for embedding matching: the track's mean. Set the RAW
        # embedding — normed_embedding is a read-only property derived from it.
        f['embedding'] = emb_mean
        f['det_score'] = np.float32(min(float(getattr(a, 'det_score', 0.6) or 0.6),
                                        float(getattr(b, 'det_score', 0.6) or 0.6)))
        f['_interpolated'] = True
        return f

    def _build_temporal_faces(self, tracks, gap_max):
        """Build {frame_idx: [Face, ...]} from tracked observations: gap-fill
        detection misses ≤ gap_max frames, then (when stabilize_face is on)
        smooth kps/lm106/bbox per track with the configured filter. Faces per
        frame are sorted by x so ordering matches get_all_faces."""
        from roop.one_euro import OneEuroFilter
        stab_on = bool(getattr(self.options, 'stabilize_face', False))
        method = getattr(self.options, 'stabilize_method', 'one_euro')
        mc = float(getattr(self.options, 'stabilize_min_cutoff', 0.05))
        bt = float(getattr(self.options, 'stabilize_beta', 0.02))
        out = {}
        self._interp_refused = 0
        for t in tracks:
            obs = t.get('obs') or {}
            if not obs:
                continue
            emb_mean = np.asarray(t['emb_mean'], dtype=np.float32)
            idxs = sorted(obs)
            merged = dict(obs)
            prev = None
            for i in idxs:
                if prev is not None and 1 < (i - prev) <= gap_max:
                    a, b = obs[prev], obs[i]
                    span = float(i - prev)
                    # Only bridge a gap the face could actually have crossed. A
                    # Re-ID reconnection carries no spatial constraint, so the two
                    # anchors can be on opposite sides of the frame — filling that
                    # in paints a swap across the background (see _bridgeable).
                    if not self._bridgeable(a, b, span):
                        self._interp_refused += (i - prev - 1)
                        prev = i
                        continue
                    for g in range(prev + 1, i):
                        merged[g] = self._interp_face(a, b, (g - prev) / span, emb_mean)
                prev = i
            if stab_on:
                # Per-track sequential smoothing. The default-arg captures keep
                # each track's filter state independent of the loop variable.
                if method == 'ema':
                    def _smooth(key, val, _t, _state={}):
                        prev_v = _state.get(key)
                        cur = val if prev_v is None else 0.3 * val + 0.7 * prev_v
                        _state[key] = cur
                        return cur
                else:
                    def _smooth(key, val, _t, _filters={}):
                        flt = _filters.get(key)
                        if flt is None:
                            flt = _filters[key] = OneEuroFilter(min_cutoff=mc, beta=bt)
                        return flt(val, _t)
                for i in sorted(merged):
                    f = merged[i]
                    if getattr(f, 'kps', None) is not None:
                        f['kps'] = _smooth('kps', np.asarray(f.kps, np.float64), i).astype(np.float32)
                    lm = getattr(f, 'landmark_2d_106', None)
                    if lm is not None:
                        f['landmark_2d_106'] = _smooth('lm106', np.asarray(lm, np.float64), i).astype(np.float32)
                    bb = getattr(f, 'bbox', None)
                    if bb is not None:
                        f['bbox'] = _smooth('bbox', np.asarray(bb, np.float64), i).astype(np.float32)
            for i, f in merged.items():
                out.setdefault(i, []).append(f)
        for i in out:
            out[i].sort(key=lambda f: f.bbox[0])
        return out

    def _precompute_stabilized_kps(self, source_video, awebp_frames, frame_start, frame_end, frame_count):
        """Pass 1 of 2-pass stabilization. Sequentially detect every frame's faces
        and run their kps through the (order-dependent) kps stabilizer, returning
        {frame_idx: [(raw_centroid, smoothed_kps), ...]} for pass 2 to look up.
        Frame indices are 0-based from frame_start, matching the pass-2 reader.
        Reads through its own capture so the main reader (pass 2) is untouched."""
        self.kps_stabilizer.reset()
        precomputed = {}

        def handle(idx, frame):
            with _gpu_guard(pooled=analysis_pooled()):
                faces = get_all_faces(frame)
            if not faces:
                return
            entries = []
            for f in faces:
                kps = getattr(f, 'kps', None)
                if kps is None:
                    continue
                raw_centroid = np.asarray(kps, dtype=np.float64).mean(axis=0)
                smoothed = self.kps_stabilizer.apply(kps, idx)
                entries.append((raw_centroid, np.asarray(smoothed, dtype=np.float32)))
            if entries:
                precomputed[idx] = entries

        if awebp_frames is not None:
            subset = awebp_frames[frame_start:frame_end] if frame_end > frame_start else awebp_frames[frame_start:]
            for idx, frame in enumerate(subset):
                if not roop.globals.processing:
                    break
                handle(idx, frame)
        else:
            cap = cv2.VideoCapture(source_video)
            try:
                if frame_start > 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
                idx = 0
                while roop.globals.processing:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        break
                    handle(idx, frame)
                    idx += 1
                    if idx >= frame_count:
                        break
            finally:
                cap.release()
        return precomputed

    def _lookup_precomputed_kps(self, frame_idx, face):
        """Pass 2: return the smoothed kps precomputed for this face, matched by
        nearest raw centroid within the frame. Falls back to the face's own kps
        when there's no precomputed entry (e.g. pass 1 could not decode)."""
        kps = getattr(face, 'kps', None)
        if kps is None or not self._precomputed_kps:
            return kps
        entries = self._precomputed_kps.get(frame_idx)
        if not entries:
            return kps
        c = np.asarray(kps, dtype=np.float64).mean(axis=0)
        best, best_d = None, float('inf')
        for raw_centroid, smoothed in entries:
            d = float(np.linalg.norm(raw_centroid - c))
            if d < best_d:
                best_d, best = d, smoothed
        return best if best is not None else kps
