import numpy as np

from roop.faceset_v2 import (FORMAT_NAME, FORMAT_VERSION, measure_lighting,
                              parse_pose_matrix_key, pose_matrix_cell,
                              select_reference_index)


def _pose_yaw_pitch(pose_entry):
    """Return yaw/pitch from a legacy two-value or full three-value pose."""
    if pose_entry is None:
        return None
    try:
        yaw = float(pose_entry[0])
        pitch = float(pose_entry[1])
    except (IndexError, TypeError, ValueError, OverflowError):
        return None
    if not np.isfinite(yaw) or not np.isfinite(pitch):
        return None
    return yaw, pitch


class FaceSet:
    faces = []
    ref_images = []
    embedding_average = 'None'
    embeddings_backup = None

    def __init__(self):
        self.faces = []
        self.ref_images = []
        self.embeddings_backup = None
        self.face_3d = None   # populated by face_3d_recon when use_3d_recon is enabled (first valid face's crop)
        # 3D recon per-face crop bank: list parallel to self.faces, each entry a
        # {'src_crop','src_M','src_lm68'} dict or None. Lets 3D recon warp the
        # source-bank-SELECTED face (not just face[0]) so the two features compose.
        self.face_3d_bank = None  # type: list[dict | None] | None
        # Multi-angle source bank: (yaw_deg, pitch_deg[, roll_deg]) or None per face
        # in self.faces. Consumers only use yaw/pitch; the optional roll is retained
        # for callers that compute all three values from the five-point solver.
        # Populated by ProcessMgr.initialize() when use_source_bank is enabled.
        self.face_poses = None  # type: list[tuple[float, ...] | None] | None
        # V2 is an additive metadata/index layer for cached embeddings & poses
        self.format_name = FORMAT_NAME
        self.format_version = 1
        self.faceset_metadata = None
        self.face_metadata = []
        self.pose_bank = None
        self.pose_bins = {}
        self.dermal_patch = None
        self.identity_embedding = None
        self.normalized_embedding = None
        self.reference_embeddings = []
        self.reference_weights = []
        self.reference_paths = []
        self.reference_rejected = []
        self.faceset_valid = True
        self.faceset_migration = None

    def AverageEmbeddings(self):
        if len(self.faces) > 1 and self.embeddings_backup is None:
            first_face = self.faces[0]
            if hasattr(first_face, 'embedding'):
                self.embeddings_backup = first_face.embedding
                embeddings = [face.embedding for face in self.faces]
                first_face.embedding = np.mean(embeddings, axis=0)
            else:
                self.embeddings_backup = first_face['embedding']
                embeddings = [face['embedding'] for face in self.faces]
                first_face['embedding'] = np.mean(embeddings, axis=0)

    def compute_face_poses(self):
        """Compute (yaw, pitch, roll) for all faces using solve_pose_5pt."""
        from roop.face_util import solve_pose_5pt
        poses = []
        for face in self.faces:
            kps = face.get('kps') if isinstance(face, dict) else getattr(face, 'kps', None)
            if kps is not None:
                p = solve_pose_5pt(kps)
                if p is not None:
                    poses.append((float(p[0]), float(p[1]), float(p[2])))
                else:
                    poses.append((0.0, 0.0, 0.0))
            else:
                poses.append((0.0, 0.0, 0.0))
        self.face_poses = poses
        return poses

    def select_best_pose_face(self, target_yaw: float, target_pitch: float):
        """Pick the source face whose head pose closest matches target (yaw, pitch)."""
        if not self.faces:
            return None, 0
        if len(self.faces) == 1:
            return self.faces[0], 0
        if self.face_poses is None or len(self.face_poses) != len(self.faces):
            self.compute_face_poses()

        best_i = 0
        best_dist = float('inf')
        for i, pose_entry in enumerate(self.face_poses):
            pose = _pose_yaw_pitch(pose_entry)
            if pose is None:
                continue
            src_yaw, src_pitch = pose
            dist = (target_yaw - src_yaw) ** 2 + (target_pitch - src_pitch) ** 2
            if dist < best_dist:
                best_dist = dist
                best_i = i
        return self.faces[best_i], best_i

    def get_best_match_distance(self, target_embedding: np.ndarray, target_yaw: float = None, target_pitch: float = None):
        """Compute the cosine distance against the best pose-matched or closest source embedding."""
        if not self.faces or target_embedding is None:
            return None, 0
        
        target_embedding = np.asarray(target_embedding, dtype=np.float32).reshape(-1)
        target_norm = float(np.linalg.norm(target_embedding))
        if target_norm <= 1e-8:
            return None, 0

        # If target pose is known and we have multiple poses, try the pose-matched face first
        if target_yaw is not None and target_pitch is not None and len(self.faces) > 1:
            best_face, best_i = self.select_best_pose_face(target_yaw, target_pitch)
            src_emb = best_face.get('embedding') if isinstance(best_face, dict) else getattr(best_face, 'embedding', None)
            if src_emb is not None:
                src_emb = np.asarray(src_emb, dtype=np.float32).reshape(-1)
                if src_emb.shape == target_embedding.shape:
                    src_norm = float(np.linalg.norm(src_emb))
                    if src_norm > 1e-8:
                        denom = max(target_norm * src_norm, 1e-6)
                        sim = float(np.clip(np.dot(target_embedding, src_emb) / denom, -1.0, 1.0))
                        return max(0.0, 1.0 - sim), best_i

        min_dist = float('inf')
        min_i = 0
        for i, f in enumerate(self.faces):
            src_emb = f.get('embedding') if isinstance(f, dict) else getattr(f, 'embedding', None)
            if src_emb is not None:
                src_emb = np.asarray(src_emb, dtype=np.float32).reshape(-1)
                if src_emb.shape != target_embedding.shape:
                    continue
                src_norm = float(np.linalg.norm(src_emb))
                if src_norm > 1e-8:
                    denom = max(target_norm * src_norm, 1e-6)
                    sim = float(np.clip(np.dot(target_embedding, src_emb) / denom, -1.0, 1.0))
                    dist = max(0.0, 1.0 - sim)
                    if dist < min_dist:
                        min_dist = dist
                        min_i = i
        if min_dist < float('inf'):
            return min_dist, min_i
        return None, 0

    def attach_v2_metadata(self, metadata):
        """Attach validated V2 metadata without replacing detector Face objects."""
        self.faceset_metadata = metadata
        self.format_name = metadata.get('schema', FORMAT_NAME)
        self.format_version = int(metadata.get('version', FORMAT_VERSION))
        self.face_metadata = list(metadata.get('sources') or [])
        self.pose_bank = metadata.get('pose_bank') or {}
        self.dermal_patch = metadata.get('dermal_patch') or None
        self.pose_bins = {}
        for key, cell in (metadata.get('pose_bins') or {}).items():
            parsed = parse_pose_matrix_key(key)
            if parsed is None or not isinstance(cell, dict):
                continue
            vector = self._unit_vector(cell.get('embedding'))
            if vector is not None:
                self.pose_bins[parsed] = vector
        identity = metadata.get('identity') or {}
        value = (metadata.get('default_embedding')
                 or identity.get('embedding')
                 or identity.get('normalized_embedding'))
        if value is not None:
            arr = np.asarray(value, dtype=np.float32).reshape(-1)
            norm = float(np.linalg.norm(arr))
            if norm > 1e-8 and np.isfinite(arr).all():
                self.identity_embedding = (arr / norm).astype(np.float32)
                self.normalized_embedding = self.identity_embedding.copy()
        poses = []
        for entry in self.face_metadata:
            geo = entry.get('geometry') or {}
            poses.append((geo.get('yaw'), geo.get('pitch')))
        self.face_poses = poses if poses else None
        for index, face in enumerate(self.faces):
            try:
                face['faceset_v2_index'] = index
            except Exception:
                try:
                    setattr(face, 'faceset_v2_index', index)
                except Exception:
                    pass

    @staticmethod
    def _unit_vector(value):
        if value is None:
            return None
        try:
            arr = np.asarray(value, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError):
            return None
        if arr.size == 0 or not np.isfinite(arr).all():
            return None
        norm = float(np.linalg.norm(arr))
        if norm <= 1e-8:
            return None
        return (arr / norm).astype(np.float32)

    @property
    def default_embedding(self):
        """The global normalized centroid, for both V2 and legacy FaceSets."""
        if self.identity_embedding is not None:
            return self.identity_embedding
        vectors = []
        for index, face in enumerate(self.faces or []):
            if index == 0 and self.embeddings_backup is not None:
                value = self.embeddings_backup
            elif isinstance(face, dict):
                value = face.get('embedding')
            else:
                value = getattr(face, 'embedding', None)
            vector = self._unit_vector(value)
            if vector is not None:
                vectors.append(vector)
        if not vectors or len({v.shape for v in vectors}) != 1:
            return None
        return self._unit_vector(np.mean(np.asarray(vectors), axis=0))

    def pose_bin_embedding(self, pose=None, fallback=True):
        """Return the 3x3 pose-cell centroid for `pose`."""
        cell = pose_matrix_cell(pose) if pose is not None else ("center", "center")
        vector = self.pose_bins.get(cell)
        if vector is not None:
            return vector
        if not fallback:
            return None
        return self.default_embedding

    def select_reference_index(self, pose=None, appearance=None, embedding=None):
        """Fast V2 lookup with legacy pose-bank fallback."""
        if self.format_version >= 2 and self.faceset_metadata:
            index = select_reference_index(self.faceset_metadata, pose=pose,
                                           appearance=appearance, embedding=embedding)
            return max(0, min(int(index), max(0, len(self.faces) - 1)))
        if pose is not None and self.face_poses:
            yaw, pitch = float(pose[0]), float(pose[1])
            valid = []
            for i, pose_entry in enumerate(self.face_poses):
                if i >= len(self.faces):
                    break
                source_pose = _pose_yaw_pitch(pose_entry)
                if source_pose is None:
                    continue
                source_yaw, source_pitch = source_pose
                valid.append((i, (yaw - source_yaw) ** 2 +
                              (pitch - source_pitch) ** 2))
            if valid:
                return min(valid, key=lambda item: item[1])[0]
        return 0

    def identity_detail_for(self, source_index=0):
        """Return the persistent V2 detail map, or a safe per-source fallback."""
        if self.format_version < 2 or not self.faceset_metadata:
            return None
        details = self.faceset_metadata.get('identity_details') or {}
        persistent = details.get('high_frequency')
        if isinstance(persistent, dict) and persistent.get('residual_q'):
            return persistent
        try:
            index = int(source_index)
        except (TypeError, ValueError):
            index = 0
        if 0 <= index < len(self.face_metadata):
            return ((self.face_metadata[index].get('identity_details') or {})
                    .get('high_frequency'))
        return None

    @staticmethod
    def lighting_for_frame(image, bbox=None):
        return measure_lighting(image, bbox)
