import numpy as np

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
        # Multi-angle source bank: list of (yaw_deg, pitch_deg) or None per face in self.faces
        # Populated by ProcessMgr.initialize() when use_source_bank is enabled.
        self.face_poses = None  # type: list[tuple[float, float] | None] | None

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
        """Compute (yaw, pitch, roll) for all faces in this FaceSet using solve_pose_5pt."""
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
            if pose_entry is None:
                continue
            src_yaw, src_pitch = pose_entry[0], pose_entry[1]
            dist = (target_yaw - src_yaw) ** 2 + (target_pitch - src_pitch) ** 2
            if dist < best_dist:
                best_dist = dist
                best_i = i
        return self.faces[best_i], best_i

    def get_best_match_distance(self, target_embedding: np.ndarray, target_yaw: float = None, target_pitch: float = None):
        """Compute the cosine distance against the best pose-matched or closest source embedding."""
        if not self.faces or target_embedding is None:
            return None, 0
        
        target_norm = float(np.linalg.norm(target_embedding))
        if target_norm <= 1e-8:
            return None, 0

        # If target pose is known and we have multiple poses, try the pose-matched face first
        if target_yaw is not None and target_pitch is not None and len(self.faces) > 1:
            best_face, best_i = self.select_best_pose_face(target_yaw, target_pitch)
            src_emb = best_face.get('embedding') if isinstance(best_face, dict) else getattr(best_face, 'embedding', None)
            if src_emb is not None:
                src_norm = float(np.linalg.norm(src_emb))
                if src_norm > 1e-8:
                    sim = float(np.dot(target_embedding, src_emb) / (target_norm * src_norm))
                    return 1.0 - sim, best_i

        min_dist = float('inf')
        min_i = 0
        for i, f in enumerate(self.faces):
            src_emb = f.get('embedding') if isinstance(f, dict) else getattr(f, 'embedding', None)
            if src_emb is not None:
                src_norm = float(np.linalg.norm(src_emb))
                if src_norm > 1e-8:
                    sim = float(np.dot(target_embedding, src_emb) / (target_norm * src_norm))
                    dist = 1.0 - sim
                    if dist < min_dist:
                        min_dist = dist
                        min_i = i
        if min_dist < float('inf'):
            return min_dist, min_i
        return None, 0
