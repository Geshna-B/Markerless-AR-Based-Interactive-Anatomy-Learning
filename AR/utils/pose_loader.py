import json
import numpy as np
from pathlib import Path

ORGAN_NAMES = ["heart", "brain", "lungs"]
DEFAULT_SCALE = 100.0


class PoseLoader:
    def __init__(self, poses_dir="member1_output", scale_factor=DEFAULT_SCALE):
        self.poses_dir = Path(poses_dir)
        self.scale_factor = scale_factor
        self._cache = {}
        self._load_all()

    def _load_all(self):
        for organ in ORGAN_NAMES:
            json_path = self.poses_dir / organ / f"{organ}_pose.json"
            if not json_path.exists():
                print(f"[PoseLoader] WARNING: Missing {json_path}")
                continue
            with open(json_path) as f:
                data = json.load(f)
            self._cache[organ] = {
                "R": np.array(data["rotation_matrix"], dtype=np.float64),
                "t": np.array(data["translation_vector"], dtype=np.float64),
                "K": np.array(data["camera_matrix"], dtype=np.float64),
                "meta": data.get("metadata", {})
            }
        print(f"[PoseLoader] Loaded: {list(self._cache.keys())}")

    def get_pose_opencv(self, organ):
        entry = self._cache.get(organ)
        if entry is None:
            return None, None, None
        R = entry["R"].copy()
        t = entry["t"].copy() * self.scale_factor
        K = entry["K"].copy()
        return R, t, K

    def get_pose_opengl(self, organ):
        R, t, K = self.get_pose_opencv(organ)
        if R is None:
            return None, None, None
        T = np.array([[1,0,0],[0,-1,0],[0,0,-1]], dtype=np.float64)
        return T @ R @ T.T, T @ t, K

    def get_pose_unity(self, organ):
        R, t, K = self.get_pose_opencv(organ)
        if R is None:
            return None, None, None
        T = np.array([[-1,0,0],[0,1,0],[0,0,1]], dtype=np.float64)
        return T @ R, T @ t, K

    def get_transform_matrix(self, organ, convention="opencv"):
        fns = {
            "opencv": self.get_pose_opencv,
            "opengl": self.get_pose_opengl,
            "unity":  self.get_pose_unity
        }
        R, t, _ = fns[convention](organ)
        if R is None:
            return None
        mat = np.eye(4)
        mat[:3, :3] = R
        mat[:3, 3] = t.flatten()
        return mat

    def get_projection_matrix(self, organ, width=1280, height=720, near=0.1, far=1000.0):
        _, _, K = self.get_pose_opencv(organ)
        if K is None:
            return None
        fx, fy = K[0,0], K[1,1]
        cx, cy = K[0,2], K[1,2]
        return np.array([
            [2*fx/width,  0,           (width - 2*cx)/width,          0],
            [0,           2*fy/height, (2*cy - height)/height,        0],
            [0,           0,           -(far+near)/(far-near), -2*far*near/(far-near)],
            [0,           0,           -1,                             0]
        ])

    def available_organs(self):
        return list(self._cache.keys())

    def quality_report(self):
        print("\n── Pose Quality ──────────────────────")
        for organ, entry in self._cache.items():
            m = entry["meta"]
            print(f"  {organ.upper():6s} | Error: {m.get('reprojection_error_px','?')} px"
                  f" | Inliers: {m.get('inlier_ratio', 0)*100:.1f}%"
                  f" | Quality: {m.get('quality','?')}")
        print("──────────────────────────────────────\n")