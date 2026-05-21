import open3d as o3d
import numpy as np
import cv2
from pathlib import Path


class OrganModelRenderer:
    ORGAN_COLORS_BGR = {
        "heart": (40,  40, 200),
        "brain": (30, 130, 220),
        "lungs": (30, 180,  60),
    }
    LIGHT_DIR = np.array([0.4, 0.6, 0.7], dtype=np.float64)

    def __init__(self, models_dir="assets/3d_models"):
        self.models_dir = Path(models_dir)
        self._meshes = {}
        self._load_all()

    def _load_all(self):
        TARGET_TRIS = 1500
        for organ in ["heart", "brain", "lungs"]:
            for ext in [".stl", ".obj", ".ply"]:
                path = self.models_dir / f"{organ}{ext}"
                if not path.exists():
                    continue
                print(f"[ModelRenderer] Loading {organ}{ext} ...")
                mesh = o3d.io.read_triangle_mesh(str(path))
                mesh.orient_triangles()
                mesh.compute_triangle_normals()
                mesh.compute_vertex_normals()

                bbox   = mesh.get_axis_aligned_bounding_box()
                center = bbox.get_center()
                mesh.translate(-center)
                extent = max(bbox.get_extent())
                mesh.scale(1.0 / extent, center=[0, 0, 0])

                current = len(mesh.triangles)
                if current > TARGET_TRIS:
                    mesh = mesh.simplify_quadric_decimation(
                        target_number_of_triangles=TARGET_TRIS)
                    mesh.orient_triangles()
                    mesh.compute_triangle_normals()

                self._meshes[organ] = {
                    "vertices":  np.asarray(mesh.vertices,         dtype=np.float64),
                    "triangles": np.asarray(mesh.triangles,        dtype=np.int32),
                    "normals":   np.asarray(mesh.triangle_normals, dtype=np.float64),
                }
                print(f"[ModelRenderer] {organ} ✅ ({len(self._meshes[organ]['triangles'])} tris)")
                break
            else:
                print(f"[ModelRenderer] WARNING: No model for {organ}")

    def render_organ_overlay(self, frame, organ, R, t, K,
                              organ_size=220, alpha=0.85):
        """
        R is always np.eye(3) from renderer — organ faces camera cleanly.
        organ_size = desired pixel radius on screen.
        """
        data = self._meshes.get(organ)
        if data is None:
            return frame

        triangles = data["triangles"]
        normals   = data["normals"]
        h, w      = frame.shape[:2]
        t_flat    = t.flatten()

        # Scale unit-sphere to pixel radius
        depth      = max(abs(t_flat[2]), 1.0)
        world_scale = organ_size * depth / K[0, 0]
        vertices   = data["vertices"] * world_scale

        # ── Vectorized projection ──────────────────────────────────────────
        pts3d = vertices @ R.T + t_flat
        valid  = pts3d[:, 2] > 1.0
        depths = np.where(valid, pts3d[:, 2], -1.0)
        denom  = np.where(valid, pts3d[:, 2], 1.0)
        px_arr = K[0, 0] * pts3d[:, 0] / denom + K[0, 2]
        py_arr = K[1, 1] * pts3d[:, 1] / denom + K[1, 2]
        pts2d  = np.stack([px_arr, py_arr], axis=1).astype(np.float64)
        pts2d[~valid] = -99999

        ai, bi, ci = triangles[:, 0], triangles[:, 1], triangles[:, 2]
        pa_all = pts2d[ai]; pb_all = pts2d[bi]; pc_all = pts2d[ci]

        # Screen-space back-face culling — works correctly with R=eye
        ax = pa_all[:,0]; ay = pa_all[:,1]
        bx = pb_all[:,0]; by = pb_all[:,1]
        cx2= pc_all[:,0]; cy2= pc_all[:,1]
        cross  = (bx - ax) * (cy2 - ay) - (by - ay) * (cx2 - ax)
        facing = cross < 0

        tri_valid  = (depths[ai] > 0) & (depths[bi] > 0) & (depths[ci] > 0) & facing
        tri_valid &= np.maximum(pa_all[:,0], np.maximum(pb_all[:,0], pc_all[:,0])) >= 0
        tri_valid &= np.minimum(pa_all[:,0], np.minimum(pb_all[:,0], pc_all[:,0])) <= w
        tri_valid &= np.maximum(pa_all[:,1], np.maximum(pb_all[:,1], pc_all[:,1])) >= 0
        tri_valid &= np.minimum(pa_all[:,1], np.minimum(pb_all[:,1], pc_all[:,1])) <= h

        vi = np.where(tri_valid)[0]
        if len(vi) == 0:
            return frame

        # Shading
        light      = self.LIGHT_DIR / np.linalg.norm(self.LIGHT_DIR)
        base_color = np.array(self.ORGAN_COLORS_BGR.get(organ, (150, 150, 150)),
                              dtype=np.float64)
        n_vecs = normals[vi]
        n_unit = n_vecs / (np.linalg.norm(n_vecs, axis=1, keepdims=True) + 1e-8)
        shades = 0.35 + 0.65 * np.clip(n_unit @ light, 0, 1)

        avg_d  = (depths[ai[vi]] + depths[bi[vi]] + depths[ci[vi]]) / 3.0
        order  = np.argsort(-avg_d)
        vi     = vi[order];  shades = shades[order]

        pa_d = pa_all[vi]; pb_d = pb_all[vi]; pc_d = pc_all[vi]

        # Draw
        overlay = frame.copy()
        for k in range(len(vi)):
            shade = float(shades[k])
            color = tuple(int(np.clip(base_color[ch] * shade, 0, 255))
                          for ch in range(3))
            pts = np.array([[pa_d[k,0], pa_d[k,1]],
                            [pb_d[k,0], pb_d[k,1]],
                            [pc_d[k,0], pc_d[k,1]]], dtype=np.int32)
            cv2.fillConvexPoly(overlay, pts, color)

        edge_color = tuple(int(c * 0.4) for c in base_color)
        for k in range(0, len(vi), 3):
            pts = np.array([[pa_d[k,0], pa_d[k,1]],
                            [pb_d[k,0], pb_d[k,1]],
                            [pc_d[k,0], pc_d[k,1]]], dtype=np.int32)
            cv2.polylines(overlay, [pts], True, edge_color, 1, cv2.LINE_AA)

        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        return frame

    def available_organs(self):
        return list(self._meshes.keys())
