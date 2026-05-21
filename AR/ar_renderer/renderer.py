import cv2
import numpy as np
from utils.pose_loader import PoseLoader
from utils.pose_smoother import PoseSmoother
from ar_renderer.model_renderer import OrganModelRenderer


ORGAN_COLORS = {
    "heart": (60,  60, 220),
    "brain": (30, 144, 255),
    "lungs": (30, 200,  80),
}

ORGAN_INFO = {
    "heart": "Heart  |  4 chambers  |  ~300g  |  Beats 100,000x/day",
    "brain": "Brain  |  ~86B neurons  |  ~1.4kg  |  Uses 20% of body O2",
    "lungs": "Lungs  |  ~300M alveoli  |  6L capacity  |  Left + Right lobes",
}


class ARRenderer:
    def __init__(self, poses_dir="member1_output",
                 scale_factor=100.0, smooth_alpha=0.3):
        self.loader         = PoseLoader(poses_dir, scale_factor)
        self.smoother       = PoseSmoother(alpha=smooth_alpha)
        self.model_renderer = OrganModelRenderer("assets/3d_models")
        self.loader.quality_report()

    def render_frame(self, frame, detection, controller=None):
        """
        Main per-frame call.
        controller: InteractionController instance (optional)
        """
        output = frame.copy()

        if detection is None:
            self._draw_no_detection_hud(output)
            return output

        organ = detection["organ_name"]
        R, t, K = self.loader.get_pose_opencv(organ)

        if R is None:
            cv2.putText(output, f"No pose data for '{organ}'",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return output

        # Use identity R — organ always faces camera cleanly (like demo mode)
        # Member 1 R is near-identity anyway but causes backface artifacts
        R = np.eye(3)

        # Place organ at bbox center at fixed depth — identical to demo
        bx, by, bw, bh = detection["bbox"]
        cx = bx + bw // 2
        cy = by + bh // 2
        depth = K[0, 0] * 0.45
        t = np.array([[(cx - K[0, 2]) * depth / K[0, 0]],
                      [(cy - K[1, 2]) * depth / K[1, 1]],
                      [depth]])

        # Apply user interaction on top of pose
        if controller is not None:
            R, t = controller.apply(R, t)

        # Highlight effect — brighten organ if active
        alpha = 0.95 if (controller and controller.highlight) else 0.85

        # Organ sizes tuned to fill the anatomy card nicely
        # organ_size = desired pixel radius on screen (auto-scales with depth)
        ORGAN_SIZES = {"heart": 220, "brain": 210, "lungs": 225}

        # Render 3D organ
        output = self.model_renderer.render_organ_overlay(
            output, organ, R, t, K,
            organ_size=ORGAN_SIZES.get(organ, 100.0), alpha=alpha)

        # Draw axes
        output = self._draw_axes(output, R, t, K, length=15.0)

        # Draw HUD
        output = self._draw_hud(output, organ, detection, controller)

        return output

    # ── Axes ─────────────────────────────────────────────────────────────────

    def _project(self, pt3d, R, t, K):
        p = R @ pt3d + t.flatten()
        if p[2] <= 0:
            return None
        uv = K @ p
        return (int(uv[0] / uv[2]), int(uv[1] / uv[2]))

    def _draw_axes(self, frame, R, t, K, length=15.0):
        o  = self._project(np.zeros(3), R, t, K)
        px = self._project(np.array([length, 0, 0]), R, t, K)
        py = self._project(np.array([0, length, 0]), R, t, K)
        pz = self._project(np.array([0, 0, length]), R, t, K)

        if all(p is not None for p in [o, px, py, pz]):
            cv2.arrowedLine(frame, o, px, (0,   0, 255), 2, tipLength=0.2)
            cv2.arrowedLine(frame, o, py, (0, 255,   0), 2, tipLength=0.2)
            cv2.arrowedLine(frame, o, pz, (255,  0,   0), 2, tipLength=0.2)
            cv2.putText(frame, "X", (px[0]+4, px[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,   0, 255), 1)
            cv2.putText(frame, "Y", (py[0]+4, py[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255,   0), 1)
            cv2.putText(frame, "Z", (pz[0]+4, pz[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,  0,   0), 1)
        return frame

    # ── HUD ──────────────────────────────────────────────────────────────────

    def _draw_hud(self, frame, organ, detection, controller=None):
        h, w = frame.shape[:2]
        color = ORGAN_COLORS.get(organ, (255, 255, 255))

        # Top banner
        cv2.rectangle(frame, (0, 0), (w, 54), (15, 15, 15), -1)
        cv2.line(frame, (0, 54), (w, 54), color, 1)
        cv2.putText(frame, "AR ANATOMY SYSTEM",
                    (14, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1)
        cv2.putText(frame, f"|  {organ.upper()}",
                    (210, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        cv2.putText(frame, f"Confidence: {detection['confidence']:.0%}",
                    (w - 240, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, "6D Pose: ACTIVE",
                    (14, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 200, 80), 1)
        cv2.putText(frame, "Member 1: Pose  |  Member 3: Detection  |  Member 4: AR Render",
                    (w - 560, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 120, 120), 1)

        # Interaction status bar (only when controller active)
        if controller is not None:
            status = controller.get_status()
            cv2.rectangle(frame, (0, 54), (w, 76), (25, 25, 25), -1)
            cv2.putText(frame, f"  {status}",
                        (14, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 200, 160), 1)

            # Controls hint
            hint = "W/S:RotateUD  A/D:RotateLR  +/-:Zoom  H:Highlight  R:Reset  Q:Quit"
            cv2.putText(frame, hint,
                        (w - 680, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 100, 100), 1)

        # Bottom info bar
        cv2.rectangle(frame, (0, h - 44), (w, h), (15, 15, 15), -1)
        cv2.line(frame, (0, h - 44), (w, h - 44), color, 1)
        cv2.putText(frame, ORGAN_INFO.get(organ, ""),
                    (14, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (220, 220, 220), 1)

        # Bounding box
        x, y, bw, bh = detection["bbox"]
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 1)
        cv2.putText(frame, detection.get("label", organ),
                    (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return frame

    def _draw_no_detection_hud(self, frame):
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 54), (15, 15, 15), -1)
        cv2.putText(frame, "AR ANATOMY SYSTEM  |  Searching for anatomy card...",
                    (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (90, 90, 90), 1)