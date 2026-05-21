import numpy as np


class InteractionController:
    """
    Tracks user input and applies rotation/zoom
    on top of Member 1's pose.

    Usage in render loop:
        key = cv2.waitKey(1) & 0xFF
        controller.handle_key(key)
        R_final, t_final = controller.apply(R, t)
    """

    def __init__(self):
        self.rot_x    = 0.0   # up/down rotation (degrees)
        self.rot_y    = 0.0   # left/right rotation (degrees)
        self.zoom     = 1.0   # zoom multiplier
        self.highlight = False

        # Tuning
        self.rot_step  = 5.0   # degrees per keypress
        self.zoom_step = 0.1   # zoom per keypress
        self.zoom_min  = 0.3
        self.zoom_max  = 3.0

    def handle_key(self, key):
        """Call this every frame with cv2.waitKey result."""
        if key == ord('a'):   self.rot_y -= self.rot_step
        elif key == ord('d'): self.rot_y += self.rot_step
        elif key == ord('w'): self.rot_x -= self.rot_step
        elif key == ord('s'): self.rot_x += self.rot_step
        elif key == ord('+') or key == ord('='): 
            self.zoom = min(self.zoom_max, self.zoom + self.zoom_step)
        elif key == ord('-'):
            self.zoom = max(self.zoom_min, self.zoom - self.zoom_step)
        elif key == ord('r'): self.reset()
        elif key == ord('h'): self.highlight = not self.highlight

    def apply(self, R, t):
        """
        Apply user rotation and zoom on top of pose R, t.
        Returns modified R_out, t_out.
        """
        # Build rotation matrices from user input
        rx = np.deg2rad(self.rot_x)
        ry = np.deg2rad(self.rot_y)

        # Rotation around X axis (up/down)
        Rx = np.array([
            [1,           0,            0],
            [0,  np.cos(rx), -np.sin(rx)],
            [0,  np.sin(rx),  np.cos(rx)]
        ])

        # Rotation around Y axis (left/right)
        Ry = np.array([
            [ np.cos(ry), 0, np.sin(ry)],
            [          0, 1,          0],
            [-np.sin(ry), 0, np.cos(ry)]
        ])

        # Apply user rotation on top of Member 1's rotation
        R_user = Ry @ Rx
        R_out  = R @ R_user

        # Apply zoom by scaling translation depth
        t_out = t.copy()
        t_out[2] = t_out[2] / self.zoom   # closer = smaller z

        return R_out, t_out

    def reset(self):
        self.rot_x    = 0.0
        self.rot_y    = 0.0
        self.zoom     = 1.0
        self.highlight = False

    def get_status(self):
        """Returns a status string for HUD display."""
        return (f"Rot X:{self.rot_x:+.0f} "
                f"Y:{self.rot_y:+.0f}  "
                f"Zoom:{self.zoom:.1f}x  "
                f"{'[HIGHLIGHT]' if self.highlight else ''}")