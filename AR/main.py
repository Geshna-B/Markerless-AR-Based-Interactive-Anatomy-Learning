import sys
import argparse
import time
import numpy as np
import cv2

# ── Toggle: stub ↔ real Member 3 detector ──────────────────────────────────
USE_STUB = True   # ← flip to False when Member 3 delivers real detector

if USE_STUB:
    from member3_stub.member3_stub import detect_anatomy_card, draw_detection
    print("[main] Using Member 3 STUB detector")
else:
    from member3_real.detector import detect_anatomy_card
    draw_detection = lambda f, d: f
    print("[main] Using Member 3 REAL detector")

from ar_renderer.renderer import ARRenderer
# ───────────────────────────────────────────────────────────────────────────
from utils.interaction import InteractionController
controller = InteractionController()

def run_webcam(source=0, save_output=False):
    """
    Live AR from webcam or video file.
    Press Q to quit.
    Press S to save a screenshot.
    Press 1/2/3 to force show heart/brain/lungs (for demo purposes).
    """
    renderer = ARRenderer(
        poses_dir="member1_output",
        scale_factor=1.0,   # t is overridden in renderer; scale_factor not used
        smooth_alpha=0.3,
    )

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[main] Cannot open source: {source}")
        sys.exit(1)

    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    writer = None
    if save_output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter("output_ar.mp4", fourcc, fps, (w, h))
        print("[main] Recording to output_ar.mp4")

    print("AR running  |  Q=quit  S=screenshot  1=heart  2=brain  3=lungs")

    prev_time    = time.time()
    forced_organ = None   # None = use real detector
    screenshot_n = 0
    import os
    os.makedirs("demo_output", exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── Detection ────────────────────────────────────────────────────
        if forced_organ:
            detection = {
                "organ_name": forced_organ,
                "bbox": [int(w*0.25), int(h*0.25), int(w*0.5), int(h*0.5)],
                "confidence": 1.0,
                "label": f"{forced_organ.capitalize()} Anatomy Card (forced)"
            }
        else:
            detection = detect_anatomy_card(frame)

        # ── AR render ────────────────────────────────────────────────────
        ar_frame = renderer.render_frame(frame, detection, controller)

        if USE_STUB and not forced_organ:
            ar_frame = draw_detection(ar_frame, detection)

        # ── FPS display ──────────────────────────────────────────────────
        cur_time  = time.time()
        fps_live  = 1.0 / max(cur_time - prev_time, 1e-6)
        prev_time = cur_time
        cv2.putText(ar_frame, f"FPS {fps_live:.1f}",
                    (w - 110, h - 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (140, 140, 140), 1)

        if writer:
            writer.write(ar_frame)

        cv2.imshow("AR Anatomy System — Member 4", ar_frame)

        key = cv2.waitKey(1) & 0xFF

        # Pass key to interaction controller first
        controller.handle_key(key)

        if key == ord("q"):
            break
        elif key == ord("s"):
            path = f"demo_output/screenshot_{screenshot_n:03d}.png"
            cv2.imwrite(path, ar_frame)
            print(f"[main] Screenshot saved → {path}")
            screenshot_n += 1
        elif key == ord("1"):
            forced_organ = "heart"
            controller.reset()
        elif key == ord("2"):
            forced_organ = "brain"
            controller.reset()
        elif key == ord("3"):
            forced_organ = "lungs"
            controller.reset()
        elif key == ord("0"):
            forced_organ = None
            controller.reset()

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("[main] AR session ended.")


def run_demo():
    import os
    os.makedirs("demo_output", exist_ok=True)

    from ar_renderer.model_renderer import OrganModelRenderer
    from utils.pose_loader import PoseLoader

    loader = PoseLoader("member1_output", scale_factor=1.0)
    model_renderer = OrganModelRenderer("assets/3d_models")

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
    organ_sizes = {"heart": 120.0, "brain": 110.0, "lungs": 105.0}

    # Fixed camera intrinsics for 1280x720 synthetic frame
    K = np.array([
        [900.0,   0.0, 640.0],
        [  0.0, 900.0, 360.0],
        [  0.0,   0.0,   1.0]
    ])

    # Fixed pose: organ centered, facing camera, at comfortable depth
    R = np.eye(3)
    t = np.array([[0.0], [0.0], [400.0]])   # z=400 puts it nicely in frame

    organs = ["heart", "brain", "lungs"]
    print("Generating demo screenshots...")

    for organ in organs:
        frame = np.full((720, 1280, 3), 40, dtype=np.uint8)

        # Background grid
        for gx in range(0, 1280, 60):
            cv2.line(frame, (gx, 0), (gx, 720), (55, 55, 55), 1)
        for gy in range(0, 720, 60):
            cv2.line(frame, (0, gy), (1280, gy), (55, 55, 55), 1)

        # Anatomy card
        cv2.rectangle(frame, (382, 214), (898, 506), (160, 160, 160), 2)
        cv2.rectangle(frame, (390, 222), (890, 498), (70, 70, 70), -1)
        cv2.putText(frame, f"[ {organ.upper()} ANATOMY CARD ]",
                    (420, 470), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)

        # Render organ directly with fixed pose
        frame = model_renderer.render_organ_overlay(
            frame, organ, R, t, K,
            organ_size=organ_sizes[organ],
            alpha=0.88
        )

        # Draw axes
        color = ORGAN_COLORS[organ]
        origin = (640, 360)
        cv2.arrowedLine(frame, origin, (780, 360), (0,   0, 255), 2, tipLength=0.2)
        cv2.arrowedLine(frame, origin, (640, 220), (0, 255,   0), 2, tipLength=0.2)
        cv2.arrowedLine(frame, origin, (720, 440), (255, 0,   0), 2, tipLength=0.2)
        cv2.putText(frame, "X", (788, 365), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,   0, 255), 1)
        cv2.putText(frame, "Y", (644, 214), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255,   0), 1)
        cv2.putText(frame, "Z", (724, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,   0, 0), 1)

        # HUD top
        cv2.rectangle(frame, (0, 0), (1280, 54), (15, 15, 15), -1)
        cv2.line(frame, (0, 54), (1280, 54), color, 1)
        cv2.putText(frame, "AR ANATOMY SYSTEM",
                    (14, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1)
        cv2.putText(frame, f"|  {organ.upper()}",
                    (210, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        cv2.putText(frame, "Confidence: 97%",
                    (1040, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, "6D Pose: ACTIVE",
                    (14, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 200, 80), 1)
        cv2.putText(frame, "Member 1: Pose  |  Member 3: Detection  |  Member 4: AR Render",
                    (720, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 120, 120), 1)

        # Bounding box
        cv2.rectangle(frame, (382, 214), (898, 506), color, 1)
        cv2.putText(frame, f"{organ.capitalize()} Anatomy Card",
                    (382, 208), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # HUD bottom
        cv2.rectangle(frame, (0, 676), (1280, 720), (15, 15, 15), -1)
        cv2.line(frame, (0, 676), (1280, 676), color, 1)
        cv2.putText(frame, ORGAN_INFO[organ],
                    (14, 704), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (220, 220, 220), 1)

        path = f"demo_output/{organ}_ar_demo.png"
        cv2.imwrite(path, frame)
        print(f"  Saved {path}")

    print("\nDemo complete!")
    
def main():
    parser = argparse.ArgumentParser(description="Member 4 — AR Anatomy System")
    parser.add_argument("--source", default="0",
                        help="Camera index (0,1,...) or path to video file")
    parser.add_argument("--demo",   action="store_true",
                        help="Generate demo screenshots (no camera needed)")
    parser.add_argument("--save",   action="store_true",
                        help="Record webcam AR to output_ar.mp4")
    args = parser.parse_args()

    if args.demo:
        run_demo()
    else:
        src = int(args.source) if args.source.isdigit() else args.source
        run_webcam(source=src, save_output=args.save)


if __name__ == "__main__":
    main()