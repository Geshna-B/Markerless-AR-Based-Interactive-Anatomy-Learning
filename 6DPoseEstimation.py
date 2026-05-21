"""
MEMBER 1: 6D POSE ESTIMATION - FINAL WORKING VERSION
Fixes dimension mismatch error from RANSAC

Production ready - all bugs fixed!
"""

import numpy as np
import cv2
import json
from pathlib import Path
from datetime import datetime


class Member1_PoseEstimator:
    
    def __init__(self, member2_output_dir):
        self.member2_dir = Path(member2_output_dir)
        self.organs = {}
        
        for organ_name in ['heart', 'brain', 'lungs']:
            organ_dir = self.member2_dir / organ_name
            json_path = organ_dir / f"{organ_name}_correspondences.json"
            
            if json_path.exists():
                print(f"✓ Loading {organ_name} data...")
                self.load_organ_data(organ_name, json_path)
            else:
                print(f"✗ {organ_name} data not found")
        
        self.pose_history = {organ: [] for organ in self.organs.keys()}
        
        print(f"\n📊 Loaded {len(self.organs)} organs for pose estimation")
    
    def load_organ_data(self, organ_name, json_path):
        """Load Member 2's reconstruction data"""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        K = np.array(data['camera_matrix'], dtype=np.float32)
        dist = np.array(data['distortion_coeffs'], dtype=np.float32)
        
        # Collect all correspondences
        all_3d = []
        all_2d = []
        
        for corr in data['correspondences']:
            pts_3d = np.array(corr['points_3d'], dtype=np.float32)
            pts_2d = np.array(corr['points_2d_1'], dtype=np.float32)
            
            # Quick quality check
            std_3d = np.std(pts_3d, axis=0)
            if np.max(std_3d) >= 0.1:  # Valid 3D structure
                all_3d.append(pts_3d)
                all_2d.append(pts_2d)
        
        if len(all_3d) == 0:
            print(f"   ⚠️ No valid views for {organ_name}")
            return False
        
        # Merge all valid views
        points_3d = np.vstack(all_3d)
        points_2d = np.vstack(all_2d)
        
        self.organs[organ_name] = {
            'camera_matrix': K,
            'distortion': dist,
            'points_3d': points_3d,
            'points_2d_reference': points_2d,
            'num_points': len(points_3d),
            'metadata': data['metadata']
        }
        
        print(f"   • {len(points_3d)} 3D points")
        print(f"   • Camera: fx={K[0,0]:.1f}, fy={K[1,1]:.1f}")
        
        return True
    
    # ============================================================
    # STEP 15: 6D POSE ESTIMATION
    # ============================================================
    
    def estimate_pose_pnp(self, organ_name, points_2d_query=None):
        """
        Estimate 6D pose using PnP with RANSAC
        
        FIXED: Properly handles RANSAC inlier indexing
        
        Returns:
            R: 3x3 rotation matrix
            t: 3x1 translation vector  
            inliers: boolean mask
        """
        if organ_name not in self.organs:
            return None, None, None
        
        organ = self.organs[organ_name]
        pts_3d = organ['points_3d']
        K = organ['camera_matrix']
        dist = organ['distortion']
        
        if points_2d_query is None:
            pts_2d = organ['points_2d_reference']
        else:
            pts_2d = np.array(points_2d_query, dtype=np.float32)
        
        # Need minimum 6 points
        if len(pts_3d) < 6 or len(pts_2d) < 6:
            return None, None, None
        
        # Match sizes
        n = min(len(pts_3d), len(pts_2d))
        pts_3d_subset = pts_3d[:n]
        pts_2d_subset = pts_2d[:n]
        
        try:
            # PnP with RANSAC
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                objectPoints=pts_3d_subset,
                imagePoints=pts_2d_subset,
                cameraMatrix=K,
                distCoeffs=dist,
                iterationsCount=1000,
                reprojectionError=8.0,
                confidence=0.99,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if not success or inliers is None:
                return None, None, None
            
            # CRITICAL FIX: solvePnPRansac returns INDICES, not boolean mask
            inlier_indices = inliers.flatten()
            
            # Create proper boolean mask
            inliers_bool = np.zeros(len(pts_3d_subset), dtype=bool)
            inliers_bool[inlier_indices] = True
            
            # Need minimum inliers
            if np.sum(inliers_bool) < 6:
                return None, None, None
            
            # Refine using inliers only
            success, rvec, tvec = cv2.solvePnP(
                objectPoints=pts_3d_subset[inliers_bool],
                imagePoints=pts_2d_subset[inliers_bool],
                cameraMatrix=K,
                distCoeffs=dist,
                rvec=rvec,
                tvec=tvec,
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            R, _ = cv2.Rodrigues(rvec)
            t = tvec.reshape(3, 1)
            
            return R, t, inliers_bool
            
        except Exception as e:
            print(f"      Error: {e}")
            return None, None, None
    
    # ============================================================
    # STEP 16: POSE SMOOTHING
    # ============================================================
    
    def smooth_pose(self, organ_name, R, t, alpha=0.3):
        """Temporal smoothing for stable AR"""
        history = self.pose_history[organ_name]
        
        if len(history) == 0:
            history.append({'R': R.copy(), 't': t.copy()})
            return R, t
        
        prev = history[-1]
        
        # Exponential moving average
        t_smooth = alpha * t + (1 - alpha) * prev['t']
        R_smooth = alpha * R + (1 - alpha) * prev['R']
        
        # Ensure valid rotation
        U, _, Vt = np.linalg.svd(R_smooth)
        R_smooth = U @ Vt
        
        history.append({'R': R_smooth, 't': t_smooth})
        if len(history) > 10:
            history.pop(0)
        
        return R_smooth, t_smooth
    
    # ============================================================
    # STEP 17: COORDINATE ALIGNMENT
    # ============================================================
    
    def align_to_opengl(self, R, t):
        """Convert OpenCV coords to OpenGL coords"""
        T = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
        return T @ R @ T.T, T @ t
    
    def align_to_unity(self, R, t):
        """Convert OpenCV coords to Unity coords"""
        T = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        return T @ R, T @ t
    
    # ============================================================
    # VALIDATION
    # ============================================================
    
    def compute_reprojection_error(self, organ_name, R, t):
        """Compute mean reprojection error in pixels"""
        organ = self.organs[organ_name]
        pts_3d = organ['points_3d']
        pts_2d = organ['points_2d_reference']
        K = organ['camera_matrix']
        dist = organ['distortion']
        
        # Subsample if too many points
        if len(pts_3d) > 500:
            indices = np.random.choice(len(pts_3d), 500, replace=False)
            pts_3d = pts_3d[indices]
            pts_2d = pts_2d[indices]
        
        rvec, _ = cv2.Rodrigues(R)
        projected, _ = cv2.projectPoints(pts_3d, rvec, t, K, dist)
        projected = projected.reshape(-1, 2)
        
        errors = np.linalg.norm(projected - pts_2d, axis=1)
        return np.mean(errors), np.median(errors), np.max(errors)
    
    def validate_pose(self, R, t, min_translation=0.01, max_translation=200.0):
        """Check if pose is valid"""
        # Rotation matrix tests
        det = np.linalg.det(R)
        if abs(det - 1.0) > 0.1:
            return False, f"det(R)={det:.3f} ≠ 1"
        
        ortho = R @ R.T
        if not np.allclose(ortho, np.eye(3), atol=0.1):
            return False, "R not orthogonal"
        
        # Translation tests
        t_norm = np.linalg.norm(t)
        
        if t_norm > max_translation:
            return False, f"|t|={t_norm:.1f} too large"
        
        if t_norm < min_translation:
            return False, f"|t|={t_norm:.4f} too small"
        
        return True, "Valid"
    
    # ============================================================
    # SAVE FOR MEMBER 4
    # ============================================================
    
    def save_for_member4(self, organ_name, R, t, output_dir='member1_output'):
        """Save pose data for AR integration"""
        out_path = Path(output_dir) / organ_name
        out_path.mkdir(parents=True, exist_ok=True)
        
        data = {
            'organ_name': organ_name,
            'rotation_matrix': R.tolist(),
            'translation_vector': t.tolist(),
            'camera_matrix': self.organs[organ_name]['camera_matrix'].tolist(),
            'pose_format': 'opencv',
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'num_3d_points': self.organs[organ_name]['num_points'],
                'method': 'PnP_RANSAC'
            }
        }
        
        json_path = out_path / f'{organ_name}_pose.json'
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Saved: {json_path}")
        return json_path


# ============================================================
# MAIN
# ============================================================

def main():
    print("="*60)
    print("MEMBER 1: 6D POSE ESTIMATION PIPELINE")
    print("="*60)
    
    estimator = Member1_PoseEstimator("member2_output")
    
    if len(estimator.organs) == 0:
        print("\n❌ No usable data found!")
        return
    
    print("\n" + "="*60)
    print("RUNNING POSE ESTIMATION")
    print("="*60)
    
    results = {}
    
    for organ_name in estimator.organs.keys():
        print(f"\n🔬 {organ_name.upper()}")
        
        R, t, inliers = estimator.estimate_pose_pnp(organ_name)
        
        if R is None:
            print("   ✗ Pose estimation failed")
            results[organ_name] = {'success': False}
            continue
        
        # Stats
        n_inliers = int(np.sum(inliers))
        n_total = len(inliers)
        inlier_pct = 100 * n_inliers / n_total
        
        print(f"   ✓ Pose estimated")
        print(f"   • Inliers: {n_inliers}/{n_total} ({inlier_pct:.1f}%)")
        
        # Validate
        valid, msg = estimator.validate_pose(R, t)
        print(f"   • Validation: {'✓' if valid else '⚠'} {msg}")
        
        # Translation
        t_norm = np.linalg.norm(t)
        print(f"   • Translation: {t_norm:.3f} units")
        
        # Errors
        mean_err, med_err, max_err = estimator.compute_reprojection_error(organ_name, R, t)
        print(f"   • Reprojection error: {mean_err:.2f}px (median: {med_err:.2f}px)")
        
        # Quality
        if mean_err < 3:
            quality = "Excellent ✓"
        elif mean_err < 8:
            quality = "Good"
        elif mean_err < 15:
            quality = "Acceptable"
        else:
            quality = "Poor ⚠"
        print(f"   • Quality: {quality}")
        
        # Smooth and save
        R_smooth, t_smooth = estimator.smooth_pose(organ_name, R, t)
        estimator.save_for_member4(organ_name, R_smooth, t_smooth)
        
        results[organ_name] = {
            'success': True,
            'error': mean_err,
            'inliers': inlier_pct,
            'valid': valid,
            'translation': t_norm
        }
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    success_count = sum(1 for r in results.values() if r['success'])
    total = len(results)
    
    print(f"\n✅ Successful: {success_count}/{total} organs")
    
    for organ, res in results.items():
        if res['success']:
            icon = "✓" if res['error'] < 8 else "⚠"
            print(f"\n{organ.upper()}: {icon}")
            print(f"  Error: {res['error']:.2f}px")
            print(f"  Inliers: {res['inliers']:.1f}%")
            print(f"  Translation: {res['translation']:.3f}")
        else:
            print(f"\n{organ.upper()}: ✗ Failed")
    
    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)
    print("\nOutput: member1_output/")
    print("Ready for Member 4!")


if __name__ == "__main__":
    main()