"""
Member 2: Feature Matching & Sparse 3D Reconstruction
Markerless AR Anatomy Learning Project

Author: Member 2
Dependencies: opencv-python, numpy, torch, kornia, open3d
"""

import os
import json
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    from kornia.feature import LoFTR
    LOFTR_AVAILABLE = True
except ImportError:
    LOFTR_AVAILABLE = False
    print("LoFTR not available, will use SIFT fallback")

try:
    import open3d as o3d
    O3D_AVAILABLE = True
except ImportError:
    O3D_AVAILABLE = False
    print("Open3D not available, PLY files will be created manually")


# CONFIGURATION
CONFIG = {
    'images_path': r'D:/SEM 6/CV&IP/Project_Asmi/dataset/anatomy_dataset/images/train/',
    'labels_path': r'D:/SEM 6/CV&IP/Project_Asmi/dataset/anatomy_dataset/labels/train/',
    'camera_matrix_path': r'D:/SEM 6/CV&IP/CameraCalibration/output/intrinsic_matrix.txt',
    'loftr_weights_path': r'C:\Users\ADMIN\.cache\torch\hub\checkpoints\loftr_outdoor.ckpt',
    'output_dir': './member2_output/',
    
    'class_names': {0: 'heart', 1: 'brain', 2: 'lungs'},
    'bbox_margin': 20,
    'min_matches': 8,
    'loftr_confidence_threshold': {0: 0.5, 1: 0.3, 2: 0.5},  
    'min_translation': 0.05,
    'min_z_variation': 0.01,
    'max_z_threshold': 1000.0,
    'ransac_threshold': 1.0,
}


# UTILITY FUNCTIONS

def load_camera_matrix(path):
    """Load 3x3 camera intrinsic matrix from text file."""
    try:
        K = np.loadtxt(path).reshape(3, 3)
        print(f" Camera matrix loaded from {path}")
        print(f"  fx={K[0,0]:.2f}, fy={K[1,1]:.2f}, cx={K[0,2]:.2f}, cy={K[1,2]:.2f}")
        return K
    except Exception as e:
        print(f" Failed to load camera matrix: {e}")
        print("  Using default camera matrix")
        return np.array([
            [9.4567122e+02, 0.0000000e+00, 4.7847478e+02],
            [0.0000000e+00, 9.4531944e+02, 6.4779593e+02],
            [0.0000000e+00, 0.0000000e+00, 1.0000000e+00]
        ])


def load_yolo_bboxes(label_path, img_width, img_height, margin=20):
    """
    Parse YOLO format labels and convert to absolute pixel coordinates.
    Format: class_id x_center y_center width height (all normalized 0-1)
    Returns: list of (class_id, x1, y1, x2, y2)
    """
    bboxes = []
    if not os.path.exists(label_path):
        return bboxes
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            class_id = int(parts[0])
            x_center = float(parts[1]) * img_width
            y_center = float(parts[2]) * img_height
            width = float(parts[3]) * img_width
            height = float(parts[4]) * img_height
            
            x1 = int(max(0, x_center - width/2 - margin))
            y1 = int(max(0, y_center - height/2 - margin))
            x2 = int(min(img_width, x_center + width/2 + margin))
            y2 = int(min(img_height, y_center + height/2 + margin))
            
            bboxes.append((class_id, x1, y1, x2, y2))
    
    return bboxes


def get_images_by_class(images_path, labels_path):
    """
    Organize images by class ID based on YOLO labels.
    Returns: dict {class_id: [(img_path, bbox), ...]}
    """
    images_path = Path(images_path)
    labels_path = Path(labels_path)
    
    class_images = {0: [], 1: [], 2: []}  # heart, brain, lungs
    
    for img_file in sorted(images_path.glob('*.jpeg')) + sorted(images_path.glob('*.jpg')):
        label_file = labels_path / (img_file.stem + '.txt')
        
        if not label_file.exists():
            continue
        
        # Read image to get dimensions
        img = cv2.imread(str(img_file))
        if img is None:
            continue
        
        h, w = img.shape[:2]
        bboxes = load_yolo_bboxes(str(label_file), w, h, CONFIG['bbox_margin'])
        
        for class_id, x1, y1, x2, y2 in bboxes:
            if class_id in class_images:
                class_images[class_id].append((str(img_file), (x1, y1, x2, y2)))
    
    # Print statistics
    print("\n Dataset Statistics:")
    for class_id, name in CONFIG['class_names'].items():
        count = len(class_images[class_id])
        print(f"  Class {class_id} ({name.upper()}): {count} images")
    
    return class_images


# FEATURE MATCHING

class FeatureMatcher:
    """Handles feature matching using LoFTR or SIFT fallback."""
    
    def __init__(self, use_loftr=True):
        self.method = None
        
        if use_loftr and LOFTR_AVAILABLE:
            try:
                self.matcher = self._init_loftr()
                self.method = 'LoFTR'
                print(" Using LoFTR for feature matching")
            except Exception as e:
                print(f" LoFTR initialization failed: {e}")
                self._init_sift()
        else:
            self._init_sift()
    
    def _init_loftr(self):
        """Initialize LoFTR with local weights."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize LoFTR
        matcher = LoFTR(pretrained='outdoor').to(device).eval()
        
        # Load local weights if available
        weights_path = CONFIG['loftr_weights_path']
        if os.path.exists(weights_path):
            try:
                checkpoint = torch.load(weights_path, map_location=device)
                # Handle 'state_dict' wrapper
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                
                # Load state dict
                matcher.load_state_dict(state_dict, strict=False)
                print(f"  Loaded weights from {weights_path}")
            except Exception as e:
                print(f"   Failed to load local weights: {e}, using pretrained")
        
        return matcher
    
    def _init_sift(self):
        """Initialize SIFT as fallback."""
        self.sift = cv2.SIFT_create(nfeatures=2000)
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        self.method = 'SIFT'
        print(" Using SIFT for feature matching")
    
    def match(self, img1, img2, bbox1, bbox2, class_id):
        """
        Match features between two images within bounding box regions.
        Returns: (pts1, pts2, confidence)
        """
        # Crop regions
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        crop1 = img1[y1_1:y2_1, x1_1:x2_1]
        crop2 = img2[y1_2:y2_2, x1_2:x2_2]
        
        if crop1.size == 0 or crop2.size == 0:
            return None, None, None
        
        if self.method == 'LoFTR':
            pts1, pts2, conf = self._match_loftr(crop1, crop2, class_id)
            
            # Auto-fallback to SIFT if LoFTR fails
            if pts1 is None:
                if not hasattr(self, '_sift_fallback_initialized'):
                    print("     Initializing SIFT fallback...")
                    self._init_sift()
                    self._sift_fallback_initialized = True
                pts1, pts2, conf = self._match_sift(crop1, crop2)
        else:
            pts1, pts2, conf = self._match_sift(crop1, crop2)
        
        if pts1 is None:
            return None, None, None
        
        # Convert coordinates back to original image space
        pts1[:, 0] += x1_1
        pts1[:, 1] += y1_1
        pts2[:, 0] += x1_2
        pts2[:, 1] += y1_2
        
        return pts1, pts2, conf
    
    def _match_loftr(self, img1, img2, class_id):
        """Match using LoFTR with proper API handling."""
        device = next(self.matcher.parameters()).device
        
        # Convert to grayscale and normalize
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2
        
        # Resize if images are too small (LoFTR works best with larger images)
        min_size = 200
        h1, w1 = gray1.shape
        h2, w2 = gray2.shape
        
        scale1 = 1.0
        scale2 = 1.0
        
        if h1 < min_size or w1 < min_size:
            scale1 = min_size / min(h1, w1)
            gray1 = cv2.resize(gray1, None, fx=scale1, fy=scale1, interpolation=cv2.INTER_LINEAR)
        
        if h2 < min_size or w2 < min_size:
            scale2 = min_size / min(h2, w2)
            gray2 = cv2.resize(gray2, None, fx=scale2, fy=scale2, interpolation=cv2.INTER_LINEAR)
        
        # Prepare input tensors
        img1_tensor = torch.from_numpy(gray1)[None, None].float() / 255.0
        img2_tensor = torch.from_numpy(gray2)[None, None].float() / 255.0
        
        input_dict = {
            'image0': img1_tensor.to(device),
            'image1': img2_tensor.to(device)
        }
        
        try:
            with torch.no_grad():
                # Run LoFTR inference - returns a dict with matches
                output = self.matcher(input_dict)
                
                # Extract matches from output dict
                mkpts0 = None
                mkpts1 = None
                mconf = None
                
                # Option 1: Direct output dict format (kornia LoFTR returns this)
                if isinstance(output, dict) and 'keypoints0' in output:
                    mkpts0 = output['keypoints0'].cpu().numpy()
                    mkpts1 = output['keypoints1'].cpu().numpy()
                    mconf = output.get('confidence', torch.ones(len(mkpts0))).cpu().numpy()
                
                # Option 2: Check if results were added to input_dict (older LoFTR versions)
                elif 'mkpts0_f' in input_dict:
                    mkpts0 = input_dict['mkpts0_f'].cpu().numpy()
                    mkpts1 = input_dict['mkpts1_f'].cpu().numpy()
                    mconf = input_dict.get('mconf', torch.ones(len(mkpts0))).cpu().numpy()
                
                # Option 3: Coarse matches in input_dict
                elif 'mkpts0_c' in input_dict:
                    mkpts0 = input_dict['mkpts0_c'].cpu().numpy()
                    mkpts1 = input_dict['mkpts1_c'].cpu().numpy()
                    mconf = np.ones(len(mkpts0))
                
                # Option 4: Keypoints format in input_dict
                elif 'keypoints0' in input_dict and 'keypoints1' in input_dict:
                    kpts0 = input_dict['keypoints0'].cpu().numpy()
                    kpts1 = input_dict['keypoints1'].cpu().numpy()
                    
                    if 'matches0' in input_dict:
                        matches = input_dict['matches0'].cpu().numpy()
                        valid = matches > -1
                        mkpts0 = kpts0[valid]
                        mkpts1 = kpts1[matches[valid].astype(int)]
                        mconf = input_dict.get('matching_scores0', torch.ones(valid.sum())).cpu().numpy()
                    else:
                        mkpts0 = kpts0
                        mkpts1 = kpts1
                        mconf = input_dict.get('confidence', torch.ones(len(mkpts0))).cpu().numpy()
                
                else:
                    print(f"     Unexpected LoFTR output format")
                    print(f"       Output type: {type(output)}")
                    if isinstance(output, dict):
                        print(f"       Output keys: {list(output.keys())}")
                    print(f"       Input dict keys: {list(input_dict.keys())}")
                    print(f"    Falling back to SIFT...")
                    return None, None, None
                
                if mkpts0 is None or len(mkpts0) == 0:
                    return None, None, None
                
        except Exception as e:
            print(f"     LoFTR inference failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None
        
        # Scale points back to original size
        if scale1 != 1.0:
            mkpts0 = mkpts0 / scale1
        if scale2 != 1.0:
            mkpts1 = mkpts1 / scale2
        
        # Reshape if needed
        if mkpts0.ndim == 3:
            mkpts0 = mkpts0.squeeze(0)
        if mkpts1.ndim == 3:
            mkpts1 = mkpts1.squeeze(0)
        if mconf.ndim == 2:
            mconf = mconf.squeeze(0)
        
        # Filter by confidence
        threshold = CONFIG['loftr_confidence_threshold'][class_id]
        mask = mconf > threshold
        
        if mask.sum() == 0:
            # If no matches pass threshold, return top matches
            if len(mconf) >= CONFIG['min_matches']:
                top_k = min(CONFIG['min_matches'] * 2, len(mconf))
                top_indices = np.argsort(mconf)[-top_k:]
                mask = np.zeros(len(mconf), dtype=bool)
                mask[top_indices] = True
            else:
                return None, None, None
        
        return mkpts0[mask], mkpts1[mask], mconf[mask]
    
    def _match_sift(self, img1, img2):
        """Match using SIFT + FLANN."""
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2
        
        kp1, des1 = self.sift.detectAndCompute(gray1, None)
        kp2, des2 = self.sift.detectAndCompute(gray2, None)
        
        if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
            return None, None, None
        
        matches = self.flann.knnMatch(des1, des2, k=2)
        
        # Lowe's ratio test
        good_matches = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < CONFIG['min_matches']:
            return None, None, None
        
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
        conf = np.ones(len(good_matches)) * 0.8  # Dummy confidence
        
        return pts1, pts2, conf


# TRIANGULATION

def triangulate_points(pts1, pts2, K, min_translation=0.05, min_z_var=0.01):
    """
    Triangulate 3D points from 2D correspondences with degeneracy checks.
    
    Returns: (points_3d, R, t, inlier_mask) or (None, None, None, None) if degenerate
    """
    if len(pts1) < 8:
        print(f"     Insufficient points: {len(pts1)} < 8")
        return None, None, None, None
    
    # Normalize points using camera matrix
    pts1_norm = cv2.undistortPoints(pts1.reshape(-1, 1, 2), K, None, None, None, K)
    pts2_norm = cv2.undistortPoints(pts2.reshape(-1, 1, 2), K, None, None, None, K)
    
    pts1_norm = pts1_norm.reshape(-1, 2)
    pts2_norm = pts2_norm.reshape(-1, 2)
    
    # Find fundamental matrix with RANSAC
    F, mask_F = cv2.findFundamentalMat(
        pts1_norm, pts2_norm, 
        cv2.FM_RANSAC, 
        CONFIG['ransac_threshold'], 
        0.99
    )
    
    if F is None or mask_F is None:
        print("     Failed to compute fundamental matrix")
        return None, None, None, None
    
    mask_F = mask_F.ravel().astype(bool)
    
    if mask_F.sum() < CONFIG['min_matches']:
        print(f"     Too few inliers after F estimation: {mask_F.sum()}")
        return None, None, None, None
    
    # Compute essential matrix: E = K.T @ F @ K
    E = K.T @ F @ K
    
    # Recover pose
    _, R, t, pose_mask = cv2.recoverPose(
        E, 
        pts1_norm[mask_F], 
        pts2_norm[mask_F], 
        K
    )
    
    # Check translation magnitude
    t_mag = np.linalg.norm(t)
    print(f"    Translation magnitude: {t_mag:.4f}", end="")
    
    if t_mag < min_translation:
        print(f"  WARNING: Small baseline (< {min_translation})")
    else:
        print(" ")
    
    # Create projection matrices
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t])
    
    # Triangulate points
    points_4d = cv2.triangulatePoints(
        P1, P2,
        pts1_norm[mask_F].T,
        pts2_norm[mask_F].T
    )
    
    # Convert from homogeneous to 3D
    points_3d = points_4d[:3, :] / points_4d[3, :]
    points_3d = points_3d.T
    
    # Filter points with extreme depth
    z_values = points_3d[:, 2]
    valid_mask = np.abs(z_values) < CONFIG['max_z_threshold']
    
    if valid_mask.sum() == 0:
        print("     All points have extreme Z values")
        return None, None, None, None
    
    points_3d = points_3d[valid_mask]
    
    # Check Z variation to detect degeneracy
    z_var = np.std(points_3d[:, 2])
    z_range = np.ptp(points_3d[:, 2])
    print(f"    Z variation: std={z_var:.4f}, range={z_range:.4f}", end="")
    
    if z_var < min_z_var:
        print(f"  WARNING: Low depth variation (< {min_z_var})")
    else:
        print(" ")
    
    # Create inlier mask for original points
    full_mask = np.zeros(len(pts1), dtype=bool)
    full_mask[mask_F] = valid_mask
    
    print(f"    Final 3D points: {len(points_3d)}")
    
    return points_3d, R, t, full_mask


# OUTPUT GENERATION

def save_ply_manual(filename, points_3d, colors=None):
    """Manually create PLY file without Open3D dependency."""
    if colors is None:
        colors = np.tile([128, 128, 128], (len(points_3d), 1))
    
    with open(filename, 'w') as f:
        # Header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points_3d)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        # Data
        for i in range(len(points_3d)):
            f.write(f"{points_3d[i,0]} {points_3d[i,1]} {points_3d[i,2]} ")
            f.write(f"{int(colors[i,0])} {int(colors[i,1])} {int(colors[i,2])}\n")


def save_ply(filename, points_3d):
    """Save point cloud to PLY file."""
    if O3D_AVAILABLE:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_3d)
        o3d.io.write_point_cloud(filename, pcd)
    else:
        save_ply_manual(filename, points_3d)
    
    print(f"   Saved PLY: {filename}")


def save_json(filename, data):
    """Save JSON with proper formatting."""
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"   Saved JSON: {filename}")


# MAIN RECONSTRUCTION PIPELINE

def reconstruct_class(class_id, image_bbox_pairs, K, matcher):
    """
    Reconstruct 3D points for a single class (organ).
    
    Returns: (all_points_3d, correspondences_list, metadata)
    """
    class_name = CONFIG['class_names'][class_id]
    print(f"\n{'='*60}")
    print(f" Processing Class {class_id}: {class_name.upper()}")
    print(f"{'='*60}")
    print(f"Total images: {len(image_bbox_pairs)}")
    
    if len(image_bbox_pairs) < 2:
        print("   Need at least 2 images for reconstruction")
        return None, None, None
    
    # Use first image as reference
    ref_img_path, ref_bbox = image_bbox_pairs[0]
    ref_img = cv2.imread(ref_img_path)
    
    print(f"\n Reference image: {Path(ref_img_path).name}")
    
    all_points_3d = []
    correspondences = []
    
    # Match reference with other views
    for i, (img_path, bbox) in enumerate(image_bbox_pairs[1:], 1):
        print(f"\n  [{i}/{len(image_bbox_pairs)-1}] Matching with {Path(img_path).name}")
        
        img = cv2.imread(img_path)
        if img is None:
            print("     Failed to load image")
            continue
        
        # Feature matching
        pts1, pts2, conf = matcher.match(ref_img, img, ref_bbox, bbox, class_id)
        
        if pts1 is None:
            print(f"     No matches found")
            continue
        
        print(f"    Matches found: {len(pts1)}")
        
        # Triangulation
        points_3d, R, t, mask = triangulate_points(
            pts1, pts2, K,
            CONFIG['min_translation'],
            CONFIG['min_z_variation']
        )
        
        if points_3d is None:
            print("     Triangulation failed")
            continue
        
        # Store correspondence
        correspondence = {
            'points_3d': points_3d.tolist(),
            'points_2d_1': pts1[mask].tolist(),
            'points_2d_2': pts2[mask].tolist(),
            'confidence': conf[mask].tolist() if conf is not None else [1.0] * len(points_3d),
            'rotation': R.tolist(),
            'translation': t.flatten().tolist(),
            'num_points': len(points_3d),
            'class_id': int(class_id),
            'image_pair': [Path(ref_img_path).name, Path(img_path).name]
        }
        
        correspondences.append(correspondence)
        all_points_3d.append(points_3d)
    
    # Combine all 3D points
    if len(all_points_3d) > 0:
        all_points_3d = np.vstack(all_points_3d)
        print(f"\n Total 3D points reconstructed: {len(all_points_3d)}")
    else:
        all_points_3d = None
        print(f"\n No 3D points reconstructed for {class_name}")
    
    # Metadata
    metadata = {
        'feature_matcher': matcher.method,
        'total_correspondences': len(all_points_3d) if all_points_3d is not None else 0,
        'num_views': len(correspondences) + 1,
        'timestamp': datetime.now().isoformat(),
        'class_id': int(class_id),
        'class_name': class_name
    }
    
    return all_points_3d, correspondences, metadata


def main():
    """Main pipeline for Member 2."""
    print("\n" + "="*60)
    print(" MEMBER 2: Feature Matching & Sparse 3D Reconstruction")
    print("="*60)
    
    # Load camera matrix
    K = load_camera_matrix(CONFIG['camera_matrix_path'])
    
    # Organize images by class
    class_images = get_images_by_class(CONFIG['images_path'], CONFIG['labels_path'])
    
    # Initialize feature matcher
    matcher = FeatureMatcher(use_loftr=LOFTR_AVAILABLE)
    
    # Create output directory
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Process each class
    for class_id in [0, 1, 2]:  # heart, brain, lungs
        class_name = CONFIG['class_names'][class_id]
        image_bbox_pairs = class_images[class_id]
        
        if len(image_bbox_pairs) == 0:
            print(f"\n No images found for class {class_id} ({class_name})")
            continue
        
        # Reconstruct
        points_3d, correspondences, metadata = reconstruct_class(
            class_id, image_bbox_pairs, K, matcher
        )
        
        if points_3d is None or len(correspondences) == 0:
            print(f"\n Skipping output for {class_name} (no valid reconstructions)")
            continue
        
        # Create class output directory
        class_dir = output_dir / class_name
        class_dir.mkdir(exist_ok=True)
        
        # Save PLY point cloud
        ply_path = class_dir / f"{class_name}_sparse_cloud.ply"
        save_ply(str(ply_path), points_3d)
        
        # Save JSON correspondences
        json_data = {
            'object_name': class_name,
            'camera_matrix': K.tolist(),
            'distortion_coeffs': [0, 0, 0, 0, 0],
            'correspondences': correspondences,
            'metadata': metadata
        }
        
        json_path = class_dir / f"{class_name}_correspondences.json"
        save_json(str(json_path), json_data)
    
    print("\n" + "="*60)
    print(" RECONSTRUCTION COMPLETE")
    print("="*60)
    print(f"Output directory: {output_dir.absolute()}")
    print("\nGenerated files:")
    for class_name in CONFIG['class_names'].values():
        class_dir = output_dir / class_name
        if class_dir.exists():
            print(f"   {class_name}/")
            for file in class_dir.glob('*'):
                print(f"    - {file.name}")


if __name__ == "__main__":
    main()