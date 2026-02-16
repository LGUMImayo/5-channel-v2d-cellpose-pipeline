#!/usr/bin/env python3
"""
Create pseudo masks for FN cells (GT cells not detected by silver masks).

By default, pseudo masks are now intensity-guided shapes extracted from local
NeuN/DAPI signal around FN centroids, with a robust fallback to circular masks.

Usage:
    python create_pseudo_masks.py
    python create_pseudo_masks.py --method circular --radius 20
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cdist
from scipy import ndimage
from skimage.draw import disk
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, binary_closing, disk as morph_disk

sys.path.insert(0, str(Path(__file__).parent))

from config import PipelineConfig
from data_utils import read_multichannel_tiff, parse_cellcounter_xml


def get_fn_centroids(
    silver_mask: np.ndarray,
    gt_points: np.ndarray,
    match_radius: float = 30.0
) -> np.ndarray:
    """
    Find GT cells that are false negatives (not matched to any silver mask).
    
    Returns:
        Array of (y, x) centroids for FN cells
    """
    # Get silver mask centroids
    from scipy.ndimage import center_of_mass
    
    unique_labels = np.unique(silver_mask)
    unique_labels = unique_labels[unique_labels > 0]
    
    if len(unique_labels) == 0:
        return gt_points  # All GT are FN

    # Fast centroid extraction for all labels at once
    ones = np.ones_like(silver_mask, dtype=np.uint8)
    silver_centroids = np.array(
        center_of_mass(ones, labels=silver_mask, index=unique_labels.tolist())
    )
    
    # Match GT to silver
    gt_points = np.asarray(gt_points)
    if len(gt_points) == 0:
        return np.array([])
    
    # GT points are (x, y), convert to (y, x)
    gt_yx = gt_points[:, ::-1]  # (x, y) -> (y, x)
    
    # Compute distances
    distances = cdist(gt_yx, silver_centroids)
    
    # Find unmatched GT (FN)
    min_dist_per_gt = distances.min(axis=1)
    fn_mask = min_dist_per_gt > match_radius
    
    fn_centroids = gt_yx[fn_mask]
    
    return fn_centroids


def _add_circular_pseudo_masks(
    base_mask: np.ndarray,
    fn_centroids: np.ndarray,
    radius: int = 18,
):
    """Legacy circular pseudo masks used as fallback or explicit method."""
    H, W = base_mask.shape
    max_label = int(base_mask.max())
    augmented_mask = base_mask.copy()
    n_added = 0

    for cy, cx in fn_centroids:
        cy, cx = int(cy), int(cx)

        if cy < radius or cy >= H - radius or cx < radius or cx >= W - radius:
            continue

        check_region = augmented_mask[
            max(0, cy - radius // 2):min(H, cy + radius // 2),
            max(0, cx - radius // 2):min(W, cx + radius // 2),
        ]
        if check_region.max() > 0:
            continue

        rr, cc = disk((cy, cx), radius, shape=(H, W))
        new_label = max_label + n_added + 1
        augmented_mask[rr, cc] = new_label
        n_added += 1

    return augmented_mask, n_added


def _extract_intensity_guided_region(
    img_hw2: np.ndarray,
    cy: int,
    cx: int,
    patch_radius: int = 48,
    min_area: int = 350,
    max_area: int = 4500,
):
    """Extract a pseudo-cell region around centroid using local NeuN/DAPI intensity."""
    H, W, C = img_hw2.shape
    if C < 2:
        return None

    y0 = max(0, cy - patch_radius)
    y1 = min(H, cy + patch_radius + 1)
    x0 = max(0, cx - patch_radius)
    x1 = min(W, cx + patch_radius + 1)

    patch = img_hw2[y0:y1, x0:x1]
    if patch.size == 0:
        return None

    neun = patch[:, :, 0].astype(np.float32) / 255.0
    dapi = patch[:, :, 1].astype(np.float32) / 255.0
    signal = 0.7 * neun + 0.3 * dapi
    signal = ndimage.gaussian_filter(signal, sigma=1.2)

    try:
        thr = threshold_otsu(signal)
    except Exception:
        thr = np.percentile(signal, 70)

    binary = signal > thr
    binary = binary_closing(binary, footprint=morph_disk(2))
    binary = remove_small_objects(binary, min_size=80)

    local_y = cy - y0
    local_x = cx - x0
    if local_y < 0 or local_y >= binary.shape[0] or local_x < 0 or local_x >= binary.shape[1]:
        return None

    labels, ncc = ndimage.label(binary)
    if ncc == 0:
        return None

    center_lbl = labels[local_y, local_x]
    if center_lbl == 0:
        # If center is not in foreground, take nearest component to center.
        best_lbl = 0
        best_dist = np.inf
        for lbl in range(1, ncc + 1):
            ys, xs = np.where(labels == lbl)
            if len(ys) == 0:
                continue
            my = np.mean(ys)
            mx = np.mean(xs)
            d2 = (my - local_y) ** 2 + (mx - local_x) ** 2
            if d2 < best_dist:
                best_dist = d2
                best_lbl = lbl
        center_lbl = best_lbl
        if center_lbl == 0:
            return None

    region_local = labels == center_lbl
    area = int(region_local.sum())
    if area < min_area or area > max_area:
        return None

    yy, xx = np.where(region_local)
    yy_global = yy + y0
    xx_global = xx + x0
    return yy_global, xx_global


def add_pseudo_masks(
    silver_mask: np.ndarray,
    img_hw2: np.ndarray,
    fn_centroids: np.ndarray,
    radius: int = 18,
    method: str = "intensity",
    patch_radius: int = 48,
) -> np.ndarray:
    """
    Add circular pseudo masks around FN centroids.
    
    Args:
        silver_mask: Existing silver mask with labeled cells
        img_hw2: CLAHE image, shape (H, W, 2) [NeuN, DAPI]
        fn_centroids: (N, 2) array of (y, x) centroids for FN cells
        radius: Fallback circle radius
        method: "intensity" (default) or "circular"
        patch_radius: Local patch half-size for intensity-guided extraction
    
    Returns:
        Updated mask with pseudo masks added
    """
    H, W = silver_mask.shape
    max_label = int(silver_mask.max())
    augmented_mask = silver_mask.copy()

    if method == "circular":
        return _add_circular_pseudo_masks(augmented_mask, fn_centroids, radius=radius)

    n_added = 0
    for cy, cx in fn_centroids:
        cy, cx = int(cy), int(cx)

        if cy < 2 or cy >= H - 2 or cx < 2 or cx >= W - 2:
            continue

        check_region = augmented_mask[
            max(0, cy - radius // 2):min(H, cy + radius // 2),
            max(0, cx - radius // 2):min(W, cx + radius // 2),
        ]
        if check_region.max() > 0:
            continue

        region = _extract_intensity_guided_region(
            img_hw2=img_hw2,
            cy=cy,
            cx=cx,
            patch_radius=patch_radius,
        )

        new_label = max_label + n_added + 1

        if region is not None:
            yy, xx = region
            # Prevent overlap into existing labels
            overlap = augmented_mask[yy, xx] > 0
            if overlap.all():
                continue
            yy = yy[~overlap]
            xx = xx[~overlap]
            if len(yy) < 200:
                # Too small after overlap pruning, fallback to circle
                rr, cc = disk((cy, cx), radius, shape=(H, W))
                free = augmented_mask[rr, cc] == 0
                rr = rr[free]
                cc = cc[free]
                if len(rr) == 0:
                    continue
                augmented_mask[rr, cc] = new_label
            else:
                augmented_mask[yy, xx] = new_label
        else:
            rr, cc = disk((cy, cx), radius, shape=(H, W))
            free = augmented_mask[rr, cc] == 0
            rr = rr[free]
            cc = cc[free]
            if len(rr) == 0:
                continue
            augmented_mask[rr, cc] = new_label

        n_added += 1

    return augmented_mask, n_added


def process_all_cases(
    silver_mask_dir: Path,
    output_dir: Path,
    radius: int = 18,
    match_radius: float = 30.0,
    method: str = "intensity",
    patch_radius: int = 48,
):
    """
    Process all cases: add pseudo masks for FN cells.
    """
    config = PipelineConfig()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load manifest
    manifest = json.load(open(silver_mask_dir / 'silver_masks_manifest.json'))
    
    new_manifest = []
    total_silver = 0
    total_pseudo = 0
    total_gt = 0
    
    print(f"Creating pseudo masks with radius={radius}px")
    print(f"Pseudo-mask method: {method}")
    print(f"Match radius for FN detection: {match_radius}px")
    print("=" * 60)
    
    for entry in manifest:
        case_id = entry['case_id']
        source = entry.get('source', 'MAYO')
        
        # Load silver mask
        silver_mask = np.load(entry['mask_path'])
        img_hw2 = np.load(entry['img_path'])
        n_silver = int(silver_mask.max())
        
        # Load GT
        if source == 'MAYO':
            case_dir = Path(config.mayo_data_dir) / case_id
        elif source == 'LR':
            case_dir = Path(config.lr_data_dir) / case_id
        else:
            print(f"  {case_id}: Unknown source '{source}', skipping")
            continue
        
        xml_files = list(case_dir.glob("CellCounter_*.xml"))
        if not xml_files:
            print(f"  {case_id}: No GT XML found, skipping")
            continue
        
        marker_types = parse_cellcounter_xml(str(xml_files[0]))
        # marker_types[1] is a dict with 'name' and 'points' keys
        gt_dict = marker_types.get(1, {})
        gt_pts = gt_dict.get('points', np.zeros((0, 2))) if isinstance(gt_dict, dict) else np.zeros((0, 2))
        n_gt = len(gt_pts)
        
        # Find FN centroids
        fn_centroids = get_fn_centroids(silver_mask, gt_pts, match_radius)
        
        # Add pseudo masks
        augmented_mask, n_added = add_pseudo_masks(
            silver_mask=silver_mask,
            img_hw2=img_hw2,
            fn_centroids=fn_centroids,
            radius=radius,
            method=method,
            patch_radius=patch_radius,
        )
        
        # Save
        mask_filename = f"{case_id}_augmented_masks.npy"
        mask_path = output_dir / mask_filename
        np.save(mask_path, augmented_mask.astype(np.uint16))
        
        # Copy image reference
        img_path = entry['img_path']
        
        # Update manifest
        new_entry = {
            'case_id': case_id,
            'source': source,
            'img_path': img_path,
            'mask_path': str(mask_path),
            'silver_count': n_silver,
            'pseudo_count': n_added,
            'total_count': n_silver + n_added,
            'gt_count': n_gt,
            'fn_count': len(fn_centroids),
        }
        new_manifest.append(new_entry)
        
        total_silver += n_silver
        total_pseudo += n_added
        total_gt += n_gt
        
        # Calculate recall improvement
        old_recall = n_silver / n_gt if n_gt > 0 else 0
        new_recall = (n_silver + n_added) / n_gt if n_gt > 0 else 0
        
        print(f"  {case_id}: silver={n_silver}, +pseudo={n_added}, total={n_silver + n_added} | "
              f"GT={n_gt}, recall: {old_recall:.3f} → {new_recall:.3f}")
    
    # Save manifest
    manifest_path = output_dir / 'augmented_masks_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(new_manifest, f, indent=2)
    
    print("=" * 60)
    print(f"SUMMARY")
    print(f"  Total silver masks: {total_silver}")
    print(f"  Total pseudo masks: {total_pseudo} (+{100*total_pseudo/total_silver:.1f}%)")
    print(f"  Total combined:     {total_silver + total_pseudo}")
    print(f"  GT total:           {total_gt}")
    print(f"  Recall improvement: {total_silver/total_gt:.3f} → {(total_silver+total_pseudo)/total_gt:.3f}")
    print(f"\nOutput: {output_dir}")
    
    return new_manifest


def main():
    parser = argparse.ArgumentParser(description="Create pseudo masks for FN cells")
    parser.add_argument("--radius", type=int, default=18,
                       help="Radius of pseudo masks (default: 18, ~diameter 35px)")
    parser.add_argument("--match_radius", type=float, default=30.0,
                       help="Radius for matching GT to silver masks")
    parser.add_argument("--method", type=str, default="intensity", choices=["intensity", "circular"],
                       help="Pseudo-mask shape method")
    parser.add_argument("--patch_radius", type=int, default=48,
                       help="Local patch radius for intensity-guided pseudo masks")
    parser.add_argument("--output_suffix", type=str, default="augmented",
                       help="Suffix for output directory")
    args = parser.parse_args()
    
    config = PipelineConfig()
    
    # Use low-threshold silver masks as base (better recall)
    silver_mask_dir = Path(config.output_dir) / 'silver_masks_v2_merged'
    
    if not silver_mask_dir.exists():
        # Fall back to original silver masks
        silver_mask_dir = Path(config.output_dir) / 'silver_masks'
        print(f"Using original silver masks: {silver_mask_dir}")
    else:
        print(f"Using merged silver masks (low-threshold): {silver_mask_dir}")
    
    output_dir = Path(config.output_dir) / f'silver_masks_{args.output_suffix}'
    
    process_all_cases(
        silver_mask_dir=silver_mask_dir,
        output_dir=output_dir,
        radius=args.radius,
        match_radius=args.match_radius,
        method=args.method,
        patch_radius=args.patch_radius,
    )


if __name__ == '__main__':
    main()
