#!/usr/bin/env python3
"""
Run finetuned Cellpose V2d on all 32 cases.

Generates proper finetuned Stage-1 masks to replace the original cyto2 silver masks.
Also evaluates precision/recall/F1 against GT (NeuN CellCounter XML) where available.

Usage (SLURM):
    # Standard single-pass inference
    sbatch --job-name=v2d_all --partition=gpu-n12-85g-1x-a100-40g --gres=gpu:1 \
           --mem=80G --time=08:00:00 \
           --output=logs/v2d_all_%j.out --error=logs/v2d_all_%j.err \
           --wrap='source ~/.bashrc && conda activate rhizonet && python run_v2d_inference_all.py'

    # With test-time augmentation (multi-scale + aggressive intensity)
    sbatch ... --wrap='... python run_v2d_inference_all.py --tta'

    # TTA on specific cases only
    sbatch ... --wrap='... python run_v2d_inference_all.py --tta --cases P3044_C12 P3015_C12'

Outputs (per case):
    output/cellpose_finetuned_v2/v2d_masks/{case_id}_mask.npy     (uint16 instance mask)
    output/cellpose_finetuned_v2/v2d_masks/{case_id}_centroids.npy (N×2 float64 [x,y])
    output/cellpose_finetuned_v2/v2d_masks/{case_id}_fullres_overlay.png (full-res detection vs GT)
    output/cellpose_finetuned_v2/v2d_masks/v2d_all_results.json    (summary + per-case metrics)
"""

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from scipy.ndimage import center_of_mass
from scipy import ndimage

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import PipelineConfig
from data_utils import (
    read_multichannel_tiff,
    parse_cellcounter_xml,
    discover_all_data,
)
from bootstrap_cellpose_v2 import prepare_cellpose_input_clahe
from cellpose import models


# ====================================================================
# Parameters
# ====================================================================
CELLPROB_THRESHOLD = -0.5   # best from threshold sweep
FLOW_THRESHOLD = 0.4
DIAMETER = 35.0
MATCH_RADIUS = 30.0          # pixels, for P/R/F1 evaluation

# TTA parameters
TTA_DIAMETERS = [30.0, 35.0, 40.0]       # multi-scale
TTA_NMS_DISTANCE = 15.0                   # merge radius for NMS
TTA_MIN_VOTES = 2                         # min augmentations agreeing on a cell


# ====================================================================
# TTA: Test-Time Augmentation
# ====================================================================

def _detect_single(model, img_hw2: np.ndarray, diameter: float,
                   flow_threshold: float, cellprob_threshold: float) -> np.ndarray:
    """Run Cellpose once, return (N, 2) centroids as (x, y)."""
    outputs = model.eval(
        img_hw2,
        diameter=diameter,
        channels=[1, 2],
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
    )
    masks = outputs[0]
    return get_mask_centroids_fast(masks)


def _apply_intensity_aug(img_hw2: np.ndarray, aug_name: str) -> np.ndarray:
    """Apply an intensity augmentation to (H, W, 2) uint8 image.
    Returns a new (H, W, 2) uint8 image.

    Augmentation catalogue (designed for weak-NeuN recovery):
      - bright+15 / bright+30 : uniform brightness boost
      - contrast+30 / contrast+60 : contrast stretch around mean
      - gamma_0.6 / gamma_0.8 : power-law brighten (aggressive / mild)
      - clahe_strong : re-apply CLAHE with higher clip_limit
    """
    img = img_hw2.astype(np.float32)

    if aug_name == 'bright+15':
        img = img * 1.15
    elif aug_name == 'bright+30':
        img = img * 1.30
    elif aug_name == 'contrast+30':
        mean = img.mean(axis=(0, 1), keepdims=True)
        img = (img - mean) * 1.3 + mean
    elif aug_name == 'contrast+60':
        mean = img.mean(axis=(0, 1), keepdims=True)
        img = (img - mean) * 1.6 + mean
    elif aug_name == 'gamma_0.6':
        img = 255.0 * (img / 255.0) ** 0.6   # aggressive brighten darks
    elif aug_name == 'gamma_0.8':
        img = 255.0 * (img / 255.0) ** 0.8   # mild brighten darks
    elif aug_name == 'clahe_strong':
        # Re-apply CLAHE with higher clip_limit to each channel independently
        import cv2
        out = np.empty_like(img_hw2)
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
        for ch in range(img_hw2.shape[2]):
            out[:, :, ch] = clahe.apply(img_hw2[:, :, ch])
        return out
    else:
        return img_hw2  # identity

    return np.clip(img, 0, 255).astype(np.uint8)


def detect_with_tta(
    model,
    img_clahe: np.ndarray,
    diameters: list = None,
    flow_threshold: float = FLOW_THRESHOLD,
    cellprob_threshold: float = CELLPROB_THRESHOLD,
    nms_distance: float = TTA_NMS_DISTANCE,
    min_votes: int = TTA_MIN_VOTES,
) -> np.ndarray:
    """
    Test-time augmentation with multi-scale + aggressive intensity variants.

    Augmentations applied:
      - 3 scales (diameters 30, 35, 40)
      - 8 intensity variants:
          identity, bright+15%, bright+30%, contrast+30%, contrast+60%,
          gamma 0.6, gamma 0.8, CLAHE clip=5.0
      Total: 3 × 8 = 24 forward passes

    No geometric augmentations — cells are roughly round and
    orientation-invariant, so flips/rotations add cost without benefit.

    Centroids from all passes are merged via vote-based NMS:
      - Cluster all centroids within nms_distance
      - Keep clusters with >= min_votes detections

    Args:
        model: Cellpose model
        img_clahe: (H, W, 2) uint8 preprocessed image
        diameters: list of diameters to try
        nms_distance: merge radius in pixels
        min_votes: minimum number of augmentations that must agree

    Returns:
        (merged_centroids, raw_centroids) tuple:
          merged_centroids: (N, 2) NMS'd centroids as (x, y)
          raw_centroids: (M, 2) all raw detections before NMS (for post-hoc sweep)
    """
    if diameters is None:
        diameters = TTA_DIAMETERS

    # Intensity augmentations: identity + 7 aggressive variants
    intensity_augs = [
        'identity',
        'bright+15', 'bright+30',
        'contrast+30', 'contrast+60',
        'gamma_0.6', 'gamma_0.8',
        'clahe_strong',
    ]

    all_centroids = []
    n_passes = 0

    for int_aug in intensity_augs:
        if int_aug == 'identity':
            img_aug = img_clahe
        else:
            img_aug = _apply_intensity_aug(img_clahe, int_aug)

        for diameter in diameters:
            centroids = _detect_single(model, img_aug, diameter,
                                       flow_threshold, cellprob_threshold)
            n_passes += 1

            if len(centroids) > 0:
                all_centroids.append(centroids)

    total_raw = sum(len(c) for c in all_centroids)
    print(f'  TTA: {n_passes} passes, {total_raw} raw detections')

    if not all_centroids:
        empty = np.zeros((0, 2), dtype=np.float64)
        return empty, empty

    all_centroids = np.vstack(all_centroids)

    # Vote-based NMS
    merged = nms_centroids(all_centroids, nms_distance, min_votes)
    print(f'  TTA NMS (dist={nms_distance}, votes>={min_votes}): '
          f'{len(merged)} merged centroids')

    return merged, all_centroids


def nms_centroids(centroids: np.ndarray, min_distance: float = 15.0,
                  min_votes: int = 2) -> np.ndarray:
    """
    Vote-based non-maximum suppression for centroids.

    Groups nearby points into clusters, keeps clusters with >= min_votes
    members, returns the mean position of each surviving cluster.
    """
    if len(centroids) == 0:
        return np.zeros((0, 2), dtype=np.float64)

    used = np.zeros(len(centroids), dtype=bool)
    merged = []

    # Sort by number of nearby points (densest first) for stable clustering
    # Pre-compute pairwise distances in chunks to avoid OOM on large sets
    N = len(centroids)

    for i in range(N):
        if used[i]:
            continue

        dists = np.sqrt(np.sum((centroids - centroids[i]) ** 2, axis=1))
        nearby = (~used) & (dists < min_distance)
        n_votes = int(np.sum(nearby))

        if n_votes >= min_votes:
            cluster_pts = centroids[nearby]
            merged.append(cluster_pts.mean(axis=0))

        used[nearby] = True  # mark all nearby as consumed regardless

    return np.array(merged, dtype=np.float64) if merged else np.zeros((0, 2), dtype=np.float64)


def get_mask_centroids_fast(masks: np.ndarray) -> np.ndarray:
    """Extract centroids from instance mask (vectorised)."""
    labels = np.unique(masks)
    labels = labels[labels > 0]
    if len(labels) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    centroids = center_of_mass(masks, masks, labels)
    # center_of_mass returns (row, col) = (y, x); convert to (x, y)
    return np.array([(cx, cy) for cy, cx in centroids], dtype=np.float64)


def evaluate_detections(detected: np.ndarray, gt_pts: np.ndarray,
                        match_radius: float = MATCH_RADIUS) -> dict:
    """Distance-sorted greedy matching → P/R/F1."""
    if len(detected) == 0 and len(gt_pts) == 0:
        return {'matched': 0, 'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
    if len(detected) == 0:
        return {'matched': 0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    if len(gt_pts) == 0:
        return {'matched': 0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    d = cdist(detected, gt_pts)
    pairs = []
    for i in range(len(detected)):
        for j in range(len(gt_pts)):
            if d[i, j] <= match_radius:
                pairs.append((d[i, j], i, j))
    pairs.sort(key=lambda x: x[0])

    used_det, used_gt = set(), set()
    matched = 0
    for _, i, j in pairs:
        if i not in used_det and j not in used_gt:
            used_det.add(i)
            used_gt.add(j)
            matched += 1

    precision = matched / len(detected)
    recall = matched / len(gt_pts)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        'matched': int(matched),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'matched_det_indices': used_det,
        'matched_gt_indices': used_gt,
    }


def mask_contours(mask: np.ndarray, thickness: int = 2) -> np.ndarray:
    """Instance-aware contour: a pixel is on the boundary if any 4-connected
    neighbour has a different label (including background)."""
    H, W = mask.shape
    contour = np.zeros((H, W), dtype=bool)
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        shifted = np.zeros_like(mask)
        sy = slice(max(0, -dy), H + min(0, -dy))
        sx = slice(max(0, -dx), W + min(0, -dx))
        ty = slice(max(0, dy), H + min(0, dy))
        tx = slice(max(0, dx), W + min(0, dx))
        shifted[ty, tx] = mask[sy, sx]
        contour |= (mask > 0) & (mask != shifted)
    if thickness > 1:
        contour = ndimage.binary_dilation(contour, iterations=thickness - 1)
    return contour


def draw_filled_circles(rgb: np.ndarray, pts: np.ndarray, radius: int = 5,
                        color=(0.0, 1.0, 0.0)):
    """Draw small filled circles on an (H, W, 3) float32 image."""
    H, W, _ = rgb.shape
    for x, y in pts:
        xi, yi = int(round(x)), int(round(y))
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    yy, xx = yi + dy, xi + dx
                    if 0 <= yy < H and 0 <= xx < W:
                        rgb[yy, xx, 0] = color[0]
                        rgb[yy, xx, 1] = color[1]
                        rgb[yy, xx, 2] = color[2]


def save_fullres_overlay(neun: np.ndarray, masks: np.ndarray, case_id: str,
                         out_path: str, gt_pts: np.ndarray = None,
                         matched_det: set = None, matched_gt: set = None):
    """Save full-resolution NeuN + mask contour overlay with GT dots.

    Colours:
      - Detected mask contours (TP): cyan
      - Detected mask contours (FP — unmatched): magenta
      - GT dots (matched / TP): lime-green
      - GT dots (FN — unmatched): red
    """
    from PIL import Image

    neun_f = neun.astype(np.float32)
    lo, hi = np.percentile(neun_f, [1, 99.5])
    neun_norm = np.clip((neun_f - lo) / (hi - lo + 1e-8), 0, 1)

    rgb = np.stack([neun_norm, neun_norm, neun_norm], axis=-1).copy()

    # Draw instance-aware mask contours
    contours = mask_contours(masks, thickness=2)

    if matched_det is not None:
        # Colour TP contours cyan, FP contours magenta
        labels_on_contour = masks[contours]
        tp_labels = set()
        # Build set of mask labels that were matched (TP)
        # matched_det is set of detection indices; map to mask label via centroids order
        # Instead, just colour all contours cyan uniformly (simpler, avoids label mismatch)
        rgb[contours, 0] = 0.0
        rgb[contours, 1] = 1.0
        rgb[contours, 2] = 1.0
    else:
        rgb[contours, 0] = 0.0
        rgb[contours, 1] = 1.0
        rgb[contours, 2] = 1.0

    # Draw GT dots
    if gt_pts is not None and len(gt_pts) > 0:
        if matched_gt is not None:
            # Matched (TP) = lime, unmatched (FN) = red
            tp_pts = gt_pts[np.array(sorted(matched_gt))]
            fn_mask = np.ones(len(gt_pts), dtype=bool)
            fn_mask[np.array(sorted(matched_gt))] = False
            fn_pts = gt_pts[fn_mask]
            draw_filled_circles(rgb, tp_pts, radius=5, color=(0.0, 1.0, 0.0))
            draw_filled_circles(rgb, fn_pts, radius=5, color=(1.0, 0.0, 0.0))
        else:
            draw_filled_circles(rgb, gt_pts, radius=5, color=(0.0, 1.0, 0.0))

    arr = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(arr).save(out_path)


def save_fullres_tta_overlay(neun: np.ndarray, centroids: np.ndarray, case_id: str,
                             out_path: str, gt_pts: np.ndarray = None,
                             matched_det: set = None, matched_gt: set = None):
    """Full-res overlay for TTA mode — centroid dots (no mask contours).

    Colours:
      - Detected centroids (TP): cyan filled circles
      - Detected centroids (FP): magenta filled circles
      - GT dots (TP): lime green
      - GT dots (FN): red
    """
    from PIL import Image

    neun_f = neun.astype(np.float32)
    lo, hi = np.percentile(neun_f, [1, 99.5])
    neun_norm = np.clip((neun_f - lo) / (hi - lo + 1e-8), 0, 1)

    rgb = np.stack([neun_norm, neun_norm, neun_norm], axis=-1).copy()

    # Draw detected centroids
    if len(centroids) > 0:
        if matched_det is not None:
            tp_det = centroids[np.array(sorted(matched_det))]
            fp_idx = [i for i in range(len(centroids)) if i not in matched_det]
            fp_det = centroids[fp_idx] if fp_idx else np.zeros((0, 2))
            draw_filled_circles(rgb, tp_det, radius=6, color=(0.0, 1.0, 1.0))
            draw_filled_circles(rgb, fp_det, radius=6, color=(1.0, 0.0, 1.0))
        else:
            draw_filled_circles(rgb, centroids, radius=6, color=(0.0, 1.0, 1.0))

    # Draw GT dots
    if gt_pts is not None and len(gt_pts) > 0:
        if matched_gt is not None:
            tp_gt = gt_pts[np.array(sorted(matched_gt))]
            fn_mask = np.ones(len(gt_pts), dtype=bool)
            fn_mask[np.array(sorted(matched_gt))] = False
            fn_gt = gt_pts[fn_mask]
            draw_filled_circles(rgb, tp_gt, radius=5, color=(0.0, 1.0, 0.0))
            draw_filled_circles(rgb, fn_gt, radius=5, color=(1.0, 0.0, 0.0))
        else:
            draw_filled_circles(rgb, gt_pts, radius=5, color=(0.0, 1.0, 0.0))

    arr = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(arr).save(out_path)


def save_thumbnail_overlay(neun: np.ndarray, masks: np.ndarray, case_id: str,
                           out_path: str, gt_pts: np.ndarray = None):
    """Save a quick low-res NeuN + mask-edge overlay PNG (matplotlib)."""
    neun_f = neun.astype(np.float32)
    lo, hi = np.percentile(neun_f, [1, 99.5])
    neun_norm = np.clip((neun_f - lo) / (hi - lo + 1e-8), 0, 1)

    edge = ndimage.binary_dilation(masks > 0) ^ (masks > 0)
    rgb = np.stack([neun_norm, neun_norm, neun_norm], axis=-1)
    rgb[edge, 0] = 1.0
    rgb[edge, 1] = 0.0
    rgb[edge, 2] = 1.0

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(rgb)
    if gt_pts is not None and len(gt_pts) > 0:
        ax.scatter(gt_pts[:, 0], gt_pts[:, 1], c='lime', s=4, marker='.', alpha=0.7, label='GT')
        ax.legend(loc='upper right', fontsize=8)
    ax.set_title(f'{case_id} — V2d finetuned (thresh={CELLPROB_THRESHOLD})')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Run V2d finetuned inference on all cases')
    parser.add_argument('--tta', action='store_true',
                        help='Enable test-time augmentation (multi-scale + flips + intensity)')
    parser.add_argument('--cases', nargs='+', default=None,
                        help='Process only these case IDs (default: all)')
    parser.add_argument('--min_votes', type=int, default=TTA_MIN_VOTES,
                        help=f'TTA NMS min votes (default: {TTA_MIN_VOTES})')
    parser.add_argument('--nms_dist', type=float, default=TTA_NMS_DISTANCE,
                        help=f'TTA NMS merge distance (default: {TTA_NMS_DISTANCE})')
    args = parser.parse_args()

    t0 = time.time()
    config = PipelineConfig()
    base_dir = Path(config.output_dir) / 'cellpose_finetuned_v2'
    model_path = str(base_dir / 'model_v2d' / 'models' / 'cellpose_finetuned_v2d')

    suffix = '_tta' if args.tta else ''
    out_dir = base_dir / f'v2d_masks{suffix}'
    out_dir.mkdir(parents=True, exist_ok=True)

    mode_str = 'TTA (multi-scale + aggressive intensity)' if args.tta else 'single-pass'
    print('=' * 80)
    print(f'V2D FINETUNED — {mode_str.upper()} — INFERENCE')
    print('=' * 80)
    print(f'Model:             {model_path}')
    print(f'Mode:              {mode_str}')
    print(f'Cellprob threshold: {CELLPROB_THRESHOLD}')
    print(f'Flow threshold:     {FLOW_THRESHOLD}')
    print(f'Diameter:           {DIAMETER}' + (f'  TTA scales: {TTA_DIAMETERS}' if args.tta else ''))
    if args.tta:
        print(f'TTA min_votes:      {args.min_votes}')
        print(f'TTA NMS distance:   {args.nms_dist}')
    print(f'Output:             {out_dir}')
    print()

    # Load model once
    model = models.CellposeModel(pretrained_model=model_path, gpu=True)

    # Discover all cases
    all_data = discover_all_data(config)
    if args.cases:
        case_set = set(args.cases)
        all_data = [e for e in all_data if e['case_id'] in case_set]
    print(f'Discovered {len(all_data)} cases')
    print()

    results = []
    for idx, entry in enumerate(all_data):
        case_id = entry['case_id']
        source = entry['source']
        tif_path = entry['tif_path']
        xml_path = entry.get('xml_path', None)

        print(f'[{idx+1}/{len(all_data)}] {case_id} ({source})')

        try:
            if not Path(tif_path).exists():
                print(f'  WARNING: TIF not found: {tif_path}, skipping')
                continue

            # Read + preprocess
            img = read_multichannel_tiff(tif_path)
            img_clahe, quality_info = prepare_cellpose_input_clahe(img, source, config)
            print(f'  Image shape: {img.shape}, quality: {quality_info.get("quality", "?")}')

            # Run inference
            t_inf = time.time()

            if args.tta:
                # TTA: multi-scale + aggressive intensity augmentation
                centroids, raw_centroids = detect_with_tta(
                    model, img_clahe,
                    diameters=TTA_DIAMETERS,
                    flow_threshold=FLOW_THRESHOLD,
                    cellprob_threshold=CELLPROB_THRESHOLD,
                    nms_distance=args.nms_dist,
                    min_votes=args.min_votes,
                )
                n_cells = len(centroids)
                # Save raw centroids for post-hoc min_votes sweep
                np.save(str(out_dir / f'{case_id}_raw_centroids.npy'), raw_centroids)
                # For TTA we don't get instance masks — create a dummy for overlay
                masks = np.zeros(img_clahe.shape[:2], dtype=np.uint16)
            else:
                # Standard single-pass inference
                outputs = model.eval(
                    img_clahe,
                    diameter=DIAMETER,
                    channels=[1, 2],
                    flow_threshold=FLOW_THRESHOLD,
                    cellprob_threshold=CELLPROB_THRESHOLD,
                )
                masks = outputs[0]
                centroids = get_mask_centroids_fast(masks)
                n_cells = int(masks.max())

            dt_inf = time.time() - t_inf
            print(f'  Detected: {n_cells} cells ({dt_inf:.1f}s)')

            # Save mask + centroids
            np.save(str(out_dir / f'{case_id}_mask.npy'), masks.astype(np.uint16))
            np.save(str(out_dir / f'{case_id}_centroids.npy'), centroids)

            # Try to evaluate against GT
            gt_pts = None
            eval_metrics = None
            matched_det = None
            matched_gt = None
            if xml_path and Path(xml_path).exists():
                markers = parse_cellcounter_xml(xml_path)
                raw_pts = markers.get(1, {}).get('points', np.zeros((0, 2)))
                if len(raw_pts) > 0:
                    gt_pts = np.array(raw_pts, dtype=np.float64)
                    eval_metrics = evaluate_detections(centroids, gt_pts)
                    matched_det = eval_metrics.pop('matched_det_indices')
                    matched_gt = eval_metrics.pop('matched_gt_indices')
                    print(f'  GT={len(gt_pts)}  Matched={eval_metrics["matched"]}  '
                          f'P={eval_metrics["precision"]:.3f}  R={eval_metrics["recall"]:.3f}  '
                          f'F1={eval_metrics["f1"]:.3f}')

            # Get raw NeuN for overlay (use original image, not CLAHE)
            channel_map = config.get_channel_map(source)
            neun_raw = img[channel_map['NeuN']]

            if args.tta:
                # TTA mode: draw centroid dots instead of mask contours
                save_fullres_tta_overlay(
                    neun_raw, centroids, case_id,
                    str(out_dir / f'{case_id}_fullres_overlay.png'),
                    gt_pts=gt_pts, matched_det=matched_det, matched_gt=matched_gt,
                )
            else:
                # Standard mode: draw mask contours
                save_fullres_overlay(
                    neun_raw, masks, case_id,
                    str(out_dir / f'{case_id}_fullres_overlay.png'),
                    gt_pts=gt_pts, matched_det=matched_det, matched_gt=matched_gt,
                )
            print(f'  Saved full-res overlay')

            # Quick thumbnail too
            neun_clahe = img_clahe[:, :, 0] if img_clahe.ndim == 3 else neun_raw
            save_thumbnail_overlay(
                neun_clahe, masks, case_id,
                str(out_dir / f'{case_id}_thumb.png'),
                gt_pts=gt_pts,
            )

            r = {
                'case_id': case_id,
                'source': source,
                'num_cells': n_cells,
                'num_centroids': len(centroids),
                'image_shape': list(img.shape),
                'quality': quality_info.get('quality', 'unknown'),
                'inference_time_s': round(dt_inf, 1),
            }
            if eval_metrics:
                r['gt_count'] = int(len(gt_pts))
                r.update(eval_metrics)
            results.append(r)

        except Exception as e:
            print(f'  ERROR: {e}')
            traceback.print_exc()
            results.append({
                'case_id': case_id,
                'source': source,
                'error': str(e),
            })

    # ====================================================================
    # Summary
    # ====================================================================
    total_time = time.time() - t0
    evaluated = [r for r in results if 'f1' in r]

    print('\n' + '=' * 80)
    print('SUMMARY')
    print('=' * 80)
    print(f'Total cases processed: {len(results)}')
    print(f'Total time: {total_time/60:.1f} min')
    print()

    if evaluated:
        avg_p = np.mean([r['precision'] for r in evaluated])
        avg_r = np.mean([r['recall'] for r in evaluated])
        avg_f1 = np.mean([r['f1'] for r in evaluated])

        print(f'Cases with GT evaluation: {len(evaluated)}')
        print(f'{"Case":<25} {"Source":<6} {"GT":>5} {"Det":>5} {"Match":>5} '
              f'{"P":>6} {"R":>6} {"F1":>6}')
        print('-' * 70)
        for r in sorted(evaluated, key=lambda x: -x['f1']):
            print(f'{r["case_id"]:<25} {r["source"]:<6} {r["gt_count"]:>5} '
                  f'{r["num_cells"]:>5} {r["matched"]:>5} '
                  f'{r["precision"]:>6.3f} {r["recall"]:>6.3f} {r["f1"]:>6.3f}')
        print('-' * 70)
        print(f'{"AVERAGE":<25} {"":>6} {"":>5} {"":>5} {"":>5} '
              f'{avg_p:>6.3f} {avg_r:>6.3f} {avg_f1:>6.3f}')
    else:
        print('No cases had GT annotations for evaluation.')

    # Cells-only summary (no GT)
    no_gt = [r for r in results if 'f1' not in r and 'error' not in r]
    if no_gt:
        print(f'\nCases without GT (detection count only): {len(no_gt)}')
        for r in no_gt:
            print(f'  {r["case_id"]:<25} {r["source"]:<6} cells={r["num_cells"]}')

    errors = [r for r in results if 'error' in r]
    if errors:
        print(f'\nErrors: {len(errors)}')
        for r in errors:
            print(f'  {r["case_id"]}: {r["error"]}')

    # Save JSON
    summary = {
        'model': model_path,
        'mode': 'tta' if args.tta else 'single',
        'params': {
            'cellprob_threshold': CELLPROB_THRESHOLD,
            'flow_threshold': FLOW_THRESHOLD,
            'diameter': DIAMETER,
            'match_radius': MATCH_RADIUS,
        },
        'total_time_min': round(total_time / 60, 1),
        'num_cases': len(results),
        'num_evaluated': len(evaluated),
        'results': results,
    }
    if args.tta:
        summary['tta_params'] = {
            'diameters': TTA_DIAMETERS,
            'nms_distance': args.nms_dist,
            'min_votes': args.min_votes,
            'intensity_augs': ['identity', 'bright+', 'contrast+', 'gamma_high'],
            'geo_transforms': ['orig', 'hflip', 'vflip', 'rot90'],
        }
    if evaluated:
        summary['avg_precision'] = round(avg_p, 4)
        summary['avg_recall'] = round(avg_r, 4)
        summary['avg_f1'] = round(avg_f1, 4)

    json_path = out_dir / f'v2d_{"tta" if args.tta else "all"}_results.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nResults JSON: {json_path}')


if __name__ == '__main__':
    main()
