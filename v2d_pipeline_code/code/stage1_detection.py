"""
Stage 1: Cell Detection using Cellpose.

This module runs Cellpose on the NeuN + DAPI channels to detect all neurons,
then filters the detected cells against manual NeuN annotations.

Three modes of operation:
  1. evaluate_cellpose(): Run Cellpose and compare against manual annotations
  2. generate_silver_masks(): Create filtered masks where manual points exist
  3. finetune_cellpose(): Fine-tune Cellpose on the silver masks (if needed)
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist

from config import PipelineConfig, CHANNEL_MAP_LR, CHANNEL_MAP_MAYO
from data_utils import (
    parse_cellcounter_xml,
    read_multichannel_tiff,
    discover_all_data,
)

logger = logging.getLogger(__name__)


def get_cellpose_input(
    img: np.ndarray,
    source: str,
    config: PipelineConfig,
) -> np.ndarray:
    """
    Extract and prepare NeuN + DAPI channels for Cellpose input.

    Args:
        img: (C, H, W) multi-channel image
        source: 'LR' or 'MAYO'
        config: PipelineConfig

    Returns:
        (H, W, 2) array with [NeuN, DAPI], normalized to uint8 or float
    """
    channel_map = config.get_channel_map(source)
    neun_idx = channel_map["NeuN"]
    dapi_idx = channel_map["DAPI"]

    neun = img[neun_idx].astype(np.float32)
    dapi = img[dapi_idx].astype(np.float32)

    # Normalize each channel to [0, 255] for Cellpose
    def norm_to_uint8(ch):
        lo = np.percentile(ch, 1)
        hi = np.percentile(ch, 99.5)
        if hi - lo < 1:
            return np.zeros_like(ch, dtype=np.uint8)
        ch = (ch - lo) / (hi - lo)
        ch = np.clip(ch * 255, 0, 255).astype(np.uint8)
        return ch

    neun_u8 = norm_to_uint8(neun)
    dapi_u8 = norm_to_uint8(dapi)

    # Cellpose expects (H, W, 2) with [cytoplasm, nucleus]
    return np.stack([neun_u8, dapi_u8], axis=-1)


def run_cellpose_detection(
    cellpose_input: np.ndarray,
    config: PipelineConfig,
) -> Tuple[np.ndarray, dict]:
    """
    Run Cellpose detection on a prepared 2-channel image.

    Args:
        cellpose_input: (H, W, 2) array [NeuN, DAPI]
        config: PipelineConfig

    Returns:
        masks: (H, W) integer array, each cell has a unique ID (0 = background)
        info: dict with 'flows', 'styles', etc.
    """
    from cellpose import models

    model = models.CellposeModel(
        gpu=config.cellpose_gpu,
        model_type=config.cellpose_model,
    )

    # channels = [cytoplasm, nucleus]
    # In our 2-channel input: channel 1 = NeuN (cytoplasm), channel 2 = DAPI (nucleus)
    # Note: cellpose v4.0.1+ changed API - eval returns 3 values now, channels param deprecated
    masks, flows, styles = model.eval(
        cellpose_input,
        diameter=config.cellpose_diameter,
        flow_threshold=config.cellpose_flow_threshold,
        cellprob_threshold=config.cellpose_cellprob_threshold,
    )

    info = {
        'diameter': config.cellpose_diameter if config.cellpose_diameter is not None else 30.0,  # Default for display
        'num_cells': int(masks.max()),
    }

    return masks, info


def get_mask_centroids(masks: np.ndarray) -> np.ndarray:
    """
    Get centroids of all cells in a mask image.

    Returns:
        (N, 2) array of (x, y) centroids, where N = number of cells
    """
    from scipy import ndimage

    n_cells = masks.max()
    if n_cells == 0:
        return np.zeros((0, 2))

    centroids = ndimage.center_of_mass(masks > 0, masks, range(1, n_cells + 1))
    # center_of_mass returns (row, col) = (y, x), convert to (x, y)
    centroids = np.array(centroids)
    return centroids[:, ::-1]  # (y, x) → (x, y)


def match_cellpose_to_manual(
    cellpose_centroids: np.ndarray,
    manual_points: np.ndarray,
    match_radius: float = 15.0,
) -> Dict:
    """
    Match Cellpose detections to manual NeuN annotations.

    Args:
        cellpose_centroids: (M, 2) array of (x, y) from Cellpose
        manual_points: (N, 2) array of (x, y) from XML
        match_radius: Maximum distance for a match

    Returns:
        dict with:
            'matched_pairs': List of (cellpose_idx, manual_idx, distance)
            'unmatched_cellpose': List of cellpose_idx (false positives)
            'unmatched_manual': List of manual_idx (missed detections)
            'precision': float
            'recall': float
            'f1': float
    """
    if len(cellpose_centroids) == 0 and len(manual_points) == 0:
        return {
            'matched_pairs': [],
            'unmatched_cellpose': [],
            'unmatched_manual': [],
            'precision': 1.0, 'recall': 1.0, 'f1': 1.0,
        }
    if len(cellpose_centroids) == 0:
        return {
            'matched_pairs': [],
            'unmatched_cellpose': [],
            'unmatched_manual': list(range(len(manual_points))),
            'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
        }
    if len(manual_points) == 0:
        return {
            'matched_pairs': [],
            'unmatched_cellpose': list(range(len(cellpose_centroids))),
            'unmatched_manual': [],
            'precision': 0.0, 'recall': 1.0, 'f1': 0.0,
        }

    # Compute pairwise distances
    dist_matrix = cdist(cellpose_centroids, manual_points)

    # Hungarian matching
    from scipy.optimize import linear_sum_assignment
    # Set distances > match_radius to a very large value
    cost = dist_matrix.copy()
    cost[cost > match_radius] = 1e6

    row_ind, col_ind = linear_sum_assignment(cost)

    matched_pairs = []
    unmatched_cellpose = set(range(len(cellpose_centroids)))
    unmatched_manual = set(range(len(manual_points)))

    for r, c in zip(row_ind, col_ind):
        if dist_matrix[r, c] <= match_radius:
            matched_pairs.append((int(r), int(c), float(dist_matrix[r, c])))
            unmatched_cellpose.discard(r)
            unmatched_manual.discard(c)

    tp = len(matched_pairs)
    fp = len(unmatched_cellpose)
    fn = len(unmatched_manual)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'matched_pairs': matched_pairs,
        'unmatched_cellpose': sorted(unmatched_cellpose),
        'unmatched_manual': sorted(unmatched_manual),
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def filter_masks_by_manual(
    masks: np.ndarray,
    cellpose_centroids: np.ndarray,
    manual_points: np.ndarray,
    match_radius: float = 15.0,
) -> np.ndarray:
    """
    Create "silver truth" masks: keep only Cellpose masks that match manual points.

    Returns:
        Filtered masks (same shape, but non-matching cells zeroed out)
    """
    match_result = match_cellpose_to_manual(cellpose_centroids, manual_points, match_radius)

    # Get the Cellpose cell IDs that matched
    matched_cp_indices = set(pair[0] for pair in match_result['matched_pairs'])

    # Cell IDs in masks are 1-indexed, centroids are 0-indexed
    # So centroid index i corresponds to mask value (i + 1)
    keep_ids = set(idx + 1 for idx in matched_cp_indices)

    filtered = masks.copy()
    for cell_id in range(1, masks.max() + 1):
        if cell_id not in keep_ids:
            filtered[filtered == cell_id] = 0

    # Re-label sequentially
    from scipy import ndimage
    filtered_relabeled, n = ndimage.label(filtered > 0)

    return filtered_relabeled


# =============================================================================
# Main evaluation function
# =============================================================================

def evaluate_cellpose_on_dataset(config: PipelineConfig) -> Dict:
    """
    Run Cellpose on all cases and evaluate against manual annotations.

    Returns:
        Summary dict with per-case and aggregate metrics.
    """
    all_cases = discover_all_data(config)
    results = []

    for case in all_cases:
        logger.info(f"Processing {case['case_id']} ({case['source']})...")

        # Read image
        img = read_multichannel_tiff(case['tif_path'])
        logger.info(f"  Image shape: {img.shape}")

        # Prepare Cellpose input
        cp_input = get_cellpose_input(img, case['source'], config)

        # Run Cellpose
        masks, info = run_cellpose_detection(cp_input, config)
        diam_str = f"{info['diameter']:.1f}" if info['diameter'] is not None else "auto"
        logger.info(f"  Cellpose found {info['num_cells']} cells (diam={diam_str})")

        # Get centroids
        centroids = get_mask_centroids(masks)

        # Load manual annotations
        marker_types = parse_cellcounter_xml(case['xml_path'])
        manual_neun = marker_types.get(1, {}).get('points', np.zeros((0, 2)))

        # Match
        match = match_cellpose_to_manual(centroids, manual_neun, config.match_radius_px)
        logger.info(f"  Precision={match['precision']:.3f}, "
                    f"Recall={match['recall']:.3f}, F1={match['f1']:.3f}")

        results.append({
            'case_id': case['case_id'],
            'source': case['source'],
            'manual_count': len(manual_neun),
            'cellpose_count': info['num_cells'],
            'tp': len(match['matched_pairs']),
            'fp': len(match['unmatched_cellpose']),
            'fn': len(match['unmatched_manual']),
            'precision': match['precision'],
            'recall': match['recall'],
            'f1': match['f1'],
            'diameter': info['diameter'],
        })

    # Aggregate
    total_tp = sum(r['tp'] for r in results)
    total_fp = sum(r['fp'] for r in results)
    total_fn = sum(r['fn'] for r in results)
    agg_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    agg_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    agg_f1 = 2 * agg_prec * agg_rec / (agg_prec + agg_rec) if (agg_prec + agg_rec) > 0 else 0

    summary = {
        'per_case': results,
        'aggregate': {
            'total_tp': total_tp,
            'total_fp': total_fp,
            'total_fn': total_fn,
            'precision': agg_prec,
            'recall': agg_rec,
            'f1': agg_f1,
        }
    }

    return summary


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stage 1: Evaluate Cellpose on NeuN detection")
    parser.add_argument("--config", type=str, default=None, help="Path to config JSON")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    config = PipelineConfig.load(args.config) if args.config else PipelineConfig()

    summary = evaluate_cellpose_on_dataset(config)

    # Print results
    print("\n=== Cellpose Evaluation Summary ===")
    print(f"{'Case':<30s} {'Manual':>6s} {'Cellpose':>8s} {'TP':>4s} {'FP':>4s} {'FN':>4s} {'P':>6s} {'R':>6s} {'F1':>6s}")
    print("-" * 90)
    for r in summary['per_case']:
        print(f"{r['case_id']:<30s} {r['manual_count']:>6d} {r['cellpose_count']:>8d} "
              f"{r['tp']:>4d} {r['fp']:>4d} {r['fn']:>4d} "
              f"{r['precision']:>6.3f} {r['recall']:>6.3f} {r['f1']:>6.3f}")
    print("-" * 90)
    agg = summary['aggregate']
    print(f"{'AGGREGATE':<30s} {'':<6s} {'':<8s} "
          f"{agg['total_tp']:>4d} {agg['total_fp']:>4d} {agg['total_fn']:>4d} "
          f"{agg['precision']:>6.3f} {agg['recall']:>6.3f} {agg['f1']:>6.3f}")

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to {args.output}")
