#!/usr/bin/env python3
"""
Sweep cellprob_threshold on V2d model to find optimal precision/recall tradeoff.

Loads the test image once, then runs inference at multiple thresholds.
Uses distance-sorted greedy matching (more accurate than naive greedy).

Usage:
    python sweep_cellprob_threshold.py
    python sweep_cellprob_threshold.py --model v1    # sweep V1 model instead
    python sweep_cellprob_threshold.py --model v2b   # sweep V2b model
    python sweep_cellprob_threshold.py --model v2e   # sweep V2e model
"""

import os
import sys
import argparse
import logging
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import PipelineConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def distance_sorted_greedy_match(detected, gt_pts, match_radius=30.0):
    """
    Match detected centroids to GT points using distance-sorted greedy.
    
    This processes the closest (det, gt) pair first, which avoids the 
    suboptimal assignments that naive iteration-order greedy can produce.
    
    Returns: number of matched pairs
    """
    if len(detected) == 0 or len(gt_pts) == 0:
        return 0
    
    from scipy.spatial.distance import cdist
    
    distances = cdist(detected, gt_pts)
    
    # Get all (det_idx, gt_idx) pairs within match_radius, sorted by distance
    pairs = []
    for i in range(len(detected)):
        for j in range(len(gt_pts)):
            if distances[i, j] <= match_radius:
                pairs.append((distances[i, j], i, j))
    
    pairs.sort(key=lambda x: x[0])  # closest first
    
    used_det = set()
    used_gt = set()
    matched = 0
    
    for dist, det_idx, gt_idx in pairs:
        if det_idx not in used_det and gt_idx not in used_gt:
            used_det.add(det_idx)
            used_gt.add(gt_idx)
            matched += 1
    
    return matched


def evaluate_at_threshold(model, img_clahe, gt_pts, cellprob_threshold,
                          diameter=35, flow_threshold=0.4, match_radius=30.0):
    """Run inference at a specific cellprob_threshold and evaluate."""
    from scipy.ndimage import center_of_mass
    
    masks, flows, styles = model.eval(
        img_clahe,
        diameter=diameter,
        channels=[1, 2],
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
    )
    
    # Get detected centroids
    unique_labels = np.unique(masks)
    unique_labels = unique_labels[unique_labels > 0]
    
    detected = []
    for lbl in unique_labels:
        cy, cx = center_of_mass(masks == lbl)
        detected.append((cx, cy))
    detected = np.array(detected) if detected else np.array([]).reshape(0, 2)
    
    # Distance-sorted greedy matching
    matched = distance_sorted_greedy_match(detected, gt_pts, match_radius)
    
    n_det = len(detected)
    n_gt = len(gt_pts)
    precision = matched / n_det if n_det > 0 else 0
    recall = matched / n_gt if n_gt > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'cellprob_threshold': cellprob_threshold,
        'detected': n_det,
        'matched': matched,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def main():
    parser = argparse.ArgumentParser(description="Sweep cellprob_threshold for V2d")
    parser.add_argument("--model", type=str, default="v2d",
                        choices=["v1", "v2b", "v2d", "v2e"],
                        help="Which model to sweep (default: v2d)")
    parser.add_argument("--test_case", type=str, default="P3044_C12")
    parser.add_argument("--thresholds", type=str, default=None,
                        help="Comma-separated thresholds (default: -3.0 to 1.0 in 0.5 steps)")
    parser.add_argument("--flow_threshold", type=float, default=0.4)
    parser.add_argument("--diameter", type=float, default=35)
    parser.add_argument("--match_radius", type=float, default=30.0)
    args = parser.parse_args()
    
    config = PipelineConfig()
    base_dir = Path(config.output_dir) / 'cellpose_finetuned_v2'
    
    # Resolve model path
    if args.model == "v1":
        model_path = str(base_dir.parent / 'cellpose_finetuned' / 'model' / 'models' / 'cellpose_finetuned_neun_dapi')
        model_label = "V1 (manual GT)"
    elif args.model == "v2b":
        model_path = str(base_dir / 'model_merged' / 'models' / 'cellpose_finetuned_v2')
        model_label = "V2b (low-thresh silver)"
    elif args.model == "v2e":
        model_path = str(base_dir / 'model_v2e' / 'models' / 'cellpose_finetuned_v2e')
        model_label = "V2e (intensity-guided pseudo FN)"
    else:
        model_path = str(base_dir / 'model_v2d' / 'models' / 'cellpose_finetuned_v2d')
        model_label = "V2d (silver + pseudo FN)"
    
    if not Path(model_path).exists():
        logger.error(f"Model not found: {model_path}")
        sys.exit(1)
    
    # Parse thresholds
    if args.thresholds:
        thresholds = [float(t) for t in args.thresholds.split(',')]
    else:
        thresholds = np.arange(-3.0, 1.5, 0.5).tolist()
    
    # Load test data (once)
    from data_utils import read_multichannel_tiff, parse_cellcounter_xml
    from bootstrap_cellpose_v2 import prepare_cellpose_input_clahe
    
    case_dir = Path(config.mayo_data_dir) / args.test_case
    tif_files = list(case_dir.glob("StitchedROI-*.tif"))
    if not tif_files:
        logger.error(f"No TIF found for {args.test_case}")
        sys.exit(1)
    
    logger.info(f"Loading test image: {tif_files[0]}")
    img = read_multichannel_tiff(str(tif_files[0]))
    img_clahe, _ = prepare_cellpose_input_clahe(img, 'MAYO', config)
    
    xml_files = list(case_dir.glob("CellCounter_*.xml"))
    marker_types = parse_cellcounter_xml(str(xml_files[0]))
    gt_pts = np.array(marker_types.get(1, {}).get('points', []))
    logger.info(f"GT cells: {len(gt_pts)}")
    
    # Load model (once)
    from cellpose import models
    logger.info(f"Loading model: {model_path}")
    model = models.CellposeModel(pretrained_model=model_path, gpu=True)
    
    # Sweep
    print(f"\n{'='*80}")
    print(f"CELLPROB THRESHOLD SWEEP: {model_label}")
    print(f"Test case: {args.test_case} ({len(gt_pts)} GT cells)")
    print(f"Flow threshold: {args.flow_threshold}, Diameter: {args.diameter}")
    print(f"Match radius: {args.match_radius}px, Matching: distance-sorted greedy")
    print(f"{'='*80}")
    print(f"{'Threshold':>10} {'Detected':>10} {'Matched':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"{'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    
    results = []
    for thresh in thresholds:
        logger.info(f"Running threshold={thresh:.1f}...")
        r = evaluate_at_threshold(
            model, img_clahe, gt_pts, thresh,
            diameter=args.diameter,
            flow_threshold=args.flow_threshold,
            match_radius=args.match_radius,
        )
        results.append(r)
        
        print(f"{thresh:>10.1f} {r['detected']:>10d} {r['matched']:>10d} "
              f"{r['precision']:>10.3f} {r['recall']:>10.3f} {r['f1']:>10.3f}")
    
    # Find best
    best = max(results, key=lambda x: x['f1'])
    print(f"\n{'='*80}")
    print(f"BEST F1: {best['f1']:.3f} at cellprob_threshold={best['cellprob_threshold']:.1f}")
    print(f"  Detected: {best['detected']}, Matched: {best['matched']}")
    print(f"  Precision: {best['precision']:.3f}, Recall: {best['recall']:.3f}")
    print(f"{'='*80}")
    
    # Also show best precision ≥ 0.7 (practical minimum)
    viable = [r for r in results if r['precision'] >= 0.7]
    if viable:
        best_viable = max(viable, key=lambda x: x['f1'])
        print(f"\nBEST F1 with precision >= 0.7: {best_viable['f1']:.3f} "
              f"at threshold={best_viable['cellprob_threshold']:.1f}")
        print(f"  P={best_viable['precision']:.3f}, R={best_viable['recall']:.3f}")
    
    # Save results
    import json
    out_dir = Path(config.output_dir) / 'cellpose_finetuned_v2' / 'threshold_sweep'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f'sweep_{args.model}_{args.test_case}.json'
    with open(out_file, 'w') as f:
        json.dump({
            'model': args.model,
            'model_path': model_path,
            'test_case': args.test_case,
            'flow_threshold': args.flow_threshold,
            'diameter': args.diameter,
            'match_radius': args.match_radius,
            'matching': 'distance_sorted_greedy',
            'results': results,
            'best_f1': best,
        }, f, indent=2)
    logger.info(f"Results saved to {out_file}")


if __name__ == '__main__':
    main()
