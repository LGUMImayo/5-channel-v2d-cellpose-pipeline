#!/usr/bin/env python3
"""
Post-hoc sweep of TTA min_votes using saved raw centroids.

Reads *_raw_centroids.npy from v2d_masks_tta/, applies NMS at various
min_votes thresholds, and evaluates P/R/F1 against GT.

No GPU needed — runs in seconds on CPU.

Usage:
    python sweep_tta_min_votes.py
    python sweep_tta_min_votes.py --votes 2 3 4 6 8 10 12
    python sweep_tta_min_votes.py --nms_dist 15
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

sys.path.insert(0, str(Path(__file__).parent))
from config import PipelineConfig
from data_utils import parse_cellcounter_xml, discover_all_data
from run_v2d_inference_all import nms_centroids

MATCH_RADIUS = 30.0


def evaluate(detected, gt_pts, match_radius=MATCH_RADIUS):
    """Greedy distance-sorted matching → P/R/F1."""
    if len(detected) == 0 and len(gt_pts) == 0:
        return 0, 1.0, 1.0, 1.0
    if len(detected) == 0:
        return 0, 0.0, 0.0, 0.0
    if len(gt_pts) == 0:
        return 0, 0.0, 0.0, 0.0

    d = cdist(detected, gt_pts)
    pairs = []
    for i in range(len(detected)):
        for j in range(len(gt_pts)):
            if d[i, j] <= match_radius:
                pairs.append((d[i, j], i, j))
    pairs.sort()
    used_d, used_g = set(), set()
    matched = 0
    for dist, di, gi in pairs:
        if di not in used_d and gi not in used_g:
            matched += 1
            used_d.add(di)
            used_g.add(gi)
    p = matched / len(detected) if len(detected) else 0
    r = matched / len(gt_pts) if len(gt_pts) else 0
    f1 = 2 * p * r / (p + r) if (p + r) else 0
    return matched, p, r, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--votes', nargs='+', type=int,
                        default=[2, 3, 4, 5, 6, 8, 10, 12],
                        help='min_votes values to sweep')
    parser.add_argument('--nms_dist', type=float, default=15.0,
                        help='NMS merge distance (pixels)')
    args = parser.parse_args()

    config = PipelineConfig()
    base_dir = Path(config.output_dir) / 'cellpose_finetuned_v2'
    tta_dir = base_dir / 'v2d_masks_tta'
    sp_json = base_dir / 'v2d_masks' / 'v2d_all_results.json'

    # Load single-pass results for comparison
    sp_data = {}
    if sp_json.exists():
        with open(sp_json) as f:
            sp_data = {r['case_id']: r for r in json.load(f)['results']}

    # Discover GT
    all_data = {e['case_id']: e for e in discover_all_data(config)}

    # Find cases with raw centroids
    raw_files = sorted([f for f in os.listdir(tta_dir)
                        if f.endswith('_raw_centroids.npy')])
    cases = [f.replace('_raw_centroids.npy', '') for f in raw_files]

    if not cases:
        print('No raw centroid files found. Run TTA inference first.')
        return

    print(f'Found {len(cases)} cases with raw TTA centroids')
    print(f'NMS distance: {args.nms_dist} px')
    print(f'Sweeping min_votes: {args.votes}')
    print(f'Match radius: {MATCH_RADIUS} px')
    print()

    # Results table: per case × per min_votes
    all_results = {}

    for case_id in cases:
        raw = np.load(str(tta_dir / f'{case_id}_raw_centroids.npy'))

        # Load GT
        entry = all_data.get(case_id)
        if not entry:
            continue
        xml_path = entry.get('xml_path')
        if not xml_path or not os.path.exists(xml_path):
            continue
        markers = parse_cellcounter_xml(xml_path)
        raw_pts = markers.get(1, {}).get('points', np.zeros((0, 2)))
        if len(raw_pts) == 0:
            continue
        gt_pts = np.array(raw_pts, dtype=np.float64)

        case_results = {'gt_count': len(gt_pts), 'raw_count': len(raw)}
        sp = sp_data.get(case_id, {})
        case_results['sp_f1'] = sp.get('f1', None)
        case_results['sp_p'] = sp.get('precision', None)
        case_results['sp_r'] = sp.get('recall', None)
        case_results['sp_det'] = sp.get('num_cells', None)

        for mv in args.votes:
            merged = nms_centroids(raw, args.nms_dist, mv)
            matched, p, r, f1 = evaluate(merged, gt_pts)
            case_results[f'v{mv}'] = {
                'det': len(merged), 'matched': matched,
                'p': round(p, 4), 'r': round(r, 4), 'f1': round(f1, 4),
            }

        all_results[case_id] = case_results

    # Print summary table
    vote_cols = args.votes
    header = f'{"Case":25s}  {"SP F1":>6s} |'
    for mv in vote_cols:
        header += f'  v={mv:>2d} F1'
    header += f'  | {"best":>5s}  {"best_v":>5s}'
    print(header)
    print('-' * len(header))

    # Per-case rows
    summary = {mv: [] for mv in vote_cols}
    sp_f1s = []

    for case_id in sorted(all_results.keys()):
        cr = all_results[case_id]
        sp_f1 = cr.get('sp_f1')
        row = f'{case_id:25s}  {sp_f1:6.3f} |' if sp_f1 else f'{case_id:25s}  {"?":>6s} |'

        best_f1, best_v = 0, 0
        for mv in vote_cols:
            v = cr[f'v{mv}']
            row += f'  {v["f1"]:7.3f}'
            summary[mv].append(v['f1'])
            if v['f1'] > best_f1:
                best_f1 = v['f1']
                best_v = mv

        if sp_f1:
            sp_f1s.append(sp_f1)
        delta = best_f1 - sp_f1 if sp_f1 else 0
        row += f'  | {best_f1:5.3f}  v={best_v:>2d}  ({delta:+.3f})'
        print(row)

    # Mean row
    print('-' * len(header))
    sp_mean = np.mean(sp_f1s) if sp_f1s else 0
    row = f'{"MEAN":25s}  {sp_mean:6.3f} |'
    best_mean_f1, best_mean_v = 0, 0
    for mv in vote_cols:
        mean_f1 = np.mean(summary[mv]) if summary[mv] else 0
        row += f'  {mean_f1:7.3f}'
        if mean_f1 > best_mean_f1:
            best_mean_f1 = mean_f1
            best_mean_v = mv
    row += f'  | {best_mean_f1:5.3f}  v={best_mean_v:>2d}  ({best_mean_f1 - sp_mean:+.3f})'
    print(row)

    # Detailed breakdown for best min_votes
    print(f'\n\n=== Best overall: min_votes={best_mean_v} (mean F1={best_mean_f1:.3f}) ===\n')
    print(f'{"Case":25s}  {"GT":>5s}  {"SP det":>6s} {"TTA det":>7s}  '
          f'{"SP P":>6s} {"TTA P":>6s}  {"SP R":>6s} {"TTA R":>6s}  '
          f'{"SP F1":>6s} {"TTA F1":>6s}  {"delta":>7s}')
    print('-' * 110)

    for case_id in sorted(all_results.keys()):
        cr = all_results[case_id]
        v = cr[f'v{best_mean_v}']
        sp_f1 = cr.get('sp_f1', 0)
        sp_p = cr.get('sp_p', 0)
        sp_r = cr.get('sp_r', 0)
        sp_det = cr.get('sp_det', 0)
        gt = cr['gt_count']
        delta = v['f1'] - sp_f1
        print(f'{case_id:25s}  {gt:5d}  {sp_det:6d} {v["det"]:7d}  '
              f'{sp_p:6.3f} {v["p"]:6.3f}  {sp_r:6.3f} {v["r"]:6.3f}  '
              f'{sp_f1:6.3f} {v["f1"]:6.3f}  {delta:+7.3f}')

    # Save results JSON
    out_path = tta_dir / 'tta_min_votes_sweep.json'
    with open(out_path, 'w') as f:
        json.dump({
            'nms_distance': args.nms_dist,
            'match_radius': MATCH_RADIUS,
            'votes_swept': vote_cols,
            'best_mean_votes': best_mean_v,
            'best_mean_f1': round(best_mean_f1, 4),
            'sp_mean_f1': round(sp_mean, 4),
            'per_case': all_results,
        }, f, indent=2)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
