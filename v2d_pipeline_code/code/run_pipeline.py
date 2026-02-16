"""
End-to-end Inference Pipeline: Detect neurons → Classify biomarkers.

This script runs the full two-stage pipeline on new images:
  Stage 1: Finetuned Cellpose V2d detects all NeuN+ neurons (CLAHE preprocessing)
  Stage 2: ResNet-18 classifies each neuron for CAMKII/PHF1/BEX1

Usage:
    python run_pipeline.py \
        --image /path/to/image.tif \
        --source LR \
        --output results/ \
        --visualize

Models are automatically loaded from default paths. Override with:
    --cellpose_model /path/to/cellpose_model
    --classifier_checkpoint /path/to/classifier.ckpt
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import PipelineConfig, BIOMARKER_LABELS
from data_utils import (
    read_multichannel_tiff,
    unify_channel_order,
    extract_cell_crop,
    normalize_crop,
)
from stage1_detection import get_mask_centroids
from bootstrap_cellpose_v2 import prepare_cellpose_input_clahe
from stage2_classifier import CellPhenotypingModule

logger = logging.getLogger(__name__)

# Default model paths
DEFAULT_CELLPOSE_MODEL = "output/cellpose_finetuned_v2/model_v2d/models/cellpose_finetuned_v2d"
DEFAULT_CLASSIFIER_CHECKPOINTS = [
    "output/training_runs/fold0_20260210_223613/checkpoints/best-epoch=04-val/macro_f1=0.8325.ckpt",
    "output/training_runs/fold1_20260210_225334/checkpoints/best-epoch=38-val/macro_f1=0.8076.ckpt",
    "output/training_runs/fold2_20260210_233127/checkpoints/best-epoch=08-val/macro_f1=0.8572.ckpt",
    "output/training_runs/fold3_20260210_235201/checkpoints/best-epoch=22-val/macro_f1=0.8251.ckpt",
    "output/training_runs/fold4_20260211_002153/checkpoints/best-epoch=20-val/macro_f1=0.7901.ckpt",
]
# Best single-fold (fold 2 with F1=0.8572)
DEFAULT_CLASSIFIER_SINGLE = DEFAULT_CLASSIFIER_CHECKPOINTS[2]

# Tuned Stage 1 parameters
DEFAULT_CELLPROB_THRESHOLD = -0.5
DEFAULT_FLOW_THRESHOLD = 0.4
DEFAULT_DIAMETER = 35.0


def load_classifier(checkpoint_path: str, config: PipelineConfig) -> CellPhenotypingModule:
    """Load a trained classifier from checkpoint."""
    # PyTorch 2.6+ defaults to weights_only=True which fails on older checkpoints
    # that saved config objects. Use weights_only=False for compatibility.
    import torch.serialization
    try:
        # Try adding PipelineConfig to safe globals for newer PyTorch
        torch.serialization.add_safe_globals([PipelineConfig])
    except AttributeError:
        pass  # Older PyTorch version
    
    model = CellPhenotypingModule.load_from_checkpoint(
        checkpoint_path,
        config=config,
        map_location='cpu',
        strict=False,  # Allow missing/unexpected keys (e.g. loss_fn.pos_weight)
        weights_only=False,  # Required for checkpoints with custom config objects
    )
    model.eval()
    model.freeze()
    return model


def load_classifier_ensemble(checkpoint_paths: List[str], config: PipelineConfig) -> List[CellPhenotypingModule]:
    """Load multiple classifiers for ensemble prediction."""
    models = []
    for ckpt in checkpoint_paths:
        if os.path.exists(ckpt):
            models.append(load_classifier(ckpt, config))
            logger.info(f"  Loaded fold: {Path(ckpt).parent.parent.parent.name}")
        else:
            logger.warning(f"  Checkpoint not found: {ckpt}")
    return models


def run_cellpose_finetuned(
    img: np.ndarray,
    source: str,
    config: PipelineConfig,
    model_path: str,
    diameter: float = DEFAULT_DIAMETER,
    flow_threshold: float = DEFAULT_FLOW_THRESHOLD,
    cellprob_threshold: float = DEFAULT_CELLPROB_THRESHOLD,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Run finetuned Cellpose V2d with CLAHE preprocessing.
    
    Returns:
        masks: (H, W) instance mask
        centroids: (N, 2) array of (x, y)
        info: dict with detection stats
    """
    from cellpose import models
    
    # CLAHE preprocessing (same as training)
    img_clahe, quality_info = prepare_cellpose_input_clahe(img, source, config)
    
    # Load finetuned model
    model = models.CellposeModel(pretrained_model=model_path, gpu=config.cellpose_gpu)
    
    # Run detection
    outputs = model.eval(
        img_clahe,
        diameter=diameter,
        channels=[1, 2],
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
    )
    masks = outputs[0]
    
    centroids = get_mask_centroids(masks)
    
    info = {
        'diameter': diameter,
        'flow_threshold': flow_threshold,
        'cellprob_threshold': cellprob_threshold,
        'num_cells': int(masks.max()),
        'quality': quality_info,
    }
    
    return masks, centroids, info


def classify_cells_batch(
    model: CellPhenotypingModule,
    crops: List[np.ndarray],
    batch_size: int = 64,
    device: str = 'cuda',
) -> np.ndarray:
    """
    Run classification on a batch of cell crops using a single model.

    Args:
        model: Trained CellPhenotypingModule
        crops: List of (5, H, W) normalized float32 arrays
        batch_size: Batch size for inference
        device: 'cuda' or 'cpu'

    Returns:
        (N, 3) array of probabilities for [CAMKII, PHF1, BEX1]
    """
    model = model.to(device)
    all_probs = []

    for i in range(0, len(crops), batch_size):
        batch_crops = crops[i:i + batch_size]
        batch_tensor = torch.from_numpy(np.stack(batch_crops)).float().to(device)

        with torch.no_grad():
            logits = model(batch_tensor)
            probs = torch.sigmoid(logits)

        all_probs.append(probs.cpu().numpy())

    return np.concatenate(all_probs, axis=0)


def classify_cells_ensemble(
    models: List[CellPhenotypingModule],
    crops: List[np.ndarray],
    batch_size: int = 64,
    device: str = 'cuda',
) -> np.ndarray:
    """
    Run classification using ensemble of models (average probabilities).

    Returns:
        (N, 3) array of averaged probabilities for [CAMKII, PHF1, BEX1]
    """
    if len(models) == 0:
        return np.zeros((len(crops), len(BIOMARKER_LABELS)))
    
    all_probs = []
    for model in models:
        probs = classify_cells_batch(model, crops, batch_size, device)
        all_probs.append(probs)
    
    # Average across models
    return np.mean(all_probs, axis=0)


def run_pipeline(
    image_path: str,
    source: str,
    config: PipelineConfig,
    cellpose_model: str = None,
    classifier_checkpoints: List[str] = None,
    use_ensemble: bool = False,
    threshold: float = 0.5,
    cellprob_threshold: float = DEFAULT_CELLPROB_THRESHOLD,
) -> Dict:
    """
    Run the full two-stage pipeline on a single image.

    Args:
        image_path: Path to multi-channel TIF
        source: 'LR' or 'MAYO'
        config: PipelineConfig
        cellpose_model: Path to finetuned Cellpose model (default: V2d)
        classifier_checkpoints: List of classifier checkpoints (for ensemble or single)
        use_ensemble: If True, average predictions from all checkpoints
        threshold: Classification threshold for binary predictions
        cellprob_threshold: Cellpose cell probability threshold

    Returns:
        dict with:
            'cells': list of dicts per detected cell:
                {
                    'x': float, 'y': float,
                    'mask_id': int,
                    'probabilities': {'CAMKII': float, 'PHF1': float, 'BEX1': float},
                    'predictions': {'CAMKII': bool, 'PHF1': bool, 'BEX1': bool},
                }
            'summary': aggregate counts
    """
    base_dir = Path(__file__).parent
    
    # Resolve default model paths
    if cellpose_model is None:
        cellpose_model = str(base_dir / DEFAULT_CELLPOSE_MODEL)
    if classifier_checkpoints is None:
        if use_ensemble:
            classifier_checkpoints = [str(base_dir / p) for p in DEFAULT_CLASSIFIER_CHECKPOINTS]
        else:
            classifier_checkpoints = [str(base_dir / DEFAULT_CLASSIFIER_SINGLE)]
    
    # =========================================================================
    # Stage 1: Cell Detection (Finetuned Cellpose V2d + CLAHE)
    # =========================================================================
    logger.info("Stage 1: Running finetuned Cellpose V2d with CLAHE...")

    img = read_multichannel_tiff(image_path)
    logger.info(f"  Image shape: {img.shape}")

    img_unified = unify_channel_order(img, source, config)

    masks, centroids, cp_info = run_cellpose_finetuned(
        img, source, config,
        model_path=cellpose_model,
        cellprob_threshold=cellprob_threshold,
    )

    logger.info(f"  Detected {cp_info['num_cells']} cells (threshold={cellprob_threshold})")
    logger.info(f"  NeuN quality: {cp_info['quality'].get('quality', 'unknown')}")

    # =========================================================================
    # Stage 2: Extract & Classify
    # =========================================================================
    logger.info("Stage 2: Classifying biomarker positivity...")

    device = 'cuda' if torch.cuda.is_available() and config.cellpose_gpu else 'cpu'
    
    if use_ensemble:
        logger.info(f"  Loading {len(classifier_checkpoints)}-fold ensemble...")
        models = load_classifier_ensemble(classifier_checkpoints, config)
    else:
        logger.info(f"  Loading single classifier: {Path(classifier_checkpoints[0]).stem}")
        models = [load_classifier(classifier_checkpoints[0], config)]

    # Extract crops for all detected cells
    valid_crops = []
    valid_indices = []
    valid_centroids = []

    for idx, (cx, cy) in enumerate(centroids):
        crop = extract_cell_crop(img_unified, cx, cy, config.crop_size)
        if crop is not None:
            crop_norm = normalize_crop(crop, config.norm_percentile_low, config.norm_percentile_high)
            valid_crops.append(crop_norm)
            valid_indices.append(idx)
            valid_centroids.append((cx, cy))

    logger.info(f"  Valid crops (non-edge): {len(valid_crops)} / {len(centroids)}")

    # Classify
    if valid_crops:
        if use_ensemble:
            probs = classify_cells_ensemble(models, valid_crops, config.batch_size, device)
        else:
            probs = classify_cells_batch(models[0], valid_crops, config.batch_size, device)
    else:
        probs = np.zeros((0, len(BIOMARKER_LABELS)))

    # =========================================================================
    # Compile Results
    # =========================================================================
    cells = []
    for i, (cx, cy) in enumerate(valid_centroids):
        cell_probs = {b: float(probs[i, j]) for j, b in enumerate(BIOMARKER_LABELS)}
        cell_preds = {b: bool(probs[i, j] >= threshold) for j, b in enumerate(BIOMARKER_LABELS)}

        cells.append({
            'x': float(cx),
            'y': float(cy),
            'mask_id': int(valid_indices[i] + 1),
            'probabilities': cell_probs,
            'predictions': cell_preds,
        })

    # Summary counts
    total_neurons = len(cells)
    summary = {
        'total_neurons': total_neurons,
        'image': os.path.basename(image_path),
        'source': source,
    }
    for b in BIOMARKER_LABELS:
        n_pos = sum(1 for c in cells if c['predictions'][b])
        summary[f'{b}_positive'] = n_pos
        summary[f'{b}_negative'] = total_neurons - n_pos
        summary[f'{b}_percent'] = 100 * n_pos / total_neurons if total_neurons > 0 else 0

    logger.info(f"\n  Results:")
    logger.info(f"    Total NeuN+ neurons: {total_neurons}")
    for b in BIOMARKER_LABELS:
        logger.info(f"    {b}+: {summary[f'{b}_positive']} ({summary[f'{b}_percent']:.1f}%)")

    return {
        'cells': cells,
        'summary': summary,
        'cellpose_info': cp_info,
        'masks_shape': list(masks.shape),
        'masks': masks,           # Keep for full-res visualization
        'centroids': centroids,   # Keep for full-res visualization
    }


# =========================================================================
# Full-resolution visualization helpers
# =========================================================================

def _norm_channel(ch: np.ndarray) -> np.ndarray:
    """Normalize a 2D channel to [0,1] float32."""
    lo, hi = np.percentile(ch, [1, 99.5])
    return np.clip((ch.astype(np.float32) - lo) / (hi - lo + 1e-8), 0, 1)


def _mask_contours(mask: np.ndarray, thickness: int = 2) -> np.ndarray:
    """Instance-aware contour extraction."""
    from scipy import ndimage
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


def _draw_circles(rgb: np.ndarray, pts, radius: int = 6,
                  color=(0.0, 1.0, 0.0)):
    """Draw filled circles on (H, W, 3) float32 image."""
    H, W, _ = rgb.shape
    for x, y in pts:
        xi, yi = int(round(x)), int(round(y))
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    yy, xx = yi + dy, xi + dx
                    if 0 <= yy < H and 0 <= xx < W:
                        rgb[yy, xx] = color


def _draw_rings(rgb: np.ndarray, pts, inner_r: int = 5, outer_r: int = 8,
                color=(1.0, 0.0, 0.0)):
    """Draw unfilled circle rings on (H, W, 3) float32 image."""
    H, W, _ = rgb.shape
    for x, y in pts:
        xi, yi = int(round(x)), int(round(y))
        for dy in range(-outer_r, outer_r + 1):
            for dx in range(-outer_r, outer_r + 1):
                r2 = dx * dx + dy * dy
                if inner_r * inner_r <= r2 <= outer_r * outer_r:
                    yy, xx = yi + dy, xi + dx
                    if 0 <= yy < H and 0 <= xx < W:
                        rgb[yy, xx] = color


def visualize_stage1_fullres(
    image_path: str,
    source: str,
    masks: np.ndarray,
    output_path: str,
    config: 'PipelineConfig',
    gt_xml_path: str = None,
):
    """
    Full-resolution Stage 1 visualization: NeuN + detected mask contours + GT dots.

    Colours:
      - Mask contours: cyan
      - GT NeuN dots (matched TP): lime green filled circles
      - GT NeuN dots (unmatched FN): red filled circles
      - Detected centroids with no GT match (FP): magenta rings
    """
    from PIL import Image
    from scipy.ndimage import center_of_mass
    from scipy.spatial.distance import cdist

    img = read_multichannel_tiff(image_path)
    channel_map = config.get_channel_map(source)
    neun = _norm_channel(img[channel_map['NeuN']])

    rgb = np.stack([neun, neun, neun], axis=-1).copy()

    # Draw mask contours in cyan
    contours = _mask_contours(masks, thickness=2)
    rgb[contours] = [0.0, 1.0, 1.0]

    # Extract centroids from masks
    labels = np.unique(masks)
    labels = labels[labels > 0]
    if len(labels) > 0:
        coms = center_of_mass(masks, masks, labels)
        det_pts = np.array([(cx, cy) for cy, cx in coms], dtype=np.float64)
    else:
        det_pts = np.zeros((0, 2))

    # Compare to GT if available
    if gt_xml_path and os.path.exists(gt_xml_path):
        from data_utils import parse_cellcounter_xml
        markers = parse_cellcounter_xml(gt_xml_path)
        gt_pts = markers.get(1, {}).get('points', np.zeros((0, 2)))
        gt_pts = np.asarray(gt_pts, dtype=np.float64)

        if len(gt_pts) > 0 and len(det_pts) > 0:
            d = cdist(det_pts, gt_pts)
            pairs = []
            for i in range(len(det_pts)):
                for j in range(len(gt_pts)):
                    if d[i, j] <= 30.0:
                        pairs.append((d[i, j], i, j))
            pairs.sort()
            used_det, used_gt = set(), set()
            for _, i, j in pairs:
                if i not in used_det and j not in used_gt:
                    used_det.add(i)
                    used_gt.add(j)

            # TP GT dots = lime
            tp_gt = gt_pts[sorted(used_gt)]
            _draw_circles(rgb, tp_gt, radius=5, color=(0.0, 1.0, 0.0))
            # FN GT dots = red
            fn_idx = [j for j in range(len(gt_pts)) if j not in used_gt]
            if fn_idx:
                _draw_circles(rgb, gt_pts[fn_idx], radius=5, color=(1.0, 0.0, 0.0))
            # FP detections = magenta rings
            fp_idx = [i for i in range(len(det_pts)) if i not in used_det]
            if fp_idx:
                _draw_rings(rgb, det_pts[fp_idx], inner_r=5, outer_r=8,
                            color=(1.0, 0.0, 1.0))
        elif len(gt_pts) > 0:
            # All GT are FN
            _draw_circles(rgb, gt_pts, radius=5, color=(1.0, 0.0, 0.0))

    arr = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(arr).save(output_path)
    logger.info(f"Stage 1 full-res visualization saved to {output_path}")


def visualize_stage2_fullres(
    image_path: str,
    source: str,
    results: Dict,
    output_path: str,
    config: 'PipelineConfig',
    gt_xml_path: str = None,
):
    """
    Full-resolution Stage 2 visualization: NeuN background + per-biomarker
    classification predictions (coloured circles) vs GT annotations (rings).

    Prediction colours (filled circles):
      - CAMKII+: green
      - PHF1+: red
      - BEX1+: blue
      - NeuN-only (all negative): white
      - Multi-positive: mixed colour

    GT annotation colours (rings, if xml provided):
      - GT CAMKII: green ring
      - GT PHF1: red ring
      - GT BEX1: blue ring
    """
    from PIL import Image

    img = read_multichannel_tiff(image_path)
    channel_map = config.get_channel_map(source)
    neun = _norm_channel(img[channel_map['NeuN']])

    rgb = np.stack([neun * 0.4, neun * 0.4, neun * 0.4], axis=-1).copy()

    biomarker_colors = {
        'CAMKII': (0.0, 1.0, 0.0),
        'PHF1':   (1.0, 0.2, 0.2),
        'BEX1':   (0.3, 0.5, 1.0),
    }

    # Draw predicted cells as filled circles
    cells = results['cells']
    for cell in cells:
        cx, cy = cell['x'], cell['y']
        preds = cell['predictions']
        positive = [b for b in BIOMARKER_LABELS if preds[b]]

        if not positive:
            color = (0.8, 0.8, 0.8)  # white-ish for NeuN-only
        else:
            mixed = np.mean([biomarker_colors[b] for b in positive], axis=0)
            color = tuple(np.clip(mixed, 0, 1))

        _draw_circles(rgb, [(cx, cy)], radius=7, color=color)

    # Draw GT biomarker annotations as rings if XML provided
    if gt_xml_path and os.path.exists(gt_xml_path):
        from data_utils import parse_cellcounter_xml
        markers = parse_cellcounter_xml(gt_xml_path)
        # Type 2=CAMKII, 3=PHF1, 4=BEX1
        gt_bio_map = {2: 'CAMKII', 3: 'PHF1', 4: 'BEX1'}
        for marker_type, bio_name in gt_bio_map.items():
            gt_pts = markers.get(marker_type, {}).get('points', np.zeros((0, 2)))
            if len(gt_pts) > 0:
                _draw_rings(rgb, gt_pts, inner_r=9, outer_r=12,
                            color=biomarker_colors[bio_name])

    arr = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(arr).save(output_path)
    logger.info(f"Stage 2 full-res visualization saved to {output_path}")


def visualize_results(
    image_path: str,
    source: str,
    results: Dict,
    output_path: str,
    config: PipelineConfig,
):
    """
    Create a visualization overlaying detection and classification results.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    img = read_multichannel_tiff(image_path)
    img_unified = unify_channel_order(img, source, config)

    # Create a composite RGB from NeuN (red), DAPI (blue), and a biomarker (green)
    neun = img_unified[0].astype(np.float32)
    dapi = img_unified[1].astype(np.float32)

    # Normalize for display
    def norm(ch):
        lo, hi = np.percentile(ch, [1, 99.5])
        return np.clip((ch - lo) / (hi - lo + 1e-8), 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    cells = results['cells']

    # Color map: NeuN-only = white, +CAMKII = green, +PHF1 = red, +BEX1 = blue
    # Combinations get mixed colors
    biomarker_colors = {
        'CAMKII': np.array([0, 1, 0]),
        'PHF1': np.array([1, 0, 0]),
        'BEX1': np.array([0, 0.5, 1]),
    }

    for ax_idx, (title, display_channel_idx) in enumerate([
        ("NeuN + DAPI", None),
        ("Biomarker Overlay", None),
        ("Classification Results", None),
    ]):
        ax = axes[ax_idx]

        if ax_idx == 0:
            # NeuN (green) + DAPI (blue) composite
            rgb = np.zeros((*neun.shape, 3), dtype=np.float32)
            rgb[..., 1] = norm(neun)        # Green = NeuN
            rgb[..., 2] = norm(dapi)        # Blue = DAPI
            ax.imshow(rgb)
            ax.set_title("NeuN (green) + DAPI (blue)")

        elif ax_idx == 1:
            # All channels composite
            camkii = norm(img_unified[2].astype(np.float32))
            phf1 = norm(img_unified[3].astype(np.float32))
            bex1 = norm(img_unified[4].astype(np.float32))
            rgb = np.zeros((*neun.shape, 3), dtype=np.float32)
            rgb[..., 0] = np.clip(norm(neun) * 0.3 + phf1 * 0.7, 0, 1)
            rgb[..., 1] = np.clip(camkii * 0.7 + bex1 * 0.3, 0, 1)
            rgb[..., 2] = np.clip(norm(dapi) * 0.3 + bex1 * 0.7, 0, 1)
            ax.imshow(rgb)
            ax.set_title("All Channels Composite")

        elif ax_idx == 2:
            # Dark background with colored dots
            rgb = np.zeros((*neun.shape, 3), dtype=np.float32)
            rgb[..., 1] = norm(neun) * 0.2  # Faint NeuN background
            ax.imshow(rgb)

            for cell in cells:
                cx, cy = cell['x'], cell['y']
                preds = cell['predictions']

                # Determine cell color
                positive_markers = [b for b in BIOMARKER_LABELS if preds[b]]
                if not positive_markers:
                    color = 'white'
                    alpha = 0.5
                else:
                    mixed = np.zeros(3)
                    for b in positive_markers:
                        mixed += biomarker_colors[b]
                    mixed = np.clip(mixed / len(positive_markers), 0, 1)
                    color = mixed
                    alpha = 0.8

                circle = Circle((cx, cy), radius=8, facecolor=color,
                               edgecolor='white', linewidth=0.5, alpha=alpha)
                ax.add_patch(circle)

            ax.set_title("Phenotype: White=NeuN only, Green=CAMKII+, Red=PHF1+, Blue=BEX1+")

        ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Visualization saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run full cell phenotyping pipeline")
    parser.add_argument("--image", type=str, required=True, help="Path to multi-channel TIF")
    parser.add_argument("--source", type=str, required=True, choices=["LR", "MAYO"],
                        help="Data source (determines channel order)")
    parser.add_argument("--cellpose_model", type=str, default=None,
                        help="Path to finetuned Cellpose model (default: V2d)")
    parser.add_argument("--classifier_checkpoint", type=str, default=None,
                        help="Path to classifier checkpoint (default: best fold 2)")
    parser.add_argument("--ensemble", action="store_true",
                        help="Use 5-fold ensemble for classification (better accuracy)")
    parser.add_argument("--config", type=str, default=None, help="Config JSON")
    parser.add_argument("--output", type=str, default="results/", help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Classification threshold (default: 0.5)")
    parser.add_argument("--cellprob_threshold", type=float, default=DEFAULT_CELLPROB_THRESHOLD,
                        help=f"Cellpose cell probability threshold (default: {DEFAULT_CELLPROB_THRESHOLD})")
    parser.add_argument("--visualize", action="store_true", help="Generate visualization")
    parser.add_argument("--gt_xml", type=str, default=None,
                        help="Path to CellCounter XML for GT overlay (auto-detected if omitted)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    config = PipelineConfig.load(args.config) if args.config else PipelineConfig()

    # Prepare classifier checkpoints
    classifier_checkpoints = None
    if args.classifier_checkpoint:
        classifier_checkpoints = [args.classifier_checkpoint]

    # Run pipeline
    results = run_pipeline(
        image_path=args.image,
        source=args.source,
        config=config,
        cellpose_model=args.cellpose_model,
        classifier_checkpoints=classifier_checkpoints,
        use_ensemble=args.ensemble,
        threshold=args.threshold,
        cellprob_threshold=args.cellprob_threshold,
    )

    # Save results
    os.makedirs(args.output, exist_ok=True)
    image_name = Path(args.image).stem

    results_path = os.path.join(args.output, f"{image_name}_results.json")
    # Remove non-serializable numpy arrays before JSON dump
    results_json = {k: v for k, v in results.items() if k not in ('masks', 'centroids')}
    with open(results_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"Results saved to {results_path}")

    # Print summary
    s = results['summary']
    print(f"\n{'='*50}")
    print(f"PIPELINE RESULTS: {s['image']}")
    print(f"{'='*50}")
    print(f"Total NeuN+ neurons: {s['total_neurons']}")
    for b in BIOMARKER_LABELS:
        print(f"  {b}+: {s[f'{b}_positive']} / {s['total_neurons']} ({s[f'{b}_percent']:.1f}%)")
    print()

    # Visualize
    if args.visualize:
        # Auto-detect GT XML if not provided
        gt_xml = args.gt_xml
        if gt_xml is None:
            # Try to find CellCounter XML in the same directory as the image
            img_dir = Path(args.image).parent
            xml_candidates = list(img_dir.glob('CellCounter_*.xml'))
            if xml_candidates:
                gt_xml = str(xml_candidates[0])
                print(f"Auto-detected GT XML: {gt_xml}")

        # Stage 1 full-res: detected masks vs GT NeuN dots
        s1_path = os.path.join(args.output, f"{image_name}_stage1_fullres.png")
        visualize_stage1_fullres(
            args.image, args.source, results['masks'], s1_path, config,
            gt_xml_path=gt_xml,
        )

        # Stage 2 full-res: classified cells vs GT biomarker annotations
        s2_path = os.path.join(args.output, f"{image_name}_stage2_fullres.png")
        visualize_stage2_fullres(
            args.image, args.source, results, s2_path, config,
            gt_xml_path=gt_xml,
        )

        # Also keep the matplotlib thumbnail overview
        vis_path = os.path.join(args.output, f"{image_name}_visualization.png")
        visualize_results(args.image, args.source, results, vis_path, config)

    # Also export as CSV for easy analysis
    csv_path = os.path.join(args.output, f"{image_name}_cells.csv")
    with open(csv_path, 'w') as f:
        headers = ['x', 'y', 'mask_id'] + \
                  [f'{b}_prob' for b in BIOMARKER_LABELS] + \
                  [f'{b}_pred' for b in BIOMARKER_LABELS]
        f.write(','.join(headers) + '\n')
        for cell in results['cells']:
            row = [
                f"{cell['x']:.1f}",
                f"{cell['y']:.1f}",
                str(cell['mask_id']),
            ]
            for b in BIOMARKER_LABELS:
                row.append(f"{cell['probabilities'][b]:.4f}")
            for b in BIOMARKER_LABELS:
                row.append(str(int(cell['predictions'][b])))
            f.write(','.join(row) + '\n')
    print(f"Cell-level CSV saved to {csv_path}")


if __name__ == "__main__":
    main()
