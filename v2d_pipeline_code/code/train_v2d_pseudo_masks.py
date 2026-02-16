#!/usr/bin/env python3
"""
Train Cellpose V2d on augmented masks (silver + pseudo masks for FN cells).

This approach:
1. Keeps original silver masks (good boundaries from V1)
2. Adds circular pseudo masks around FN centroids (imperfect but teaches "cell here")
3. Trains on full images (no tile boundary issues)
4. More epochs to learn from both precise and approximate masks

Usage:
    # Generate augmented masks and train
    python train_v2d_pseudo_masks.py --train --n_epochs 300
    
    # Just evaluate existing model
    python train_v2d_pseudo_masks.py --eval_only
"""

import os
import sys
import json
import argparse
import logging
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import PipelineConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def apply_intensity_augmentation(img: np.ndarray, seed: int = None) -> np.ndarray:
    """Apply V2c-style intensity augmentation to HWC image.

    - Brightness: per-channel scale in [0.8, 1.2]
    - Contrast: per-channel scale in [0.8, 1.2]
    - Gaussian noise: sigma=0.02 (applied 50% probability)
    """
    if seed is not None:
        np.random.seed(seed)

    out = img.astype(np.float32).copy()

    brightness = np.random.uniform(0.8, 1.2, size=(1, 1, out.shape[2]))
    out = out * brightness

    contrast = np.random.uniform(0.8, 1.2, size=(1, 1, out.shape[2]))
    mean = out.mean(axis=(0, 1), keepdims=True)
    out = (out - mean) * contrast + mean

    if np.random.rand() > 0.5:
        noise = np.random.normal(0, 0.02, out.shape)
        out = out + noise * max(float(out.max()), 1.0)

    return np.clip(out, 0, 255).astype(np.float32)


def _centroids_from_label_mask(mask: np.ndarray) -> np.ndarray:
    """Extract (x, y) centroids from instance label mask."""
    from scipy.ndimage import center_of_mass

    labels = np.unique(mask)
    labels = labels[labels > 0]
    if len(labels) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    centroids = []
    for lbl in labels:
        cy, cx = center_of_mass(mask == lbl)
        centroids.append((float(cx), float(cy)))
    return np.array(centroids, dtype=np.float32)


def _match_distance_sorted(detected_xy: np.ndarray, gt_xy: np.ndarray, match_radius: float = 30.0) -> int:
    """Distance-sorted greedy matching count."""
    from scipy.spatial.distance import cdist

    if len(detected_xy) == 0 or len(gt_xy) == 0:
        return 0

    dists = cdist(detected_xy, gt_xy)
    pairs = []
    for i in range(len(detected_xy)):
        for j in range(len(gt_xy)):
            if dists[i, j] <= match_radius:
                pairs.append((dists[i, j], i, j))
    pairs.sort(key=lambda x: x[0])

    used_det = set()
    used_gt = set()
    matched = 0
    for _, det_idx, gt_idx in pairs:
        if det_idx not in used_det and gt_idx not in used_gt:
            used_det.add(det_idx)
            used_gt.add(gt_idx)
            matched += 1
    return matched


def evaluate_dataset_detection_metrics(
    model_path: str,
    images: list,
    gt_masks: list,
    diameter: float = 35,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = -0.5,
    match_radius: float = 30.0,
    max_images: int = 0,
):
    """Evaluate detection-style metrics on image/mask dataset.

    Returns object-level metrics aggregated across images.
    """
    from cellpose import models

    n_eval = len(images) if max_images <= 0 else min(len(images), int(max_images))
    if n_eval == 0:
        return {
            'n_images': 0,
            'tp': 0,
            'fp': 0,
            'fn': 0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'accuracy': 0.0,
        }

    model = models.CellposeModel(pretrained_model=model_path, gpu=True)

    tp = fp = fn = 0
    for image, gt_mask in zip(images[:n_eval], gt_masks[:n_eval]):
        gt_xy = _centroids_from_label_mask(gt_mask)
        pred_masks, _, _ = model.eval(
            image,
            diameter=diameter,
            channels=[1, 2],
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
        )
        pred_xy = _centroids_from_label_mask(pred_masks)

        matched = _match_distance_sorted(pred_xy, gt_xy, match_radius=match_radius)
        tp += matched
        fp += max(0, len(pred_xy) - matched)
        fn += max(0, len(gt_xy) - matched)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    return {
        'n_images': n_eval,
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'accuracy': float(accuracy),
    }


def train_on_augmented_masks(
    mask_dir: Path,
    output_model_dir: Path,
    pretrained_model: str,
    model_name: str = "cellpose_finetuned_v2d",
    n_epochs: int = 300,
    learning_rate: float = 0.0001,
    weight_decay: float = 0.1,
    batch_size: int = 8,
    crops_per_image: int = 3,
    test_fraction: float = 0.15,
    seed: int = 42,
    min_silver_recall: float = 0.0,
    intensity_aug: bool = True,
    aug_prob: float = 1.0,
    report_accuracy: bool = False,
    accuracy_max_images: int = 0,
    accuracy_cellprob: float = -0.5,
):
    """Train Cellpose on augmented masks (silver + pseudo FN)."""
    from cellpose import models, train
    
    output_model_dir.mkdir(parents=True, exist_ok=True)
    
    # Load manifest
    manifest_path = mask_dir / 'augmented_masks_manifest.json'
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    # Filter cases with too few cells
    manifest = [e for e in manifest if e['total_count'] >= 10]

    # Optional: exclude cases where silver masks have very low recall vs GT
    if min_silver_recall > 0:
        kept_manifest = []
        n_excluded = 0
        for entry in manifest:
            gt_count = float(entry.get('gt_count', 0))
            silver_count = float(entry.get('silver_count', 0))
            silver_recall = (silver_count / gt_count) if gt_count > 0 else 0.0
            if silver_recall >= min_silver_recall:
                kept_manifest.append(entry)
            else:
                n_excluded += 1
        manifest = kept_manifest
        logger.info(
            f"Applied min_silver_recall={min_silver_recall:.3f}: excluded {n_excluded} low-recall cases"
        )

    logger.info(f"Using {len(manifest)} images with >= 10 cells")
    if len(manifest) < 2:
        raise RuntimeError(
            f"Not enough training cases after filtering: {len(manifest)}. "
            f"Lower min_silver_recall or add more data."
        )
    
    # Split train/test
    np.random.seed(seed)
    np.random.shuffle(manifest)
    n_test = max(1, int(len(manifest) * test_fraction))
    
    train_images, train_masks = [], []
    test_images, test_masks = [], []
    
    for entry in manifest[:-n_test]:
        img = np.load(entry['img_path'])
        mask = np.load(entry['mask_path'])
        train_images.append(img)
        train_masks.append(mask)
    
    for entry in manifest[-n_test:]:
        img = np.load(entry['img_path'])
        mask = np.load(entry['mask_path'])
        test_images.append(img)
        test_masks.append(mask)

    # Apply intensity augmentation in-place (no dataset size increase)
    n_augmented = 0
    if intensity_aug and len(train_images) > 0:
        aug_prob = float(np.clip(aug_prob, 0.0, 1.0))
        augmented_images = []
        for idx, img in enumerate(train_images):
            if np.random.rand() <= aug_prob:
                augmented_images.append(apply_intensity_augmentation(img, seed=seed + idx))
                n_augmented += 1
            else:
                augmented_images.append(img.astype(np.float32))
        train_images = augmented_images
        logger.info(f"Intensity augmentation enabled: augmented {n_augmented}/{len(train_images)} train images")
    
    total_train_cells = sum(int(m.max()) for m in train_masks)
    total_test_cells = sum(int(m.max()) for m in test_masks)
    
    # Count silver vs pseudo masks
    train_silver = sum(e['silver_count'] for e in manifest[:-n_test])
    train_pseudo = sum(e['pseudo_count'] for e in manifest[:-n_test])
    
    logger.info(f"Train: {len(train_images)} images, {total_train_cells} cells")
    logger.info(f"  Silver masks: {train_silver}")
    logger.info(f"  Pseudo masks: {train_pseudo} (+{100*train_pseudo/train_silver:.1f}%)")
    logger.info(f"Test: {len(test_images)} images, {total_test_cells} cells")
    
    # Initialize model
    model = models.CellposeModel(gpu=True, pretrained_model=pretrained_model)
    
    # Use nimg_per_epoch to get multiple crops per image
    nimg_per_epoch = len(train_images) * crops_per_image
    
    logger.info(f"Training for {n_epochs} epochs...")
    logger.info(f"  LR={learning_rate}, WD={weight_decay}, BS={batch_size}")
    logger.info(f"  Crops/image: {crops_per_image}, nimg_per_epoch: {nimg_per_epoch}")
    logger.info(f"  Intensity augmentation: {intensity_aug} (prob={aug_prob:.2f})")
    
    filename, train_losses, test_losses = train.train_seg(
        net=model.net,
        train_data=train_images,
        train_labels=train_masks,
        test_data=test_images if test_images else None,
        test_labels=test_masks if test_masks else None,
        save_path=str(output_model_dir),
        save_every=50,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        nimg_per_epoch=nimg_per_epoch,
        min_train_masks=1,
        model_name=model_name,
    )
    
    logger.info(f"Model saved to: {filename}")
    logger.info(f"Final train loss: {train_losses[-1]:.4f}")
    if test_losses is not None and len(test_losses) > 0:
        best_epoch = np.argmin(test_losses)
        logger.info(f"Best test loss: {test_losses[best_epoch]:.4f} (epoch {best_epoch + 1})")

    train_metrics = None
    test_metrics = None
    if report_accuracy:
        logger.info(
            "Computing train/test detection metrics "
            f"(cellprob={accuracy_cellprob}, max_images={accuracy_max_images if accuracy_max_images > 0 else 'all'})"
        )
        train_metrics = evaluate_dataset_detection_metrics(
            model_path=str(filename),
            images=train_images,
            gt_masks=train_masks,
            cellprob_threshold=accuracy_cellprob,
            max_images=accuracy_max_images,
        )
        test_metrics = evaluate_dataset_detection_metrics(
            model_path=str(filename),
            images=test_images,
            gt_masks=test_masks,
            cellprob_threshold=accuracy_cellprob,
            max_images=accuracy_max_images,
        )
        logger.info(
            "Train metrics: "
            f"acc={train_metrics['accuracy']:.3f}, p={train_metrics['precision']:.3f}, "
            f"r={train_metrics['recall']:.3f}, f1={train_metrics['f1']:.3f}"
        )
        logger.info(
            "Test metrics: "
            f"acc={test_metrics['accuracy']:.3f}, p={test_metrics['precision']:.3f}, "
            f"r={test_metrics['recall']:.3f}, f1={test_metrics['f1']:.3f}"
        )
    
    # Save training info
    info = {
        'model': 'v2d',
        'approach': 'silver + pseudo FN masks',
        'base_model': pretrained_model,
        'n_epochs': n_epochs,
        'n_train_images': len(train_images),
        'n_test_images': len(test_images),
        'total_train_cells': total_train_cells,
        'train_silver_masks': train_silver,
        'train_pseudo_masks': train_pseudo,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'min_silver_recall': min_silver_recall,
        'intensity_aug': intensity_aug,
        'aug_prob': aug_prob,
        'n_augmented_train_images': n_augmented,
        'report_accuracy': report_accuracy,
        'accuracy_cellprob': accuracy_cellprob,
        'accuracy_max_images': accuracy_max_images,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'model_path': str(filename),
        'final_train_loss': float(train_losses[-1]),
        'final_test_loss': float(test_losses[-1]) if len(test_losses) > 0 else None,
    }
    with open(output_model_dir / 'training_info.json', 'w') as f:
        json.dump(info, f, indent=2)
    
    return filename, train_losses, test_losses


def evaluate_model(model_path: str, test_case: str = "P3044_C12"):
    """Evaluate model on test case."""
    from cellpose import models
    from scipy.spatial.distance import cdist
    
    from data_utils import read_multichannel_tiff, parse_cellcounter_xml
    from bootstrap_cellpose_v2 import prepare_cellpose_input_clahe
    
    config = PipelineConfig()
    
    # Load test image
    case_dir = Path(config.mayo_data_dir) / test_case
    tif_files = list(case_dir.glob("StitchedROI-*.tif"))
    if not tif_files:
        logger.error(f"No TIF found for {test_case}")
        return
    
    img = read_multichannel_tiff(str(tif_files[0]))
    img_clahe, _ = prepare_cellpose_input_clahe(img, 'MAYO', config)
    
    # Load GT
    xml_files = list(case_dir.glob("CellCounter_*.xml"))
    marker_types = parse_cellcounter_xml(str(xml_files[0]))
    gt_pts = np.array(marker_types.get(1, {}).get('points', []))
    
    # Run inference
    model = models.CellposeModel(pretrained_model=model_path, gpu=True)
    masks, flows, styles = model.eval(img_clahe, diameter=35, channels=[1, 2])
    
    # Get detected centroids
    from scipy.ndimage import center_of_mass
    unique_labels = np.unique(masks)
    unique_labels = unique_labels[unique_labels > 0]
    
    detected = []
    for lbl in unique_labels:
        cy, cx = center_of_mass(masks == lbl)
        detected.append((cx, cy))  # (x, y) format
    detected = np.array(detected) if detected else np.array([]).reshape(0, 2)
    
    # Match (distance-sorted greedy; closest valid pairs first)
    if len(detected) > 0 and len(gt_pts) > 0:
        distances = cdist(detected, gt_pts)

        pairs = []
        for i in range(len(detected)):
            for j in range(len(gt_pts)):
                if distances[i, j] <= 30:
                    pairs.append((distances[i, j], i, j))
        pairs.sort(key=lambda x: x[0])

        matched = 0
        used_det = set()
        used_gt = set()

        for _, det_idx, gt_idx in pairs:
            if det_idx not in used_det and gt_idx not in used_gt:
                matched += 1
                used_det.add(det_idx)
                used_gt.add(gt_idx)
        
        precision = matched / len(detected) if len(detected) > 0 else 0
        recall = matched / len(gt_pts) if len(gt_pts) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    else:
        precision, recall, f1 = 0, 0, 0
        matched = 0
    
    print(f"\n{'='*60}")
    print(f"EVALUATION: {test_case}")
    print(f"{'='*60}")
    print(f"  GT cells:   {len(gt_pts)}")
    print(f"  Detected:   {len(detected)}")
    print(f"  Matched:    {matched}")
    print(f"  Precision:  {precision:.3f}")
    print(f"  Recall:     {recall:.3f}")
    print(f"  F1:         {f1:.3f}")
    
    return {
        'test_case': test_case,
        'gt_count': len(gt_pts),
        'detected': len(detected),
        'matched': matched,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--create_masks", action="store_true",
                       help="Create augmented masks (silver + pseudo FN)")
    parser.add_argument("--train", action="store_true",
                       help="Train on augmented masks")
    parser.add_argument("--eval_only", action="store_true",
                       help="Only evaluate existing model")
    parser.add_argument("--n_epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--wd", type=float, default=0.1)
    parser.add_argument("--min_silver_recall", type=float, default=0.0,
                       help="Exclude training cases with silver_count/gt_count below this threshold")
    parser.add_argument("--aug_prob", type=float, default=1.0,
                       help="Probability of applying intensity augmentation per train image")
    parser.add_argument("--intensity_aug", dest="intensity_aug", action="store_true",
                       help="Enable intensity augmentation (brightness/contrast/noise)")
    parser.add_argument("--no_intensity_aug", dest="intensity_aug", action="store_false",
                       help="Disable intensity augmentation")
    parser.add_argument("--report_accuracy", action="store_true",
                       help="Report train/test object-level accuracy, precision, recall, F1 after training")
    parser.add_argument("--accuracy_max_images", type=int, default=0,
                       help="Max images per split for metric reporting (0=all)")
    parser.add_argument("--accuracy_cellprob", type=float, default=-0.5,
                       help="cellprob threshold used when reporting train/test metrics")
    parser.add_argument("--radius", type=int, default=18,
                       help="Radius for pseudo masks")
    parser.set_defaults(intensity_aug=True)
    args = parser.parse_args()
    
    config = PipelineConfig()
    base_dir = Path(config.output_dir) / 'cellpose_finetuned_v2'
    
    # V1 model path
    v1_model = str(base_dir.parent / 'cellpose_finetuned' / 'model' / 'models' / 'cellpose_finetuned_neun_dapi')
    
    augmented_mask_dir = base_dir / 'silver_masks_augmented'
    output_model_dir = base_dir / 'model_v2d'
    
    if args.create_masks or (args.train and not augmented_mask_dir.exists()):
        print("="*60)
        print("STEP 1: Creating augmented masks (silver + pseudo FN)")
        print("="*60)
        
        # Run pseudo mask creation
        from create_pseudo_masks import process_all_cases
        
        silver_mask_dir = base_dir / 'silver_masks_v2_merged'
        if not silver_mask_dir.exists():
            silver_mask_dir = base_dir / 'silver_masks'
            
        print(f"Using silver masks from: {silver_mask_dir}")
        
        process_all_cases(
            silver_mask_dir=silver_mask_dir,
            output_dir=augmented_mask_dir,
            radius=args.radius,
        )
    
    if args.train:
        print("\n" + "="*60)
        print("STEP 2: Training V2d on augmented masks")
        print("="*60)
        
        model_path, train_losses, test_losses = train_on_augmented_masks(
            mask_dir=augmented_mask_dir,
            output_model_dir=output_model_dir,
            pretrained_model=v1_model,
            model_name="cellpose_finetuned_v2d",
            n_epochs=args.n_epochs,
            learning_rate=args.lr,
            weight_decay=args.wd,
            min_silver_recall=args.min_silver_recall,
            intensity_aug=args.intensity_aug,
            aug_prob=args.aug_prob,
            report_accuracy=args.report_accuracy,
            accuracy_max_images=args.accuracy_max_images,
            accuracy_cellprob=args.accuracy_cellprob,
        )
        
        print("\n" + "="*60)
        print("STEP 3: Evaluating V2d")
        print("="*60)
        
        # Evaluate
        evaluate_model(model_path, "P3044_C12")
        
        # Compare models
        print("\n" + "="*60)
        print("MODEL COMPARISON on P3044_C12")
        print("="*60)
        
        print("\nV1 (manual GT):")
        v1_result = evaluate_model(v1_model, "P3044_C12")
        
        v2b_model = str(base_dir / 'model_merged' / 'models' / 'cellpose_finetuned_v2')
        if Path(v2b_model).exists():
            print("\nV2b (low-thresh silver):")
            v2b_result = evaluate_model(v2b_model, "P3044_C12")
        
        print("\nV2d (silver + pseudo FN):")
        v2d_result = evaluate_model(model_path, "P3044_C12")
    
    elif args.eval_only:
        v2d_model = str(output_model_dir / 'models' / 'cellpose_finetuned_v2d')
        if not Path(v2d_model).exists():
            logger.error(f"V2d model not found: {v2d_model}")
            return
        
        print("="*60)
        print("MODEL COMPARISON on P3044_C12")
        print("="*60)
        
        print("\nV1 (manual GT) + CLAHE:")
        evaluate_model(v1_model, "P3044_C12")
        
        v2b_model = str(base_dir / 'model_merged' / 'models' / 'cellpose_finetuned_v2')
        if Path(v2b_model).exists():
            print("\nV2b (low-thresh silver) + CLAHE:")
            evaluate_model(v2b_model, "P3044_C12")
        
        print("\nV2d (silver + pseudo FN) + CLAHE:")
        evaluate_model(v2d_model, "P3044_C12")
    
    else:
        print("Use --create_masks to generate augmented masks")
        print("Use --train to train on augmented masks")
        print("Use --eval_only to evaluate existing model")


if __name__ == '__main__':
    main()
