"""
Fine-tune Cellpose on your NeuN + DAPI data using "silver truth" masks.

Cellpose v4.0.8 API notes:
  - model_type is IGNORED; always loads 'cpsam' (SAM-based Transformer)
  - CellposeModel has NO .train() method
  - Training uses train.train_seg(net, ...) instead
  - channels parameter removed from eval (auto-detected from input shape)
  - AdamW is always used (SGD param deprecated)

Workflow:
  1. Run pre-trained Cellpose on all images → get initial masks
  2. Filter masks against manual NeuN annotations → "silver truth" masks
  3. Fine-tune Cellpose on these filtered masks using train.train_seg()

The "silver truth" approach:
  - Your FIJI XML gives POINT locations, not outlines
  - Cellpose gives OUTLINES but may miss some cells or detect artifacts
  - By keeping only Cellpose masks that match a manual point, you get
    reasonable outlines for confirmed neurons WITHOUT hand-drawing masks

Usage:
    python finetune_cellpose.py [--n_epochs 100] [--output_dir output/cellpose_finetuned]
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import PipelineConfig
from data_utils import (
    parse_cellcounter_xml,
    read_multichannel_tiff,
    discover_all_data,
)
from stage1_detection import (
    get_cellpose_input,
    run_cellpose_detection,
    get_mask_centroids,
    filter_masks_by_manual,
    match_cellpose_to_manual,
)

logger = logging.getLogger(__name__)


def generate_silver_masks(
    config: PipelineConfig,
    output_dir: str,
) -> List[Dict]:
    """
    Step 1-2: Run Cellpose → filter against manual annotations → save silver masks.

    For each case, saves:
      - {case_id}_img.npy:  (H, W, 2) uint8 array [NeuN, DAPI]
      - {case_id}_mask.npy: (H, W) int32 filtered mask

    Returns:
        List of dicts with paths and stats
    """
    os.makedirs(output_dir, exist_ok=True)
    all_cases = discover_all_data(config)
    results = []

    for case in all_cases:
        case_id = case['case_id']
        logger.info(f"Processing {case_id} ({case['source']})...")

        # Read image and prepare Cellpose input
        img = read_multichannel_tiff(case['tif_path'])
        cp_input = get_cellpose_input(img, case['source'], config)

        # Run Cellpose
        masks, info = run_cellpose_detection(cp_input, config)
        centroids = get_mask_centroids(masks)
        logger.info(f"  Cellpose detected {info['num_cells']} cells")

        # Load manual NeuN points
        marker_types = parse_cellcounter_xml(case['xml_path'])
        manual_neun = marker_types.get(1, {}).get('points', np.zeros((0, 2)))

        # Match and filter
        match = match_cellpose_to_manual(centroids, manual_neun, config.match_radius_px)
        filtered_masks = filter_masks_by_manual(
            masks, centroids, manual_neun, config.match_radius_px
        )

        n_kept = filtered_masks.max()
        logger.info(f"  Kept {n_kept} / {info['num_cells']} masks "
                    f"(recall={match['recall']:.3f}, {len(match['unmatched_manual'])} manual pts unmatched)")

        # Save
        img_path = os.path.join(output_dir, f"{case_id}_img.npy")
        mask_path = os.path.join(output_dir, f"{case_id}_mask.npy")
        np.save(img_path, cp_input)         # (H, W, 2) uint8
        np.save(mask_path, filtered_masks)   # (H, W) int32

        results.append({
            'case_id': case_id,
            'source': case['source'],
            'img_path': img_path,
            'mask_path': mask_path,
            'cellpose_count': info['num_cells'],
            'manual_count': len(manual_neun),
            'silver_count': int(n_kept),
            'recall': match['recall'],
            'precision': match['precision'],
        })

    # Save manifest
    manifest_path = os.path.join(output_dir, "silver_masks_manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Manifest saved to {manifest_path}")

    return results


def finetune_cellpose(
    silver_mask_dir: str,
    output_model_dir: str,
    n_epochs: int = 100,
    learning_rate: float = 0.0001,
    weight_decay: float = 0.1,
    batch_size: int = 8,
    pretrained_model: str = "cpsam",
    use_gpu: bool = True,
    test_fraction: float = 0.15,
    seed: int = 42,
):
    """
    Step 3: Fine-tune Cellpose on silver truth masks.

    Uses Cellpose v4.0.8 train.train_seg() API.

    Args:
        silver_mask_dir: Directory with *_img.npy and *_mask.npy files
        output_model_dir: Where to save the fine-tuned model
        n_epochs: Training epochs
        learning_rate: Learning rate (v4 default is 1e-5, we use 1e-4 for faster convergence)
        weight_decay: L2 regularization (v4 default is 0.1)
        batch_size: Training batch size
        pretrained_model: Base model (v4 only has 'cpsam' available)
        use_gpu: Whether to use GPU
        test_fraction: Fraction of images to hold out for validation
        seed: Random seed for train/test split
    """
    from cellpose import models, train

    os.makedirs(output_model_dir, exist_ok=True)

    # Load manifest
    manifest_path = os.path.join(silver_mask_dir, "silver_masks_manifest.json")
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    # Load all images and masks
    train_images = []
    train_masks = []

    np.random.seed(seed)
    np.random.shuffle(manifest)

    n_test = max(1, int(len(manifest) * test_fraction))
    n_train = len(manifest) - n_test

    logger.info(f"Train: {n_train} images, Test: {n_test} images")

    for entry in manifest[:n_train]:
        img = np.load(entry['img_path'])    # (H, W, 2)
        mask = np.load(entry['mask_path'])  # (H, W)
        train_images.append(img)
        train_masks.append(mask)

    test_images = []
    test_masks = []
    for entry in manifest[n_train:]:
        img = np.load(entry['img_path'])
        mask = np.load(entry['mask_path'])
        test_images.append(img)
        test_masks.append(mask)

    logger.info(f"Loaded {len(train_images)} train, {len(test_images)} test images")

    # Initialize Cellpose model
    # In v4.0.8, CellposeModel only provides eval - training uses train.train_seg()
    model = models.CellposeModel(
        gpu=use_gpu,
        pretrained_model=pretrained_model,
    )

    # Fine-tune using train.train_seg() - the correct v4.0.8 API
    logger.info(f"Starting fine-tuning from {pretrained_model} for {n_epochs} epochs...")
    logger.info(f"  LR={learning_rate}, weight_decay={weight_decay}, batch_size={batch_size}")
    logger.info(f"  Saving to: {output_model_dir}")

    # train.train_seg() returns (filename, train_losses, test_losses)
    filename, train_losses, test_losses = train.train_seg(
        net=model.net,
        train_data=train_images,
        train_labels=train_masks,
        test_data=test_images if test_images else None,
        test_labels=test_masks if test_masks else None,
        save_path=output_model_dir,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        min_train_masks=1,
        model_name=f"cellpose_finetuned_neun_dapi",
    )

    logger.info(f"Fine-tuned model saved to: {filename}")
    logger.info(f"Final train loss: {train_losses[-1]:.4f}")
    if test_losses is not None and len(test_losses) > 0:
        logger.info(f"Final test loss: {test_losses[-1]:.4f}")

    # Save training info
    info = {
        'base_model': pretrained_model,
        'n_epochs': n_epochs,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'batch_size': batch_size,
        'n_train_images': len(train_images),
        'n_test_images': len(test_images),
        'model_path': str(filename),
        'final_train_loss': float(train_losses[-1]) if train_losses else None,
        'final_test_loss': float(test_losses[-1]) if test_losses else None,
    }
    info_path = os.path.join(output_model_dir, "finetune_info.json")
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)

    return filename


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Cellpose on NeuN + DAPI data")
    parser.add_argument("--config", type=str, default=None, help="Pipeline config JSON")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: output/cellpose_finetuned)")
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="Learning rate (v4 default: 1e-5, we use 1e-4 for faster convergence)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--skip_mask_gen", action="store_true",
                        help="Skip silver mask generation (use existing masks)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    config = PipelineConfig.load(args.config) if args.config else PipelineConfig()

    output_dir = args.output_dir or os.path.join(config.output_dir, "cellpose_finetuned")
    silver_mask_dir = os.path.join(output_dir, "silver_masks")
    model_dir = os.path.join(output_dir, "model")

    # =========================================================================
    # Step 1-2: Generate silver truth masks
    # =========================================================================
    if not args.skip_mask_gen:
        print("=" * 60)
        print("STEP 1-2: Generating silver truth masks")
        print("=" * 60)
        results = generate_silver_masks(config, silver_mask_dir)

        total_silver = sum(r['silver_count'] for r in results)
        total_manual = sum(r['manual_count'] for r in results)
        print(f"\nSilver masks: {total_silver} cells from {len(results)} images")
        print(f"Coverage: {total_silver}/{total_manual} manual points matched "
              f"({100*total_silver/total_manual:.1f}%)")
    else:
        print("Skipping mask generation (using existing masks)")

    # =========================================================================
    # Step 3: Fine-tune Cellpose
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 3: Fine-tuning Cellpose")
    print("=" * 60)

    model_path = finetune_cellpose(
        silver_mask_dir=silver_mask_dir,
        output_model_dir=model_dir,
        n_epochs=args.n_epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        pretrained_model='cpsam',  # v4.0.8 only has cpsam
        use_gpu=config.cellpose_gpu,
    )

    print(f"\nFine-tuned model saved to: {model_path}")
    print("To use this model for inference, update config.cellpose_model to the model path")
    print(f'  e.g., config.cellpose_model = "{model_path}"')


if __name__ == "__main__":
    main()
