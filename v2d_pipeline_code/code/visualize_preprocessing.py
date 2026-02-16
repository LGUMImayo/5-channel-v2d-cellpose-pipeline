#!/usr/bin/env python3
"""
Visualize preprocessing stages: Raw → CLAHE → TTA intensity variants.
Produces a comparison grid for a sample case.
"""

import sys
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from config import PipelineConfig
from data_utils import read_multichannel_tiff, discover_all_data
from bootstrap_cellpose_v2 import prepare_cellpose_input_clahe


def apply_intensity_aug(img_hw2, aug_name):
    """Same augmentations as in run_v2d_inference_all.py TTA."""
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
        img = 255.0 * (img / 255.0) ** 0.6
    elif aug_name == 'gamma_0.8':
        img = 255.0 * (img / 255.0) ** 0.8
    elif aug_name == 'clahe_strong':
        import cv2
        out = np.empty_like(img_hw2)
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
        for ch in range(img_hw2.shape[2]):
            out[:, :, ch] = clahe.apply(img_hw2[:, :, ch])
        return out
    return np.clip(img, 0, 255).astype(np.uint8)


def norm_for_display(ch):
    lo, hi = np.percentile(ch, [1, 99.5])
    return np.clip((ch.astype(np.float32) - lo) / (hi - lo + 1e-8), 0, 1)


def main():
    config = PipelineConfig()
    all_data = discover_all_data(config)

    # Pick two cases: one MAYO, one LR
    targets = ['P3044_C12', 'P3015_C12.5_LR']
    cases = [e for e in all_data if e['case_id'] in targets]

    if not cases:
        cases = all_data[:2]

    out_dir = Path(config.output_dir) / 'cellpose_finetuned_v2' / 'preprocessing_viz'
    out_dir.mkdir(parents=True, exist_ok=True)

    intensity_augs = ['identity', 'bright+15', 'bright+30', 'contrast+30', 'contrast+60',
                      'gamma_0.6', 'gamma_0.8', 'clahe_strong']

    for entry in cases:
        case_id = entry['case_id']
        source = entry['source']
        print(f'\nProcessing {case_id} ({source})')

        img = read_multichannel_tiff(entry['tif_path'])
        channel_map = config.get_channel_map(source)

        # Raw channels
        neun_raw = img[channel_map['NeuN']]
        dapi_raw = img[channel_map['DAPI']]

        # CLAHE preprocessed
        img_clahe, quality_info = prepare_cellpose_input_clahe(img, source, config)
        neun_clahe = img_clahe[:, :, 0]  # (H, W) uint8
        dapi_clahe = img_clahe[:, :, 1]

        print(f'  Quality: {quality_info}')
        print(f'  Raw NeuN range: [{neun_raw.min()}, {neun_raw.max()}]')
        print(f'  CLAHE NeuN range: [{neun_clahe.min()}, {neun_clahe.max()}]')

        # ============================================================
        # Figure 1: Raw vs CLAHE (NeuN and DAPI side by side)
        # ============================================================
        fig, axes = plt.subplots(2, 2, figsize=(16, 16))

        axes[0, 0].imshow(norm_for_display(neun_raw), cmap='gray')
        axes[0, 0].set_title('Raw NeuN', fontsize=14)

        axes[0, 1].imshow(norm_for_display(neun_clahe), cmap='gray')
        axes[0, 1].set_title('CLAHE NeuN', fontsize=14)

        axes[1, 0].imshow(norm_for_display(dapi_raw), cmap='gray')
        axes[1, 0].set_title('Raw DAPI', fontsize=14)

        axes[1, 1].imshow(norm_for_display(dapi_clahe), cmap='gray')
        axes[1, 1].set_title('CLAHE DAPI', fontsize=14)

        for ax in axes.flat:
            ax.axis('off')

        fig.suptitle(f'{case_id} ({source}) — Raw vs CLAHE\n'
                     f'Quality: {quality_info.get("quality", "?")}, '
                     f'DAPI weight: {quality_info.get("dapi_weight_applied", 0)}',
                     fontsize=16)
        plt.tight_layout()
        p1 = out_dir / f'{case_id}_raw_vs_clahe.png'
        plt.savefig(p1, dpi=120, bbox_inches='tight')
        plt.close()
        print(f'  Saved: {p1}')

        # ============================================================
        # Figure 2: Zoomed crop — Raw vs CLAHE (center 800×800 region)
        # ============================================================
        H, W = neun_raw.shape
        cy, cx = H // 2, W // 2
        r = 400
        y0, y1 = max(0, cy - r), min(H, cy + r)
        x0, x1 = max(0, cx - r), min(W, cx + r)

        fig, axes = plt.subplots(2, 2, figsize=(16, 16))

        axes[0, 0].imshow(norm_for_display(neun_raw[y0:y1, x0:x1]), cmap='gray')
        axes[0, 0].set_title('Raw NeuN (zoom)', fontsize=14)

        axes[0, 1].imshow(norm_for_display(neun_clahe[y0:y1, x0:x1]), cmap='gray')
        axes[0, 1].set_title('CLAHE NeuN (zoom)', fontsize=14)

        axes[1, 0].imshow(norm_for_display(dapi_raw[y0:y1, x0:x1]), cmap='gray')
        axes[1, 0].set_title('Raw DAPI (zoom)', fontsize=14)

        axes[1, 1].imshow(norm_for_display(dapi_clahe[y0:y1, x0:x1]), cmap='gray')
        axes[1, 1].set_title('CLAHE DAPI (zoom)', fontsize=14)

        for ax in axes.flat:
            ax.axis('off')

        fig.suptitle(f'{case_id} — Zoom (center {x1-x0}×{y1-y0})', fontsize=16)
        plt.tight_layout()
        p2 = out_dir / f'{case_id}_raw_vs_clahe_zoom.png'
        plt.savefig(p2, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  Saved: {p2}')

        # ============================================================
        # Figure 3: All TTA intensity variants (NeuN channel only, zoomed)
        # ============================================================
        n_augs = len(intensity_augs)
        cols = 4
        rows = (n_augs + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        axes = axes.flat

        for i, aug_name in enumerate(intensity_augs):
            if aug_name == 'identity':
                aug_img = img_clahe
            else:
                aug_img = apply_intensity_aug(img_clahe, aug_name)

            neun_aug = aug_img[:, :, 0]
            crop = neun_aug[y0:y1, x0:x1]

            axes[i].imshow(norm_for_display(crop), cmap='gray')
            axes[i].set_title(f'{aug_name}\n[{crop.min()}, {crop.max()}]', fontsize=12)
            axes[i].axis('off')

        # Hide unused axes
        for j in range(i + 1, len(list(axes))):
            pass  # axes is an iterator, can't hide

        fig.suptitle(f'{case_id} — TTA Intensity Variants (NeuN, zoomed center)',
                     fontsize=16)
        plt.tight_layout()
        p3 = out_dir / f'{case_id}_tta_intensity_variants.png'
        plt.savefig(p3, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  Saved: {p3}')

        # ============================================================
        # Figure 4: Geometric transforms (NeuN CLAHE, full image, downsampled)
        # ============================================================
        geo_names = ['original', 'hflip', 'vflip', 'rot90']
        geo_ops = [
            lambda x: x,
            lambda x: np.flip(x, axis=1),
            lambda x: np.flip(x, axis=0),
            lambda x: np.rot90(x, 1),
        ]

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        for i, (gname, gfn) in enumerate(zip(geo_names, geo_ops)):
            transformed = gfn(img_clahe)
            neun_t = transformed[:, :, 0] if transformed.ndim == 3 else transformed
            # Downsample for display
            ds = max(1, neun_t.shape[0] // 600)
            axes[i].imshow(norm_for_display(neun_t[::ds, ::ds]), cmap='gray')
            axes[i].set_title(gname, fontsize=14)
            axes[i].axis('off')

        fig.suptitle(f'{case_id} — TTA Geometric Transforms', fontsize=16)
        plt.tight_layout()
        p4 = out_dir / f'{case_id}_tta_geometric.png'
        plt.savefig(p4, dpi=120, bbox_inches='tight')
        plt.close()
        print(f'  Saved: {p4}')

        # ============================================================
        # Figure 5: Cellpose 2-channel input visualization (what the model sees)
        # ============================================================
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Channel 0 = NeuN (cytoplasm)
        axes[0].imshow(norm_for_display(neun_clahe), cmap='gray')
        axes[0].set_title('Ch0: NeuN (cytoplasm)', fontsize=14)

        # Channel 1 = DAPI (nucleus)
        axes[1].imshow(norm_for_display(dapi_clahe), cmap='gray')
        axes[1].set_title('Ch1: DAPI (nucleus)', fontsize=14)

        # Composite: NeuN=green, DAPI=blue (what Cellpose "sees")
        rgb = np.zeros((*neun_clahe.shape, 3), dtype=np.float32)
        rgb[..., 1] = norm_for_display(neun_clahe)   # green = cytoplasm
        rgb[..., 2] = norm_for_display(dapi_clahe)    # blue = nucleus
        # Downsample for display
        ds = max(1, rgb.shape[0] // 800)
        axes[2].imshow(rgb[::ds, ::ds])
        axes[2].set_title('Composite: NeuN(G) + DAPI(B)', fontsize=14)

        for ax in axes:
            ax.axis('off')

        fig.suptitle(f'{case_id} — Cellpose 2-Channel Input', fontsize=16)
        plt.tight_layout()
        p5 = out_dir / f'{case_id}_cellpose_input.png'
        plt.savefig(p5, dpi=120, bbox_inches='tight')
        plt.close()
        print(f'  Saved: {p5}')

    print(f'\nAll visualizations saved to: {out_dir}')


if __name__ == '__main__':
    main()
