"""
Utility functions for parsing FIJI CellCounter XML files, reading multi-channel
TIFFs, and matching biomarker annotations to NeuN neurons.
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist

from config import (
    MARKER_TYPE_MAP,
    BIOMARKER_LABELS,
    CHANNEL_MAP_LR,
    CHANNEL_MAP_MAYO,
    PipelineConfig,
)


# =============================================================================
# XML Parsing
# =============================================================================

def parse_cellcounter_xml(xml_path: str) -> Dict[int, Dict]:
    """
    Parse a FIJI CellCounter XML file.

    Returns:
        dict mapping marker type (int) to:
            {'name': str, 'points': np.ndarray of shape (N, 2) as (x, y)}
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    marker_types = {}
    for mt in root.findall('.//Marker_Type'):
        typ = int(mt.find('Type').text)
        name = mt.find('Name').text
        markers = []
        for m in mt.findall('Marker'):
            x = int(m.find('MarkerX').text)
            y = int(m.find('MarkerY').text)
            markers.append((x, y))

        points = np.array(markers, dtype=np.float64) if markers else np.zeros((0, 2), dtype=np.float64)
        marker_types[typ] = {'name': name, 'points': points}

    return marker_types


def match_biomarkers_to_neurons(
    marker_types: Dict[int, Dict],
    match_radius: float = 15.0,
) -> List[Dict]:
    """
    Match biomarker annotations (Types 2-4) to NeuN neurons (Type 1).

    For each NeuN neuron, check if there's a CAMKII/PHF1/BEX1 marker within
    `match_radius` pixels. Produces a multi-label annotation per neuron.

    Args:
        marker_types: Output of parse_cellcounter_xml()
        match_radius: Max pixel distance for a match

    Returns:
        List of dicts, one per NeuN neuron:
        {
            'x': float, 'y': float,
            'labels': {'CAMKII': 0 or 1, 'PHF1': 0 or 1, 'BEX1': 0 or 1}
        }
    """
    neun_points = marker_types.get(1, {}).get('points', np.zeros((0, 2)))
    if len(neun_points) == 0:
        return []

    neurons = []
    for i, (nx, ny) in enumerate(neun_points):
        labels = {}
        for marker_type_id, biomarker_name in MARKER_TYPE_MAP.items():
            if biomarker_name == "NeuN":
                continue  # Skip — NeuN is the base
            bio_points = marker_types.get(marker_type_id, {}).get('points', np.zeros((0, 2)))
            if len(bio_points) == 0:
                labels[biomarker_name] = 0
            else:
                # Distance from this NeuN point to all biomarker points
                dists = np.sqrt((bio_points[:, 0] - nx) ** 2 + (bio_points[:, 1] - ny) ** 2)
                labels[biomarker_name] = 1 if dists.min() <= match_radius else 0

        neurons.append({
            'x': float(nx),
            'y': float(ny),
            'labels': labels,
        })

    return neurons


# =============================================================================
# TIFF Reading
# =============================================================================

def read_multichannel_tiff(tif_path: str) -> np.ndarray:
    """
    Read a multi-frame TIFF where each IFD/frame is one channel.

    Returns:
        np.ndarray of shape (C, H, W), dtype uint16
    """
    try:
        import tifffile
        img = tifffile.imread(tif_path)
        # tifffile may return (C, H, W) or (H, W) for single frame
        if img.ndim == 2:
            img = img[np.newaxis, ...]
        return img
    except ImportError:
        # Fallback: use PIL to read frame by frame
        from PIL import Image
        pil_img = Image.open(tif_path)
        frames = []
        try:
            for i in range(100):  # Safety limit
                pil_img.seek(i)
                frame = np.array(pil_img)
                frames.append(frame)
        except EOFError:
            pass

        if not frames:
            raise ValueError(f"Could not read any frames from {tif_path}")

        return np.stack(frames, axis=0)


def unify_channel_order(
    img: np.ndarray,
    source: str,
    config: PipelineConfig,
) -> np.ndarray:
    """
    Reorder channels from source-specific order to the unified order.

    Unified order: [NeuN, DAPI, CAMKII, PHF1, BEX1]

    Args:
        img: (C, H, W) array in the source's native channel order
        source: 'LR' or 'MAYO'
        config: PipelineConfig

    Returns:
        (C, H, W) array in unified channel order
    """
    source_map = config.get_channel_map(source)
    unified = config.unified_channel_order
    indices = [source_map[ch_name] for ch_name in unified]
    return img[indices]


# =============================================================================
# Crop Extraction
# =============================================================================

def extract_cell_crop(
    img: np.ndarray,
    cx: float,
    cy: float,
    crop_size: int = 96,
) -> Optional[np.ndarray]:
    """
    Extract a square crop centered on (cx, cy) from a (C, H, W) image.

    Returns:
        (C, crop_size, crop_size) array, or None if the crop would go out of bounds.
    """
    C, H, W = img.shape
    half = crop_size // 2

    x_int = int(round(cx))
    y_int = int(round(cy))

    x0 = x_int - half
    y0 = y_int - half
    x1 = x0 + crop_size
    y1 = y0 + crop_size

    # Skip cells too close to the edge
    if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
        return None

    return img[:, y0:y1, x0:x1].copy()


def normalize_crop(
    crop: np.ndarray,
    pct_low: float = 1.0,
    pct_high: float = 99.0,
) -> np.ndarray:
    """
    Per-channel percentile normalization to [0, 1].

    Args:
        crop: (C, H, W) array, any dtype
        pct_low: Lower percentile
        pct_high: Upper percentile

    Returns:
        (C, H, W) float32 array, clipped to [0, 1]
    """
    crop = crop.astype(np.float32)
    C = crop.shape[0]
    for c in range(C):
        channel = crop[c]
        lo = np.percentile(channel, pct_low)
        hi = np.percentile(channel, pct_high)
        if hi - lo < 1e-6:
            crop[c] = 0.0
        else:
            crop[c] = (channel - lo) / (hi - lo)
    return np.clip(crop, 0.0, 1.0)


# =============================================================================
# Data Discovery
# =============================================================================

def discover_cases(data_dir: str) -> List[Dict[str, str]]:
    """
    Discover all case directories in a data directory.

    Each case directory should contain one .tif and one .xml file.

    Returns:
        List of dicts: {'case_id': str, 'tif_path': str, 'xml_path': str}
    """
    cases = []
    data_path = Path(data_dir)

    for sub in sorted(data_path.iterdir()):
        if not sub.is_dir():
            continue

        tifs = list(sub.glob("*.tif"))
        xmls = list(sub.glob("*.xml"))

        if len(tifs) == 1 and len(xmls) == 1:
            cases.append({
                'case_id': sub.name,
                'tif_path': str(tifs[0]),
                'xml_path': str(xmls[0]),
            })
        elif len(tifs) > 0 and len(xmls) > 0:
            # Take first match
            cases.append({
                'case_id': sub.name,
                'tif_path': str(tifs[0]),
                'xml_path': str(xmls[0]),
            })

    return cases


def discover_all_data(config: PipelineConfig) -> List[Dict]:
    """
    Discover all cases from both LR and MAYO data sources.

    Returns:
        List of dicts with 'case_id', 'tif_path', 'xml_path', 'source'
    """
    all_cases = []

    for source, data_dir in [("LR", config.lr_data_dir), ("MAYO", config.mayo_data_dir)]:
        if not os.path.isdir(data_dir):
            print(f"Warning: {source} data dir not found: {data_dir}")
            continue

        cases = discover_cases(data_dir)
        for c in cases:
            c['source'] = source
        all_cases.extend(cases)
        print(f"Found {len(cases)} cases in {source} ({data_dir})")

    return all_cases


# =============================================================================
# Summary / Statistics
# =============================================================================

def print_annotation_summary(all_cases: List[Dict], config: PipelineConfig):
    """
    Print a summary of annotations across all cases.
    """
    total_neurons = 0
    total_labels = {b: 0 for b in BIOMARKER_LABELS}

    for case in all_cases:
        marker_types = parse_cellcounter_xml(case['xml_path'])
        neurons = match_biomarkers_to_neurons(marker_types, config.match_radius_px)
        total_neurons += len(neurons)

        for n in neurons:
            for b in BIOMARKER_LABELS:
                total_labels[b] += n['labels'].get(b, 0)

    print(f"\n=== Annotation Summary ===")
    print(f"Total cases: {len(all_cases)}")
    print(f"Total NeuN+ neurons: {total_neurons}")
    for b in BIOMARKER_LABELS:
        pos = total_labels[b]
        pct = 100.0 * pos / total_neurons if total_neurons > 0 else 0
        print(f"  {b}+: {pos} ({pct:.1f}%)")
    print()
