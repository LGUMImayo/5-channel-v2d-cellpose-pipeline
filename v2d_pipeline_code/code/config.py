"""
Configuration for the Cell Phenotyping Pipeline.

Two-stage pipeline:
  Stage 1: Cell Detection (Cellpose on NeuN + DAPI)
  Stage 2: Multi-label Classification (ResNet-18 on all channels)

Supports two data sources with different channel orders:
  LR Data:   [NeuN, CAMKII, PHF1, BEX1, DAPI] → indices [0, 1, 2, 3, 4]
  MAYO Data: [NeuN, DAPI, CAMKII, PHF1, BEX1] → indices [0, 1, 2, 3, 4]
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json


# =============================================================================
# Channel definitions per data source
# =============================================================================

# Maps channel name → frame index in the multi-frame TIFF
CHANNEL_MAP_LR = {
    "NeuN": 0,
    "CAMKII": 1,
    "PHF1": 2,
    "BEX1": 3,
    "DAPI": 4,
}

CHANNEL_MAP_MAYO = {
    "NeuN": 0,
    "DAPI": 1,
    "CAMKII": 2,
    "PHF1": 3,
    "BEX1": 4,
}

# FIJI CellCounter XML marker type → biomarker name
# Type 1 = NeuN (all neurons), Types 2-4 = co-localized biomarkers
MARKER_TYPE_MAP = {
    1: "NeuN",
    2: "CAMKII",
    3: "PHF1",
    4: "BEX1",
}

# The biomarkers to classify (Stage 2 output labels)
BIOMARKER_LABELS = ["CAMKII", "PHF1", "BEX1"]


@dataclass
class PipelineConfig:
    """Master configuration for the two-stage pipeline."""

    # --- Data paths ---
    lr_data_dir: str = "/fslustre/qhs/ext_chen_yuheng_mayo_edu/Cell_Counting_Proj/multiple_channel_model/Counting_Atypical_LR"
    mayo_data_dir: str = "/fslustre/qhs/ext_chen_yuheng_mayo_edu/Cell_Counting_Proj/multiple_channel_model/Counting_Atypical_LR_MAYO"
    output_dir: str = "/fslustre/qhs/ext_chen_yuheng_mayo_edu/Cell_Counting_Proj/multiple_channel_model/cell_phenotyping_pipeline/output"

    # --- Stage 1: Cell Detection (Cellpose) ---
    cellpose_model: str = "cyto2"          # Pre-trained Cellpose model
    cellpose_diameter: Optional[float] = None  # None = auto-estimate
    cellpose_flow_threshold: float = 0.4
    cellpose_cellprob_threshold: float = 0.0
    cellpose_gpu: bool = True
    cellpose_channels: List[str] = field(default_factory=lambda: ["NeuN", "DAPI"])
    # For Cellpose: channels=[cytoplasm, nucleus] → [NeuN, DAPI]
    # Cellpose channel format: [cyto=1|2, nuc=0|1|2] where 0=none, 1=red/gray, 2=green

    # --- Point matching ---
    match_radius_px: float = 30.0  # Max distance (px) to match biomarker → NeuN point

    # --- Stage 2: Crop Extraction ---
    crop_size: int = 96              # Crop size in pixels (centered on cell)
    num_channels: int = 5            # Total channels in unified crop
    # Unified channel order for crops (regardless of source):
    # [NeuN, DAPI, CAMKII, PHF1, BEX1] — always this order
    unified_channel_order: List[str] = field(
        default_factory=lambda: ["NeuN", "DAPI", "CAMKII", "PHF1", "BEX1"]
    )

    # --- Stage 2: Classifier Architecture ---
    classifier_backbone: str = "resnet18"  # resnet18 or efficientnet_b0
    classifier_in_channels: int = 5
    classifier_num_labels: int = 3          # CAMKII, PHF1, BEX1
    classifier_pretrained: bool = True      # Use ImageNet pre-trained (adapt first conv)
    classifier_dropout: float = 0.3

    # --- Stage 2: Training ---
    batch_size: int = 64
    num_workers: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    max_epochs: int = 100
    patience: int = 15                      # Early stopping patience
    n_folds: int = 5                        # K-fold cross-validation
    seed: int = 42

    # Class weights for BCEWithLogitsLoss (compensate imbalance).
    # These will be computed from data; these are reasonable defaults.
    # Higher weight for rarer positives means the model pays more attention to them.
    pos_weight: Optional[List[float]] = None  # [CAMKII, PHF1, BEX1], computed from data

    # --- Normalization ---
    norm_percentile_low: float = 1.0   # Lower percentile for per-crop normalization
    norm_percentile_high: float = 99.0 # Upper percentile for per-crop normalization

    # --- Augmentation ---
    augment_train: bool = True
    aug_flip_prob: float = 0.5
    aug_rotate_prob: float = 0.5
    aug_brightness_range: Tuple[float, float] = (0.8, 1.2)
    aug_noise_std: float = 0.02

    def get_channel_map(self, source: str) -> Dict[str, int]:
        """Get channel index mapping for a data source."""
        if source.upper() == "LR":
            return CHANNEL_MAP_LR
        elif source.upper() == "MAYO":
            return CHANNEL_MAP_MAYO
        else:
            raise ValueError(f"Unknown source: {source}. Use 'LR' or 'MAYO'.")

    def save(self, path: str):
        """Save config to JSON."""
        import dataclasses
        d = dataclasses.asdict(self)
        with open(path, 'w') as f:
            json.dump(d, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "PipelineConfig":
        """Load config from JSON."""
        with open(path, 'r') as f:
            d = json.load(f)
        # Convert tuples back
        if 'aug_brightness_range' in d and isinstance(d['aug_brightness_range'], list):
            d['aug_brightness_range'] = tuple(d['aug_brightness_range'])
        return cls(**d)

    def __repr__(self):
        import dataclasses
        lines = ["PipelineConfig("]
        for f in dataclasses.fields(self):
            lines.append(f"  {f.name}={getattr(self, f.name)!r},")
        lines.append(")")
        return "\n".join(lines)
