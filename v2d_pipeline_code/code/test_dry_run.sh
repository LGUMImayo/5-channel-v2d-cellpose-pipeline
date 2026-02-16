#!/bin/bash
# Quick dry run test before submitting to Slurm

set -e
source ~/.bashrc
conda activate rhizonet

cd /fslustre/qhs/ext_chen_yuheng_mayo_edu/Cell_Counting_Proj/multiple_channel_model/cell_phenotyping_pipeline

echo "=== Running Dry Run Test ==="
echo "Testing cellpose detection on first case only..."
echo ""

python << 'EOFPYTHON'
import sys
import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

from config import PipelineConfig
from data_utils import discover_all_data, read_multichannel_tiff, parse_cellcounter_xml
from stage1_detection import get_cellpose_input, run_cellpose_detection, get_mask_centroids

config = PipelineConfig()
config.cellpose_gpu = False  # CPU for quick test

all_data = discover_all_data(config)
if not all_data:
    print("ERROR: No data!")
    sys.exit(1)

case = all_data[0]
print(f"\n✓ Testing on: {case['case_id']}")

img = read_multichannel_tiff(case['tif_path'])
print(f"✓ Image: {img.shape}")

cp_input = get_cellpose_input(img, case['source'], config)
print(f"✓ Prepared: {cp_input.shape}")

print("✓ Running Cellpose (CPU)...")
masks, info = run_cellpose_detection(cp_input, config)
print(f"✓ Found {info['num_cells']} cells")

centroids = get_mask_centroids(masks)
print(f"✓ Extracted {len(centroids)} centroids")

marker_types = parse_cellcounter_xml(case['xml_path'])
manual_neun = marker_types.get(1, {}).get('points', None)
if manual_neun is not None:
    print(f"✓ Manual: {len(manual_neun)} annotations")

print("\n✅ SUCCESS - Ready to submit to Slurm!")
EOFPYTHON

echo ""
echo "Dry run completed successfully!"
