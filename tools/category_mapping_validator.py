#!/usr/bin/env python
"""
Category Mapping Validator for Vehicle Lights Dataset

This script validates that the COCO JSON category_id mapping aligns correctly
with the metainfo['classes'] order in the training config.

Usage:
    python tools/category_mapping_validator.py \
        --config configs/vehicle_lights/rtmdet_m_vehicle_lights.py \
        --ann-file data/vehicle_lights/annotations/train.json
"""

import argparse
import json
import os
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Validate COCO category mapping')
    parser.add_argument('--config', required=True, help='Path to training config file')
    parser.add_argument('--ann-file', required=True, help='Path to COCO annotation JSON')
    args = parser.parse_args()
    return args


def load_config_classes(config_path):
    """Load class list from MMDetection config."""
    print(f"Loading config from: {config_path}")
    
    if not os.path.exists(config_path):
        print(f"❌ ERROR: Config file not found: {config_path}")
        sys.exit(1)
    
    try:
        from mmengine.config import Config
        cfg = Config.fromfile(config_path)
        
        # Extract classes from metainfo
        if hasattr(cfg, 'metainfo') and 'classes' in cfg.metainfo:
            classes = cfg.metainfo['classes']
        elif hasattr(cfg, 'train_dataloader'):
            classes = cfg.train_dataloader.dataset.metainfo.get('classes', [])
        else:
            print("❌ ERROR: Cannot find classes in config")
            sys.exit(1)
        
        return list(classes)
    
    except Exception as e:
        print(f"❌ ERROR: Failed to load config: {e}")
        sys.exit(1)


def load_coco_categories(ann_file):
    """Load categories from COCO JSON."""
    print(f"Loading annotations from: {ann_file}")
    
    if not os.path.exists(ann_file):
        print(f"❌ ERROR: Annotation file not found: {ann_file}")
        sys.exit(1)
    
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)
    
    if 'categories' not in coco_data:
        print("❌ ERROR: No categories found in COCO JSON")
        sys.exit(1)
    
    return coco_data['categories']


def validate_mapping(config_classes, coco_categories):
    """Validate category ID mapping."""
    print("\n" + "=" * 80)
    print("Validating Category Mapping")
    print("=" * 80)
    
    # Build COCO category mapping (sorted by ID)
    coco_cats_sorted = sorted(coco_categories, key=lambda x: x['id'])
    
    print(f"\nConfig classes: {len(config_classes)}")
    print(f"COCO categories: {len(coco_cats_sorted)}")
    
    if len(config_classes) != len(coco_cats_sorted):
        print(f"\n❌ ERROR: Number of classes mismatch!")
        print(f"   Config has {len(config_classes)} classes")
        print(f"   COCO has {len(coco_cats_sorted)} categories")
        return False
    
    # Print mapping table
    print("\n" + "-" * 80)
    print(f"{'Index':<8} {'Config Class':<35} {'COCO ID':<10} {'COCO Name':<35}")
    print("-" * 80)
    
    all_match = True
    mismatches = []
    
    for idx, (config_class, coco_cat) in enumerate(zip(config_classes, coco_cats_sorted)):
        coco_id = coco_cat['id']
        coco_name = coco_cat['name']
        
        match_status = "✅" if config_class == coco_name else "❌"
        
        print(f"{idx:<8} {config_class:<35} {coco_id:<10} {coco_name:<35} {match_status}")
        
        if config_class != coco_name:
            all_match = False
            mismatches.append({
                'index': idx,
                'config': config_class,
                'coco_id': coco_id,
                'coco_name': coco_name
            })
    
    print("-" * 80)
    
    # Report results
    if all_match:
        print("\n✅ Category mapping is CORRECT!")
        print("   - All config classes match COCO category names in order")
        print("   - Category IDs are properly aligned")
        return True
    else:
        print(f"\n❌ Category mapping has {len(mismatches)} MISMATCHES!")
        print("\nMismatched entries:")
        for mm in mismatches:
            print(f"   Index {mm['index']}: Config='{mm['config']}' vs COCO='{mm['coco_name']}' (ID={mm['coco_id']})")
        
        print("\n⚠️  IMPORTANT:")
        print("   The order of classes in config metainfo MUST match the COCO category order.")
        print("   MMDetection uses the class index (0-based) to map category IDs.")
        print("   Misalignment will cause incorrect predictions!")
        
        return False


def main():
    args = parse_args()
    
    # Load config classes
    config_classes = load_config_classes(args.config)
    print(f"✅ Loaded {len(config_classes)} classes from config")
    
    # Load COCO categories
    coco_categories = load_coco_categories(args.ann_file)
    print(f"✅ Loaded {len(coco_categories)} categories from COCO JSON")
    
    # Validate mapping
    is_valid = validate_mapping(config_classes, coco_categories)
    
    print("\n" + "=" * 80)
    if is_valid:
        print("✅ Validation PASSED - Category mapping is correct")
    else:
        print("❌ Validation FAILED - Please fix category mapping before training")
    print("=" * 80 + "\n")
    
    sys.exit(0 if is_valid else 1)


if __name__ == '__main__':
    main()
