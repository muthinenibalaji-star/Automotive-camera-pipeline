#!/usr/bin/env python
"""
Dataset Sanity Check Tool for COCO-format Vehicle Lights Dataset

This script validates:
- COCO JSON structure (images, annotations, categories)
- Category ID mapping alignment
- Bbox validity (within image bounds, non-zero area)
- Class distribution statistics
- File existence verification

Usage:
    python tools/dataset_sanity_check.py --ann-file data/vehicle_lights/annotations/train.json \
                                          --img-dir data/vehicle_lights/train/
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='COCO dataset sanity check')
    parser.add_argument('--ann-file', required=True, help='Path to COCO annotation JSON file')
    parser.add_argument('--img-dir', required=True, help='Path to image directory')
    parser.add_argument('--sample-size', type=int, default=200, 
                       help='Number of random samples to check for file existence')
    args = parser.parse_args()
    return args


def load_coco_json(ann_file):
    """Load COCO JSON file."""
    print(f"Loading annotations from: {ann_file}")
    
    if not os.path.exists(ann_file):
        print(f"❌ ERROR: Annotation file not found: {ann_file}")
        sys.exit(1)
    
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)
    
    return coco_data


def check_coco_structure(coco_data):
    """Check if COCO JSON has required keys."""
    print("\n" + "=" * 80)
    print("Checking COCO JSON structure...")
    print("=" * 80)
    
    required_keys = ['images', 'annotations', 'categories']
    missing_keys = [key for key in required_keys if key not in coco_data]
    
    if missing_keys:
        print(f"❌ ERROR: Missing required keys: {missing_keys}")
        return False
    
    print(f"✅ All required keys present: {required_keys}")
    print(f"   - Images: {len(coco_data['images'])}")
    print(f"   - Annotations: {len(coco_data['annotations'])}")
    print(f"   - Categories: {len(coco_data['categories'])}")
    
    return True


def check_categories(coco_data):
    """Check category definitions."""
    print("\n" + "=" * 80)
    print("Checking categories...")
    print("=" * 80)
    
    categories = coco_data['categories']
    
    if not categories:
        print("❌ ERROR: No categories defined!")
        return False
    
    # Expected classes in order
    expected_classes = [
        'front_headlight_left',
        'front_headlight_right',
        'front_indicator_left',
        'front_indicator_right',
        'front_all_weather_left',
        'front_all_weather_right',
        'rear_brake_left',
        'rear_brake_right',
        'rear_indicator_left',
        'rear_indicator_right',
        'rear_tailgate_left',
        'rear_tailgate_right'
    ]
    
    print(f"\nTotal categories: {len(categories)}")
    print(f"Expected categories: {len(expected_classes)}")
    
    if len(categories) != len(expected_classes):
        print(f"⚠️  WARNING: Number of categories ({len(categories)}) doesn't match expected ({len(expected_classes)})")
    
    # Build category mapping
    cat_id_to_name = {}
    for cat in categories:
        cat_id = cat['id']
        cat_name = cat['name']
        cat_id_to_name[cat_id] = cat_name
        print(f"   Category ID {cat_id}: {cat_name}")
    
    # Check for expected classes
    category_names = [cat['name'] for cat in categories]
    for expected_class in expected_classes:
        if expected_class not in category_names:
            print(f"⚠️  WARNING: Expected class not found: {expected_class}")
    
    print(f"\n✅ Categories defined")
    
    return cat_id_to_name


def check_bbox_validity(coco_data):
    """Check bbox validity."""
    print("\n" + "=" * 80)
    print("Checking bbox validity...")
    print("=" * 80)
    
    # Build image lookup
    img_id_to_info = {img['id']: img for img in coco_data['images']}
    
    annotations = coco_data['annotations']
    invalid_bboxes = []
    zero_area_bboxes = []
    out_of_bounds_bboxes = []
    
    for ann in annotations:
        bbox = ann['bbox']  # [x, y, w, h] in COCO format
        x, y, w, h = bbox
        
        # Check for zero area
        if w <= 0 or h <= 0:
            zero_area_bboxes.append(ann['id'])
            continue
        
        # Check if bbox is within image bounds
        img_info = img_id_to_info.get(ann['image_id'])
        if img_info:
            img_w, img_h = img_info['width'], img_info['height']
            
            if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
                out_of_bounds_bboxes.append({
                    'ann_id': ann['id'],
                    'bbox': bbox,
                    'img_size': (img_w, img_h)
                })
    
    total_annotations = len(annotations)
    
    print(f"Total annotations checked: {total_annotations}")
    
    if zero_area_bboxes:
        print(f"❌ ERROR: {len(zero_area_bboxes)} bboxes with zero or negative area!")
        print(f"   First few: {zero_area_bboxes[:5]}")
        return False
    else:
        print(f"✅ No zero-area bboxes")
    
    if out_of_bounds_bboxes:
        print(f"⚠️  WARNING: {len(out_of_bounds_bboxes)} bboxes out of image bounds!")
        print(f"   First few: {out_of_bounds_bboxes[:3]}")
    else:
        print(f"✅ All bboxes within image bounds")
    
    return True


def check_class_distribution(coco_data, cat_id_to_name):
    """Check class distribution."""
    print("\n" + "=" * 80)
    print("Checking class distribution...")
    print("=" * 80)
    
    annotations = coco_data['annotations']
    
    # Count annotations per category
    category_counts = defaultdict(int)
    for ann in annotations:
        cat_id = ann['category_id']
        category_counts[cat_id] += 1
    
    # Print distribution
    print(f"\nClass distribution:")
    total = sum(category_counts.values())
    
    for cat_id in sorted(category_counts.keys()):
        count = category_counts[cat_id]
        percentage = (count / total) * 100
        cat_name = cat_id_to_name.get(cat_id, f'Unknown_{cat_id}')
        print(f"   {cat_name:30s}: {count:6d} ({percentage:5.2f}%)")
    
    # Check for severe imbalance
    if category_counts:
        max_count = max(category_counts.values())
        min_count = min(category_counts.values())
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        print(f"\nImbalance ratio (max/min): {imbalance_ratio:.2f}")
        
        if imbalance_ratio > 10:
            print(f"⚠️  WARNING: Severe class imbalance detected (ratio > 10)")
            print(f"   Consider collecting more samples for underrepresented classes")
        else:
            print(f"✅ Class distribution is acceptable")
    
    return True


def check_file_existence(coco_data, img_dir, sample_size):
    """Check if image files exist."""
    print("\n" + "=" * 80)
    print(f"Checking file existence (random sample of {sample_size})...")
    print("=" * 80)
    
    images = coco_data['images']
    
    if len(images) < sample_size:
        sample_size = len(images)
        print(f"   Dataset has fewer images than sample size, checking all {sample_size} images")
    
    # Random sample
    sample_indices = np.random.choice(len(images), size=sample_size, replace=False)
    sample_images = [images[i] for i in sample_indices]
    
    missing_files = []
    
    for img in sample_images:
        img_path = os.path.join(img_dir, img['file_name'])
        if not os.path.exists(img_path):
            missing_files.append(img['file_name'])
    
    if missing_files:
        print(f"❌ ERROR: {len(missing_files)} image files not found!")
        print(f"   First few: {missing_files[:5]}")
        return False
    else:
        print(f"✅ All {sample_size} sampled image files exist")
    
    return True


def main():
    args = parse_args()
    
    # Load COCO JSON
    coco_data = load_coco_json(args.ann_file)
    
    # Run checks
    all_passed = True
    
    # 1. Check structure
    if not check_coco_structure(coco_data):
        all_passed = False
        sys.exit(1)  # Fatal error, cannot continue
    
    # 2. Check categories
    cat_id_to_name = check_categories(coco_data)
    if not cat_id_to_name:
        all_passed = False
        sys.exit(1)
    
    # 3. Check bbox validity
    if not check_bbox_validity(coco_data):
        all_passed = False
    
    # 4. Check class distribution
    if not check_class_distribution(coco_data, cat_id_to_name):
        all_passed = False
    
    # 5. Check file existence
    if not check_file_existence(coco_data, args.img_dir, args.sample_size):
        all_passed = False
    
    # Final report
    print("\n" + "=" * 80)
    if all_passed:
        print("✅ All sanity checks PASSED!")
    else:
        print("❌ Some sanity checks FAILED. Please fix the issues above.")
    print("=" * 80 + "\n")
    
    sys.exit(0 if all_passed else 1)


if __name__ == '__main__':
    main()
