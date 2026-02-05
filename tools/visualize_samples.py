#!/usr/bin/env python
"""
Visualize Random Samples from COCO Dataset

This script visualizes random samples with bboxes and labels overlay.
Useful for verifying annotations before training.

Usage:
    python tools/visualize_samples.py \
        --ann-file data/vehicle_lights/annotations/train.json \
        --img-dir data/vehicle_lights/train/ \
        --output-dir visualizations/ \
        --num-samples 200
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize COCO dataset samples')
    parser.add_argument('--ann-file', required=True, help='Path to COCO annotation JSON')
    parser.add_argument('--img-dir', required=True, help='Path to image directory')
    parser.add_argument('--output-dir', default='visualizations', help='Output directory for visualizations')
    parser.add_argument('--num-samples', type=int, default=200, help='Number of samples to visualize')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    return args


def load_coco_data(ann_file):
    """Load COCO JSON."""
    print(f"Loading annotations from: {ann_file}")
    
    if not os.path.exists(ann_file):
        print(f"❌ ERROR: Annotation file not found: {ann_file}")
        sys.exit(1)
    
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)
    
    return coco_data


def build_lookup_tables(coco_data):
    """Build lookup tables for fast access."""
    
    # Image ID to image info
    img_id_to_info = {img['id']: img for img in coco_data['images']}
    
    # Category ID to category info
    cat_id_to_info = {cat['id']: cat for cat in coco_data['categories']}
    
    # Image ID to annotations
    img_id_to_anns = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_id_to_anns:
            img_id_to_anns[img_id] = []
        img_id_to_anns[img_id].append(ann)
    
    return img_id_to_info, cat_id_to_info, img_id_to_anns


def draw_bbox(img, bbox, label, color, thickness=2):
    """Draw bounding box with label."""
    x, y, w, h = [int(v) for v in bbox]
    
    # Draw rectangle
    cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
    
    # Draw label background
    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    label_w, label_h = label_size
    
    cv2.rectangle(img, (x, y - label_h - 10), (x + label_w + 10, y), color, -1)
    
    # Draw label text
    cv2.putText(img, label, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    return img


def visualize_sample(img_path, annotations, cat_id_to_info):
    """Visualize a single sample."""
    
    # Read image
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️  WARNING: Failed to load image: {img_path}")
        return None
    
    # Color palette (12 colors for 12 classes)
    colors = [
        (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
        (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
        (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0)
    ]
    
    # Draw annotations
    for ann in annotations:
        bbox = ann['bbox']
        cat_id = ann['category_id']
        
        cat_info = cat_id_to_info.get(cat_id, {})
        cat_name = cat_info.get('name', f'Cat_{cat_id}')
        
        # Get color (cycle through palette)
        color = colors[cat_id % len(colors)]
        
        # Draw
        img = draw_bbox(img, bbox, cat_name, color)
    
    return img


def main():
    args = parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Load COCO data
    coco_data = load_coco_data(args.ann_file)
    
    # Build lookup tables
    img_id_to_info, cat_id_to_info, img_id_to_anns = build_lookup_tables(coco_data)
    
    print(f"Total images: {len(img_id_to_info)}")
    print(f"Total annotations: {len(coco_data['annotations'])}")
    print(f"Total categories: {len(cat_id_to_info)}")
    
    # Select random samples
    all_img_ids = list(img_id_to_info.keys())
    
    # Only select images that have annotations
    img_ids_with_anns = [img_id for img_id in all_img_ids if img_id in img_id_to_anns]
    
    if len(img_ids_with_anns) < args.num_samples:
        print(f"⚠️  WARNING: Only {len(img_ids_with_anns)} images with annotations available")
        num_samples = len(img_ids_with_anns)
    else:
        num_samples = args.num_samples
    
    sample_img_ids = random.sample(img_ids_with_anns, num_samples)
    
    print(f"\nVisualizing {num_samples} random samples...")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Visualize samples
    success_count = 0
    
    for img_id in tqdm(sample_img_ids, desc="Visualizing"):
        img_info = img_id_to_info[img_id]
        img_filename = img_info['file_name']
        img_path = os.path.join(args.img_dir, img_filename)
        
        if not os.path.exists(img_path):
            print(f"⚠️  WARNING: Image not found: {img_path}")
            continue
        
        # Get annotations
        annotations = img_id_to_anns.get(img_id, [])
        
        # Visualize
        vis_img = visualize_sample(img_path, annotations, cat_id_to_info)
        
        if vis_img is not None:
            # Save visualization
            output_filename = f"sample_{img_id:06d}_{Path(img_filename).stem}.jpg"
            output_path = os.path.join(args.output_dir, output_filename)
            
            cv2.imwrite(output_path, vis_img)
            success_count += 1
    
    print(f"\n✅ Visualized {success_count} / {num_samples} samples")
    print(f"   Output directory: {args.output_dir}")
    
    # Create a summary grid (if we have samples)
    if success_count > 0:
        print("\nCreating summary grid...")
        create_summary_grid(args.output_dir, max_images=50)


def create_summary_grid(vis_dir, max_images=50, grid_size=(5, 10)):
    """Create a summary grid of visualizations."""
    
    # Get all visualization files
    vis_files = sorted([f for f in os.listdir(vis_dir) if f.endswith('.jpg')])
    
    if not vis_files:
        return
    
    # Limit to max_images
    if len(vis_files) > max_images:
        vis_files = vis_files[:max_images]
    
    # Read first image to get size
    first_img = cv2.imread(os.path.join(vis_dir, vis_files[0]))
    if first_img is None:
        return
    
    img_h, img_w = first_img.shape[:2]
    
    # Resize images to fit grid
    cell_w, cell_h = 384, 216  # 16:9 aspect ratio
    
    rows, cols = grid_size
    grid_img = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
    
    for idx, vis_file in enumerate(vis_files[:rows * cols]):
        row = idx // cols
        col = idx % cols
        
        img = cv2.imread(os.path.join(vis_dir, vis_file))
        if img is None:
            continue
        
        # Resize to fit cell
        img_resized = cv2.resize(img, (cell_w, cell_h))
        
        # Place in grid
        y1 = row * cell_h
        y2 = y1 + cell_h
        x1 = col * cell_w
        x2 = x1 + cell_w
        
        grid_img[y1:y2, x1:x2] = img_resized
    
    # Save grid
    grid_path = os.path.join(vis_dir, '_summary_grid.jpg')
    cv2.imwrite(grid_path, grid_img)
    
    print(f"✅ Created summary grid: {grid_path}")


if __name__ == '__main__':
    main()
