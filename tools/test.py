#!/usr/bin/env python
"""
Test/Evaluation Script for RTMDet-m Vehicle Lights Detection

This script evaluates a trained model on the validation or test set.

Usage:
    python tools/test.py configs/vehicle_lights/rtmdet_m_vehicle_lights.py \
        work_dirs/rtmdet_m_vehicle_lights/epoch_300.pth
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def parse_args():
    parser = argparse.ArgumentParser(description='Test RTMDet-m vehicle lights detector')
    parser.add_argument('config', help='Test config file path')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--work-dir', help='Directory to save evaluation results')
    parser.add_argument(
        '--show',
        action='store_true',
        help='Show prediction results'
    )
    parser.add_argument(
        '--show-dir',
        help='Directory to save visualization results'
    )
    parser.add_argument(
        '--out',
        help='Output file to save predictions (pickle format)'
    )
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='Job launcher'
    )
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    
    # Verify files exist
    if not os.path.exists(args.config):
        print(f"❌ ERROR: Config file not found: {args.config}")
        sys.exit(1)
    
    if not os.path.exists(args.checkpoint):
        print(f"❌ ERROR: Checkpoint file not found: {args.checkpoint}")
        sys.exit(1)
    
    # Import MMDetection
    try:
        from mmengine.config import Config
        from mmengine.runner import Runner
    except ImportError as e:
        print(f"❌ ERROR: Failed to import MMDetection: {e}")
        print("   Please install MMDetection: pip install mmdet")
        sys.exit(1)
    
    # Load config
    cfg = Config.fromfile(args.config)
    
    # Set work directory
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = os.path.join('./work_dirs',
                                    os.path.splitext(os.path.basename(args.config))[0])
    
    # Set checkpoint
    cfg.load_from = args.checkpoint
    
    # Set launcher
    cfg.launcher = args.launcher
    if cfg.launcher == 'none':
        cfg.launcher = None
    
    # Print configuration summary
    print("\n" + "=" * 80)
    print("Evaluation Configuration Summary")
    print("=" * 80)
    print(f"Config file: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Work directory: {cfg.work_dir}")
    print(f"Dataset root: {cfg.data_root}")
    print(f"Number of classes: {cfg.model.bbox_head.num_classes}")
    print("=" * 80 + "\n")
    
    # Build the runner
    runner = Runner.from_cfg(cfg)
    
    # Load checkpoint
    runner.load_checkpoint(args.checkpoint)
    
    # Start evaluation
    print("🔍 Starting evaluation...\n")
    metrics = runner.test()
    
    print("\n" + "=" * 80)
    print("Evaluation Results")
    print("=" * 80)
    
    # Print metrics
    if metrics:
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
    
    print("=" * 80 + "\n")
    print(f"✅ Evaluation completed!")
    print(f"Results saved to: {cfg.work_dir}")


if __name__ == '__main__':
    main()
