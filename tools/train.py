#!/usr/bin/env python
"""
Training Script for RTMDet-m Vehicle Lights Detection

This script:
1. Runs dataset sanity checks before training
2. Validates category mapping
3. Launches MMDetection training with proper configuration

Usage:
    python tools/train.py configs/vehicle_lights/rtmdet_m_vehicle_lights.py [--auto-scale-lr]
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def parse_args():
    parser = argparse.ArgumentParser(description='Train RTMDet-m for vehicle lights detection')
    parser.add_argument('config', help='Training config file path')
    parser.add_argument('--work-dir', help='Directory to save logs and models')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='Auto scale learning rate based on actual batch size'
    )
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='Resume training from checkpoint'
    )
    parser.add_argument(
        '--skip-sanity-checks',
        action='store_true',
        help='Skip pre-flight dataset sanity checks (not recommended)'
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


def run_sanity_checks(config_path):
    """Run dataset sanity checks before training."""
    print("\n" + "=" * 80)
    print("Running pre-flight dataset sanity checks...")
    print("=" * 80 + "\n")
    
    try:
        from mmengine.config import Config
        
        # Load config
        cfg = Config.fromfile(config_path)
        data_root = cfg.data_root
        
        # Check if dataset exists
        if not os.path.exists(data_root):
            print(f"❌ ERROR: Dataset root not found: {data_root}")
            print(f"Please create the dataset directory and populate it with COCO-format data.")
            return False
        
        # Check annotation files
        train_ann = os.path.join(data_root, 'annotations', 'train.json')
        val_ann = os.path.join(data_root, 'annotations', 'val.json')
        
        if not os.path.exists(train_ann):
            print(f"❌ ERROR: Training annotations not found: {train_ann}")
            return False
        
        if not os.path.exists(val_ann):
            print(f"❌ ERROR: Validation annotations not found: {val_ann}")
            return False
        
        print(f"✅ Dataset root exists: {data_root}")
        print(f"✅ Training annotations found: {train_ann}")
        print(f"✅ Validation annotations found: {val_ann}")
        
        # Run detailed sanity checks
        print("\nRunning detailed dataset validation...")
        sanity_check_script = project_root / 'tools' / 'dataset_sanity_check.py'
        
        if sanity_check_script.exists():
            import subprocess
            result = subprocess.run(
                [sys.executable, str(sanity_check_script), 
                 '--ann-file', train_ann,
                 '--img-dir', os.path.join(data_root, 'train')],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                print(f"❌ Dataset sanity check failed!")
                print(result.stdout)
                print(result.stderr)
                return False
            
            print(result.stdout)
        
        # Run category mapping validation
        print("\nValidating category mapping...")
        mapping_validator = project_root / 'tools' / 'category_mapping_validator.py'
        
        if mapping_validator.exists():
            import subprocess
            result = subprocess.run(
                [sys.executable, str(mapping_validator),
                 '--config', config_path,
                 '--ann-file', train_ann],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                print(f"❌ Category mapping validation failed!")
                print(result.stdout)
                print(result.stderr)
                return False
            
            print(result.stdout)
        
        print("\n✅ All sanity checks passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error during sanity checks: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    args = parse_args()
    
    # Verify config file exists
    if not os.path.exists(args.config):
        print(f"❌ ERROR: Config file not found: {args.config}")
        sys.exit(1)
    
    # Run sanity checks (unless explicitly skipped)
    if not args.skip_sanity_checks:
        if not run_sanity_checks(args.config):
            print("\n❌ Pre-flight checks failed. Please fix the issues above before training.")
            print("   (Use --skip-sanity-checks to bypass, but this is NOT recommended)")
            sys.exit(1)
    else:
        print("\n⚠️  WARNING: Skipping sanity checks. Training may fail if dataset is invalid.")
    
    # Import MMDetection training tools
    try:
        from mmdet.apis import init_detector, train_detector
        from mmengine.config import Config, DictAction
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
        # Use config filename as default work_dir
        cfg.work_dir = os.path.join('./work_dirs',
                                    os.path.splitext(os.path.basename(args.config))[0])
    
    # Auto-scale learning rate
    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and cfg.auto_scale_lr.get('enable', False):
            pass  # Already enabled in config
        else:
            cfg.auto_scale_lr = dict(enable=True, base_batch_size=16)
    
    # Resume training
    if args.resume:
        cfg.resume = True
        if args.resume != 'auto':
            cfg.load_from = args.resume
    
    # Set launcher
    cfg.launcher = args.launcher
    if cfg.launcher == 'none':
        cfg.launcher = None
    
    # Print configuration summary
    print("\n" + "=" * 80)
    print("Training Configuration Summary")
    print("=" * 80)
    print(f"Config file: {args.config}")
    print(f"Work directory: {cfg.work_dir}")
    print(f"Dataset root: {cfg.data_root}")
    print(f"Number of classes: {cfg.model.bbox_head.num_classes}")
    print(f"Max epochs: {cfg.train_cfg.max_epochs}")
    print(f"Batch size: {cfg.train_dataloader.batch_size}")
    print(f"Base learning rate: {cfg.optim_wrapper.optimizer.lr}")
    print(f"Auto-scale LR: {args.auto_scale_lr}")
    print("=" * 80 + "\n")
    
    # Build the runner
    runner = Runner.from_cfg(cfg)
    
    # Start training
    print("🚀 Starting training...\n")
    runner.train()
    
    print("\n✅ Training completed!")
    print(f"Checkpoints saved to: {cfg.work_dir}")


if __name__ == '__main__':
    main()
