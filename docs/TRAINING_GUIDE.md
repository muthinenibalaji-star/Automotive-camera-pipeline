# RTMDet-m Training Guide for Vehicle Lights Detection

This guide explains how to train the RTMDet-m model on the custom vehicle lights dataset for robust lamp lens detection.

## Overview

**Model**: RTMDet-m (medium variant)  
**Task**: Detect 12 vehicle lamp lens classes  
**Input Resolution**: 1920×1080  
**Framework**: MMDetection (PyTorch)  

### 12 Vehicle Light Classes (Exact Order)

```
1. front_headlight_left
2. front_headlight_right
3. front_indicator_left
4. front_indicator_right
5. front_all_weather_left
6. front_all_weather_right
7. rear_brake_left
8. rear_brake_right
9. rear_indicator_left
10. rear_indicator_right
11. rear_tailgate_left
12. rear_tailgate_right
```

**CRITICAL**: This order must match exactly in your COCO annotations.

---

## Prerequisites

### 1. Environment Setup

Ensure you have installed:
- Python 3.8+
- PyTorch + CUDA (GPU required)
- MMEngine, MMCV, MMDetection

```powershell
# Install MMDetection ecosystem
pip install -U openmim
mim install "mmcv>=2.0.0" "mmdet>=3.0.0"
```

### 2. Dataset Preparation

Your dataset must be in **COCO format** and organized as:

```
data/vehicle_lights/
├── annotations/
│   ├── train.json
│   ├── val.json
│   └── test.json
├── train/          # Training images
├── val/            # Validation images
└── test/           # Test images
```

See `data/vehicle_lights/README.md` for detailed format requirements.

---

## Pre-Training Validation (MANDATORY)

Before training, **always** run these sanity checks:

### 1. Dataset Sanity Check

```bash
python tools/dataset_sanity_check.py \
    --ann-file data/vehicle_lights/annotations/train.json \
    --img-dir data/vehicle_lights/train/
```

This validates:
- COCO JSON structure
- Bbox validity (within bounds, non-zero area)
- Class distribution
- File existence

### 2. Category Mapping Validation

```bash
python tools/category_mapping_validator.py \
    --config configs/vehicle_lights/rtmdet_m_vehicle_lights.py \
    --ann-file data/vehicle_lights/annotations/train.json
```

This ensures COCO `category_id` mapping aligns with config class order.

### 3. Visual Inspection (Recommended)

```bash
python tools/visualize_samples.py \
    --ann-file data/vehicle_lights/annotations/train.json \
    --img-dir data/vehicle_lights/train/ \
    --output-dir visualizations/ \
    --num-samples 200
```

Inspect `visualizations/` to verify:
- Labels follow **lens boundary** (not just bright core)
- Consistent labeling across ON/OFF states
- No mislabeled bboxes

---

## Training

### Basic Training

```bash
python tools/train.py configs/vehicle_lights/rtmdet_m_vehicle_lights.py
```

The script will:
1. Run pre-flight sanity checks automatically
2. Start training with default settings
3. Save checkpoints to `work_dirs/rtmdet_m_vehicle_lights/`

### Advanced Options

**Auto-scale learning rate** (if changing batch size):
```bash
python tools/train.py configs/vehicle_lights/rtmdet_m_vehicle_lights.py --auto-scale-lr
```

**Resume training** from checkpoint:
```bash
python tools/train.py configs/vehicle_lights/rtmdet_m_vehicle_lights.py --resume
```

**Custom work directory**:
```bash
python tools/train.py configs/vehicle_lights/rtmdet_m_vehicle_lights.py --work-dir custom_output/
```

**Skip sanity checks** (NOT recommended):
```bash
python tools/train.py configs/vehicle_lights/rtmdet_m_vehicle_lights.py --skip-sanity-checks
```

---

## Training Configuration Details

The config `configs/vehicle_lights/rtmdet_m_vehicle_lights.py` includes:

- **Model**: RTMDet-m with 12 classes
- **Input size**: 1920×1080
- **Batch size**: 8 (adjust based on GPU memory)
- **Epochs**: 300 (with pipeline switch at epoch 280)
- **Augmentations**: 
  - CachedMosaic + RandomResize (conservative ranges)
  - Mild color jitter (YOLOXHSVRandomAug)
  - RandomFlip
  - **No aggressive crops** (to preserve lamp visibility)

**Why conservative augmentations?**  
OFF-state lenses are small and low-contrast. Aggressive crops or rotations can cut off lamps or make them undetectable.

---

## Evaluation

### Evaluate on Validation Set

```bash
python tools/test.py \
    configs/vehicle_lights/rtmdet_m_vehicle_lights.py \
    work_dirs/rtmdet_m_vehicle_lights/epoch_300.pth
```

### Evaluate on Test Set

Modify config to use test dataloader, then:
```bash
python tools/test.py \
    configs/vehicle_lights/rtmdet_m_vehicle_lights.py \
    work_dirs/rtmdet_m_vehicle_lights/best_coco_bbox_mAP_epoch_*.pth
```

### Metrics to Monitor

- **Per-class AP** (Average Precision)
- **Per-class Recall** (especially for OFF states)
- **Confusion Matrix** (check for brake vs indicator confusion)

---

## Common Issues & Troubleshooting

### 1. Low OFF-State Recall
**Problem**: Model detects ON lenses well but misses OFF lenses.  
**Solutions**:
- Ensure OFF coverage ≥30% in training data
- Label lens boundary consistently (not just bright core)
- Increase input resolution (already at 1920×1080)
- Reduce NMS threshold in test config

### 2. Class Imbalance
**Problem**: Some classes have 10x fewer samples than others.  
**Solutions**:
- Collect more data for underrepresented classes
- Use weighted sampling (modify config)

### 3. Category Mapping Errors
**Problem**: Category IDs don't match config class order.  
**Solution**: Run `category_mapping_validator.py` and fix COCO JSON or config order.

### 4. Bbox Annotation Issues
**Problem**: Bboxes cut off lamp edges or include too much background.  
**Solution**: 
- Re-annotate with lens boundary guidelines
- Run `visualize_samples.py` to audit annotations

---

## Dataset Requirements (Checklist)

Before claiming "dataset ready for training":

- [ ] **COCO format** with `images`, `annotations`, `categories` keys
- [ ] **Category IDs** match config class order (validated)
- [ ] **Lens boundary labels** (not just bright core)
- [ ] **Scenario-based split** (not random frame split)
- [ ] **Coverage matrix**:
  - [ ] Multiple camera angles
  - [ ] Multiple lighting conditions (normal, low, glare)
  - [ ] OFF-state coverage ≥30% per class
- [ ] **Validation passed**:
  - [ ] `dataset_sanity_check.py` 
  - [ ] `category_mapping_validator.py`
  - [ ] `visualize_samples.py` (manual review)

---

## Next Steps After Training

1. **Evaluate** on test set
2. **Analyze** per-class metrics and confusion matrix
3. **Deploy** checkpoint for inference:
   ```python
   from mmdet.apis import DetInferencer
   
   inferencer = DetInferencer(
       model='configs/vehicle_lights/rtmdet_m_vehicle_lights.py',
       weights='work_dirs/rtmdet_m_vehicle_lights/best_coco_bbox_mAP_epoch_*.pth'
   )
   
   result = inferencer('test_image.jpg', out_dir='output/')
   ```

4. **Integrate** with pipeline (see `pipeline/detection/` modules)

---

## References

- [RTMDet Paper](https://arxiv.org/abs/2212.07784)
- [MMDetection Docs](https://mmdetection.readthedocs.io/)
- [COCO Dataset Format](https://cocodataset.org/#format-data)
- Training Guide Source: `RTMDet-m_Training_Guide_Custom_Vehicle_Lights_v1.1_clean.md`
- Pipeline Playbook: `Claude_4_5_Playbook_Vehicle_Lights_Pipeline_Team_Guide_v1.3_clean.md`
