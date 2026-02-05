# Vehicle Lights Dataset (COCO Format)

This directory contains the vehicle lights detection dataset in COCO format.

## Directory Structure

```
data/vehicle_lights/
├── annotations/
│   ├── train.json      # Training set annotations
│   ├── val.json        # Validation set annotations
│   └── test.json       # Test set annotations
├── train/              # Training images
├── val/                # Validation images
├── test/               # Test images
└── README.md          # This file
```

## COCO JSON Format

Each annotation file (`train.json`, `val.json`, `test.json`) must follow the COCO format:

```json
{
  "images": [
    {
      "id": 1,
      "width": 1920,
      "height": 1080,
      "file_name": "000001.jpg"
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [x, y, width, height],
      "area": width * height,
      "iscrowd": 0
    }
  ],
  "categories": [
    {"id": 1, "name": "front_headlight_left", "supercategory": "light"},
    {"id": 2, "name": "front_headlight_right", "supercategory": "light"},
    {"id": 3, "name": "front_indicator_left", "supercategory": "light"},
    {"id": 4, "name": "front_indicator_right", "supercategory": "light"},
    {"id": 5, "name": "front_all_weather_left", "supercategory": "light"},
    {"id": 6, "name": "front_all_weather_right", "supercategory": "light"},
    {"id": 7, "name": "rear_brake_left", "supercategory": "light"},
    {"id": 8, "name": "rear_brake_right", "supercategory": "light"},
    {"id": 9, "name": "rear_indicator_left", "supercategory": "light"},
    {"id": 10, "name": "rear_indicator_right", "supercategory": "light"},
    {"id": 11, "name": "rear_tailgate_left", "supercategory": "light"},
    {"id": 12, "name": "rear_tailgate_right", "supercategory": "light"}
  ]
}
```

## Important Notes

### Bbox Format
- COCO uses `[x, y, width, height]` format (top-left corner + dimensions)
- `x, y` is the top-left corner of the bounding box
- Coordinates are in pixels

### Category IDs
**CRITICAL**: Category IDs must match the order in the training config:
- Category ID 1 → `front_headlight_left` (index 0)
- Category ID 2 → `front_headlight_right` (index 1)
- ... and so on

The category ID order **must** align with the `metainfo['classes']` in the training config.

### Labeling Guidelines
1. **Label the lens boundary** (not just the bright core when ON)
2. **Consistent labeling** across ON and OFF states
3. **Split by scenario/session** (not random frames)
   - Train: 70%
   - Val: 15%
   - Test: 15%

4. **Coverage requirements**:
   - Multiple bench configurations
   - Multiple camera angles
   - Multiple lighting conditions (normal, low, glare)
   - Both ON and OFF states for all classes
   - Ensure OFF-state coverage ≥30% per class

## Dataset Preparation Steps

### 1. Collect and Annotate Data
Use tools like CVAT, LabelImg, or Roboflow to annotate your images.
Export in COCO JSON format.

### 2. Organize Files
Place images in `train/`, `val/`, `test/` directories.
Place COCO JSON files in `annotations/` directory.

### 3. Validate Dataset
Run sanity checks before training:

```bash
# Check dataset validity
python tools/dataset_sanity_check.py \
    --ann-file data/vehicle_lights/annotations/train.json \
    --img-dir data/vehicle_lights/train/

# Validate category mapping
python tools/category_mapping_validator.py \
    --config configs/vehicle_lights/rtmdet_m_vehicle_lights.py \
    --ann-file data/vehicle_lights/annotations/train.json

# Visualize random samples
python tools/visualize_samples.py \
    --ann-file data/vehicle_lights/annotations/train.json \
    --img-dir data/vehicle_lights/train/ \
    --output-dir visualizations/ \
    --num-samples 200
```

### 4. Start Training
Once validation passes:

```bash
python tools/train.py configs/vehicle_lights/rtmdet_m_vehicle_lights.py
```

## Minimum Dataset Size

For good results, aim for:
- **500-1000 images per class** (minimum)
- **Balanced distribution** across all 12 classes
- **Diverse scenarios** (angles, lighting, ON/OFF states)

## References

- [RTMDet-m Training Guide](../../docs/TRAINING_GUIDE.md)
- [COCO Dataset Format](https://cocodataset.org/#format-data)
