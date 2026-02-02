# Training Guide: Customizing RTMDet for Your Data

To improve detection accuracy or add new classes (e.g., specific traffic signs, different vehicle types), you'll need to fine-tune the model on your own data.

## 1. Data Collection
Capture diverse scenarios to make the model robust.
- **Varied Conditions**: Day, night, rain, tunnels.
- **Angles**: Front, rear, side views of vehicles.
- **Edge Cases**: Partially occluded vehicles, distant lights.

**Goal**: Aim for ~500-1000 labeled images per class for good results.

## 2. Annotation
You need to label your images. RTMDet (via MMDetection) typically uses **COCO format**.

### Recommended Tools
1. **CVAT (Computer Vision Annotation Tool)** - Best for video
   - Fast interpolation (label one frame, skip 10, it fills in between).
   - Export format: `COCO 1.0`
2. **LabelImg** - Good for single images
   - Simple, lightweight.
   - Saves as Pascal VOC (XML) or YOLO (TXT). You'll need to convert to COCO.
3. **Roboflow** - Online platform
   - Easy drag-and-drop labeling.
   - **One-click export to MMDetection (COCO) format.** (Highly recommended for beginners).

## 3. Training Workflow (MMDetection)

### Step A: Prepare Dataset
Organize folders like this:
```
data/
  coco/
    annotations/
      instances_train2017.json
      instances_val2017.json
    train2017/   # Images
    val2017/     # Images
```

### Step B: Create Config File
Create `configs/rtmdet_custom_training.py` that inherits from the base config but modifies classes.

```python
# Inherit from base config
_base_ = 'models/rtmdet_tiny_8xb32-300e_coco.py'

# 1. Dataset settings
dataset_type = 'CocoDataset'
classes = ('my_class_1', 'my_class_2', 'brake_light')
data_root = 'data/coco/'

# 2. Model head (adjust num_classes)
model = dict(
    bbox_head=dict(
        num_classes=3  # Change to your number of classes
    )
)

# 3. Dataloader settings
train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=dict(classes=classes),
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/')
    )
)

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=dict(classes=classes),
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/')
    )
)

test_dataloader = val_dataloader
```

### Step C: Run Training
```powershell
# Train using mim (automatically handles dependencies)
mim train mmdet configs/rtmdet_custom_training.py
```

This will produce a new `.pth` weight file in `work_dirs/`.

## 4. Improving Performance
- **Data Augmentation**: Flip, rotate, change brightness (MMDetection does this automatically in the config's pipeline).
- **Hard Negative Mining**: If the model frequently mistakes a red mailbox for a brake light, add many images of red mailboxes to the dataset *without* labels (background images) or label them effectively.
- **Model Size**: If RTMDet-Tiny is too inaccurate, switch to **RTMDet-s** or **RTMDet-m** (larger, slower, but smarter).

## 5. Deployment
Once trained:
1. Copy the new `.pth` file to `models/`.
2. Update `configs/pipeline_config.yaml`:
   ```yaml
   detection:
     model:
       config_path: "configs/rtmdet_custom_training.py"
       weights_path: "models/best_epoch_300.pth"
   ```
