# HandGrid Gesture Control

This project trains a hand gesture classifier and uses it in a real-time webcam demo that controls the mouse with hand gestures.

The workflow is:

1. Train a static gesture classifier on a HaGRID-style classification dataset
2. Export the trained classifier to ONNX
3. Run the webcam demo
4. Use recognized gestures to move the mouse, click, drag, scroll, zoom, and pause/resume control

The runtime pipeline is inspired by the `dynamic_gestures` repository structure:

- ONNX hand detector
- ONNX static gesture classifier
- tracking/controller logic
- desktop action mapping on top of stable gesture predictions

## Requirements

Install the main dependencies:

```powershell
pip install -r requirements-handgrid.txt
```

If `filterpy`, `onnx`, `onnxruntime`, `opencv-python`, or `pynput` are missing, install them in the same Python environment you use to run the scripts.

## Dataset Format

The training script supports more than one local dataset format.

### Preferred format used in this project

`hagrid-classification-512p/`

This dataset is a flat classification dataset:

```text
hagrid-classification-512p/
  call/
  dislike/
  fist/
  ...
  two_up_inverted/
```

Each folder name is treated as the class label.

### Important note

Datasets are ignored by `.gitignore` and should stay local.

## Training

### Train the model

This command trains the classifier, saves the PyTorch checkpoint, and exports ONNX automatically unless ONNX export is explicitly disabled.

```powershell
python train_static_classifier.py --dataset hagrid-classification-512p --architecture mobilenet_v3_small --run-name robust_run
```

Current training defaults:

- `30` epochs
- MobileNetV3 Small classifier
- image augmentations for blur, tilt, perspective, crop variation, and pixelation
- automatic timestamped save names

### Override epochs manually

```powershell
python train_static_classifier.py --dataset hagrid-classification-512p --architecture mobilenet_v3_small --run-name robust_run --epochs 30
```

### Output files

Each run creates unique timestamped files in `checkpoints/`, for example:

```text
checkpoints/
  robust_run_20260419_220600.pt
  robust_run_20260419_220600.onnx
  robust_run_20260419_220600.json
```

These files are not overwritten because the timestamp changes for each run.

## Export ONNX Later

If you already have a `.pt` checkpoint and want to export ONNX later:

```powershell
python export_classifier_onnx.py --checkpoint "checkpoints\robust_run_20260419_220600.pt"
```

## Run the Demo

### Run the latest ONNX demo

```powershell
python run_handgrid_onnx_demo.py --classifier "checkpoints\robust_run_20260419_220600.onnx" --debug
```

Without debug overlays:

```powershell
python run_handgrid_onnx_demo.py --classifier "checkpoints\robust_run_20260419_220600.onnx"
```

If your camera is not the default one:

```powershell
python run_handgrid_onnx_demo.py --classifier "checkpoints\robust_run_20260419_220600.onnx" --debug --camera 1
```

### Debug vs normal mode

`--debug` shows extra visual information:

- FPS
- cursor coordinates
- active camera area used for mouse mapping
- bounding boxes
- gesture labels and confidence
- hand center point

Normal mode keeps the window cleaner.

## Gesture Controls

The current desktop-control demo uses these gesture mappings:

- `one` -> move mouse
- `ok` -> left click
- `three` -> right click
- `fist` -> click and hold drag
- `two_up` -> scroll up/down based on hand motion
- `two_up_inverted` -> scroll up/down based on hand motion
- `thumb_index` -> zoom in
- `thumb_index2` -> zoom out
- `call` -> pause/resume all mouse actions

### Pause behavior

`call` is a toggle:

- first `call` pauses the gesture controls
- second `call` resumes them

This helps prevent accidental mouse movement or clicks.

## Labels the Classifier Knows

The current model trained on `hagrid-classification-512p` knows these static gesture labels:

1. `call`
2. `dislike`
3. `fist`
4. `four`
5. `like`
6. `mute`
7. `ok`
8. `one`
9. `palm`
10. `peace`
11. `peace_inverted`
12. `point`
13. `rock`
14. `stop`
15. `stop_inverted`
16. `three`
17. `three2`
18. `thumb_index`
19. `thumb_index2`
20. `two_up`
21. `two_up_inverted`

Not every label is currently mapped to a desktop action, but the classifier can still recognize them.

## Main Files

### Training and export

- `train_static_classifier.py`
  Trains the gesture classifier, applies augmentation, saves `.pt`, `.onnx`, and `.json`
- `export_classifier_onnx.py`
  Exports a saved `.pt` checkpoint to ONNX later if needed
- `requirements-handgrid.txt`
  Python packages needed for training and the ONNX demo

### Training support package

- `gesture_pipeline/data.py`
  Loads local datasets in supported formats
- `gesture_pipeline/models.py`
  Builds the classifier architecture
- `gesture_pipeline/checkpoints.py`
  Saves and loads checkpoints
- `gesture_pipeline/constants.py`
  Shared constants such as normalization values
- `gesture_pipeline/dynamic_logic.py`
  Earlier dynamic gesture logic module kept as a helper/reference

### ONNX runtime and demo

- `run_handgrid_onnx_demo.py`
  Main webcam demo for gesture-driven mouse control
- `handgrid_dynamic/main_controller.py`
  Runs detection, classification, and hand tracking
- `handgrid_dynamic/onnx_models.py`
  Loads and runs the ONNX detector and ONNX classifier
- `handgrid_dynamic/utils/action_controller.py`
  Buffers gesture history and handles swipe-style action detection
- `handgrid_dynamic/utils/hand.py`
  Stores hand state such as box, center, size, and gesture label
- `handgrid_dynamic/utils/enums.py`
  Action enums used by the runtime controller

### Reference code

- `dynamic_gestures/`
  Reference repository used for inspiration and reused tracking components

### Utility script

- `resize_dataset.py`
  Local helper script for dataset processing experiments

## Git / Repo Notes

The `.gitignore` is set up so GitHub does not accidentally include:

- local datasets
- training checkpoints
- ONNX exports
- archives like `.zip`
- cache folders

Recommended GitHub strategy:

- commit the code
- do not commit datasets
- do not commit large trained models into the repo
- if you want to share a final model, upload it separately and link it

## Typical Workflow

### 1. Install dependencies

```powershell
pip install -r requirements-handgrid.txt
```

### 2. Train

```powershell
python train_static_classifier.py --dataset hagrid-classification-512p --architecture mobilenet_v3_small --run-name robust_run
```

### 3. If needed, export ONNX later

```powershell
python export_classifier_onnx.py --checkpoint "checkpoints\robust_run_20260419_220600.pt"
```

### 4. Run the webcam demo

```powershell
python run_handgrid_onnx_demo.py --classifier "checkpoints\robust_run_20260419_220600.onnx" --debug
```

## Notes

- The webcam demo uses an ONNX hand detector plus your trained ONNX classifier
- The current active mapping region in the camera view is intentionally large so mouse control feels less constrained
- If OpenCV window display fails, make sure you installed `opencv-python` and not only a headless build
