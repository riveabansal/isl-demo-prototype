# ISL Live: trained static alphabet baseline

This prototype performs **real inference**, with no presenter-selected or simulated labels. It uses MediaPipe to detect hand landmarks and a PyTorch neural network trained from scratch on selected RealSign Indian Sign Language alphabet images.

Supported letters: **A, B, C, D, L, V, Y**. Start with **L, V, and C**, which use one hand. A, B, D, and Y use two hands in this dataset; keep both visible. The reference images in the app show the dataset's sign forms.

## How to use

On Mac

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python app.py --port 8001
```

On Windows, use `.venv\Scripts\python` instead of `.venv/bin/python`. `run.command` is a macOS convenience script that sets up `.venv` on first use and starts the server. The trained weights and hand-landmarker asset are included, so no dataset download is needed to run the recognizer. Python packages need an internet connection on first installation.

## Camera demonstration

1. Click **Start camera**. Allow camera access for localhost if your browser asks.
2. Use good lighting and a simple background. Keep hands large enough in the frame; include the whole hand and wrist. Avoid overlapping fingers more than the reference sign requires.
3. Hold **V**, **L**, or **C** steadily. Try the two-hand letters after these.
4. The overlay shows actual detected landmarks. Prediction, top three model scores, hand count and inference latency update automatically.
5. After four consistent accepted predictions, **Add letter** becomes available. Click it to build text. Space, backspace and clear are manual.
6. Click **Stop camera** to release the camera.

If camera access is unavailable in the in-app browser, open the same localhost address in Chrome, Edge, Firefox or Safari and check camera permissions. No camera recording is saved. JPEG frames are posted to the local Python server and discarded after inference; no cloud service receives them.

Click a held-out reference image to test real recognition without a camera. You can also upload a local JPEG or PNG. These controls run the same detector and classifier endpoint as live frames. The image mode bypasses the four-frame stabilization requirement, since it is a single image.

## What was trained

- Source: [RealSign Indian Sign Language Dataset](https://github.com/RealSign62/RealSign-Indian-Sign-Language-Dataset), with its original Training / Validation / Testing folders.
- Scope: seven static alphabet classes. Up to 300 training, 100 validation and 200 testing images per class were attempted, chosen deterministically with seed 42.
- Exact duplicate image hashes are excluded across selected splits. Near-duplicate frames and signer overlap may remain.
- Samples with missing landmarks or an incorrect detected hand count for the class are excluded from classifier training and feature-level evaluation.
- Features: one or two hands, screen-x order; wrist-centered, palm-size-scaled coordinates; aspect-ratio correction; 126 coordinates plus two hand-count flags = 128 values.
- Classifier: `128 → 128 → 64 → 7` MLP, ReLU and dropout. AdamW, learning rate 0.001, weighted cross-entropy, validation-loss checkpoint selection and early stopping.
- Augmentation: reflected training features (with hand slots reordered) and small feature noise. Validation/test features are not used to fit normalization or model weights.
- Sign classifier is trained from scratch. MediaPipe's hand detection/landmark model is pretrained; we did not train that detector.

Measured run: 1571 usable training images, 518 validation images and 1044 usable test images. Feature-level test accuracy was 100.0%; across all 1330 attempted test images, 78.3% yielded an accepted correct prediction. The model trained for 137 epochs with validation-loss early stopping.

See `models/isl_mlp.json` for measured class metrics, confusion matrix, training curve, feature coverage and rejection coverage. `models/extraction_report.json` records detection counts. `models/data_manifest.json` records source paths and hashes. `DATA_SOURCE.json` records archive and detector provenance and checksums. The source archive checksum was verified.

**Evaluation limits:** feature-level accuracy applies only to usable detected test samples. The report also includes the fraction of attempted test images that generated accepted correct predictions. Dataset scores are not webcam accuracy, not verified new-signer performance, and not evidence of continuous ISL recognition. Similar dataset frames can inflate results. The 75% softmax threshold, distance check and expected-hand-count gate are conservative heuristics, not a calibrated “unknown sign” detector. Unsupported signs can still be misclassified.

## Reproduce extraction and training

Download `Dataset.zip` from the source repository, verify it against `DATA_SOURCE.json`, and extract it to a directory containing `Training`, `Validation`, and `Testing`.

```sh
.venv/bin/python extract_dataset.py /path/to/RealSign
.venv/bin/python train.py
.venv/bin/python -m unittest -v
```

The included feature cache allows retraining without downloading the raw dataset:

```sh
.venv/bin/python train.py --features models/features.npz
```

`models/isl_mlp.pt` contains the checkpoint, vocabulary, normalization statistics and rejection metadata. The server loads it with PyTorch `weights_only=True`. `models/test_split.npz` identifies the feature test images. Reference images are held-out examples, not training inputs.

## Files

- `app.py`: local HTTP server, frame decoding, metadata and recognition endpoints.
- `features.py`: real MediaPipe extraction and shared normalization.
- `model.py`: PyTorch model architecture.
- `train.py`: reproducible training and evaluation.
- `extract_dataset.py`: extraction with original dataset splits and duplicate checks.
- `inference.py`: checkpoint loading, prediction and rejection logic.
- `static/index.html`: camera capture, skeleton overlay, stability gate and text output.
- `models/`: trained weights, extraction cache, provenance manifests and measured reports.
- `test_features.py`: invariance and feature-contract tests.

This is an academic prototype with a limited vocabulary, intended for a demonstration. Dataset authors and license are preserved in `DATASET_README.md` and `DATASET_LICENSE.txt`.
