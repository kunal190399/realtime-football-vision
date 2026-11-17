# Evaluation protocol (planned)

Do not publish performance claims until they are measured on a documented machine and test set.

## Detection
- Report mAP@0.5 and mAP@0.5:0.95 for each labelled class.
- Include precision-recall curves and per-class support.

## Tracking
- Report ID switches, IDF1 or HOTA where ground-truth tracks are available.
- Inspect identity stability across camera cuts and occlusion.

## Events
- Define event timestamp tolerance before scoring.
- Report precision, recall and F1 per event class.
- Report event detection delay from annotated event onset.

## Runtime
- Record GPU/CPU, driver, CUDA, model weights, input resolution and batch size.
- Report mean and P95 per-frame latency, throughput, and end-to-end processing speed separately.
