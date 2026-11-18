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

## Opening-goal clip check

The test used the first 600 consecutive frames of a clean 1280 × 720, 50 fps France–Australia broadcast clip. Run the dashboard with possession enabled, a confidence threshold of 0.25, the experimental ByteTrack configuration, and the default pose weights. Compare the default detector with the football-specific preset on identical source frames. The source footage is supplied separately.

Count a ball candidate when the exported timeline has a ball centre, and a possession candidate when it has a player track ID. The denominators include the celebration and camera cuts. For a visual spot check, use the unmarked source frame and crop around each predicted position. Check the ball centre as well as player socks, pitch markings and objects off the pitch that can trigger detections.

| Detector | Frames with a ball candidate | Frames with a possession candidate |
| --- | ---: | ---: |
| COCO YOLO11n | 0 / 600 | 0 / 600 |
| reco YOLO26s ONNX | 30 / 600 | 0 / 600 |

Visual review of all 30 exported positions showed the ball at the reported centre. Two frames were at 2.92–2.94 s, seven were within 5.24–5.38 s, and 21 were within 7.56–8.00 s. Coverage is 5% of all frames. Ball visibility in the remaining frames has not been exhaustively labelled, so recall is unknown. This single-clip check provides no validation of player possession.
