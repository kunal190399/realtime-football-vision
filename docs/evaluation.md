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

### Recovering short gaps during the attack

Run the same clip and settings. Compare the new `ball_source` column against the previous timeline using the zero-based frame index. In frames 144–152 (2.88–3.04 s), the direct detector supplied two untracked observations, ByteTrack supplied two tracked positions, and local image matching continued the ball for five frames. At frame 150 the estimate was (675, 221) in the original 1280 × 720 image. Review of the source frame places that point on the moving ball.

A separate one-second excerpt starting 56.00 seconds into the same source video exercised a close-up with a fresh tracker. Of 50 frames, 11 had tracked positions, one had a motion estimate and 38 were unknown. The ball was visible at all 12 reported positions. Without the pitch-colour check, this close-up yielded six additional correct positions. The check trades those positions for fewer advertising-board matches in the opening clip. Neither excerpt has exhaustive ground truth for visibility or possession.

Across the opening clip, all 30 previous tracked positions were retained. Fourteen additional frames had untracked detections, 15 had motion estimates, and 541 remained unknown. All 29 new positions were visually reviewed on the original footage. No player possession was confirmed. The pitch-colour check was selected using this clip, so it needs evaluation on other matches.

## Scoreboard display changes

The fixed FRA–AUS preset was checked through the Streamlit upload workflow using all 3,334 frames of the supplied 50 fps highlights. OCR returned scores on 278 of 278 samples. The five confirmed changes were 0–1 at 9.84 s, 1–1 at 24.72 s, 2–1 at 36.96 s, 3–1 at 48.00 s and 4–1 at 63.36 s. Before-and-after score crops were checked for each transition. Times identify the first sample of a stable reading on the video timeline. Camera edits and the broadcast update delay preclude using these values as goal-line crossing times.

### Custom layout check

Both the built-in and manually configured regions reproduced all five timestamps on the France–Australia footage. The custom reader returned a score for all 278 samples. A separate synthetic 640 × 360 clip moved the digits to the bottom right, inverted their colours and used RED and BLU labels. The dashboard preview read 0–0. Scanning reported BLU at 2.00 s and RED at 4.00 s and produced both CSV and highlight downloads. An independent broadcast with a different scoreboard design still needs evaluation.

### Match-clock continuity on edited footage

A second 1280 × 720, 50 fps highlights edit used the same broadcast graphic with Argentina on the left and France on the right. The reader compared three-sample score runs across the full 62.06 seconds. Six increments were visible from 0–0 to 3–3. At 20.40 s the new score was 2–1. The previous 2–0 sample at 18.48 s displayed 79:25 and the new sample displayed 80:54. A two-second scoreboard-gap rule alone accepted this change. The clock check sends it to review. The other five changes span scoreboard gaps exceeding two seconds. None qualify for automatic highlights. In the France–Australia clip all five previously recorded updates pass the check. Neither clip validates the timing of the ball crossing the goal line.
