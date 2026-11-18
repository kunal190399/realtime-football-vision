# Real-Time Football Event Detection & Pose Estimation

A prototype computer-vision pipeline for analysing football video-game broadcast footage. The first milestone provides video ingestion, player tracking, human pose overlays, annotated-video export and inference profiling.

> **Status:** Experimental player tracking and possession prototype. This is an engineering prototype, not a validated goal/shot detector. Generic COCO weights are not specialised for small footballs, broadcast overlays or game-specific UI elements.

## Project scope

This KTP-oriented prototype explores real-time computer vision for football broadcast analysis. Stage 1 establishes the video-processing foundation; it does not yet claim validated match-event recognition. The initial demonstrator can be exercised with football gameplay footage while the later work will target real broadcast footage and sport-specific datasets.

## Stage 1 dashboard

The dashboard layout is shown below. The public illustration intentionally omits the original broadcast preview; use footage you are licensed to share when reproducing the demonstration.

![Stage 1 dashboard illustration](docs/images/stage1-dashboard.svg)

## Features

- Video upload and processing through a Streamlit interface
- Ultralytics YOLO tracking with ByteTrack persistence
- Human pose estimation overlays
- Annotated MP4 export
- Per-frame inference time and average throughput metrics
- Modular event-detector scaffold for future temporal logic

## Architecture

`Video → OpenCV → YOLO tracking → Pose estimation → Annotated video + metrics → Streamlit`

## Quick start

Python 3.10–3.12 is recommended.

```bash
git clone https://github.com/kunal190399/realtime-football-vision.git
cd realtime-football-vision
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m streamlit run app/dashboard.py
```

The first run downloads the configured model weights. For GPU inference, install a PyTorch build compatible with your CUDA environment and verify that Ultralytics can access the GPU. Leave the device field blank for automatic selection.

## Review ball tracks and possession candidates

Enable **Estimate player possession (experimental)** before processing a clip. Use weights with a `person` or `player` class and a `sports ball`, `ball` or `football` class. The default COCO model includes `sports ball`. Small or occluded footballs can be missed.

Download the frame timeline for zero-based frame indices, source-video timestamps in seconds, ball centre coordinates in pixels, and track IDs. Empty fields mean unknown. When multiple balls are visible, the pipeline follows the previous ball ID while visible, then chooses the highest-confidence ball. A player becomes a possession candidate after three consecutive frames with the same ball and nearest player, measured from the ball centre to the bottom centre of the player box in units of player height (maximum 0.5). Missing observations and identity changes reset confirmation. This image-space heuristic can confuse nearby players, referees and contested balls. Compare candidate intervals with manually labelled footage before interpreting possession.

### Reading the ball timeline

The CSV includes a `ball_source` column. A `tracked` position has a ball track ID. A `detected` position comes from a detector box that ByteTrack did not assign an ID. Untracked seeds require a majority of turf-coloured pixels in a 49 × 49 neighbourhood to reduce matches on advertising boards. Airborne balls, close-ups and pitches with different colours can be missed. Both appear as yellow rings. If both are missing, the pipeline searches nearby pixels for the last observed ball. A distinctive match appears as a blue ring and a `motion` row. The search stops after a failed or ambiguous match, a scene change, or approximately 0.1 seconds (at most five frames). An `unknown` row has empty coordinates. Only tracked observations can confirm possession.

### Football-specific detector

Select **Use football-specific detector (experimental)** and **Estimate player possession (experimental)** to try the reco YOLO26s model. The first run downloads its 40 MB ONNX file and checks its SHA-256 digest. Later runs reuse the verified file. The preset uses a fixed 1920 × 1920 input and tracks the person and ball classes. The referee class is excluded. The dashboard reports ball and possession coverage as frame counts.

The model was trained on amateur football filmed with fixed cameras. Broadcast camera movement, close-ups and cuts can affect both the detector and the tracker. Model author, training conditions and AGPL-3.0 model licence are documented in the [pinned model card](https://huggingface.co/Dura-S/reco-yolo26s-football/blob/d20231484951d4ebf3f9f2a7437aa922a4c1b395/README.md).

## Scoreboard updates

Choose **Scoreboard updates**, upload a clip and click **Scan scoreboard**. This mode scans the full video independently of the player models and exports CSV rows with the team, old and new scores, first stable sample time and confirmation time. It requires Tesseract OCR with English language data on the system PATH (Ubuntu: `sudo apt install tesseract-ocr tesseract-ocr-eng`; macOS with Homebrew: `brew install tesseract`).

To create match highlights, select **Export one highlight per score update** before scanning. The default window starts eight seconds before the update and ends two seconds after it. Adjust either window to suit the broadcast delay. The ZIP contains one H.264 MP4 per event, with source audio when available, and a CSV mapping each clip to its source timestamps. Exports stop at the beginning and end of the input. Edited footage can contain a cut inside a highlight. Highlight export requires FFmpeg with libx264 on PATH.

The preset expects the single-digit FRA–AUS score graphic in the same position and team order as the supplied 16:9 footage. For another fixed graphic, select **Custom layout**. Enter the two team codes and each digit box as `x, y, width, height` in the 1280 × 720 preview, even if your video has another resolution. Pick light-on-dark or dark-on-light digits, move the preview to a time with the scoreboard visible, and use **Check preview score** before scanning. The same settings apply to the entire upload and flow through to the CSV and highlight filenames. Custom mode supports single-digit scores with contrasting backgrounds. Moving graphics and changes of team order require separate scans. The reader samples approximately four times per second and accepts a one-goal increment after three consecutive matching score pairs. It ignores downward changes, rebases on larger forward jumps, and establishes a fresh baseline after a prolonged unreadable interval. A timestamp is the first sample of the confirmed display change. The ball can cross the goal line earlier.

## Current limitations

1. COCO-pretrained YOLO detects people but does not provide a reliable, dedicated football detector for broadcast footage.
2. Broadcast overlays, camera cuts, miniature players and stylised game graphics can differ substantially from natural-image training data.
3. Visual goal-line crossing and shot events are not yet implemented. These require a sports-specific ball/player detector, temporal tracking and usually scoreboard/goal-region calibration.
4. Reported FPS is model inference throughput and excludes some I/O and UI overhead. Results depend on hardware, model size, resolution and video codec.

## Roadmap

- [x] Stage 1 scaffold: video ingestion, tracking, pose overlay and metrics
- [x] Stage 2 experimental person-only ByteTrack configuration and tracking evaluation protocol
- [ ] Add sports-specific player/ball dataset and detector training
- [x] Add experimental temporal ball tracking and player possession candidates
- [ ] Add calibrated shot/goal candidate detection and event evaluation
- [ ] Add mAP, precision-recall, event F1 and latency benchmark reports
- [ ] Add demo clips and reproducible experiment configuration

## Responsible use and data

Use footage and datasets you have permission to process and redistribute. Do not commit copyrighted match footage or model weights unless their licences permit redistribution. Record dataset provenance and licences before training.

## License

MIT. See `LICENSE`.
