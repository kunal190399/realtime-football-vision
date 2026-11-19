# Real-Time Football Event Detection & Pose Estimation

Analyse football footage in Streamlit. Track players, inspect ball and possession estimates, review scoreboard changes and export highlights. Fixed-camera clips can also be checked for candidate shots and projected goal-line crossings.

The football detector, timestamp benchmarks and six-second reproducible demo are documented in the [prototype evaluation](docs/football-experiment.md). The dashboard reports end-to-end processing speed for each input.

## Stage 1 dashboard

The dashboard layout is shown below. The public illustration intentionally omits the original broadcast preview; use footage you are licensed to share when reproducing the demonstration.

![Stage 1 dashboard illustration](docs/images/stage1-dashboard.svg)

## Features

- Video upload and processing through a Streamlit interface
- Ultralytics YOLO tracking with ByteTrack persistence
- Human pose estimation overlays
- Annotated MP4 export
- Per-frame inference time and average throughput metrics
- Ball tracking, possession estimates and calibrated shot and crossing candidates
- Scoreboard OCR, clock continuity checks and highlight export

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

### Recover small balls

With possession or fixed-camera analysis enabled, **Recover small balls with the appearance model** adds a CPU fallback when the primary detector misses the ball. Its observations are labelled `appearance` in the timeline. On 39 held-out images it found 7 of 35 labelled balls with one extra box. See the [model evaluation and reproduction steps](docs/football-experiment.md#small-ball-appearance-model).

### Football-specific detector

Select **Use football-specific detector (experimental)** and **Estimate player possession (experimental)** to try the reco YOLO26s model. The first run downloads its 40 MB ONNX file and checks its SHA-256 digest. Later runs reuse the verified file. The preset uses a fixed 1920 × 1920 input and tracks the person and ball classes. The referee class is excluded. The dashboard reports ball and possession coverage as frame counts.

The model was trained on amateur football filmed with fixed cameras. Broadcast camera movement, close-ups and cuts can affect both the detector and the tracker. Model author, training conditions and AGPL-3.0 model licence are documented in the [pinned model card](https://huggingface.co/Dura-S/reco-yolo26s-football/blob/d20231484951d4ebf3f9f2a7437aa922a4c1b395/README.md).

## Scoreboard updates

Choose **Scoreboard updates**, upload a clip and click **Scan scoreboard**. This mode scans the full video independently of the player models and exports CSV rows with the team, old and new scores, first stable sample time and confirmation time. It requires Tesseract OCR with English language data on the system PATH (Ubuntu: `sudo apt install tesseract-ocr tesseract-ocr-eng`; macOS with Homebrew: `brew install tesseract`).

**Check match-clock continuity before exporting highlights** is enabled by default. For custom layouts, enter the clock box in the same 1280 × 720 reference coordinates as the score boxes (the default is `64,40,98,25`). The clock must show minutes and seconds with a colon and use the selected digit polarity. For each stable one-goal change, the scanner reads the clock on the last old-score sample and the first new-score sample. A backward clock, a difference greater than two seconds between match-clock and video elapsed time, an unreadable clock, or a score gap longer than two seconds sends the change to a separate review table and CSV. Automatic highlights use only accepted events. This checks continuity across the score change; other cuts elsewhere in a highlight are possible. Turn the clock check off to use the earlier score-only scanner.

After a scan, use **Change to replay** to choose a flagged score update and **Preview flagged clip** to watch the source interval. Set the lead-in and follow-up for the displayed clip in seconds. The dashboard shows its source interval and keeps each change’s trim while you review other changes in the same scan. Preview and export use the same interval. Editing a selected trim clears the previous ZIP. Select any clips to keep, then choose **Build selected highlights**. This separate ZIP includes only the selected changes. Its CSV records the manual selection, original warning, clock readings and previous sample time. Exporting a clip does not establish when the goal occurred. Changing the upload or scan settings clears the previous review. The automatic export continues to include only accepted changes.

To create match highlights, select **Export one highlight per score update** before scanning. The default window starts eight seconds before the update and ends two seconds after it. Adjust either window to suit the broadcast delay. The ZIP contains one H.264 MP4 per event, with source audio when available, and a CSV mapping each clip to its source timestamps. Exports stop at the beginning and end of the input. Edited footage can contain a cut inside a highlight. Highlight export requires FFmpeg with libx264 on PATH.

The preset expects the single-digit FRA–AUS score graphic in the same position and team order as the supplied 16:9 footage. For another fixed graphic, select **Custom layout**. Enter the two team codes and each digit box as `x, y, width, height` in the 1280 × 720 preview, even if your video has another resolution. Pick light-on-dark or dark-on-light digits, move the preview to a time with the scoreboard visible, and use **Check preview score** before scanning. The same settings apply to the entire upload and flow through to the CSV and highlight filenames. Custom mode supports single-digit scores with contrasting backgrounds. Moving graphics and changes of team order require separate scans. The reader samples approximately four times per second and accepts a one-goal increment after three consecutive matching score pairs. It ignores downward changes, rebases on larger forward jumps, and establishes a fresh baseline after a prolonged unreadable interval. A timestamp is the first sample of the confirmed display change. The ball can cross the goal line earlier.

## Measure processing speed

After running either analysis, expand **Runtime benchmark** and download the JSON. Playback speed is the duration of the processed frames divided by elapsed processing time. A value of 1× means processing took as long as playing the source. A frame limit uses only the processed portion of the video.

The timer begins when the pipeline or scanner opens the input and includes resource finalisation, model warm-up in the processing loop, and video writing where applicable. Upload, model construction or download, highlight export and browser rendering are outside this measurement. Stage totals cover timed operations, with remaining wall time spent in setup, finalisation and other work. Mean and P95 are per stage call. In scoreboard mode score OCR runs on sampled frames and clock OCR runs around changes. In vision mode the pose stage includes drawing the model overlays.

## Evaluate scoreboard timestamps

From the repository root, compare a dashboard CSV with hand-labelled updates for one source video:

```bash
python -m src.evaluation.score_updates examples/scoreboard-labels/france-australia.csv score-updates.csv --tolerance 0.25
```

Supply the clock-review CSV separately to measure reviewed changes, or pass both exports to evaluate all recognised changes together. With no prediction file the evaluator scores zero predictions. It prints JSON with precision, recall, F1, timing errors and the unmatched rows. The [evaluation protocol and results](docs/evaluation.md#timestamp-benchmark) describe the reference labels and units.

## Current limitations

1. COCO-pretrained YOLO detects people but does not provide a reliable, dedicated football detector for broadcast footage.
2. Broadcast overlays, camera cuts, miniature players and stylised game graphics can differ substantially from natural-image training data.
3. Visual event candidates require a stationary camera, a user-calibrated goal and successive direct ball observations. A centre crossing in the image cannot establish whether the whole ball passed between the posts below the crossbar.
4. The inference FPS display excludes some I/O and UI overhead. The runtime report separately measures end-to-end processing. Results depend on hardware, model size, resolution and video codec.

## Roadmap

- [x] Stage 1 scaffold: video ingestion, tracking, pose overlay and metrics
- [x] Stage 2 experimental person-only ByteTrack configuration and tracking evaluation protocol
- [x] Add sports-specific player/ball dataset and detector training
- [x] Add experimental temporal ball tracking and player possession candidates
- [x] Add calibrated shot/goal candidate detection and event evaluation
- [x] Add mAP, precision-recall, event F1 and latency benchmark reports
- [x] Add demo clips and reproducible experiment configuration

## Responsible use and data

Use footage and datasets you have permission to process and redistribute. Do not commit copyrighted match footage or model weights unless their licences permit redistribution. Record dataset provenance and licences before training.

## License

MIT. See `LICENSE`.
