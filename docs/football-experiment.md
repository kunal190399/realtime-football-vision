# Football detector and fixed-camera events

## Run the demo

Install the project requirements, FFmpeg with libx264, Tesseract with English data and DejaVu Sans Bold. From the repository root run:

```sh
python -m pytest -q
python -m examples.fixture outputs/demo
```

Open `outputs/demo/fixed-camera-demo.mp4` to see the ball path. The command also generates reference and predicted event CSVs, runs OCR on the encoded video, and exports a highlight ZIP. The schematic is generated locally and contains no broadcast footage.

| Check | Reference | Observed |
| --- | ---: | ---: |
| Start of directed ball motion | 1.04 s | Shot candidate at 1.08 s |
| Ball centre reaches goal line | 2.88 s | Crossing candidate at 2.88 s |
| RED score increases to 2–1 | 3.40 s | Accepted score update at 3.60 s |

The geometry regression feeds the scripted centres to the monitor. It checks timing independently of visual detection. OCR reads all 25 samples from the encoded pixels. For this scoreboard choose custom light digits, RED and BLU, boxes `263,26,26,33` and `332,26,26,33`, and clock box `43,26,100,36`.

## Analyse a fixed-camera recording

In **Players and ball**, enable **Check shot and crossing candidates (fixed camera)**. Enter the two points where the posts meet the ground and any point clearly on the field side. Coordinates refer to a 1280 × 720 view. The annotated video shows the chosen line. Keep the same camera and framing throughout. Split the input before pans, zooms or angle changes and recalibrate each view.

The monitor uses adjacent directly observed centres. It rejects gaps over 0.15 seconds (with 1 ms rounding tolerance), identity changes between known tracks, abrupt position jumps and motion extrapolations. A shot candidate needs two successive approaches at 80 reference pixels/s or faster, projected to reach the segment between the posts within two seconds. A crossing requires observations on the field side then on or beyond that segment. The timestamp interpolates the crossing between frames. A pass or airborne ball can also satisfy these image-space conditions.

For independent evaluation, label one stationary view at a time. Record `event,seconds` using source video time, including misses and clips without events. Use `shot_candidate` for the onset of a shot and `goal_line_crossing_candidate` for the visible centre crossing. Compare these labels with the dashboard CSV:

```sh
python -m src.evaluation.visual_events labels.csv goal-candidates.csv --tolerance 0.5
```

Each prediction can match one reference of the same kind. The report gives per-kind precision, recall, F1 and signed timing errors. The scripted example has one match per kind. Independent real-match event accuracy remains unmeasured.

## Reproduce detector training

Dataset attribution: football-project-pifbc, *football-players-detection*, version 1, CC BY 4.0. The mirror retains the upstream attribution. The experiment reassigns the 372 images by filename prefix across the original folders. Sorted prefix groups cycle through eight slots with the first reserved for testing and second for validation. There are 298 training, 35 validation and 39 held-out test images. This guards against sharing a prefix between splits. Match identities across different prefixes are unknown.

[Original dataset](https://universe.roboflow.com/football-project-pifbc/football-players-detection-3zvbc-yyhdl/dataset/1) · [Dataset mirror](https://huggingface.co/datasets/martinjolif/football-player-detection) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

```sh
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/martinjolif/football-player-detection data/football
git -C data/football lfs install --local
git -C data/football checkout e8b8cea002692efd74c945fcdad63e729adc5671
git -C data/football lfs pull
python -m src.training.football data/football/data outputs/football --epochs 5 --weights yolo11n.pt
```

The run starts from Ultralytics YOLO11n and freezes the first ten layers. Seed 42, batch size 4, 640-pixel images, AdamW, an initial learning rate of 0.001 and disabled mosaic are set by the script. The best validation checkpoint is evaluated once on the held-out set. To use your checkpoint, enter its local path in **Detection / tracking weights** with the experimental preset unchecked. Model weights use the applicable [Ultralytics model licence](https://www.ultralytics.com/license).

The reference run used Python 3.12, Ultralytics 8.4.215, CPU PyTorch 2.8.0 and torchvision 0.23.0. The saved run includes the training arguments and a held-out precision–recall curve.

## Held-out results

| Class | Labelled objects | Recall | AP50 | AP50–95 |
| --- | ---: | ---: | ---: | ---: |
| Player | 777 | 0.964 | 0.933 | 0.590 |
| Goalkeeper | 18 | 0.111 | 0.127 | 0.050 |
| Referee | 85 | 0.318 | 0.253 | 0.125 |
| Ball | 35 | 0.000 | 0.000 | 0.000 |

Mean AP across classes was 0.328 at IoU 0.50 and 0.191 across IoU 0.50–0.95. The checkpoint is useful for testing player detection. Its ball recall was zero, so it cannot support reliable possession or visual goal analysis on this test set.

On the first 150 frames of France–Australia, the trained detector and pose model produced a decodable annotated video at 10.36 frames/s end to end on CPU (50 fps input). All 150 ball positions were unknown. Full-clip scoreboard scans accepted five changes on France–Australia and sent all six Argentina–France changes for review. Their processing speeds were 11.79× and 1.41× playback on this host, respectively, measured during training. See the [reference timestamp evaluation](evaluation.md#timestamp-benchmark) for labelled comparisons.

As a comparison, the existing reco YOLO26s dashboard preset returned one correct ball box, zero extra ball boxes and 34 missed balls on the same 39 images at confidence 0.25 and IoU 0.50. One correct prediction gives little evidence about precision. The preset's earlier broadcast observations and this small held-out set both support treating ball-based events as review candidates. Reproduce this operating-point check with:

```sh
python -m src.evaluation.ball_boxes outputs/football/test.txt
```
