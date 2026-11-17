# Stage 2 – Player tracking evaluation

## Purpose

Stage 2 introduces an experimental person-only ByteTrack configuration. It is intended to reduce irrelevant tracks and provide a more suitable starting point for football footage. It is not yet evidence of reliable player identity tracking.

## Changes

- Restrict tracking to COCO class 0 (person).
- Use `configs/bytetrack_football.yaml` with a longer track buffer.
- Keep the tracker thresholds configurable and explicitly treat them as starting values.

## Reproducible test

1. Use the same 10–30 second clip for the baseline and Stage 2 runs.
2. Keep the detector weights, confidence threshold, input resolution and device unchanged.
3. Record the number of unique track IDs, average detections per frame, inference throughput and visible ID switches/fragmentation.
4. Inspect several fixed timestamps (early, middle and late) to check whether the same visible player retains an ID.
5. Repeat on at least three clips with different camera movement and player density.

## Interpretation

A lower number of IDs alone does not prove better tracking: missed detections can also reduce the count. Review continuity visually and, where annotated ground truth is available, calculate IDF1 or HOTA. Do not report tracking accuracy without ground-truth annotations.

## Current limitation

The detector still uses generic COCO weights. It does not provide a reliable football detector or robustly distinguish players from all broadcast/gameplay elements. The ByteTrack values are experimental and should be tuned against representative data.
