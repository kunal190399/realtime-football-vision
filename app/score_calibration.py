"""Streamlit controls for fixed scoreboard crops in a 1280 by 720 reference frame."""
import subprocess
import tempfile
from pathlib import Path

import cv2
import streamlit as st

from src.events.scoreboard import ScoreLayout, ScoreReader


def select_layout(uploaded):
    st.caption("Coordinates use the 1280 × 720 preview. Frame each digit tightly, with a small margin.")
    teams, boxes = [], []
    columns = st.columns(2)
    for column, side, team, box in zip(columns, ("Left", "Right"), ("FRA", "AUS"),
                                       ("289,43,18,20", "358,43,18,20")):
        teams.append(column.text_input(f"{side} team code", team).strip().upper())
        boxes.append(column.text_input(f"{side} digit box (x, y, width, height)", box))
    light = st.radio("Digit appearance", ["Light on dark", "Dark on light"], horizontal=True)
    try:
        coords = [tuple(map(int, value.split(","))) for value in boxes]
        if any(len(box) != 4 for box in coords):
            raise ValueError("Each box needs four comma-separated integers.")
        layout = ScoreLayout(tuple(teams), tuple(coords), light == "Light on dark")
    except ValueError as exc:
        st.warning(str(exc))
        return None
    if uploaded:
        seconds = st.number_input("Preview time (seconds)", min_value=0.0, value=0.0, step=1.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ("input" + Path(uploaded.name).suffix)
            path.write_bytes(uploaded.getbuffer())
            cap = cv2.VideoCapture(str(path))
            try:
                cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
                ok, frame = cap.read()
            finally:
                cap.release()
        if ok:
            preview = cv2.resize(frame, (1280, 720))
            for x, y, w, h in layout.boxes:
                cv2.rectangle(preview, (x-1, y-1), (x+w, y+h), (0, 255, 255), 1)
            st.image(preview, channels="BGR", caption="Score regions on the selected frame")
            if st.button("Check preview score"):
                try:
                    score = ScoreReader(layout=layout).read(frame)
                except (OSError, subprocess.SubprocessError) as exc:
                    st.warning(f"Preview OCR failed. Check the Tesseract installation. {exc}")
                    return layout
                st.write(f"{teams[0]} {score[0]}–{score[1]} {teams[1]}" if score else "Score unreadable on this frame.")
        else:
            st.warning("No frame at that time. Choose an earlier preview time.")
    return layout
