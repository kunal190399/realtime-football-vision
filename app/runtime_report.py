import json

import streamlit as st


def show_runtime(report):
    with st.expander("Runtime benchmark"):
        st.write(f"{report['source_seconds']:.2f} seconds of footage processed in "
                 f"{report['wall_seconds']:.2f} seconds ({report['playback_speed']:.2f}× playback speed).")
        st.caption("Includes video processing and file finalisation. Uploads, model construction, "
                   "highlight exports and dashboard rendering are outside this timer.")
        st.table([{"Stage": name, "Calls": row["calls"], "Total (s)": round(row["total_seconds"], 3),
                   "Mean (ms)": round(row["mean_ms"], 2), "P95 (ms)": round(row["p95_ms"], 2)}
                  for name, row in report["stages"].items()])
        st.download_button("Download runtime JSON", json.dumps(report, indent=2),
                           file_name="runtime.json", mime="application/json", on_click="ignore")
