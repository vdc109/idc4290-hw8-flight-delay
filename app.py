"""Streamlit data product for the IDC HW8 assignment."""
from __future__ import annotations

from datetime import date, time
from pathlib import Path

import pandas as pd
import streamlit as st

from predictor import Flight, load_artifact, predict_delay

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
BASELINE = 0.24319085529037324
METRICS = {"Flights": "592,255", "ROC AUC": "0.630", "Average precision": "0.344"}

st.set_page_config(page_title="Flight Delay Risk Explorer", page_icon="✈️", layout="wide")


@st.cache_data
def read_table(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / f"delay_by_{name}.csv")


def draw_ranking(name: str, label: str, minimum: int = 0) -> None:
    table = read_table(name)
    field = name.upper()
    if minimum:
        table = table[table["flights"] >= minimum]
    lookup = st.text_input(f"Filter {label.lower()}", key=f"filter_{name}").strip().upper()
    if lookup:
        table = table[table[field].astype(str).str.upper().str.contains(lookup, regex=False)]
    if table.empty:
        st.info("No groups match those filters.")
        return
    sort = st.radio("Sort by", ["Late arrival rate", "Flight count"], horizontal=True, key=f"sort_{name}")
    column = "late_rate" if sort == "Late arrival rate" else "flights"
    show = table.sort_values(column, ascending=False).head(20).copy()
    show["late_rate"] = show["late_rate"] * 100
    chart = show.set_index(field)[["late_rate"]]
    st.caption("Bars show flights arriving 15+ minutes late (%)")
    st.bar_chart(chart)
    show = show.rename(columns={field: label, "flights": "Flights", "late": "Late flights", "late_rate": "Late rate (%)"})
    st.dataframe(show, hide_index=True, use_container_width=True, column_config={"Late rate (%)": st.column_config.NumberColumn(format="%.1f%%")})
    st.download_button("Download filtered data", table.to_csv(index=False), file_name=f"{name}_summary.csv", mime="text/csv", key=f"download_{name}")


st.title("Flight Delay Risk Explorer")
st.caption("Q1 2026 scheduled flights • historical estimates and interactive summaries")
st.info("Predicts the chance of arriving at least 15 minutes late, conditional on a flight being completed without diversion. The model does not predict cancellations.")

predict_tab, explore_tab, method_tab = st.tabs(["Predict a flight", "Explore delays", "About the data"])

with predict_tab:
    artifact = load_artifact()
    vocab = artifact["vocabulary"]
    carriers = sorted(vocab["OP_UNIQUE_CARRIER"])
    airports = sorted(set(vocab["ORIGIN"]) | set(vocab["DEST"]))
    st.subheader("Enter scheduled flight details")
    st.caption("Example values are filled in. Scheduled times are local clock times at their respective airports.")
    with st.form("flight_form"):
        c1, c2, c3 = st.columns(3)
        flight_date = c1.date_input("Flight date", value=date(2026, 3, 1), min_value=date(2026, 1, 1), max_value=date(2026, 3, 31))
        carrier = c2.selectbox("Airline code", carriers, index=carriers.index("AA") if "AA" in carriers else 0)
        origin = c3.selectbox("Origin airport", airports, index=airports.index("ABQ") if "ABQ" in airports else 0)
        c4, c5, c6 = st.columns(3)
        destination = c4.selectbox("Destination airport", airports, index=airports.index("DFW") if "DFW" in airports else 1)
        departure = c5.time_input("Scheduled departure", value=time(5, 0))
        arrival = c6.time_input("Scheduled arrival", value=time(7, 58))
        c7, c8 = st.columns(2)
        elapsed = c7.number_input("Scheduled duration (minutes)", min_value=1, max_value=1440, value=118, step=1)
        distance = c8.number_input("Distance (miles)", min_value=1.0, max_value=10000.0, value=569.0, step=1.0)
        submitted = st.form_submit_button("Estimate delay risk", type="primary")
    if submitted:
        try:
            flight = Flight(flight_date, carrier, origin, destination, departure, arrival, int(elapsed), float(distance))
            probability = predict_delay(flight)
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.metric("Estimated late arrival probability", f"{probability:.1%}")
            st.progress(min(max(probability, 0.0), 1.0))
            st.caption(f"March 2026 completed-flight average: {BASELINE:.1%}. This is a model estimate, not a live operational forecast.")
            if f"{origin}-{destination}" not in vocab["ROUTE"]:
                st.warning("This route was uncommon in training and maps to the model's OTHER category. Interpret the estimate cautiously.")

with explore_tab:
    st.subheader("Historical arrival delays")
    st.caption("Rates describe completed, nondiverted Q1 2026 flights. Airport and route summaries exclude low-count groups.")
    choice = st.selectbox("Group by", ["Airline", "Origin", "Destination", "Route", "Departure hour", "Day of week", "Month"])
    mapping = {
        "Airline": ("op_unique_carrier", "Airline", 0),
        "Origin": ("origin", "Origin", 5000),
        "Destination": ("dest", "Destination", 5000),
        "Route": ("route", "Route", 1000),
        "Departure hour": ("dep_hour", "Hour", 0),
        "Day of week": ("day_of_week", "Day (Monday=0)", 0),
        "Month": ("month", "Month", 0),
    }
    name, label, floor = mapping[choice]
    draw_ranking(name, label, floor)
    st.caption("The chart shows up to 20 groups; the download contains all matching groups.")

with method_tab:
    st.subheader("Data and model")
    st.write("The source is 1,671,142 scheduled U.S. flight records from January–March 2026. The model uses 1,611,873 valid completed flights. January trained candidate models, February selected the model and threshold, and March was held out for testing.")
    cols = st.columns(3)
    for col, (label, value) in zip(cols, METRICS.items()):
        col.metric(label, value)
    st.write("The selected logistic regression uses only scheduled information: airline, airports, route, date, scheduled times, duration, and distance. It reached 0.630 ROC AUC and 0.344 average precision on March flights, versus the March late arrival rate of 0.243.")
    st.write("The source data contain no weather, aircraft rotation, live congestion, or actual departure time. Flight patterns can change, and predictions outside Q1 2026 have not been validated. Do not use this tool for current travel decisions.")
    st.caption("Source: IDC4290 HW5 Q1 2026 Flight Delay Report, Group 17.")


