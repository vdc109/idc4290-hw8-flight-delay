# IDC4290 HW8 submission — Flight Delay Risk Explorer

**App URL:** Add the public Streamlit Community Cloud URL here after deployment.

## What the tool does

The app estimates the probability that a scheduled Q1 2026 flight will arrive at least 15 minutes late, conditional on completion without diversion. It also lets users explore historical delay rates by airline, airport, route, departure hour, weekday, and month.

## How to use it

1. Open the app URL. If Streamlit says the app is sleeping, select the option to wake it.
2. Open **Predict a flight**. Use the prefilled example or enter a date in January–March 2026, airline, airports, scheduled local departure and arrival times, scheduled duration, and distance.
3. Select **Estimate delay risk** and read the probability below the form.
4. Open **Explore delays** to filter and sort historical rates. Use **Download filtered data** for a CSV.
5. Open **About the data** for model performance and limitations.

The model does not predict cancellation or diversion and is not a live travel forecast. On the held-out March data, its ROC AUC was 0.630 and average precision was 0.344.

## Run from source

With Python 3.12 installed, run `python -m pip install -r requirements.txt` and `streamlit run app.py` from the project folder. See `README.md` for details.
