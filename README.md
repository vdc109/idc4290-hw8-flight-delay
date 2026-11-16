# Flight Delay Risk Explorer

Interactive Streamlit app for IDC4290 HW8. It reuses the trained logistic regression model and compact summary tables from HW5.

## Run locally

Install Python 3.12, then run:

    python -m pip install -r requirements.txt
    streamlit run app.py

Open the local URL printed by Streamlit. The app needs no API key and no raw flight CSV files.

## Use the app

1. In **Predict a flight**, change the prefilled scheduled flight details and select **Estimate delay risk**.
2. Read the estimated probability of arriving at least 15 minutes late. The probability applies to completed, nondiverted flights, not cancellation risk.
3. In **Explore delays**, choose an airline, airport, route, departure hour, weekday, or month. Filter groups, sort the ranking, and download its CSV.
4. In **About the data**, read the validation results and limits.

The prediction form accepts dates from January 1 through March 31, 2026, the period covered by the source data. Scheduled departure and arrival times are each local to their airport. Duration and distance must be the scheduled values for the selected flight. The prefilled example is AA, ABQ to DFW, March 1, 2026, 05:00 to 07:58, 118 minutes, 569 miles.

## Deploy

1. Push this folder to a GitHub repository. Keep `app.py`, `predictor.py`, `requirements.txt`, `data/`, and `model/` together at repository root.
2. Sign in at [Streamlit Community Cloud](https://share.streamlit.io/) and connect the GitHub repository.
3. Select the repository and branch, set the main file to `app.py`, and deploy.
4. Make the app public and test its URL in a private browser window before submitting it.
5. Put the deployed URL in `SUBMISSION.md`.

## Source and limitations

The model and summaries were produced in IDC4290 HW5 from 1,671,142 Q1 2026 flight records. Of these, 1,611,873 valid completed flights entered the analysis. The selected model achieved March ROC AUC 0.630 and average precision 0.344. Cancelled and diverted flights are excluded from its outcome. It has no live weather, congestion, aircraft, or departure data, and it has not been tested beyond March 2026. It is an educational historical estimator, not a live travel forecast.

The airport and route exploration tables were prefiltered in HW5: airports have at least 5,000 flights and routes at least 1,000. The model vocabulary is broader than those display tables. Rare categories map to OTHER, as in HW5.
