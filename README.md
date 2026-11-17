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

## Deploy on Render with access limited to this repository

The source repository is https://github.com/vdc109/idc4290-hw8-flight-delay.

1. Sign in at [Render](https://dashboard.render.com/) and choose **New → Web Service**.
2. When GitHub asks which repositories Render may access, choose **Selected Repositories** and select only `vdc109/idc4290-hw8-flight-delay`.
3. Select that repository, branch `main`, runtime **Python**, and instance type **Free**.
4. Use build command `pip install -r requirements.txt`.
5. Use start command `streamlit run app.py --server.address 0.0.0.0 --server.port $PORT --server.headless true`.
6. Create the service. The repository's `.python-version` file pins Python 3.12. No secrets or environment variables are needed.
7. Test the public `onrender.com` URL in a private browser window and add it to `SUBMISSION.md`.

The `render.yaml` file contains these service settings if you prefer Render Blueprints. Render's free web services spin down after inactivity; the first visit after idle may take about a minute.

## Source and limitations

The model and summaries were produced in IDC4290 HW5 from 1,671,142 Q1 2026 flight records. Of these, 1,611,873 valid completed flights entered the analysis. The selected model achieved March ROC AUC 0.630 and average precision 0.344. Cancelled and diverted flights are excluded from its outcome. It has no live weather, congestion, aircraft, or departure data, and it has not been tested beyond March 2026. It is an educational historical estimator, not a live travel forecast.

The airport and route exploration tables were prefiltered in HW5: airports have at least 5,000 flights and routes at least 1,000. The model vocabulary is broader than those display tables. Rare categories map to OTHER, as in HW5.
