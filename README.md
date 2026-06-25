# Ozone Guardian

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ozone-guardian.streamlit.app/)

An interactive dashboard analyzing industrial pollution's impact on ozone layer depletion using machine learning.

## What it does

- Fetches real pollution data from the World Bank API (falls back to synthetic data if unavailable)
- Trains a Random Forest Regressor to predict ozone depletion from CFCs, NOx, Methane, and CO2 levels
- Interactive sliders let users test custom pollution scenarios and get instant depletion predictions
- Projects ozone health 10 years forward based on current emission trends
- Feature importance chart shows which pollutants contribute most to depletion

## Tech Stack

Python, Streamlit, scikit-learn, Plotly, Pandas, World Bank API

Model: RandomForestRegressor -- ~88.9% R2 accuracy on test set

## Run locally

git clone https://github.com/shauryaguptagit/ozone-guardian.git
cd ozone-guardian
pip install -r requirements.txt
streamlit run dashboard.py