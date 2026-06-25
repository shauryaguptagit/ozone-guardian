<div align="center">
  <h1>Ozone Guardian</h1>
  <p>Interactive ML dashboard analyzing industrial pollution's impact on ozone layer depletion.</p>
  <a href="https://ozone-guardian.streamlit.app/">
    <img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg"/>
  </a>
  <br/><br/>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white"/>
</div>

---

## What It Does

Fetches real industrial pollution data from the World Bank API, trains a Random Forest Regressor on it, and lets you interactively explore how changes in CFC, NOx, Methane, and CO₂ levels affect ozone depletion — with a 10-year forward projection baked in.

**Key features:**
- Live data from World Bank API (synthetic fallback if unavailable)
- Random Forest Regressor — ~88.9% R² on test set
- Interactive sliders: adjust pollutant levels, get instant depletion predictions
- Feature importance chart — shows which pollutants drive depletion most
- 10-year trend projection based on current emission trajectories

## Tech Stack

| Layer | Tools |
|-------|-------|
| Frontend | Streamlit, Plotly |
| ML Model | scikit-learn (RandomForestRegressor) |
| Data | Pandas, World Bank API |
| Language | Python 3.11+ |

## Run Locally

```bash
git clone https://github.com/shauryaguptagit/ozone-guardian.git
cd ozone-guardian
pip install -r requirements.txt
streamlit run dashboard.py
```

## Live Demo

[ozone-guardian.streamlit.app](https://ozone-guardian.streamlit.app/)