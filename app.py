
import streamlit as st
import joblib
import pandas as pd

# Load the model
model, team_mapping = joblib.load("over_3_5_rf_model.pkl")

st.title("England Virtual Football Over 3.5 Goals Predictor")

# Sample match input
uploaded = st.file_uploader("Upload Match Schedule JSON", type="json")
threshold = st.slider("Confidence Threshold (%)", 50, 100, 85)

if uploaded:
    schedule = pd.read_json(uploaded)
    results = []
    for i, row in schedule.iterrows():
        home, away = row['home'], row['away']
        if home in team_mapping and away in team_mapping:
            features = [[team_mapping[home], team_mapping[away], 1.5, 1.5]]
            prob = model.predict_proba(features)[0][1]
            results.append({
                "Match": f"{home} vs {away}",
                "Confidence": round(prob * 100, 2),
                "Bet": "✅" if prob >= threshold / 100 else "❌"
            })
    df = pd.DataFrame(results)
    st.dataframe(df)
    st.download_button("Download Predictions", df.to_csv(index=False), file_name="predictions.csv")
else:
    st.info("Please upload a JSON file with match data in format: [{"home": "MUN", "away": "ARS"}, ...]")
