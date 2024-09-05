import streamlit as st
import pandas as pd
import pickle
import os
import requests

# URL of your model stored on Google Drive
MODEL_URL = 'https://drive.google.com/uc?export=download&id=1D1kPHNLC1MpVirOp-jhU3ViXkDJVUS_N'
MODEL_PATH = 'fantasy_score_model.pkl'

def download_model():
    if not os.path.exists(MODEL_PATH):
        st.write("Downloading model from Google Drive...")
        with requests.Session() as session:
            response = session.get(MODEL_URL, stream=True)
            # Check if Google Drive sends a 'confirm' link for large files
            if "text/html" in response.headers.get("content-type", ""):
                # Parse the confirmation page and extract the download link
                for key, value in response.cookies.items():
                    if key.startswith('download_warning'):
                        MODEL_URL_confirm = MODEL_URL + "&confirm=" + value
                        response = session.get(MODEL_URL_confirm, stream=True)
                        break
            # Write the model file locally
            with open(MODEL_PATH, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # Filter out keep-alive new chunks
                        f.write(chunk)
            st.write("Model downloaded successfully.")

# Ensure the model is downloaded
download_model()

# Load your data and model
df = pd.read_csv('fantasy_scores.csv')
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# Streamlit UI components
st.title("Fantasy Score Predictor")

# Dropdown for inputs
venue = st.selectbox('Select Venue', df['venue'].unique())
player_name = st.selectbox('Select Player Name', df['player_name'].unique())
player_team = st.selectbox('Select Player Team', df['player_team'].unique())
batting_first_team = st.selectbox('Select Batting First Team', df['batting_first_team'].unique())
bowling_first_team = st.selectbox('Select Bowling First Team', df['bowling_first_team'].unique())

# Button to predict score
if st.button('Predict Fantasy Score'):
    # Create input data for the model
    input_data = pd.DataFrame({
        'venue': [venue],
        'player_name': [player_name],
        'player_team': [player_team],
        'batting_first_team': [batting_first_team],
        'bowling_first_team': [bowling_first_team]
    })

    # Encoding categorical features
    input_data_encoded = pd.get_dummies(input_data)
    input_data_encoded = input_data_encoded.reindex(columns=model.feature_names_in_, fill_value=0)

    # Make prediction
    predicted_score = model.predict(input_data_encoded)

    # Display the predicted score
    st.write(f"Predicted Fantasy Score for {player_name}: {predicted_score[0]}")

# Options Route
st.sidebar.title("Options")
st.sidebar.write("Unique Options from the Dataset:")

st.sidebar.write("**Venues:**")
st.sidebar.write(df['venue'].unique().tolist())

st.sidebar.write("**Player Names:**")
st.sidebar.write(df['player_name'].unique().tolist())

st.sidebar.write("**Player Teams:**")
st.sidebar.write(df['player_team'].unique().tolist())

st.sidebar.write("**Batting First Teams:**")
st.sidebar.write(df['batting_first_team'].unique().tolist())

st.sidebar.write("**Bowling First Teams:**")
st.sidebar.write(df['bowling_first_team'].unique().tolist())
