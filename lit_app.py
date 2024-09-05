import streamlit as st
import pandas as pd
#import pickle
import joblib
import os
import requests


# URL of your model stored on Google Drive
MODEL_URL = 'https://drive.google.com/uc?export=download&id=1D1kPHNLC1MpVirOp-jhU3ViXkDJVUS_N'
MODEL_PATH = 'fantasy_score_model.pkl'

EXPECTED_MODEL_SIZE = 188946758  # Replace with the actual model size in bytes

def download_model():
    if not os.path.exists(MODEL_PATH):
        st.write("Downloading model from Google Drive...")
        with requests.Session() as session:
            response = session.get(MODEL_URL, stream=True)
            if response.status_code == 200:
                with open(MODEL_PATH, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                st.write("Model downloaded successfully.")
            else:
                st.error("Failed to download the model.")

        # Verify the model file size
        model_size = os.path.getsize(MODEL_PATH)
        if model_size != EXPECTED_MODEL_SIZE:
            st.error("Downloaded model size does not match the expected size. The file might be corrupted.")
            os.remove(MODEL_PATH)  # Delete the corrupted file


# Ensure the model is downloaded
download_model()

# Load your data and model
df = pd.read_csv('fantasy_scores.csv')
with open(MODEL_PATH, 'rb') as f:
    k = joblib.dump(model, f) 
    model = joblib.load(k) #model = pickle.load(f)

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
