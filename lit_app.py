import streamlit as st
import pandas as pd
#import pickle
import joblib
import os
import requests
from google_drive_downloader import GoogleDriveDownloader as gdd


# URL and destination of the model
#MODEL_URL = 'https://drive.google.com/uc?export=download&id=1D1kPHNLC1MpVirOp-jhU3ViXkDJVUS_N'
#MODEL_PATH = 'fantasy_score_model.pkl'

@st.cache
def load_model():
    model_path = 'D:\\pkl model\fantasy_score_modell.pkl'
    if not os.path.exists(model_path):
        gdd.download_file_from_google_drive(file_id='1D1kPHNLC1MpVirOp-jhU3ViXkDJVUS_N',
                                            dest_path=model_path,
                                            unzip=False)
    model = joblib.load(model_path)
    return model

model = load_model()
    




df = pd.read_csv('fantasy_scores.csv')

# Load the model
#try:
    #with open(MODEL_PATH, 'rb') as f:
       #k = joblib.dump(model, f) 
       #model = joblib.load(k) #model = pickle.load(f)    
#except Exception as e:
    #st.error(f"Error loading model: {e}")

    
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
