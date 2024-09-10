import streamlit as st
import pandas as pd
#import pickle
import joblib
import os
import requests
#from google_drive_downloader import GoogleDriveDownloader as gdd


# URL and destination of the model
#MODEL_URL = 'https://drive.google.com/uc?export=download&id=1D1kPHNLC1MpVirOp-jhU3ViXkDJVUS_N'
#MODEL_PATH = 'fantasy_score_model.pkl'

##@st.cache
##def load_model():
    ##model_path = 'D:\\pkl model\fantasy_score_modell.pkl'
    ##if not os.path.exists(model_path):
        ##gdd.download_file_from_google_drive(file_id='1D1kPHNLC1MpVirOp-jhU3ViXkDJVUS_N',
                                            dest_path=model_path,
                                            unzip=False)
    ##model = joblib.load(model_path)
    ##return model

##model = load_model()
    




df = pd.read_csv('fantasy_scores.csv')


import streamlit as st
import joblib
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io

# Path to your service account key file
SERVICE_ACCOUNT_FILE = 'client_secret_297131267424-eeir38415dq5p16h7hl8h1f1ql6fgbun.apps.googleusercontent.com.json'

# Authenticate and create the service
credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE,
    scopes=['https://www.googleapis.com/auth/drive']
)
service = build('drive', 'v3', credentials=credentials)

# Function to download a file from Google Drive
def download_file(file_id):
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
        print(f"Download {int(status.progress() * 100)}%.")
    fh.seek(0)
    return fh

# Example usage
file_id = '1D1kPHNLC1MpVirOp-jhU3ViXkDJVUS_N'
file_content = download_file(file_id)

# Load the .pkl file using joblib
model = joblib.load(file_content)
st.write("Model loaded successfully.")






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
