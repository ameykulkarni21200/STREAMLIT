import streamlit as st
import pandas as pd
import joblib

# Load your dataset
df = pd.read_csv('fantasy_scores.csv')

# Streamlit UI components
st.title("Fantasy Score Predictor")

# File uploader widget for the model
uploaded_model = st.file_uploader("Choose a .pkl model file", type="pkl")

if uploaded_model is not None:
    # Load the uploaded model
    model = joblib.load(uploaded_model)
    st.success("Model loaded successfully!")

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
else:
    st.info("Please upload a .pkl model file to proceed.")
