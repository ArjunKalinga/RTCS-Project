import streamlit as st
import pandas as pd
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Session Report Viewer",
    layout="wide"
)

# --- Constants ---
LOG_DIRECTORY = 'data'
LOG_FILE = os.path.join(LOG_DIRECTORY, 'session_log.csv')

# --- Main App ---
st.title("Classroom Engagement Session Report")
st.write("This report displays the summary of all past monitoring sessions.")

# Check if the log file exists
if os.path.isfile(LOG_FILE):
    try:
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(LOG_FILE)
        
        st.header("Session Data")
        
        # Display the data in an interactive table
        st.dataframe(df)

        st.header("Engagement Over Time")
        
        # Create a simple line chart to visualize trends
        # We'll use the 'Timestamp' for the x-axis and 'Avg Engagement (%)' for the y-axis
        st.line_chart(df.rename(columns={'Timestamp': 'index'}).set_index('index')['Avg Engagement (%)'])

    except pd.errors.EmptyDataError:
        st.warning("The log file is empty. Please run a session in `main.py` to generate data.")
    except Exception as e:
        st.error(f"An error occurred while reading the log file: {e}")
else:
    st.info(f"No session log file found at '{LOG_FILE}'. Please run a session using `main.py` to create a log.")

