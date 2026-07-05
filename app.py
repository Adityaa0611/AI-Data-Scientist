import streamlit as st
import pandas as pd

# This creates the main heading on the web page
st.title("Autonomous AI Data Scientist")

# This creates a file upload box for the user
uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])

# This checks if the user has actually uploaded a file
if uploaded_file is not None:
    # Pandas reads the uploaded CSV file and stores it in a variable called 'df'
    df = pd.read_csv(uploaded_file)
    
    # This prints a message to the screen
    st.write("Data successfully loaded! Here are the first 5 rows:")
    
    # This displays the actual spreadsheet on the web page
    st.dataframe(df.head())