import streamlit as st
import pandas as pd

# NEW: Import your cleaning agent function
from agents.cleaning_agent import clean_data 

st.title("Autonomous AI Data Scientist")

uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])

if uploaded_file is not None:
    # Load the raw data
    raw_df = pd.read_csv(uploaded_file)
    
    st.write("Original Data Preview:")
    st.dataframe(raw_df.head())
    
    # NEW: Add a button to trigger the Cleaning Agent
    if st.button("Run Cleaning Agent"):
        
        # Shows a loading spinner while the agent processes the data
        with st.spinner("Cleaning Agent is analyzing and fixing data..."):
            
            # Pass the raw data to the agent
            cleaned_df = clean_data(raw_df)
            
            st.success("Data cleaning complete!")
            st.write("Cleaned Data Preview:")
            st.dataframe(cleaned_df.head())
            
            # Print a summary of what the agent did
            st.write(f"Rows before cleaning: {len(raw_df)}")
            st.write(f"Rows after cleaning: {len(cleaned_df)}")