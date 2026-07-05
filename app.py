import streamlit as st
import pandas as pd

# Import both of your agents
from agents.cleaning_agent import clean_data 
from agents.viz_agent import generate_correlation_heatmap

st.title("Autonomous AI Data Scientist")

uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])

if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)
    
    # NEW: Create a memory space to hold the cleaned data
    if "cleaned_df" not in st.session_state:
        st.session_state.cleaned_df = None
        
    st.write("Original Data Preview:")
    st.dataframe(raw_df.head())
    
    if st.button("Run Cleaning Agent"):
        with st.spinner("Cleaning Agent is analyzing and fixing data..."):
            # Save the cleaned data into the memory space
            st.session_state.cleaned_df = clean_data(raw_df)
            st.success("Data cleaning complete!")
            
    # NEW: If the data has been cleaned, show it and offer the next step
    if st.session_state.cleaned_df is not None:
        st.write("Cleaned Data Preview:")
        st.dataframe(st.session_state.cleaned_df.head())
        
        # Add a button to trigger the Visualization Agent
        if st.button("Run Visualization Agent"):
            with st.spinner("Visualization Agent is drawing graphs..."):
                
                # Pass the cleaned data from memory to the agent
                fig = generate_correlation_heatmap(st.session_state.cleaned_df)
                
                if fig is not None:
                    # Display the picture on the website
                    st.pyplot(fig)
                    st.success("Graph generated successfully!")
                else:
                    st.warning("Not enough numeric data to draw a correlation graph.")