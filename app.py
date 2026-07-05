import streamlit as st
import pandas as pd

from agents.cleaning_agent import clean_data 
from agents.viz_agent import generate_correlation_heatmap
# NEW: Import the ML agent
from agents.ml_agent import train_model

st.title("Autonomous AI Data Scientist")

uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])

if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)
    
    if "cleaned_df" not in st.session_state:
        st.session_state.cleaned_df = None
        
    st.write("Original Data Preview:")
    st.dataframe(raw_df.head())
    
    if st.button("Run Cleaning Agent"):
        with st.spinner("Cleaning Agent is analyzing and fixing data..."):
            st.session_state.cleaned_df = clean_data(raw_df)
            st.success("Data cleaning complete!")
            
    if st.session_state.cleaned_df is not None:
        st.write("Cleaned Data Preview:")
        st.dataframe(st.session_state.cleaned_df.head())
        
        if st.button("Run Visualization Agent"):
            with st.spinner("Visualization Agent is drawing graphs..."):
                fig = generate_correlation_heatmap(st.session_state.cleaned_df)
                if fig is not None:
                    st.pyplot(fig)
                    st.success("Graph generated successfully!")
                else:
                    st.warning("Not enough numeric data to draw a correlation graph.")
                    
        # NEW: ML Agent Section
        st.write("---")
        st.subheader("Machine Learning Agent")
        
        # Create a dropdown menu containing all the column names
        columns = st.session_state.cleaned_df.columns.tolist()
        target_column = st.selectbox("Select the column you want the AI to predict (Target):", columns)
        
        if st.button("Train ML Model"):
            with st.spinner("ML Agent is testing multiple algorithms (XGBoost, Random Forest, etc.)..."):
                
                # UPDATED: We now catch 4 things, including the report_df
                score, trained_model, metric_name, report_df = train_model(st.session_state.cleaned_df, target_column)
                
                st.success("Auto-Training complete! The ML Agent has selected the best model.")
                
                # NEW: Display the comparison report as a table on the website
                st.write("### 🏆 Model Comparison Report")
                st.dataframe(report_df)
                
                # Update the main metric to show the winner's name and score
                score_percentage = round(score * 100, 2)
                best_model_name = report_df.iloc[0]['Model']
                st.metric(label=f"Winner: {best_model_name} ({metric_name})", value=f"{score_percentage}%")