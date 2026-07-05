import streamlit as st
import pandas as pd

# Import ALL your agents
from agents.cleaning_agent import clean_data 
from agents.viz_agent import generate_correlation_heatmap
from agents.ml_agent import train_model
from agents.reporting_agent import generate_pdf_report # NEW
from agents.notebook_agent import generate_notebook

st.title("Autonomous AI Data Scientist")

uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])

if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)
    
    # 1. Memory Storage: Save things we need for the final report
    st.session_state.raw_rows = len(raw_df)
    
    if "cleaned_df" not in st.session_state:
        st.session_state.cleaned_df = None
    if "viz_fig" not in st.session_state:
        st.session_state.viz_fig = None
        
    st.write("Original Data Preview:")
    st.dataframe(raw_df.head())
    
    # 2. Cleaning Phase
    if st.button("Run Cleaning Agent"):
        with st.spinner("Cleaning Agent is analyzing and fixing data..."):
            st.session_state.cleaned_df = clean_data(raw_df)
            st.success("Data cleaning complete!")
            
    # 3. Visualization Phase
    if st.session_state.cleaned_df is not None:
        st.write("Cleaned Data Preview:")
        st.dataframe(st.session_state.cleaned_df.head())
        
        if st.button("Run Visualization Agent"):
            with st.spinner("Visualization Agent is drawing graphs..."):
                # Save the graph to memory so the PDF can use it later
                st.session_state.viz_fig = generate_correlation_heatmap(st.session_state.cleaned_df)
                if st.session_state.viz_fig is not None:
                    st.pyplot(st.session_state.viz_fig)
                    st.success("Graph generated successfully!")
                else:
                    st.warning("Not enough numeric data to draw a correlation graph.")
                    
        # 4. Machine Learning Phase
        st.write("---")
        st.subheader("Machine Learning Agent")
        
        columns = st.session_state.cleaned_df.columns.tolist()
        target_column = st.selectbox("Select the column you want the AI to predict (Target):", columns)
        
        if st.button("Train ML Models"):
            with st.spinner("ML Agent is testing multiple algorithms..."):
                score, trained_model, metric_name, report_df = train_model(st.session_state.cleaned_df, target_column)
                st.success("Auto-Training complete! The ML Agent has selected the best model.")
                
                st.write("### 🏆 Model Comparison Report")
                st.dataframe(report_df)
                
                score_percentage = round(score * 100, 2)
                best_model_name = report_df.iloc[0]['Model']
                st.metric(label=f"Winner: {best_model_name} ({metric_name})", value=f"{score_percentage}%")
                
                # 5. UPDATED: Reporting & Code Generation Phase
                with st.spinner("Generating your PDF and Jupyter Notebook..."):
                    clean_rows = len(st.session_state.cleaned_df)
                    
                    # Save the cleaned dataset so the Notebook can read it
                    st.session_state.cleaned_df.to_csv("cleaned_dataset.csv", index=False)
                    
                    # Generate the PDF
                    pdf_filename = generate_pdf_report(
                        st.session_state.raw_rows, 
                        clean_rows, 
                        st.session_state.viz_fig, 
                        report_df, 
                        metric_name
                    )
                    
                    # Generate the Notebook
                    notebook_filename = generate_notebook(target_column, metric_name)
                    
                    # Create two columns to put the download buttons side-by-side
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        with open(pdf_filename, "rb") as pdf_file:
                            st.download_button(
                                label="⬇️ Download PDF Report",
                                data=pdf_file,
                                file_name="AI_Data_Scientist_Report.pdf",
                                mime="application/pdf"
                            )
                            
                    with col2:
                        with open(notebook_filename, "rb") as nb_file:
                            st.download_button(
                                label="⬇️ Download Jupyter Notebook",
                                data=nb_file,
                                file_name="Generated_ML_Code.ipynb",
                                mime="application/x-ipynb+json"
                            )