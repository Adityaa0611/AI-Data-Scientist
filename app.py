import streamlit as st
import pandas as pd
import os
from datetime import datetime # NEW: To create timestamps

from agents.cleaning_agent import clean_data 
from agents.viz_agent import generate_correlation_heatmap
from agents.ml_agent import train_model
from agents.reporting_agent import generate_pdf_report
from agents.notebook_agent import generate_notebook
from agents.qa_agent import ask_dataset_question # NEW

# NEW: Automatically create the run_history folder if it doesn't exist
if not os.path.exists("run_history"):
    os.makedirs("run_history")

st.title("Autonomous AI Data Scientist")

# NEW: Initialize the permanent memory flags so the website remembers everything
if "run_complete" not in st.session_state:
    st.session_state.run_complete = False

uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])

if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)
    st.session_state.raw_rows = len(raw_df)
    
    if "cleaned_df" not in st.session_state:
        st.session_state.cleaned_df = None
    if "viz_fig" not in st.session_state:
        st.session_state.viz_fig = None
        
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
                st.session_state.viz_fig = generate_correlation_heatmap(st.session_state.cleaned_df)
                if st.session_state.viz_fig is not None:
                    st.pyplot(st.session_state.viz_fig)
                    st.success("Graph generated successfully!")
                else:
                    st.warning("Not enough numeric data to draw a correlation graph.")
                    
        st.write("---")
        st.subheader("Machine Learning Agent")
        
        columns = st.session_state.cleaned_df.columns.tolist()
        target_column = st.selectbox("Select the column you want the AI to predict (Target):", columns)
        
        # When clicked, do the work and save everything to permanent memory
        if st.button("Train ML Models"):
            with st.spinner("ML Agent is testing multiple algorithms..."):
                score, trained_model, metric_name, report_df = train_model(st.session_state.cleaned_df, target_column)
                
                # Create a unique timestamp for this exact run
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # NEW: Isolate the exact CSV filename so we can pass it to the notebook
                csv_filename = f"Cleaned_Data_{timestamp}.csv"
                
                # Define unique paths inside the run_history folder
                pdf_path = f"run_history/AI_Report_{timestamp}.pdf"
                nb_path = f"run_history/ML_Code_{timestamp}.ipynb"
                csv_path = f"run_history/{csv_filename}" # Use the variable here!
                
                # Save the CSV to the history folder
                st.session_state.cleaned_df.to_csv(csv_path, index=False)
                
                # Generate the files and save them to the history folder
                clean_rows = len(st.session_state.cleaned_df)
                generate_pdf_report(st.session_state.raw_rows, clean_rows, st.session_state.viz_fig, report_df, metric_name, target_column, pdf_path)
                
                # UPDATED: We now pass the csv_filename as the final argument
                generate_notebook(target_column, metric_name, nb_path, csv_filename)
                
                # Save all the results into permanent memory
                st.session_state.report_df = report_df
                st.session_state.metric_name = metric_name
                st.session_state.score = score
                st.session_state.best_model_name = report_df.iloc[0]['Model']
                st.session_state.pdf_path = pdf_path
                st.session_state.nb_path = nb_path
                st.session_state.csv_path = csv_path
                
                # Flip the switch! The training is complete.
                st.session_state.run_complete = True

        # NEW: Because this block is OUTSIDE the button, it will not disappear when you click download!
        if st.session_state.run_complete:
            st.success("Auto-Training complete! The ML Agent has selected the best model.")
            st.write("### 🏆 Model Comparison Report")
            st.dataframe(st.session_state.report_df)
            
            score_percentage = round(st.session_state.score * 100, 2)
            st.metric(label=f"Winner: {st.session_state.best_model_name} ({st.session_state.metric_name})", value=f"{score_percentage}%")
            
            st.write("### 📂 Download Run History Files")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                with open(st.session_state.pdf_path, "rb") as pdf_file:
                    st.download_button(
                        label="📄 Download PDF",
                        data=pdf_file,
                        file_name=os.path.basename(st.session_state.pdf_path),
                        mime="application/pdf"
                    )
                    
            with col2:
                with open(st.session_state.nb_path, "rb") as nb_file:
                    st.download_button(
                        label="📓 Download Notebook",
                        data=nb_file,
                        file_name=os.path.basename(st.session_state.nb_path),
                        mime="application/x-ipynb+json"
                    )
                    
            with col3:
                with open(st.session_state.csv_path, "rb") as csv_file:
                    st.download_button(
                        label="📊 Download Cleaned Data",
                        data=csv_file,
                        file_name=os.path.basename(st.session_state.csv_path),
                        mime="text/csv"
                    )
                    # -----------------------------------------
        # NEW: Q&A Chatbot Section
        # This sits outside the training loop so you can chat anytime after data is cleaned
        # -----------------------------------------
        st.write("---")
        st.subheader("💬 Chat with your Dataset")
        st.write("Ask questions like: 'What is the average value of the target column?' or 'Which category appears most often?'")
        
        # Secure text input for the user's Gemini API Key
        api_key = st.text_input("Enter your Gemini API Key to enable chat:", type="password")
        
        # Initialize memory for the chat interface
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
            
        # Display all previous chat messages on the screen
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                
        # The text input box where the user types their questions
        user_question = st.chat_input("Ask a question about the cleaned data...")
        
        if user_question:
            if not api_key:
                st.warning("Please enter your Gemini API Key first.")
            else:
                # 1. Save and display the user's question immediately
                st.session_state.chat_history.append({"role": "user", "content": user_question})
                with st.chat_message("user"):
                    st.write(user_question)
                    
                # 2. Pass the question to the Q&A Agent and display the response
                with st.chat_message("assistant"):
                    with st.spinner("AI is inspecting the spreadsheet..."):
                        # Pass the cleaned data to the agent
                        answer = ask_dataset_question(st.session_state.cleaned_df, user_question, api_key)
                        st.write(answer)
                        # Save the AI's response to memory
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})