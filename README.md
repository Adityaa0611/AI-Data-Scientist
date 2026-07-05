🤖 Autonomous AI Data Scientist
An end-to-end, multi-agent automated machine learning pipeline built with Streamlit, Scikit-Learn, LangChain, and the Google Gemini API.

This application acts as a virtual data scientist. It ingests raw CSV datasets, cleans them, performs exploratory data analysis (EDA), automatically trains and evaluates multiple machine learning models, and generates downloadable reports and Jupyter Notebooks. Finally, it provides an interactive AI chatbot to query the dataset using natural language.

✨ Key Features & Agents
This project utilizes a multi-agent architecture to break down the data science lifecycle into automated steps:

🧹 Cleaning Agent: Automatically analyzes raw data, handles missing values, encodes categorical variables, and prepares the dataset for machine learning.

📊 Visualization Agent: Generates correlation heatmaps to provide immediate Exploratory Data Analysis (EDA) insights.

🧠 Machine Learning Agent: Automatically detects whether the target variable requires Classification or Regression. It then scales the data and evaluates 6 different algorithms (Random Forest, XGBoost, SVM, Logistic/Linear Regression, Decision Trees, and Naive Bayes), outputting a ranked leaderboard of accuracy/R-squared scores.

📄 Reporting Agent: Compiles the entire pipeline—data shape, EDA visuals, model leaderboard, and the exact Python training code—into a professional, downloadable PDF Document.

📓 Notebook Agent: Dynamically generates a downloadable Jupyter Notebook (.ipynb) containing the exact Python code used to train the winning models, ensuring complete transparency and reproducibility.

💬 Q&A Chatbot Agent: Powered by LangChain and Google Gemini, this agent allows users to "talk" to their dataset. Users can ask natural language questions (e.g., "What is the average age in the dataset?") and receive immediate, data-driven answers.

📂 Automated Run History: Every training loop automatically saves timestamped versions of the Cleaned CSV, PDF Report, and Jupyter Notebook into a local run_history/ folder to prevent data overwriting.

🛠️ Tech Stack
Frontend UI: Streamlit

Data Manipulation: Pandas, NumPy

Machine Learning: Scikit-Learn, XGBoost

Generative AI & NLP: LangChain, Google Gemini API

Document Generation: FPDF (PDFs), nbformat (Jupyter Notebooks)
