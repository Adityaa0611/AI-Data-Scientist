from fpdf import FPDF
import os

# UPDATED: Added output_path to the function
def generate_pdf_report(raw_rows, clean_rows, viz_fig, ml_report_df, metric_name, target_column, output_path):
    pdf = FPDF()
    pdf.add_page()
    
    pdf.set_font("Arial", style="B", size=16)
    pdf.cell(200, 10, txt="Autonomous AI Data Scientist - Final Report", ln=True, align="C")
    pdf.ln(10)

    pdf.set_font("Arial", style="B", size=14)
    pdf.cell(200, 10, txt="1. Data Cleaning Summary", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Rows before cleaning (Raw Data): {raw_rows}", ln=True)
    pdf.cell(200, 10, txt=f"Rows after cleaning (Clean Data): {clean_rows}", ln=True)
    pdf.ln(10)

    pdf.set_font("Arial", style="B", size=14)
    pdf.cell(200, 10, txt="2. Exploratory Data Analysis (EDA)", ln=True)
    if viz_fig is not None:
        viz_fig.savefig("temp_heatmap.png")
        pdf.image("temp_heatmap.png", x=10, w=150)
        os.remove("temp_heatmap.png")
    else:
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="No correlation heatmap was generated.", ln=True)
    
    pdf.add_page()
    pdf.set_font("Arial", style="B", size=14)
    pdf.cell(200, 10, txt="3. Machine Learning Leaderboard", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Evaluation Metric Used: {metric_name}", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", style="B", size=12)
    pdf.cell(100, 10, txt="Algorithm Name", border=1)
    pdf.cell(50, 10, txt="Score", border=1, ln=True)
    
    pdf.set_font("Arial", size=12)
    for index, row in ml_report_df.iterrows():
        pdf.cell(100, 10, txt=str(row['Model']), border=1)
        score_rounded = str(round(row[metric_name], 4))
        pdf.cell(50, 10, txt=score_rounded, border=1, ln=True)

    pdf.add_page()
    pdf.set_font("Arial", style="B", size=14)
    pdf.cell(200, 10, txt="4. Auto-Generated Training Code", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Courier", size=9)
    
    if metric_name == "Accuracy":
        algo_code = """models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Naive Bayes": GaussianNB(),
    "SVM": SVC(random_state=42),
    "XGBoost": XGBClassifier(random_state=42, eval_metric='logloss')
}"""
    else:
        algo_code = """models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "SVM": SVR(),
    "XGBoost": XGBRegressor(random_state=42)
}"""

    code_block = f"""import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier, XGBRegressor

# Load and prepare data
df = pd.read_csv('cleaned_dataset.csv')
X = df.drop(columns=['{target_column}'])
y = df['{target_column}']

for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

if y.dtype == 'object':
    y = LabelEncoder().fit_transform(y.astype(str))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

{algo_code}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
"""
    pdf.multi_cell(0, 5, txt=code_block)

    # UPDATED: Use the dynamic output_path instead of a hardcoded name
    pdf.output(output_path)
    return output_path