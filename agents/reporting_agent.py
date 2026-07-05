from fpdf import FPDF
import os

# UPDATED: We added target_column to the inputs
def generate_pdf_report(raw_rows, clean_rows, viz_fig, ml_report_df, metric_name, target_column):
    """
    This is the Reporting Agent. It creates a formatted PDF document,
    now including the generated Python code used for training.
    """
    pdf = FPDF()
    pdf.add_page()
    
    # 1. Main Title
    pdf.set_font("Arial", style="B", size=16)
    pdf.cell(200, 10, txt="Autonomous AI Data Scientist - Final Report", ln=True, align="C")
    pdf.ln(10)

    # 2. Data Cleaning Section
    pdf.set_font("Arial", style="B", size=14)
    pdf.cell(200, 10, txt="1. Data Cleaning Summary", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Rows before cleaning (Raw Data): {raw_rows}", ln=True)
    pdf.cell(200, 10, txt=f"Rows after cleaning (Clean Data): {clean_rows}", ln=True)
    pdf.ln(10)

    # 3. Visualization Section
    pdf.set_font("Arial", style="B", size=14)
    pdf.cell(200, 10, txt="2. Exploratory Data Analysis (EDA)", ln=True)
    if viz_fig is not None:
        viz_fig.savefig("temp_heatmap.png")
        pdf.image("temp_heatmap.png", x=10, w=150)
        os.remove("temp_heatmap.png")
    else:
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="No correlation heatmap was generated.", ln=True)
    
    # 4. Machine Learning Leaderboard (New Page)
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

    # 5. NEW: Generated Machine Learning Code (New Page)
    pdf.add_page()
    pdf.set_font("Arial", style="B", size=14)
    pdf.cell(200, 10, txt="4. Auto-Generated Training Code", ln=True)
    pdf.ln(5)
    
    # Set font to Courier (Monospace) so it looks like a real code editor
    pdf.set_font("Courier", size=9)
    
    # Prepare the exact code string based on what metric was used
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

# Encode text features
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

if y.dtype == 'object':
    y = LabelEncoder().fit_transform(y.astype(str))

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize Models
{algo_code}

# Train and evaluate
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
    # Scores are calculated here...
"""
    # multi_cell allows long text to wrap properly within the PDF margins
    pdf.multi_cell(0, 5, txt=code_block)

    report_filename = "AI_Report.pdf"
    pdf.output(report_filename)
    return report_filename