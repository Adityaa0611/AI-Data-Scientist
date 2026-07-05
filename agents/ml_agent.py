from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, r2_score
import pandas as pd

def train_model(df, target_column):
    """
    This ML Agent tests multiple algorithms, compares them, 
    and returns a leaderboard report along with the best model.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # 1. Convert text features to numbers
    label_encoders = {}
    for column in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[column] = X[column].astype(str)
        X[column] = le.fit_transform(X[column])
        
    # 2. Detect Classification vs Regression
    is_category = False
    if y.dtype == 'object' or len(y.unique()) < 20:
        is_category = True
        
    if is_category and y.dtype == 'object':
        le_y = LabelEncoder()
        y = le_y.fit_transform(y.astype(str))
        
    # 3. Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Scale the data (Crucial for SVM and Logistic Regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. Define the arenas (Which models to test)
    if is_category:
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
            "SVM": SVC(random_state=42),
            "XGBoost": XGBClassifier(random_state=42, eval_metric='logloss')
        }
        metric_name = "Accuracy"
    else:
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(random_state=42),
            "SVM": SVR(),
            "XGBoost": XGBRegressor(random_state=42)
        }
        metric_name = "R-Squared Score"
        
    # 6. Train all models and keep track of scores
    results = []
    best_model = None
    best_score = -float('inf')
    
    for name, model in models.items():
        # Train the model
        model.fit(X_train_scaled, y_train)
        
        # Test the model
        predictions = model.predict(X_test_scaled)
        
        # Calculate the score
        if is_category:
            score = accuracy_score(y_test, predictions)
        else:
            score = r2_score(y_test, predictions)
            
        # Add to the report
        results.append({"Model": name, metric_name: score})
        
        # Check if this is the new best model
        if score > best_score:
            best_score = score
            best_model = model
            
    # 7. Create a clean Pandas DataFrame report, sorted from best to worst
    report_df = pd.DataFrame(results).sort_values(by=metric_name, ascending=False)
    
    # We now return 4 items, including the report_df
    return best_score, best_model, metric_name, report_df