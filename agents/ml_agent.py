from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, r2_score

def train_model(df, target_column):
    """
    This function acts as a smart ML Agent.
    It automatically detects if it should perform Classification or Regression.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Convert text features into numbers
    label_encoders = {}
    for column in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[column] = X[column].astype(str)
        X[column] = le.fit_transform(X[column])
        label_encoders[column] = le
        
    # Step 1: Detect the type of prediction needed
    # If the target is text, or has less than 20 unique numbers, treat it as a Category
    is_category = False
    if y.dtype == 'object' or len(y.unique()) < 20:
        is_category = True
        
    # If it is a category made of text, encode it to numbers
    if is_category and y.dtype == 'object':
        le_y = LabelEncoder()
        y = le_y.fit_transform(y.astype(str))
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Step 2: Choose the right algorithm automatically
    if is_category:
        # Classification for categories
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        score = accuracy_score(y_test, predictions)
        metric_name = "Accuracy"
    else:
        # Regression for continuous numbers
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        score = r2_score(y_test, predictions)
        metric_name = "R-Squared Score (Accuracy for numbers)"
        
    # We now return 3 things, including the name of the score used
    return score, model, metric_name