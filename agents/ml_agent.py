from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

def train_model(df, target_column):
    """
    This function acts as the ML Agent.
    It prepares the data, trains a Random Forest model, and calculates accuracy.
    """
    # Step 1: Separate the data into Features (X) and Target (y)
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Step 2: Convert any text columns in X into numbers automatically
    label_encoders = {}
    for column in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[column] = X[column].astype(str)
        X[column] = le.fit_transform(X[column])
        label_encoders[column] = le
        
    # Convert target column (y) to numbers if it is text
    if y.dtype == 'object':
        le_y = LabelEncoder()
        y = le_y.fit_transform(y.astype(str))
    
    # Step 3: Split the data into Training (80%) and Testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Step 4: Initialize and train the algorithm
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Step 5: Test the model and calculate accuracy
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    return accuracy, model