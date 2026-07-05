import matplotlib.pyplot as plt
import seaborn as sns

def generate_correlation_heatmap(df):
    """
    This function acts as the Visualization Agent.
    It isolates numeric data and generates a relationship graph.
    """
    # Step 1: Find only the columns that contain numbers
    numeric_df = df.select_dtypes(include=['number'])
    
    # Step 2: If there are fewer than 2 numeric columns, we can't compare them
    if len(numeric_df.columns) < 2:
        return None
        
    # Step 3: Create a blank canvas for the graph
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Step 4: Draw the heatmap onto the canvas
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax, fmt=".2f")
    plt.title("Automated Correlation Heatmap")
    
    # Return the finished picture
    return fig