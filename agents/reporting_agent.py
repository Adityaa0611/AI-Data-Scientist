from fpdf import FPDF
import os

def generate_pdf_report(raw_rows, clean_rows, viz_fig, ml_report_df, metric_name):
    """
    This is the Reporting Agent. It gathers data from all previous agents
    and creates a formatted PDF document.
    """
    # 1. Create the blank PDF page
    pdf = FPDF()
    pdf.add_page()
    
    # 2. Add the Main Title
    pdf.set_font("Arial", style="B", size=16)
    pdf.cell(200, 10, txt="Autonomous AI Data Scientist - Final Report", ln=True, align="C")
    pdf.ln(10) # Adds a blank line for spacing

    # 3. Add Data Cleaning Section
    pdf.set_font("Arial", style="B", size=14)
    pdf.cell(200, 10, txt="1. Data Cleaning Summary", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Rows before cleaning (Raw Data): {raw_rows}", ln=True)
    pdf.cell(200, 10, txt=f"Rows after cleaning (Clean Data): {clean_rows}", ln=True)
    pdf.ln(10)

    # 4. Add the Visualization Section
    pdf.set_font("Arial", style="B", size=14)
    pdf.cell(200, 10, txt="2. Exploratory Data Analysis (EDA)", ln=True)
    if viz_fig is not None:
        # We must save the graph as a picture file first so the PDF can read it
        viz_fig.savefig("temp_heatmap.png")
        pdf.image("temp_heatmap.png", x=10, w=150)
        os.remove("temp_heatmap.png") # Delete the picture after putting it in the PDF
    else:
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="No correlation heatmap was generated.", ln=True)
    
    # 5. Add the Machine Learning Leaderboard (on a new page)
    pdf.add_page()
    pdf.set_font("Arial", style="B", size=14)
    pdf.cell(200, 10, txt="3. Machine Learning Leaderboard", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Evaluation Metric Used: {metric_name}", ln=True)
    pdf.ln(5)

    # Draw the Table Header
    pdf.set_font("Arial", style="B", size=12)
    pdf.cell(100, 10, txt="Algorithm Name", border=1)
    pdf.cell(50, 10, txt="Score", border=1, ln=True)
    
    # Loop through the ML Report and draw the rows of the table
    pdf.set_font("Arial", size=12)
    for index, row in ml_report_df.iterrows():
        pdf.cell(100, 10, txt=str(row['Model']), border=1)
        score_rounded = str(round(row[metric_name], 4)) # Keep it to 4 decimal places
        pdf.cell(50, 10, txt=score_rounded, border=1, ln=True)

    # 6. Save the PDF to a file
    report_filename = "AI_Report.pdf"
    pdf.output(report_filename)
    
    # Return the name of the file so the website can offer it for download
    return report_filename