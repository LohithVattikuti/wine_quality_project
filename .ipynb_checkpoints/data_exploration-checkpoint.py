import sqlite3
import pandas as pd
from ydata_profiling import ProfileReport

# Path to SQLite database
db_path = '/Users/lohithvattikuti/Desktop/Wine_Quality_project/data/wine_quality.db'

# Connect to the database
conn = sqlite3.connect(db_path)

# SQL query to fetch data
query = """
SELECT 
    fixed_acidity, 
    volatile_acidity, 
    citric_acid, 
    residual_sugar, 
    chlorides, 
    free_sulfur_dioxide, 
    total_sulfur_dioxide, 
    density, 
    pH, 
    sulphates, 
    alcohol 
FROM wine_features;
"""

# Fetch data
data = pd.read_sql_query(query, conn)

# Close the database connection
conn.close()

# Generate a profile report
print("Generating data profile report...")
profile = ProfileReport(data, title="Wine Features Profile Report", explorative=True)
profile.to_file("data_exploration_report.html")

print("Data exploration completed! Report saved as 'data_exploration_report.html'.")