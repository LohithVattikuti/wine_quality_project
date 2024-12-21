import sqlite3
import pandas as pd
from ydata_profiling import ProfileReport

# Use wine_data.db for data exploration
db_path = "/Users/lohithvattikuti/Desktop/Wine_Quality_project/data/wine_data.db"

# SQL query to fetch data
query = """
SELECT * 
FROM wine_features;
"""

# Connect to the database and fetch data
try:
    conn = sqlite3.connect(db_path)
    data = pd.read_sql_query(query, conn)
    conn.close()

    # Check if data is fetched
    if data.empty:
        raise ValueError("The 'wine_features' table is empty.")

    print(f"Data fetched successfully: {data.shape[0]} rows and {data.shape[1]} columns.")

    # Generate a profile report
    print("Generating data profile report...")
    profile = ProfileReport(data, title="Wine Features Profile Report", explorative=True)
    profile.to_file("wine_features_profile.html")
    print("Profile report generated: wine_features_profile.html")

except Exception as e:
    print(f"Error: {e}")