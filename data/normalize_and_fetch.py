import pandas as pd
import sqlite3

# Paths to the CSV files
features_file = "/Users/lohithvattikuti/Desktop/Wine_Quality_project/data/wine_features.csv"
labels_file = "/Users/lohithvattikuti/Desktop/Wine_Quality_project/data/wine_labels.csv"

# Load the CSV files
wine_features = pd.read_csv(features_file)
wine_labels = pd.read_csv(labels_file)

# Combine features and labels into a single DataFrame
if 'id' in wine_features.columns and 'id' in wine_labels.columns:
    wine_data = wine_features.merge(wine_labels, on="id")
else:
    wine_data = pd.concat([wine_features, wine_labels], axis=1)

print(f"Features columns: {wine_features.columns}")
print(f"Labels columns: {wine_labels.columns}")
print(f"Final dataset columns: {wine_data.columns}")

# Connect to the SQLite database
db_path = "/Users/lohithvattikuti/Desktop/Wine_Quality_project/data/wine_data.db"
conn = sqlite3.connect(db_path)

# Optionally drop the table if it exists
drop_table = input("Do you want to drop the 'wine_features' table? (yes/no): ").strip().lower()
if drop_table == "yes":
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS wine_features")
    conn.commit()
    print("Dropped the 'wine_features' table.")

# Write the DataFrame to the SQLite database, replacing the table if it exists
wine_features.to_sql("wine_features", conn, if_exists="replace", index=False)
print("Data has been written to the 'wine_features' table in the SQLite database.")

# Fetch data back from the database and load it into a Pandas DataFrame
query = "SELECT * FROM wine_features;"
retrieved_data = pd.read_sql(query, conn)

print("Data fetched from the database:")
print(retrieved_data.head())

# Close the connection
conn.close()