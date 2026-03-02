import os
import pandas as pd
import sqlite3

# 1. Define Production Paths
CSV_PATH = os.path.join("data", "messy_sales.csv")
DB_PATH = os.path.join("database", "structured_data.db")

def build_sql_database():
    print(f"Loading raw tabular data from: {CSV_PATH}")
    
    # 2. Extract (Load the messy CSV)
    df = pd.read_csv(CSV_PATH)
    print("Raw Data Layout:")
    print(df.head())

    # 3. Transform (Clean the data)
    print("\nExecuting Pandas Data Cleaning...")
    # Fill missing revenue and units with 0 so math operations don't break
    df['revenue'] = df['revenue'].fillna(0)
    df['units_sold'] = df['units_sold'].fillna(0)
    # Fill missing text fields with 'Unknown'
    df['department'] = df['department'].fillna('Unknown')
    
    print("Cleaned Data Layout:")
    print(df.head())

    # 4. Load (Push to SQLite)
    print(f"\nConnecting to local SQLite database at: {DB_PATH}")
    # SQLite is built into Python, no external server required
    conn = sqlite3.connect(DB_PATH)
    
    # Write the Pandas DataFrame directly into a SQL table named 'uploaded_data'
    df.to_sql('uploaded_data', conn, if_exists='replace', index=False)
    
    # Verify the table was created
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM uploaded_data")
    row_count = cursor.fetchone()[0]
    
    conn.close()
    print(f"SUCCESS: SQL database built. {row_count} rows inserted into 'uploaded_data' table.")

if __name__ == "__main__":
    # Ensure database directory exists
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    build_sql_database()